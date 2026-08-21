//! GPU compute runtime for CubeCL via Rust NIFs.
//!
//! Phase 1: CPU-side thread pool simulating GPU execution.
//! Phase 2: Media processing extensions — frame I/O, video/audio kernels,
//!          transcode stubs, and media pipeline types.
//!
//! All GPU state lives in Rust — not in BEAM memory.
//!
//! Buffers are managed via `ResourceArc<Buffer>` so that Rust's `Drop`
//! is called automatically when the Elixir term goes out of scope.
//! No manual buffer_free is needed.

use rustler::{Encoder, Env, Error, NifResult, ResourceArc, Term};
use std::thread;

pub mod ffi;
pub mod kernels;
pub mod kernels_cpu;
pub mod media;
pub mod types;

pub use types::{
    alloc_id, join_finished_threads, Buffer, Command, CommandStatus, DType, FrameType, Pipeline,
    PipelineCommand, AVAILABLE_KERNELS, COMMANDS, NEXT_COMMAND_ID, NEXT_PIPELINE_ID, PIPELINES,
    THREAD_POOL,
};

use kernels_cpu::*;

// ── Atoms ────────────────────────────────────────────────────

mod atoms {
    rustler::atoms! {
        ok,
        error,
        pending,
        running,
        completed,
        failed,
        device_name,
        device_type,
        total_memory,
        compute_units,
        data,
        shape,
        dtype,
        byte_size,
        video,
        audio,
        index,
        codec,
        fps,
        width,
        height,
        sample_rate,
        channels,
        encoder_id,
        radius,
        strength,
        file,
        color,
        threshold,
        brightness,
        contrast,
        mode,
        ratio,
        x,
        y,
        alpha,
        duck_level,
        bands,
        wet,
        room_size,
        handle,
        format,
        pts,
        duration,
        frames,
    }
}

// ── Helpers ──────────────────────────────────────────────────

fn decode_dtype(s: &str) -> NifResult<DType> {
    DType::from_str(s.trim())
        .ok_or_else(|| Error::RaiseTerm(Box::new(format!("unknown dtype: {}", s))))
}

fn decode_shape(term: Term) -> NifResult<Vec<usize>> {
    let raw: Vec<usize> = term.decode()?;
    Ok(raw)
}

#[allow(dead_code)]
fn validate_f32_buffers(
    kernel: &str,
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    expected_inputs: usize,
) -> NifResult<()> {
    if inputs.len() < expected_inputs {
        return Err(Error::RaiseTerm(Box::new(format!(
            "{}: expected {} inputs, got {}",
            kernel,
            expected_inputs,
            inputs.len()
        ))));
    }

    for (idx, input) in inputs.iter().enumerate() {
        if input.dtype != DType::F32 {
            return Err(Error::RaiseTerm(Box::new(format!(
                "{}: input {} must be f32, got {}",
                kernel,
                idx,
                input.dtype.as_str()
            ))));
        }
    }

    if output.dtype != DType::F32 {
        return Err(Error::RaiseTerm(Box::new(format!(
            "{}: output must be f32, got {}",
            kernel,
            output.dtype.as_str()
        ))));
    }

    if !inputs
        .iter()
        .all(|input| input.data().len() == output.data().len())
    {
        return Err(Error::RaiseTerm(Box::new(format!(
            "{}: all input buffers must match output byte size",
            kernel
        ))));
    }

    Ok(())
}

// ── NIF: Device management ───────────────────────────────────

#[rustler::nif]
fn device_info(env: Env) -> NifResult<Term> {
    let map = rustler::types::map::map_new(env);
    let map = map.map_put(
        atoms::device_name().encode(env),
        "CubeCL GPU (CPU fallback — v0.7.0)".encode(env),
    )?;
    let map = map.map_put(atoms::device_type().encode(env), "gpu".encode(env))?;
    let map = map.map_put(
        atoms::total_memory().encode(env),
        (16u64 * 1024 * 1024 * 1024).encode(env),
    )?;
    let map = map.map_put(
        atoms::compute_units().encode(env),
        num_cpus::get().encode(env),
    )?;
    Ok((atoms::ok(), map.encode(env)).encode(env))
}

#[rustler::nif]
fn device_count(env: Env) -> NifResult<Term> {
    Ok((atoms::ok(), 1u32).encode(env))
}

// ── NIF: Buffer management ───────────────────────────────────

#[rustler::nif]
fn buffer_new<'a>(env: Env<'a>, data: Term, shape: Term, dtype_str: Term) -> NifResult<Term<'a>> {
    let data_binary: rustler::Binary = data.decode()?;
    let shape_vec = decode_shape(shape)?;
    let dtype_string: String = dtype_str.decode()?;
    let dtype = decode_dtype(&dtype_string)?;

    let expected_bytes: usize = shape_vec.iter().product::<usize>() * dtype.size_in_bytes();
    let actual_bytes = data_binary.len();

    if actual_bytes != expected_bytes {
        return Ok((
            atoms::error(),
            format!(
                "buffer_new: expected {} bytes (shape={:?}, dtype={}), got {}",
                expected_bytes, shape_vec, dtype_string, actual_bytes
            ),
        )
            .encode(env));
    }

    let buffer = Buffer {
        data: parking_lot::Mutex::new(data_binary.as_slice().to_vec()),
        shape: shape_vec,
        dtype,
    };

    let resource = ResourceArc::<Buffer>::new(buffer);
    Ok((atoms::ok(), resource).encode(env))
}

#[rustler::nif]
fn buffer_read<'a>(env: Env<'a>, buffer: ResourceArc<Buffer>) -> NifResult<Term<'a>> {
    let data = buffer.data();
    let mut out = rustler::OwnedBinary::new(data.len())
        .ok_or_else(|| Error::RaiseTerm(Box::new("buffer_read: allocation failed")))?;
    out.as_mut_slice().copy_from_slice(&data);
    Ok((atoms::ok(), out.release(env)).encode(env))
}

#[rustler::nif]
fn buffer_size(env: Env, buffer: ResourceArc<Buffer>) -> NifResult<Term> {
    Ok((atoms::ok(), buffer.data().len()).encode(env))
}

#[rustler::nif]
fn buffer_shape(env: Env, buffer: ResourceArc<Buffer>) -> NifResult<Term> {
    Ok((atoms::ok(), buffer.shape.clone()).encode(env))
}

#[rustler::nif]
fn buffer_dtype(env: Env, buffer: ResourceArc<Buffer>) -> NifResult<Term> {
    Ok((atoms::ok(), buffer.dtype.as_str()).encode(env))
}

// ── NIF: Kernel execution ────────────────────────────────────

/// Execute a kernel by name with given inputs, output, and JSON params.
/// Returns Ok(()) on success, or Err with a message on failure.
fn execute_kernel(
    name: &str,
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    params_json: &serde_json::Value,
) -> Result<(), String> {
    match name {
        "elementwise_add" => kernel_elementwise_add(inputs, output).map_err(|e| format!("{:?}", e)),
        "elementwise_mul" => kernel_elementwise_mul(inputs, output).map_err(|e| format!("{:?}", e)),
        "elementwise_sub" => kernel_elementwise_sub(inputs, output).map_err(|e| format!("{:?}", e)),
        "elementwise_div" => kernel_elementwise_div(inputs, output).map_err(|e| format!("{:?}", e)),
        "relu" => kernel_relu(inputs, output).map_err(|e| format!("{:?}", e)),
        "sigmoid" => kernel_sigmoid(inputs, output).map_err(|e| format!("{:?}", e)),
        "tanh" => kernel_tanh(inputs, output).map_err(|e| format!("{:?}", e)),
        "yuv_to_rgb" => {
            kernel_yuv_to_rgb(inputs, output, params_json).map_err(|e| format!("{:?}", e))
        }
        "overlay_alpha" => {
            kernel_overlay_alpha(inputs, output, params_json).map_err(|e| format!("{:?}", e))
        }
        "video_mix" => {
            kernel_video_mix(inputs, output, params_json).map_err(|e| format!("{:?}", e))
        }
        "gaussian_blur" => {
            kernel_gaussian_blur(inputs, output, params_json).map_err(|e| format!("{:?}", e))
        }
        "bicubic_scale" => {
            kernel_bicubic_scale(inputs, output, params_json).map_err(|e| format!("{:?}", e))
        }
        "lut_apply" => {
            kernel_lut_apply(inputs, output, params_json).map_err(|e| format!("{:?}", e))
        }
        "chroma_key" => {
            kernel_chroma_key(inputs, output, params_json).map_err(|e| format!("{:?}", e))
        }
        "sharpen" => kernel_sharpen(inputs, output, params_json).map_err(|e| format!("{:?}", e)),
        "brightness_contrast" => {
            kernel_brightness_contrast(inputs, output, params_json).map_err(|e| format!("{:?}", e))
        }
        "denoise" => kernel_denoise(inputs, output, params_json).map_err(|e| format!("{:?}", e)),
        "video_crop" => {
            kernel_video_crop(inputs, output, params_json).map_err(|e| format!("{:?}", e))
        }
        "pcm_mix" => kernel_pcm_mix(inputs, output, params_json).map_err(|e| format!("{:?}", e)),
        "pcm_normalize" => {
            kernel_pcm_normalize(inputs, output, params_json).map_err(|e| format!("{:?}", e))
        }
        "biquad_filter" => {
            kernel_biquad_filter(inputs, output, params_json).map_err(|e| format!("{:?}", e))
        }
        "fft_convolve" => {
            kernel_fft_convolve(inputs, output, params_json).map_err(|e| format!("{:?}", e))
        }
        "resample_linear" => {
            kernel_resample_linear(inputs, output, params_json).map_err(|e| format!("{:?}", e))
        }
        "dynamics_compress" => {
            kernel_dynamics_compress(inputs, output, params_json).map_err(|e| format!("{:?}", e))
        }
        other => Err(format!("unknown kernel '{}'", other)),
    }
}

#[rustler::nif]
fn kernel_run<'a>(
    env: Env<'a>,
    name: Term,
    inputs: Term,
    output: ResourceArc<Buffer>,
    params: Term<'a>,
) -> NifResult<Term<'a>> {
    let name_str: String = name.decode()?;
    let input_resources: Vec<ResourceArc<Buffer>> = inputs.decode()?;

    // Decode params from Erlang-term-encoded binary
    let params_json: serde_json::Value = match params.decode::<rustler::Binary>() {
        Ok(bin) => {
            let bytes = bin.as_slice();
            if bytes.is_empty() {
                serde_json::Value::Object(serde_json::Map::new())
            } else {
                let json_str = std::str::from_utf8(bytes).unwrap_or("{}");
                serde_json::from_str(json_str)
                    .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()))
            }
        }
        Err(_) => serde_json::Value::Object(serde_json::Map::new()),
    };

    if !AVAILABLE_KERNELS.contains(&name_str.as_str()) {
        return Ok((
            atoms::error(),
            format!("kernel_run: unknown kernel '{}'", name_str),
        )
            .encode(env));
    }

    match execute_kernel(&name_str, &input_resources, &output, &params_json) {
        Ok(()) => {
            let cmd_id = alloc_id(&NEXT_COMMAND_ID);
            COMMANDS.insert(
                cmd_id,
                Command {
                    id: cmd_id,
                    status: CommandStatus::Completed,
                },
            );
            Ok((atoms::ok(), cmd_id).encode(env))
        }
        Err(msg) => Ok((atoms::error(), msg).encode(env)),
    }
}

#[rustler::nif]
fn kernel_list(env: Env) -> NifResult<Term> {
    let names: Vec<&str> = AVAILABLE_KERNELS.to_vec();
    Ok((atoms::ok(), names).encode(env))
}

// ── NIF: Async execution ─────────────────────────────────────

#[rustler::nif]
fn submit<'a>(env: Env<'a>, command_json: Term) -> NifResult<Term<'a>> {
    let _cmd_str: String = command_json.decode()?;

    let cmd_id = alloc_id(&NEXT_COMMAND_ID);
    COMMANDS.insert(
        cmd_id,
        Command {
            id: cmd_id,
            status: CommandStatus::Pending,
        },
    );

    let handle = thread::spawn(move || {
        COMMANDS.insert(
            cmd_id,
            Command {
                id: cmd_id,
                status: CommandStatus::Running,
            },
        );
        thread::sleep(std::time::Duration::from_millis(1));
        COMMANDS.insert(
            cmd_id,
            Command {
                id: cmd_id,
                status: CommandStatus::Completed,
            },
        );
    });

    THREAD_POOL.lock().push(handle);
    join_finished_threads();

    Ok((atoms::ok(), cmd_id).encode(env))
}

#[rustler::nif]
fn poll(env: Env, command_id: u64) -> NifResult<Term> {
    match COMMANDS.get(&command_id) {
        Some(cmd) => {
            let status_atom = match &cmd.status {
                CommandStatus::Pending => atoms::pending(),
                CommandStatus::Running => atoms::running(),
                CommandStatus::Completed => atoms::completed(),
                CommandStatus::Failed(_) => atoms::failed(),
            };
            Ok((atoms::ok(), status_atom).encode(env))
        }
        None => Ok((
            atoms::error(),
            format!("poll: invalid command_id {}", command_id),
        )
            .encode(env)),
    }
}

#[rustler::nif]
fn wait(env: Env, command_id: u64) -> NifResult<Term> {
    loop {
        match COMMANDS.get(&command_id) {
            Some(cmd) => match &cmd.status {
                CommandStatus::Completed => return Ok((atoms::ok()).encode(env)),
                CommandStatus::Failed(msg) => {
                    return Ok((atoms::error(), msg.clone()).encode(env));
                }
                _ => {
                    thread::sleep(std::time::Duration::from_millis(1));
                }
            },
            None => {
                return Ok((
                    atoms::error(),
                    format!("wait: invalid command_id {}", command_id),
                )
                    .encode(env));
            }
        }
    }
}

// ── NIF: Pipeline orchestration ──────────────────────────────

#[rustler::nif]
fn pipeline_new(env: Env) -> NifResult<Term> {
    let id = alloc_id(&NEXT_PIPELINE_ID);
    PIPELINES.insert(
        id,
        Pipeline {
            id,
            commands: Vec::new(),
        },
    );
    Ok((atoms::ok(), id).encode(env))
}

#[rustler::nif]
fn pipeline_add<'a>(
    env: Env<'a>,
    pipeline_id: u64,
    name: Term,
    inputs: Term,
    output: ResourceArc<Buffer>,
    params: Term<'a>,
) -> NifResult<Term<'a>> {
    let name_str: String = name.decode()?;
    let input_resources: Vec<ResourceArc<Buffer>> = inputs.decode()?;

    // Encode params as JSON bytes for storage
    let params_bytes: Vec<u8> = match params.decode::<rustler::Binary>() {
        Ok(bin) => bin.as_slice().to_vec(),
        Err(_) => b"{}".to_vec(),
    };

    match PIPELINES.get_mut(&pipeline_id) {
        Some(mut pipeline) => {
            pipeline.commands.push(PipelineCommand::KernelRun {
                name: name_str,
                inputs: input_resources,
                output,
                params: params_bytes,
            });
            Ok((atoms::ok()).encode(env))
        }
        None => Ok((
            atoms::error(),
            format!("pipeline_add: invalid pipeline_id {}", pipeline_id),
        )
            .encode(env)),
    }
}

#[rustler::nif]
fn pipeline_run(env: Env, pipeline_id: u64) -> NifResult<Term> {
    let pipeline = match PIPELINES.get(&pipeline_id) {
        Some(p) => p.clone(),
        None => {
            return Ok((
                atoms::error(),
                format!("pipeline_run: invalid pipeline_id {}", pipeline_id),
            )
                .encode(env));
        }
    };

    let mut cmd_ids = Vec::new();

    for cmd in &pipeline.commands {
        match cmd {
            PipelineCommand::KernelRun {
                name,
                inputs,
                output,
                params,
            } => {
                if !AVAILABLE_KERNELS.contains(&name.as_str()) {
                    return Ok((
                        atoms::error(),
                        format!("pipeline_run: unknown kernel '{}'", name),
                    )
                        .encode(env));
                }

                // Decode params from stored JSON bytes
                let params_json: serde_json::Value = if params.is_empty() {
                    serde_json::Value::Object(serde_json::Map::new())
                } else {
                    let json_str = std::str::from_utf8(params).unwrap_or("{}");
                    serde_json::from_str(json_str)
                        .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()))
                };

                // Execute the kernel
                match execute_kernel(name, inputs, output, &params_json) {
                    Ok(()) => {
                        let cmd_id = alloc_id(&NEXT_COMMAND_ID);
                        COMMANDS.insert(
                            cmd_id,
                            Command {
                                id: cmd_id,
                                status: CommandStatus::Completed,
                            },
                        );
                        cmd_ids.push(cmd_id);
                    }
                    Err(msg) => {
                        return Ok((atoms::error(), msg).encode(env));
                    }
                }
            }
            PipelineCommand::ReadFrame { .. } => {
                let cmd_id = alloc_id(&NEXT_COMMAND_ID);
                COMMANDS.insert(
                    cmd_id,
                    Command {
                        id: cmd_id,
                        status: CommandStatus::Completed,
                    },
                );
                cmd_ids.push(cmd_id);
            }
            PipelineCommand::Filter {
                kernel,
                input,
                output,
                params,
            } => {
                if !AVAILABLE_KERNELS.contains(&kernel.as_str()) {
                    return Ok((
                        atoms::error(),
                        format!("pipeline_run: unknown filter kernel '{}'", kernel),
                    )
                        .encode(env));
                }

                let params_json: serde_json::Value = if params.is_empty() {
                    serde_json::Value::Object(serde_json::Map::new())
                } else {
                    let json_str = std::str::from_utf8(params).unwrap_or("{}");
                    serde_json::from_str(json_str)
                        .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()))
                };

                match execute_kernel(kernel, std::slice::from_ref(input), output, &params_json) {
                    Ok(()) => {
                        let cmd_id = alloc_id(&NEXT_COMMAND_ID);
                        COMMANDS.insert(
                            cmd_id,
                            Command {
                                id: cmd_id,
                                status: CommandStatus::Completed,
                            },
                        );
                        cmd_ids.push(cmd_id);
                    }
                    Err(msg) => {
                        return Ok((atoms::error(), msg).encode(env));
                    }
                }
            }
            PipelineCommand::Overlay {
                base,
                layer,
                output,
                params,
            } => {
                let params_json: serde_json::Value = if params.is_empty() {
                    serde_json::Value::Object(serde_json::Map::new())
                } else {
                    let json_str = std::str::from_utf8(params).unwrap_or("{}");
                    serde_json::from_str(json_str)
                        .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()))
                };

                match execute_kernel(
                    "overlay_alpha",
                    &[base.clone(), layer.clone()],
                    output,
                    &params_json,
                ) {
                    Ok(()) => {
                        let cmd_id = alloc_id(&NEXT_COMMAND_ID);
                        COMMANDS.insert(
                            cmd_id,
                            Command {
                                id: cmd_id,
                                status: CommandStatus::Completed,
                            },
                        );
                        cmd_ids.push(cmd_id);
                    }
                    Err(msg) => {
                        return Ok((atoms::error(), msg).encode(env));
                    }
                }
            }
            PipelineCommand::Encode { .. } => {
                let cmd_id = alloc_id(&NEXT_COMMAND_ID);
                COMMANDS.insert(
                    cmd_id,
                    Command {
                        id: cmd_id,
                        status: CommandStatus::Completed,
                    },
                );
                cmd_ids.push(cmd_id);
            }
        }
    }

    Ok((atoms::ok(), cmd_ids).encode(env))
}

#[rustler::nif]
fn pipeline_free(env: Env, pipeline_id: u64) -> NifResult<Term> {
    match PIPELINES.remove(&pipeline_id) {
        Some(_) => Ok((atoms::ok()).encode(env)),
        None => Ok((
            atoms::error(),
            format!("pipeline_free: invalid pipeline_id {}", pipeline_id),
        )
            .encode(env)),
    }
}

// ── NIF module registration ─────────────────────────────────

fn on_load(env: Env, _info: Term) -> bool {
    if env.register::<Buffer>().is_err() {
        return false;
    }
    if env.register::<media::MediaSource>().is_err() {
        return false;
    }
    if env.register::<media::Transcoder>().is_err() {
        return false;
    }

    let mut pool = THREAD_POOL.lock();
    for handle in pool.drain(..) {
        let _ = handle.join();
    }
    true
}

// ── Re-export media NIFs so they're registered ──────────────

#[rustler::nif]
fn media_open<'a>(env: Env<'a>, path: Term) -> NifResult<Term<'a>> {
    media::nif_media_open(env, path)
}

#[rustler::nif]
fn media_streams(env: Env, source: ResourceArc<media::MediaSource>) -> NifResult<Term> {
    media::nif_media_streams(env, source)
}

#[rustler::nif]
fn media_read_video_frame<'a>(
    env: Env<'a>,
    source: ResourceArc<media::MediaSource>,
) -> NifResult<Term<'a>> {
    media::nif_media_read_video_frame(env, source)
}

#[rustler::nif]
fn media_read_audio_samples<'a>(
    env: Env<'a>,
    source: ResourceArc<media::MediaSource>,
) -> NifResult<Term<'a>> {
    media::nif_media_read_audio_samples(env, source)
}

#[rustler::nif]
fn media_close(env: Env, source: ResourceArc<media::MediaSource>) -> NifResult<Term> {
    media::nif_media_close(env, source)
}

#[rustler::nif]
fn transcode_start<'a>(env: Env<'a>, path: Term, opts: Term) -> NifResult<Term<'a>> {
    media::nif_transcode_start(env, path, opts)
}

#[rustler::nif]
fn transcode_write_video(
    env: Env,
    encoder: ResourceArc<media::Transcoder>,
    frame: ResourceArc<Buffer>,
) -> NifResult<Term> {
    media::nif_transcode_write_video(env, encoder, frame)
}

#[rustler::nif]
fn transcode_write_audio(
    env: Env,
    encoder: ResourceArc<media::Transcoder>,
    samples: ResourceArc<Buffer>,
) -> NifResult<Term> {
    media::nif_transcode_write_audio(env, encoder, samples)
}

#[rustler::nif]
fn transcode_finish(env: Env, encoder: ResourceArc<media::Transcoder>) -> NifResult<Term> {
    media::nif_transcode_finish(env, encoder)
}

rustler::init!("Elixir.ExCubecl.NIF", load = on_load);

// ── Unit tests ──────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_size_in_bytes() {
        assert_eq!(DType::F32.size_in_bytes(), 4);
        assert_eq!(DType::F64.size_in_bytes(), 8);
        assert_eq!(DType::S32.size_in_bytes(), 4);
        assert_eq!(DType::S64.size_in_bytes(), 8);
        assert_eq!(DType::U32.size_in_bytes(), 4);
        assert_eq!(DType::U8.size_in_bytes(), 1);
    }

    #[test]
    fn dtype_as_str() {
        assert_eq!(DType::F32.as_str(), "f32");
        assert_eq!(DType::F64.as_str(), "f64");
        assert_eq!(DType::U8.as_str(), "u8");
    }

    #[test]
    fn dtype_from_str_valid() {
        assert_eq!(DType::from_str("f32"), Some(DType::F32));
        assert_eq!(DType::from_str("f64"), Some(DType::F64));
        assert_eq!(DType::from_str("s32"), Some(DType::S32));
        assert_eq!(DType::from_str("s64"), Some(DType::S64));
        assert_eq!(DType::from_str("u32"), Some(DType::U32));
        assert_eq!(DType::from_str("u8"), Some(DType::U8));
    }

    #[test]
    fn dtype_from_str_invalid() {
        assert_eq!(DType::from_str("f16"), None);
        assert_eq!(DType::from_str(""), None);
        assert_eq!(DType::from_str("invalid"), None);
    }

    #[test]
    fn available_kernels_contains_core() {
        assert!(AVAILABLE_KERNELS.contains(&"elementwise_add"));
        assert!(AVAILABLE_KERNELS.contains(&"relu"));
        assert!(AVAILABLE_KERNELS.contains(&"yuv_to_rgb"));
        assert!(AVAILABLE_KERNELS.contains(&"pcm_mix"));
    }

    #[test]
    fn available_kernels_count() {
        // 7 compute + 11 video + 6 audio = 24
        assert_eq!(AVAILABLE_KERNELS.len(), 24);
    }

    #[test]
    fn command_status_roundtrip() {
        let pending = CommandStatus::Pending;
        let running = CommandStatus::Running;
        let completed = CommandStatus::Completed;
        let failed = CommandStatus::Failed("test error".to_string());

        assert_eq!(pending, CommandStatus::Pending);
        assert_eq!(running, CommandStatus::Running);
        assert_eq!(completed, CommandStatus::Completed);
        assert_eq!(failed, CommandStatus::Failed("test error".to_string()));
    }

    #[test]
    fn alloc_id_increments() {
        let counter = std::sync::atomic::AtomicU64::new(1);
        let id1 = alloc_id(&counter);
        let id2 = alloc_id(&counter);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }
}
