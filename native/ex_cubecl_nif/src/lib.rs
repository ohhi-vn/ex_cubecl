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

use dashmap::DashMap;
use parking_lot::Mutex;
use rustler::{Encoder, Env, Error, NifResult, ResourceArc, Term};
// Explicit NIF function list is required by rustler 0.37
#[allow(deprecated)]
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

pub mod ffi;
pub mod kernels;
pub mod media;

// ── DType ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F64,
    S32,
    S64,
    U32,
    U8,
}

impl DType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "f32" => Some(DType::F32),
            "f64" => Some(DType::F64),
            "s32" => Some(DType::S32),
            "s64" => Some(DType::S64),
            "u32" => Some(DType::U32),
            "u8" => Some(DType::U8),
            _ => None,
        }
    }

    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::S32 => 4,
            DType::S64 => 8,
            DType::U32 => 4,
            DType::U8 => 1,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::S32 => "s32",
            DType::S64 => "s64",
            DType::U32 => "u32",
            DType::U8 => "u8",
        }
    }
}

// ── Buffer resource ───────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Buffer {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl Buffer {
    pub fn byte_size(&self) -> usize {
        self.data.len()
    }
}

impl rustler::Resource for Buffer {}

// ── Command / Async ───────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandStatus {
    Pending,
    Running,
    Completed,
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct Command {
    pub id: u64,
    pub status: CommandStatus,
}

// ── Pipeline ──────────────────────────────────────────────────

#[derive(Clone)]
pub struct Pipeline {
    pub id: u64,
    pub commands: Vec<PipelineCommand>,
}

#[derive(Clone)]
pub enum PipelineCommand {
    KernelRun {
        name: String,
        inputs: Vec<ResourceArc<Buffer>>,
        output: ResourceArc<Buffer>,
        params: Vec<u8>,
    },
    /// Phase 2: Read a decoded video/audio frame from a media source.
    ReadFrame {
        source_id: u64,
        stream_type: FrameType,
    },
    /// Phase 2: Apply a GPU filter kernel to a frame.
    Filter {
        kernel: String,
        input: ResourceArc<Buffer>,
        output: ResourceArc<Buffer>,
        params: Vec<u8>,
    },
    /// Phase 2: Overlay one frame onto another (alpha composite).
    Overlay {
        base: ResourceArc<Buffer>,
        layer: ResourceArc<Buffer>,
        output: ResourceArc<Buffer>,
        params: Vec<u8>,
    },
    /// Phase 2: Write a frame/sample to a transcoder.
    Encode {
        encoder_id: u64,
        frame: ResourceArc<Buffer>,
    },
}

/// Distinguishes video frames from audio samples in pipeline commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    Video,
    Audio,
}

// ── Global state ──────────────────────────────────────────────

lazy_static::lazy_static! {
    /// Command store: command_id -> Command
    static ref COMMANDS: DashMap<u64, Command> = DashMap::new();

    /// Pipeline store: pipeline_id -> Pipeline
    static ref PIPELINES: DashMap<u64, Pipeline> = DashMap::new();

    /// Monotonic ID generators
    static ref NEXT_COMMAND_ID: AtomicU64 = AtomicU64::new(1);
    static ref NEXT_PIPELINE_ID: AtomicU64 = AtomicU64::new(1);

    /// Thread pool for CPU-side work (simulating GPU)
    static ref THREAD_POOL: Mutex<Vec<thread::JoinHandle<()>>> = Mutex::new(Vec::new());
}

fn alloc_id(counter: &AtomicU64) -> u64 {
    counter.fetch_add(1, Ordering::SeqCst)
}

// ── Available kernels (Phase 1 + Phase 2) ────────────────────

const AVAILABLE_KERNELS: &[&str] = &[
    // Phase 1 — compute
    "elementwise_add",
    "elementwise_mul",
    "elementwise_sub",
    "elementwise_div",
    "relu",
    "sigmoid",
    "tanh",
    "matmul",
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "softmax",
    "layer_norm",
    "conv2d",
    "transpose",
    "reshape",
    // Phase 2 — video
    "yuv_to_rgb",
    "overlay_alpha",
    "video_mix",
    "gaussian_blur",
    "bicubic_scale",
    "lut_apply",
    "chroma_key",
    "sharpen",
    // Phase 2 — audio
    "pcm_mix",
    "pcm_normalize",
    "biquad_filter",
    "fft_convolve",
    "resample_linear",
    "dynamics_compress",
];

// ── Atoms ────────────────────────────────────────────────────

mod atoms {
    rustler::atoms! {
        ok,
        error,
        pending,
        running,
        completed,
        failed,
        // device info keys
        device_name,
        device_type,
        total_memory,
        compute_units,
        // buffer info
        data,
        shape,
        dtype,
        byte_size,
        // Phase 2 — media
        video,
        audio,
        // Phase 2 — stream info
        index,
        codec,
        fps,
        width,
        height,
        sample_rate,
        channels,
        // Phase 2 — transcode
        encoder_id,
        // Phase 2 — filter params
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
        // Phase 2 — video frame / audio samples
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

// ── NIF: Device management ───────────────────────────────────

#[rustler::nif]
fn device_info(env: Env) -> NifResult<Term> {
    let map = rustler::types::map::map_new(env);
    let map = map.map_put(
        atoms::device_name().encode(env),
        "CubeCL GPU (Phase 2 — media extensions)".encode(env),
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
        return Err(Error::RaiseTerm(Box::new(format!(
            "buffer_new: expected {} bytes (shape={:?}, dtype={}), got {}",
            expected_bytes, shape_vec, dtype_string, actual_bytes
        ))));
    }

    let buffer = Buffer {
        data: data_binary.as_slice().to_vec(),
        shape: shape_vec,
        dtype,
    };

    let resource = ResourceArc::<Buffer>::new(buffer);
    Ok((atoms::ok(), resource).encode(env))
}

#[rustler::nif]
fn buffer_read<'a>(env: Env<'a>, buffer: ResourceArc<Buffer>) -> NifResult<Term<'a>> {
    let mut out = rustler::OwnedBinary::new(buffer.data.len())
        .ok_or_else(|| Error::RaiseTerm(Box::new("buffer_read: allocation failed")))?;
    out.as_mut_slice().copy_from_slice(&buffer.data);
    Ok((atoms::ok(), out.release(env)).encode(env))
}

#[rustler::nif]
fn buffer_size(env: Env, buffer: ResourceArc<Buffer>) -> NifResult<Term> {
    Ok((atoms::ok(), buffer.byte_size()).encode(env))
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

#[rustler::nif]
fn kernel_run<'a>(
    env: Env<'a>,
    name: Term,
    inputs: Term,
    output: ResourceArc<Buffer>,
    params: Term,
) -> NifResult<Term<'a>> {
    let name_str: String = name.decode()?;
    let input_resources: Vec<ResourceArc<Buffer>> = inputs.decode()?;
    let _params_binary: rustler::Binary = params.decode()?;

    if !AVAILABLE_KERNELS.contains(&name_str.as_str()) {
        return Err(Error::RaiseTerm(Box::new(format!(
            "kernel_run: unknown kernel '{}'",
            name_str
        ))));
    }

    let cmd_id = alloc_id(&NEXT_COMMAND_ID);
    COMMANDS.insert(
        cmd_id,
        Command {
            id: cmd_id,
            status: CommandStatus::Completed,
        },
    );

    // Phase 2 stub: no-op. Real implementation dispatches to CubeCL GPU kernels.
    let _ = (input_resources, output, name_str);

    Ok((atoms::ok(), cmd_id).encode(env))
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

    let cmd_id_clone = cmd_id;
    let handle = thread::spawn(move || {
        COMMANDS.insert(
            cmd_id_clone,
            Command {
                id: cmd_id_clone,
                status: CommandStatus::Running,
            },
        );
        thread::sleep(std::time::Duration::from_millis(1));
        COMMANDS.insert(
            cmd_id_clone,
            Command {
                id: cmd_id_clone,
                status: CommandStatus::Completed,
            },
        );
    });

    THREAD_POOL.lock().push(handle);

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
                CommandStatus::Failed(ref msg) => {
                    return Ok((atoms::error(), msg.clone()).encode(env));
                }
            };
            Ok((atoms::ok(), status_atom).encode(env))
        }
        None => Err(Error::RaiseTerm(Box::new(format!(
            "poll: invalid command_id {}",
            command_id
        )))),
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
                return Err(Error::RaiseTerm(Box::new(format!(
                    "wait: invalid command_id {}",
                    command_id
                ))));
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
    params: Term,
) -> NifResult<Term<'a>> {
    let name_str: String = name.decode()?;
    let input_resources: Vec<ResourceArc<Buffer>> = inputs.decode()?;
    let _params_binary: rustler::Binary = params.decode()?;

    match PIPELINES.get_mut(&pipeline_id) {
        Some(mut pipeline) => {
            pipeline.commands.push(PipelineCommand::KernelRun {
                name: name_str,
                inputs: input_resources,
                output,
                params: Vec::new(),
            });
            Ok((atoms::ok()).encode(env))
        }
        None => Err(Error::RaiseTerm(Box::new(format!(
            "pipeline_add: invalid pipeline_id {}",
            pipeline_id
        )))),
    }
}

#[rustler::nif]
fn pipeline_run(env: Env, pipeline_id: u64) -> NifResult<Term> {
    let pipeline = match PIPELINES.get(&pipeline_id) {
        Some(p) => p.clone(),
        None => {
            return Err(Error::RaiseTerm(Box::new(format!(
                "pipeline_run: invalid pipeline_id {}",
                pipeline_id
            ))));
        }
    };

    let mut cmd_ids = Vec::new();

    for cmd in &pipeline.commands {
        match cmd {
            PipelineCommand::KernelRun {
                name,
                inputs,
                output,
                params: _params,
            } => {
                if !AVAILABLE_KERNELS.contains(&name.as_str()) {
                    return Err(Error::RaiseTerm(Box::new(format!(
                        "pipeline_run: unknown kernel '{}'",
                        name
                    ))));
                }

                let cmd_id = alloc_id(&NEXT_COMMAND_ID);
                COMMANDS.insert(
                    cmd_id,
                    Command {
                        id: cmd_id,
                        status: CommandStatus::Completed,
                    },
                );

                if let Some(first) = inputs.first() {
                    let _copy_len = output.data.len().min(first.data.len());
                }

                cmd_ids.push(cmd_id);
            }
            // Phase 2: media pipeline commands are validated but no-op in stub.
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
            PipelineCommand::Filter { kernel, .. } => {
                if !AVAILABLE_KERNELS.contains(&kernel.as_str()) {
                    return Err(Error::RaiseTerm(Box::new(format!(
                        "pipeline_run: unknown filter kernel '{}'",
                        kernel
                    ))));
                }
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
            PipelineCommand::Overlay { .. } => {
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
        None => Err(Error::RaiseTerm(Box::new(format!(
            "pipeline_free: invalid pipeline_id {}",
            pipeline_id
        )))),
    }
}

// ── NIF module registration ─────────────────────────────────

fn on_load(env: Env, _info: Term) -> bool {
    if env.register::<Buffer>().is_err() {
        return false;
    }
    // Register media resource types
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

/// Delegates to `media::media_open` etc. so all NIF functions live in `lib.rs`.
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

rustler::init!(
    "Elixir.ExCubecl.NIF",
    [
        // Device
        device_info,
        device_count,
        // Buffers
        buffer_new,
        buffer_read,
        buffer_size,
        buffer_shape,
        buffer_dtype,
        // Kernels
        kernel_run,
        kernel_list,
        // Async
        submit,
        poll,
        wait,
        // Pipelines
        pipeline_new,
        pipeline_add,
        pipeline_run,
        pipeline_free,
        // Media I/O
        media_open,
        media_streams,
        media_read_video_frame,
        media_read_audio_samples,
        media_close,
        // Transcode
        transcode_start,
        transcode_write_video,
        transcode_write_audio,
        transcode_finish,
    ],
    load = on_load
);
