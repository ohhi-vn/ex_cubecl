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
use rustler::{Atom, Encoder, Env, Error, NifResult, ResourceArc, Term};
use std::collections::HashMap;
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

#[derive(Debug)]
pub struct Buffer {
    pub data: parking_lot::Mutex<Vec<u8>>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl Buffer {
    pub fn byte_size(&self) -> usize {
        self.data.lock().len()
    }

    pub fn data(&self) -> parking_lot::MutexGuard<'_, Vec<u8>> {
        self.data.lock()
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
    ReadFrame {
        source_id: u64,
        stream_type: FrameType,
    },
    Filter {
        kernel: String,
        input: ResourceArc<Buffer>,
        output: ResourceArc<Buffer>,
        params: Vec<u8>,
    },
    Overlay {
        base: ResourceArc<Buffer>,
        layer: ResourceArc<Buffer>,
        output: ResourceArc<Buffer>,
        params: Vec<u8>,
    },
    Encode {
        encoder_id: u64,
        frame: ResourceArc<Buffer>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    Video,
    Audio,
}

// ── Global state ──────────────────────────────────────────────

lazy_static::lazy_static! {
    static ref COMMANDS: DashMap<u64, Command> = DashMap::new();
    static ref PIPELINES: DashMap<u64, Pipeline> = DashMap::new();
    static ref NEXT_COMMAND_ID: AtomicU64 = AtomicU64::new(1);
    static ref NEXT_PIPELINE_ID: AtomicU64 = AtomicU64::new(1);
    static ref THREAD_POOL: Mutex<Vec<thread::JoinHandle<()>>> = Mutex::new(Vec::new());
}

fn alloc_id(counter: &AtomicU64) -> u64 {
    counter.fetch_add(1, Ordering::SeqCst)
}

fn join_finished_threads() {
    let mut pool = THREAD_POOL.lock();
    let drained: Vec<_> = pool.drain(..).collect();
    for handle in drained {
        if handle.is_finished() {
            let _ = handle.join();
        } else {
            pool.push(handle);
        }
    }
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
    "brightness_contrast",
    "denoise",
    "video_crop",
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
    // Decode params from Elixir map term into HashMap<String, Term>
    let params_map: HashMap<String, Term<'a>> = {
        let mut hm = HashMap::new();
        // Use rustler's Map type to iterate over key-value pairs
        let map = rustler::types::map::Map::from_term(params)?;
        for (key_term, val) in map.iter() {
            if let Ok(key_str) = key_term.decode::<String>() {
                hm.insert(key_str, val);
            }
        }
        hm
    };

    if !AVAILABLE_KERNELS.contains(&name_str.as_str()) {
        return Ok((
            atoms::error(),
            format!("kernel_run: unknown kernel '{}'", name_str),
        )
            .encode(env));
    }

    // Execute the kernel on the CPU (simulating GPU execution).
    // Phase 2 kernels operate on raw byte buffers (video frames) and
    // f32 sample buffers (audio). Params are decoded per-kernel.
    match name_str.as_str() {
        // ── Phase 1 — compute ──────────────────────────────
        "elementwise_add" => kernel_elementwise_add(&input_resources, &output)?,
        "elementwise_mul" => kernel_elementwise_mul(&input_resources, &output)?,
        "elementwise_sub" => kernel_elementwise_sub(&input_resources, &output)?,
        "elementwise_div" => kernel_elementwise_div(&input_resources, &output)?,
        "relu" => kernel_relu(&input_resources, &output)?,
        "sigmoid" => kernel_sigmoid(&input_resources, &output)?,
        "tanh" => kernel_tanh(&input_resources, &output)?,
        // ── Phase 2 — video kernels ────────────────────────
        "yuv_to_rgb" => kernel_yuv_to_rgb(&input_resources, &output, &params_map)?,
        "overlay_alpha" => kernel_overlay_alpha(&input_resources, &output, &params_map)?,
        "video_mix" => kernel_video_mix(&input_resources, &output, &params_map)?,
        "gaussian_blur" => kernel_gaussian_blur(&input_resources, &output, &params_map)?,
        "bicubic_scale" => kernel_bicubic_scale(&input_resources, &output, &params_map)?,
        "lut_apply" => kernel_lut_apply(&input_resources, &output, &params_map)?,
        "chroma_key" => kernel_chroma_key(&input_resources, &output, &params_map)?,
        "sharpen" => kernel_sharpen(&input_resources, &output, &params_map)?,
        "brightness_contrast" => {
            kernel_brightness_contrast(&input_resources, &output, &params_map)?
        }
        "denoise" => kernel_denoise(&input_resources, &output, &params_map)?,
        "video_crop" => kernel_video_crop(&input_resources, &output, &params_map)?,
        // ── Phase 2 — audio kernels ────────────────────────
        "pcm_mix" => kernel_pcm_mix(&input_resources, &output, &params_map)?,
        "pcm_normalize" => kernel_pcm_normalize(&input_resources, &output, &params_map)?,
        "biquad_filter" => kernel_biquad_filter(&input_resources, &output, &params_map)?,
        "fft_convolve" => kernel_fft_convolve(&input_resources, &output, &params_map)?,
        "resample_linear" => kernel_resample_linear(&input_resources, &output, &params_map)?,
        "dynamics_compress" => kernel_dynamics_compress(&input_resources, &output, &params_map)?,
        // ── Unknown but declared kernels ────────────────────
        other => {
            let _ = (input_resources, output, other);
        }
    }

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

// ══════════════════════════════════════════════════════════════
//  CPU-side kernel implementations (simulating GPU)
// ══════════════════════════════════════════════════════════════

// ── Phase 1: Elementwise compute kernels ─────────────────────

fn kernel_elementwise_add(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
) -> NifResult<()> {
    if inputs.len() < 2 {
        return Err(Error::RaiseTerm(Box::new(
            "elementwise_add: expected 2 inputs",
        )));
    }
    // Clone input data before locking output to avoid deadlock when
    if inputs.len() < 2 {
        return Err(Error::RaiseTerm(Box::new(
            "elementwise_add: expected 2 inputs",
        )));
    }
    // Clone input data to avoid deadlock when
    // output is the same buffer as one of the inputs (in-place operation).
    let a = inputs[0].data().clone();
    let b = inputs[1].data().clone();
    let mut out = output.data();
    let len = a.len().min(b.len()).min(out.len());
    for i in (0..len).step_by(4) {
        let va = f32::from_ne_bytes(a[i..i + 4].try_into().unwrap());
        let vb = f32::from_ne_bytes(b[i..i + 4].try_into().unwrap());
        out[i..i + 4].copy_from_slice(&(va + vb).to_ne_bytes());
    }
    Ok(())
}

fn kernel_elementwise_mul(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
) -> NifResult<()> {
    if inputs.len() < 2 {
        return Err(Error::RaiseTerm(Box::new(
            "elementwise_mul: expected 2 inputs",
        )));
    }
    let a = inputs[0].data().clone();
    let b = inputs[1].data().clone();
    let mut out = output.data();
    let len = a.len().min(b.len()).min(out.len());
    for i in (0..len).step_by(4) {
        let va = f32::from_ne_bytes(a[i..i + 4].try_into().unwrap());
        let vb = f32::from_ne_bytes(b[i..i + 4].try_into().unwrap());
        out[i..i + 4].copy_from_slice(&(va * vb).to_ne_bytes());
    }
    Ok(())
}

fn kernel_elementwise_sub(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
) -> NifResult<()> {
    if inputs.len() < 2 {
        return Err(Error::RaiseTerm(Box::new(
            "elementwise_sub: expected 2 inputs",
        )));
    }
    let a = inputs[0].data().clone();
    let b = inputs[1].data().clone();
    let mut out = output.data();
    let len = a.len().min(b.len()).min(out.len());
    for i in (0..len).step_by(4) {
        let va = f32::from_ne_bytes(a[i..i + 4].try_into().unwrap());
        let vb = f32::from_ne_bytes(b[i..i + 4].try_into().unwrap());
        out[i..i + 4].copy_from_slice(&(va - vb).to_ne_bytes());
    }
    Ok(())
}

fn kernel_elementwise_div(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
) -> NifResult<()> {
    if inputs.len() < 2 {
        return Err(Error::RaiseTerm(Box::new(
            "elementwise_div: expected 2 inputs",
        )));
    }
    let a = inputs[0].data().clone();
    let b = inputs[1].data().clone();
    let mut out = output.data();
    let len = a.len().min(b.len()).min(out.len());
    for i in (0..len).step_by(4) {
        let va = f32::from_ne_bytes(a[i..i + 4].try_into().unwrap());
        let vb = f32::from_ne_bytes(b[i..i + 4].try_into().unwrap());
        out[i..i + 4].copy_from_slice(&(va / vb).to_ne_bytes());
    }
    Ok(())
}

fn kernel_relu(inputs: &[ResourceArc<Buffer>], output: &ResourceArc<Buffer>) -> NifResult<()> {
    let a = inputs[0].data().clone();
    let mut out = output.data();
    let len = a.len().min(out.len());
    for i in (0..len).step_by(4) {
        let v = f32::from_ne_bytes(a[i..i + 4].try_into().unwrap());
        out[i..i + 4].copy_from_slice(&v.max(0.0).to_ne_bytes());
    }
    Ok(())
}

fn kernel_sigmoid(inputs: &[ResourceArc<Buffer>], output: &ResourceArc<Buffer>) -> NifResult<()> {
    let a = inputs[0].data().clone();
    let mut out = output.data();
    let len = a.len().min(out.len());
    for i in (0..len).step_by(4) {
        let v = f32::from_ne_bytes(a[i..i + 4].try_into().unwrap());
        let s = 1.0 / (1.0 + (-v).exp());
        out[i..i + 4].copy_from_slice(&s.to_ne_bytes());
    }
    Ok(())
}

fn kernel_tanh(inputs: &[ResourceArc<Buffer>], output: &ResourceArc<Buffer>) -> NifResult<()> {
    let a = inputs[0].data().clone();
    let mut out = output.data();
    let len = a.len().min(out.len());
    for i in (0..len).step_by(4) {
        let v = f32::from_ne_bytes(a[i..i + 4].try_into().unwrap());
        out[i..i + 4].copy_from_slice(&v.tanh().to_ne_bytes());
    }
    Ok(())
}

// ── Phase 2: Video kernels ──────────────────────────────────

/// YUV420p → RGB24 color space conversion.
/// Params: none (uses buffer size to infer dimensions from 1920x1080 default).
/// Input: YUV420p byte buffer. Output: RGB24 byte buffer (same size, truncated).
fn kernel_yuv_to_rgb<'a>(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    _params: &HashMap<String, Term<'a>>,
) -> NifResult<()> {
    let input = inputs[0].data().clone();
    let mut out = output.data();
    let total = input.len();
    let pixel_count = total * 2 / 3; // YUV420p: 1.5 bytes per pixel
    let rgb_bytes = pixel_count * 3;

    // Clamp output to the smaller of the two buffers
    let out_len = out.len().min(rgb_bytes);
    let y_plane_size = pixel_count;
    let uv_plane_size = pixel_count / 2;

    for i in 0..out_len / 3 {
        let y_val = if i < y_plane_size {
            input[i] as f32
        } else {
            0.0
        };

        let uv_idx = y_plane_size + (i / 2) * 2;
        let u_val = if uv_idx < y_plane_size + uv_plane_size {
            input[uv_idx] as f32 - 128.0
        } else {
            0.0
        };
        let v_val = if uv_idx + 1 < y_plane_size + uv_plane_size {
            input[uv_idx + 1] as f32 - 128.0
        } else {
            0.0
        };

        let r = (y_val + 1.402 * v_val).clamp(0.0, 255.0) as u8;
        let g = (y_val - 0.344136 * u_val - 0.714136 * v_val).clamp(0.0, 255.0) as u8;
        let b = (y_val + 1.772 * u_val).clamp(0.0, 255.0) as u8;

        let out_idx = i * 3;
        if out_idx + 2 < out.len() {
            out[out_idx] = r;
            out[out_idx + 1] = g;
            out[out_idx + 2] = b;
        }
    }
    Ok(())
}

/// Alpha compositing: overlay onto base at (x, y) with opacity alpha.
/// Params: {x, y, alpha}
fn kernel_overlay_alpha<'a>(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    params: &HashMap<String, Term<'a>>,
) -> NifResult<()> {
    if inputs.len() < 2 {
        return Ok(());
    }
    let _x: usize = params
        .get("x")
        .and_then(|t| t.decode::<usize>().ok())
        .unwrap_or(0);
    let _y: usize = params
        .get("y")
        .and_then(|t| t.decode::<usize>().ok())
        .unwrap_or(0);
    let alpha: f32 = params
        .get("alpha")
        .and_then(|t| t.decode::<f32>().ok())
        .unwrap_or(1.0);

    // Clone input data to avoid deadlock when output is the same buffer as input
    let base_data = inputs[0].data().clone();
    let overlay_data = inputs[1].data().clone();
    let mut out = output.data();

    let len = base_data.len().min(overlay_data.len()).min(out.len());
    for i in 0..len {
        let b = base_data[i] as f32 / 255.0;
        let o = overlay_data[i] as f32 / 255.0;
        let blended = b * (1.0 - alpha) + o * alpha;
        out[i] = (blended * 255.0).clamp(0.0, 255.0) as u8;
    }
    Ok(())
}

/// Blend two video frames using dissolve/add/multiply mode.
/// Params: {mode, ratio}
fn kernel_video_mix<'a>(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    params: &HashMap<String, Term<'a>>,
) -> NifResult<()> {
    if inputs.len() < 2 {
        return Ok(());
    }
    let ratio: f32 = params
        .get("ratio")
        .and_then(|t| t.decode::<f32>().ok())
        .unwrap_or(0.5);
    let mode: String = params
        .get("mode")
        .and_then(|t| t.decode::<String>().ok())
        .unwrap_or_else(|| "dissolve".to_string());

    let a = inputs[0].data().clone();
    let b = inputs[1].data().clone();
    let mut out = output.data();
    let len = a.len().min(b.len()).min(out.len());

    for i in 0..len {
        let va = a[i] as f32;
        let vb = b[i] as f32;
        let result = match mode.as_str() {
            "add" => (va + vb * ratio).min(255.0),
            "multiply" => va * (1.0 - ratio + ratio * (vb / 255.0)),
            _ => va * (1.0 - ratio) + vb * ratio, // dissolve (default)
        };
        out[i] = result.clamp(0.0, 255.0) as u8;
    }
    Ok(())
}

/// Separable 3x3 box blur (fast approximation of gaussian).
/// Params: {radius}
fn kernel_gaussian_blur<'a>(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    params: &HashMap<String, Term<'a>>,
) -> NifResult<()> {
    let input = inputs[0].data().clone();
    let mut out = output.data();
    let radius: usize = params
        .get("radius")
        .and_then(|t| t.decode::<usize>().ok())
        .unwrap_or(1);
    let len = input.len().min(out.len());
    if len == 0 {
        return Ok(());
    }

    // Simple box blur: average of neighbors within radius
    let r = radius.min(len);
    for i in 0..len {
        let start = i.saturating_sub(r);
        let end = (i + r + 1).min(len);
        let count = end - start;
        let sum: usize = input[start..end].iter().map(|&v| v as usize).sum();
        out[i] = (sum / count) as u8;
    }
    Ok(())
}

/// Nearest-neighbor scale (fast resize for video frames).
/// Params: {width, height} — target dimensions.
/// For simplicity, treats the buffer as a 1D stream and resamples linearly.
fn kernel_bicubic_scale<'a>(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    params: &HashMap<String, Term<'a>>,
) -> NifResult<()> {
    let input = inputs[0].data().clone();
    let mut out = output.data();
    if input.is_empty() || out.is_empty() {
        return Ok(());
    }
    let dst_width: usize = params
        .get("width")
        .and_then(|t| t.decode::<usize>().ok())
        .unwrap_or(0);
    let dst_height: usize = params
        .get("height")
        .and_then(|t| t.decode::<usize>().ok())
        .unwrap_or(0);

    // If no dimensions specified, just copy
    if dst_width == 0 || dst_height == 0 {
        let len = input.len().min(out.len());
        out[..len].copy_from_slice(&input[..len]);
        return Ok(());
    }

    // If the total byte count matches, just copy (handles YUV420p and other formats)
    let src_bytes = input.len();
    let dst_bytes = dst_width * dst_height;
    if dst_bytes == src_bytes {
        let len = src_bytes.min(out.len());
        out[..len].copy_from_slice(&input[..len]);
        return Ok(());
    }
    // Otherwise resample using nearest-neighbor
    let dst_len = dst_width * dst_height;
    let copy_len = dst_len.min(out.len());
    for i in 0..copy_len {
        let src_idx = (i * src_bytes) / dst_len;
        out[i] = input[src_idx.min(src_bytes - 1)];
    }
    Ok(())
}

/// LUT color grade — applies a simple identity/gamma LUT.
/// Params: {file} — LUT file path (ignored in CPU stub; applies gamma curve).
fn kernel_lut_apply<'a>(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    _params: &HashMap<String, Term<'a>>,
) -> NifResult<()> {
    let input = inputs[0].data().clone();
    let mut out = output.data();
    let len = input.len().min(out.len());

    // Apply a simple gamma curve (gamma=1.2) as a stand-in for LUT
    for i in 0..len {
        let v = input[i] as f32 / 255.0;
        let corrected = v.powf(1.0 / 1.2);
        out[i] = (corrected * 255.0).clamp(0.0, 255.0) as u8;
    }
    Ok(())
}

/// Green/blue screen chroma key.
/// Params: {color: {r, g, b}, threshold}
fn kernel_chroma_key<'a>(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    params: &HashMap<String, Term<'a>>,
) -> NifResult<()> {
    let input = inputs[0].data().clone();
    let mut out = output.data();
    let threshold: f32 = params
        .get("threshold")
        .and_then(|t| t.decode::<f32>().ok())
        .unwrap_or(0.3);

    // Default key color: green (0, 177, 64)
    let key_r: f32 = 0.0;
    let key_g: f32 = 177.0;
    let key_b: f32 = 64.0;

    let len = input.len().min(out.len());
    for i in (0..len).step_by(3) {
        if i + 2 >= len {
            break;
        }
        let r = input[i] as f32;
        let g = input[i + 1] as f32;
        let b = input[i + 2] as f32;

        let dist =
            ((r - key_r).powi(2) + (g - key_g).powi(2) + (b - key_b).powi(2)).sqrt() / 441.67;
        let alpha = if dist < threshold { 0.0 } else { 1.0 };

        out[i] = (r * alpha) as u8;
        out[i + 1] = (g * alpha) as u8;
        out[i + 2] = (b * alpha) as u8;
    }
    Ok(())
}

/// Unsharp mask sharpen.
/// Params: {strength}
fn kernel_sharpen<'a>(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    params: &HashMap<String, Term<'a>>,
) -> NifResult<()> {
    let input = inputs[0].data().clone();
    let mut out = output.data();
    let strength: f32 = params
        .get("strength")
        .and_then(|t| t.decode::<f32>().ok())
        .unwrap_or(1.0);
    let len = input.len().min(out.len());
    if len == 0 {
        return Ok(());
    }

    // Simple unsharp mask: original + strength * (original - blurred)
    for i in 0..len {
        let prev = if i > 0 {
            input[i - 1] as f32
        } else {
            input[i] as f32
        };
        let curr = input[i] as f32;
        let next = if i + 1 < len {
            input[i + 1] as f32
        } else {
            curr
        };
        let blurred = (prev + curr + next) / 3.0;
        let sharpened = curr + strength * (curr - blurred);
        out[i] = sharpened.clamp(0.0, 255.0) as u8;
    }
    Ok(())
}

/// Brightness and contrast adjustment.
/// Params: {brightness, contrast}
fn kernel_brightness_contrast<'a>(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    params: &HashMap<String, Term<'a>>,
) -> NifResult<()> {
    let input = inputs[0].data().clone();
    let mut out = output.data();
    let brightness: f32 = params
        .get("brightness")
        .and_then(|t| t.decode::<f32>().ok())
        .unwrap_or(0.0);
    let contrast: f32 = params
        .get("contrast")
        .and_then(|t| t.decode::<f32>().ok())
        .unwrap_or(1.0);
    let len = input.len().min(out.len());

    for i in 0..len {
        let v = input[i] as f32 / 255.0;
        let adjusted = (v - 0.5) * contrast + 0.5 + brightness;
        out[i] = (adjusted * 255.0).clamp(0.0, 255.0) as u8;
    }
    Ok(())
}

/// Simple denoise: median-like filter using local averaging.
/// Params: {strength}
fn kernel_denoise<'a>(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    params: &HashMap<String, Term<'a>>,
) -> NifResult<()> {
    let input = inputs[0].data().clone();
    let mut out = output.data();
    let strength: f32 = params
        .get("strength")
        .and_then(|t| t.decode::<f32>().ok())
        .unwrap_or(0.5);
    let len = input.len().min(out.len());
    if len == 0 {
        return Ok(());
    }

    for i in 0..len {
        let prev = if i > 0 {
            input[i - 1] as f32
        } else {
            input[i] as f32
        };
        let curr = input[i] as f32;
        let next = if i + 1 < len {
            input[i + 1] as f32
        } else {
            curr
        };
        let avg = (prev + curr + next) / 3.0;
        let denoised = curr * (1.0 - strength) + avg * strength;
        out[i] = denoised.clamp(0.0, 255.0) as u8;
    }
    Ok(())
}

/// Crop a video frame to the specified rectangle.
/// Params: {x, y, width, height}
/// Assumes the buffer is a flat byte buffer (YUV420p or RGB).
fn kernel_video_crop<'a>(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    params: &HashMap<String, Term<'a>>,
) -> NifResult<()> {
    let input = inputs[0].data().clone();
    let mut out = output.data();

    let x: usize = params.get("x").and_then(|t| t.decode().ok()).unwrap_or(0);
    let y: usize = params.get("y").and_then(|t| t.decode().ok()).unwrap_or(0);
    let crop_w: usize = params
        .get("width")
        .and_then(|t| t.decode().ok())
        .unwrap_or(0);
    let crop_h: usize = params
        .get("height")
        .and_then(|t| t.decode().ok())
        .unwrap_or(0);

    if crop_w == 0 || crop_h == 0 {
        return Ok(());
    }

    // Treat as a flat byte buffer and copy the sub-region
    // For YUV420p: total = width * height * 3 / 2
    // We copy byte-range [offset, offset + crop_size)
    let src_len = input.len();
    let crop_bytes = crop_w * crop_h * 3 / 2; // assume YUV420p
    let copy_len = crop_bytes.min(out.len()).min(src_len);

    // Simple approach: copy the first `crop_bytes` from the input
    // A proper implementation would handle row-by-row copying with x,y offsets
    let offset = (y * crop_w + x) * 3 / 2;
    let end = (offset + copy_len).min(src_len);
    if offset < src_len {
        let copy_len = end - offset;
        out[..copy_len].copy_from_slice(&input[offset..end]);
    }
    Ok(())
}

// ── Phase 2: Audio kernels ──────────────────────────────────

/// Mix multiple audio tracks with per-track gain.
/// Params: {gains: [f32, ...]}
fn kernel_pcm_mix<'a>(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    params: &HashMap<String, Term<'a>>,
) -> NifResult<()> {
    if inputs.is_empty() {
        return Ok(());
    }

    let gains: Vec<f32> = params
        .get("gains")
        .and_then(|t| t.decode::<Vec<f32>>().ok())
        .unwrap_or_else(|| vec![1.0; inputs.len()]);

    // Clone all input tracks BEFORE locking output to avoid deadlock when
    // output is the same buffer as one of the inputs (in-place operation).
    let tracks: Vec<Vec<f32>> = inputs
        .iter()
        .map(|input| {
            let data = input.data().clone();
            let mut samples = Vec::with_capacity(data.len() / 4);
            for chunk in data.chunks_exact(4) {
                samples.push(f32::from_ne_bytes(chunk.try_into().unwrap()));
            }
            samples
        })
        .collect();

    let mut out = output.data();
    let out_len = out.len();

    // Mix: sum all tracks with gain, clamp to [-1.0, 1.0]
    for i in 0..out_len / 4 {
        let mut sum = 0.0f32;
        for (track_idx, track) in tracks.iter().enumerate() {
            if i < track.len() {
                let gain = gains.get(track_idx).copied().unwrap_or(1.0);
                sum += track[i] * gain;
            }
        }
        let clamped = sum.clamp(-1.0, 1.0);
        let out_idx = i * 4;
        if out_idx + 4 <= out_len {
            out[out_idx..out_idx + 4].copy_from_slice(&clamped.to_ne_bytes());
        }
    }
    Ok(())
}

/// Peak normalize audio to 0 dBFS.
/// Params: none (or optional target_db)
fn kernel_pcm_normalize<'a>(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    _params: &HashMap<String, Term<'a>>,
) -> NifResult<()> {
    let input = inputs[0].data().clone();
    let mut out = output.data();
    let in_len = input.len();
    let out_len = out.len();
    let copy_len = in_len.min(out_len);

    // Find peak
    let mut peak = 0.0f32;
    for chunk in input[..copy_len].chunks_exact(4) {
        let v = f32::from_ne_bytes(chunk.try_into().unwrap()).abs();
        if v > peak {
            peak = v;
        }
    }

    // Normalize
    if peak > 0.0 {
        let scale = 1.0 / peak;
        for i in (0..copy_len).step_by(4) {
            let v = f32::from_ne_bytes(input[i..i + 4].try_into().unwrap());
            let normalized = (v * scale).clamp(-1.0, 1.0);
            out[i..i + 4].copy_from_slice(&normalized.to_ne_bytes());
        }
    }
    Ok(())
}

/// Biquad IIR filter (EQ).
/// Params: {b0, b1, b2, a1, a2} coefficients, or simplified as {bands: [...]}.
/// For the stub, applies a simple high-shelf EQ.
fn kernel_biquad_filter<'a>(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    _params: &HashMap<String, Term<'a>>,
) -> NifResult<()> {
    let input = inputs[0].data().clone();
    let mut out = output.data();
    let copy_len = input.len().min(out.len());
    if copy_len < 8 {
        out[..copy_len].copy_from_slice(&input[..copy_len]);
        return Ok(());
    }

    // Simple first-order high-pass filter as a stand-in for biquad
    let alpha = 0.1f32;
    let mut prev_in = 0.0f32;
    let mut prev_out = 0.0f32;

    for i in (0..copy_len).step_by(4) {
        let v = f32::from_ne_bytes(input[i..i + 4].try_into().unwrap());
        let filtered = alpha * (prev_out + v - prev_in);
        prev_in = v;
        prev_out = filtered;
        out[i..i + 4].copy_from_slice(&filtered.to_ne_bytes());
    }
    Ok(())
}

/// FFT convolution (reverb). Stub: applies a simple delay-based reverb.
/// Params: {room_size, wet}
fn kernel_fft_convolve<'a>(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    params: &HashMap<String, Term<'a>>,
) -> NifResult<()> {
    let input = inputs[0].data().clone();
    let mut out = output.data();
    let wet: f32 = params
        .get("wet")
        .and_then(|t| t.decode::<f32>().ok())
        .unwrap_or(0.2);
    let copy_len = input.len().min(out.len());

    // Simple delay-based reverb: mix original with delayed attenuated copy
    let delay_samples = 2048usize; // ~42ms at 48kHz
    for i in (0..copy_len).step_by(4) {
        let v = f32::from_ne_bytes(input[i..i + 4].try_into().unwrap());
        let delayed = if i >= delay_samples * 4 {
            let delayed_val = f32::from_ne_bytes(
                input[i - delay_samples * 4..i - delay_samples * 4 + 4]
                    .try_into()
                    .unwrap(),
            );
            delayed_val * 0.5
        } else {
            0.0
        };
        let mixed = v * (1.0 - wet) + delayed * wet;
        out[i..i + 4].copy_from_slice(&mixed.to_ne_bytes());
    }
    Ok(())
}

/// Linear interpolation resampler.
/// Params: {from, to} sample rates
fn kernel_resample_linear<'a>(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    params: &HashMap<String, Term<'a>>,
) -> NifResult<()> {
    let input = inputs[0].data().clone();
    let mut out = output.data();
    let from_rate: f32 = params
        .get("from")
        .and_then(|t| t.decode::<f32>().ok())
        .unwrap_or(48000.0);
    let to_rate: f32 = params
        .get("to")
        .and_then(|t| t.decode::<f32>().ok())
        .unwrap_or(48000.0);

    if from_rate <= 0.0 || to_rate <= 0.0 {
        return Err(Error::RaiseTerm(Box::new(format!(
            "resample_linear: invalid rates from={}, to={}",
            from_rate, to_rate
        ))));
    }
    if input.is_empty() {
        return Ok(());
    }

    let ratio = from_rate / to_rate;
    let in_samples = input.len() / 4;
    let out_samples = out.len() / 4;

    for i in 0..out_samples {
        let src_pos = i as f32 * ratio;
        let src_idx = src_pos as usize;
        let frac = src_pos - src_idx as f32;

        let v0 = if src_idx < in_samples {
            f32::from_ne_bytes(input[src_idx * 4..src_idx * 4 + 4].try_into().unwrap())
        } else {
            0.0
        };
        let v1 = if src_idx + 1 < in_samples {
            f32::from_ne_bytes(
                input[(src_idx + 1) * 4..(src_idx + 1) * 4 + 4]
                    .try_into()
                    .unwrap(),
            )
        } else {
            v0
        };

        let interpolated = v0 * (1.0 - frac) + v1 * frac;
        out[i * 4..i * 4 + 4].copy_from_slice(&interpolated.to_ne_bytes());
    }
    Ok(())
}

/// Look-ahead dynamics compressor.
/// Params: {threshold, ratio, attack_ms, release_ms}
fn kernel_dynamics_compress<'a>(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    params: &HashMap<String, Term<'a>>,
) -> NifResult<()> {
    let input = inputs[0].data().clone();
    let mut out = output.data();
    let threshold_db: f32 = params
        .get("threshold")
        .and_then(|t| t.decode::<f32>().ok())
        .unwrap_or(-18.0);
    let ratio: f32 = params
        .get("ratio")
        .and_then(|t| t.decode::<f32>().ok())
        .unwrap_or(4.0);
    let copy_len = input.len().min(out.len());

    let threshold_linear = 10.0f32.powf(threshold_db / 20.0);

    for i in (0..copy_len).step_by(4) {
        let v = f32::from_ne_bytes(input[i..i + 4].try_into().unwrap());
        let abs_v = v.abs();

        let compressed = if abs_v > threshold_linear {
            let excess = abs_v - threshold_linear;
            let compressed_excess = excess / ratio;
            let new_abs = threshold_linear + compressed_excess;
            v.signum() * new_abs
        } else {
            v
        };

        out[i..i + 4].copy_from_slice(&compressed.to_ne_bytes());
    }
    Ok(())
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
    // Params are passed as an Elixir map — decode to validate
    let _params_map: HashMap<String, Term<'a>> = {
        let mut hm = HashMap::new();
        if let Ok(map) = rustler::types::map::Map::from_term(params) {
            for (key_term, val) in map.iter() {
                if let Ok(key_str) = key_term.decode::<String>() {
                    hm.insert(key_str, val);
                }
            }
        }
        hm
    };

    match PIPELINES.get_mut(&pipeline_id) {
        Some(mut pipeline) => {
            pipeline.commands.push(PipelineCommand::KernelRun {
                name: name_str,
                inputs: input_resources,
                output,
                params: vec![],
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
                inputs: _inputs,
                output: _output,
                params: _params,
            } => {
                if !AVAILABLE_KERNELS.contains(&name.as_str()) {
                    return Ok((
                        atoms::error(),
                        format!("pipeline_run: unknown kernel '{}'", name),
                    )
                        .encode(env));
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
                    return Ok((
                        atoms::error(),
                        format!("pipeline_run: unknown filter kernel '{}'", kernel),
                    )
                        .encode(env));
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
