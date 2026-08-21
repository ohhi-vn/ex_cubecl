//! Core types and shared state for the ExCubecl NIF.
//!
//! DType, Buffer resource, command/pipeline types, global registries,
//! and shared validation helpers.

use dashmap::DashMap;
use parking_lot::Mutex;
use rustler::{Error, NifResult, ResourceArc};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::LazyLock;
use std::thread;

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
    #[allow(clippy::should_implement_trait)]
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


pub static COMMANDS: LazyLock<DashMap<u64, Command>> = LazyLock::new(DashMap::new);
pub static PIPELINES: LazyLock<DashMap<u64, Pipeline>> = LazyLock::new(DashMap::new);
pub static NEXT_COMMAND_ID: LazyLock<AtomicU64> = LazyLock::new(|| AtomicU64::new(1));
pub static NEXT_PIPELINE_ID: LazyLock<AtomicU64> = LazyLock::new(|| AtomicU64::new(1));
pub static THREAD_POOL: LazyLock<Mutex<Vec<thread::JoinHandle<()>>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

pub fn alloc_id(counter: &AtomicU64) -> u64 {
    counter.fetch_add(1, Ordering::SeqCst)
}

pub fn join_finished_threads() {
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

pub const AVAILABLE_KERNELS: &[&str] = &[
    // Phase 1 — compute
    "elementwise_add",
    "elementwise_mul",
    "elementwise_sub",
    "elementwise_div",
    "relu",
    "sigmoid",
    "tanh",
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


pub fn validate_f32_buffers(
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
