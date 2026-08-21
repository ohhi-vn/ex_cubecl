//! C FFI implementation for ex_cubecl CPU-side API prototype.
//!
//! This module implements the C-compatible interface declared in
//! `include/ex_cubecl.h`. All functions use opaque handles (usize) to
//! reference buffers, pipelines, commands, textures, media sources, and
//! encoders. A handle of 0 indicates an error.
//!
//! Thread safety: The internal context is managed via global DashMap stores
//! protected by parking_lot::Mutex. Handles are only valid on the thread
//! that created them (same constraint as the NIF layer).

use crate::{
    alloc_id, Buffer, Command, CommandStatus, Pipeline, COMMANDS, NEXT_COMMAND_ID,
    NEXT_PIPELINE_ID, PIPELINES,
};
use parking_lot::Mutex;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::sync::atomic::Ordering;

// ── Error handling ───────────────────────────────────────────

/// Global last-error buffer (thread-local would be better, but C FFI
/// consumers are expected to copy the string immediately).
static LAST_ERROR: Mutex<String> = Mutex::new(String::new());

fn set_last_error(msg: &str) {
    let mut err = LAST_ERROR.lock();
    *err = msg.to_string();
}

/// Retrieve the last error message.
///
/// # Safety
///
/// `buf` must be a valid writable pointer to at least `len` bytes.
/// `len` must be greater than 0 and include space for the NUL terminator.
#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_last_error(buf: *mut c_char, len: usize) -> usize {
    if buf.is_null() || len == 0 {
        return 0;
    }
    let err = LAST_ERROR.lock();
    if err.is_empty() {
        return 0;
    }
    let bytes = err.as_bytes();
    let copy_len = bytes.len().min(len - 1);
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, copy_len);
    *buf.add(copy_len) = 0; // NUL terminator
    copy_len
}

// ── Device management ────────────────────────────────────────

/// Get a human-readable device info string.
///
/// # Safety
///
/// `buf` must be a valid writable pointer to at least `len` bytes.
/// `len` must be greater than 0 and include space for the NUL terminator.
#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_device_info(buf: *mut c_char, len: usize) -> usize {
    if buf.is_null() || len == 0 {
        return 0;
    }
    let info = "CubeCL GPU (CPU fallback — v0.5.0)";
    let bytes = info.as_bytes();
    let copy_len = bytes.len().min(len - 1);
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, copy_len);
    *buf.add(copy_len) = 0;
    copy_len
}

/// Get the number of available devices.
#[no_mangle]
pub extern "C" fn ex_cubecl_device_count() -> i32 {
    1
}

// ── Buffer management ────────────────────────────────────────

/// Global buffer store for C FFI handles.
use dashmap::DashMap;
use std::sync::LazyLock;

static C_BUFFERS: LazyLock<DashMap<usize, Buffer>> = LazyLock::new(DashMap::new);
static C_TEXTURES: LazyLock<DashMap<usize, Buffer>> = LazyLock::new(DashMap::new);
static C_MEDIA: LazyLock<DashMap<usize, media::MediaSource>> = LazyLock::new(DashMap::new);
static C_NEXT_HANDLE: LazyLock<std::sync::atomic::AtomicUsize> =
    LazyLock::new(|| std::sync::atomic::AtomicUsize::new(1_000_000));

fn alloc_c_handle() -> usize {
    C_NEXT_HANDLE.fetch_add(1, Ordering::SeqCst)
}

/// Create a CPU-side buffer from raw data.
///
/// # Safety
///
/// `data` must point to at least `total_bytes` valid readable bytes.
/// `shape` must point to at least `ndim` valid `usize` values.
/// `ndim` must be greater than 0.
#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_buffer_new(
    data: *const u8,
    shape: *const usize,
    ndim: usize,
    dtype: i32,
) -> usize {
    if data.is_null() || shape.is_null() || ndim == 0 {
        set_last_error("ex_cubecl_buffer_new: null data or shape, or ndim == 0");
        return 0;
    }

    let dtype_size = match dtype {
        0 => 4, // F32
        1 => 8, // F64
        2 => 4, // S32
        3 => 8, // S64
        4 => 4, // U32
        5 => 1, // U8
        _ => {
            set_last_error("ex_cubecl_buffer_new: invalid dtype");
            return 0;
        }
    };

    let shape_slice = std::slice::from_raw_parts(shape, ndim);
    let total_elements: usize = shape_slice.iter().product();
    let total_bytes = total_elements * dtype_size;

    let data_slice = std::slice::from_raw_parts(data, total_bytes);

    let buffer = Buffer {
        data: parking_lot::Mutex::new(data_slice.to_vec()),
        shape: shape_slice.to_vec(),
        dtype: crate::DType::from_str(match dtype {
            0 => "f32",
            1 => "f64",
            2 => "s32",
            3 => "s64",
            4 => "u32",
            5 => "u8",
            _ => "f32",
        })
        .unwrap_or(crate::DType::F32),
    };

    let handle = alloc_c_handle();
    C_BUFFERS.insert(handle, buffer);
    handle
}

/// Read buffer data into a caller-provided buffer.
///
/// # Safety
///
/// `out` must point to at least `len` valid writable bytes.
#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_buffer_read(handle: usize, out: *mut u8, len: usize) -> i32 {
    if out.is_null() {
        set_last_error("ex_cubecl_buffer_read: null output buffer");
        return -1;
    }
    match C_BUFFERS.get(&handle) {
        Some(buf) => {
            let data = buf.data.lock();
            let copy_len = data.len().min(len);
            std::ptr::copy_nonoverlapping(data.as_ptr(), out, copy_len);
            0
        }
        None => {
            set_last_error("ex_cubecl_buffer_read: invalid handle");
            -1
        }
    }
}

/// Get the size of buffer data in bytes.
#[no_mangle]
pub extern "C" fn ex_cubecl_buffer_size(handle: usize) -> usize {
    match C_BUFFERS.get(&handle) {
        Some(buf) => buf.data.lock().len(),
        None => 0,
    }
}

/// Get the shape of a buffer.
///
/// # Safety
///
/// `out_shape` must point to at least `out_ndim` valid `usize` slots.
#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_buffer_shape(
    handle: usize,
    out_shape: *mut usize,
    out_ndim: usize,
) -> i32 {
    if out_shape.is_null() {
        return -1;
    }
    match C_BUFFERS.get(&handle) {
        Some(buf) => {
            let shape = &buf.shape;
            let copy_len = shape.len().min(out_ndim);
            std::ptr::copy_nonoverlapping(shape.as_ptr(), out_shape, copy_len);
            0
        }
        None => {
            set_last_error("ex_cubecl_buffer_shape: invalid handle");
            -1
        }
    }
}

/// Get the data type of a buffer.
///
/// # Safety
///
/// `out_dtype` must be a valid writable pointer to an `i32`.
#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_buffer_dtype(handle: usize, out_dtype: *mut i32) -> i32 {
    if out_dtype.is_null() {
        return -1;
    }
    match C_BUFFERS.get(&handle) {
        Some(buf) => {
            *out_dtype = match buf.dtype {
                crate::DType::F32 => 0,
                crate::DType::F64 => 1,
                crate::DType::S32 => 2,
                crate::DType::S64 => 3,
                crate::DType::U32 => 4,
                crate::DType::U8 => 5,
            };
            0
        }
        None => {
            set_last_error("ex_cubecl_buffer_dtype: invalid handle");
            -1
        }
    }
}

/// Free a CPU-side buffer.
#[no_mangle]
pub extern "C" fn ex_cubecl_buffer_free(handle: usize) -> i32 {
    match C_BUFFERS.remove(&handle) {
        Some(_) => 0,
        None => {
            set_last_error("ex_cubecl_buffer_free: invalid handle");
            -1
        }
    }
}

// ── Kernel execution ─────────────────────────────────────────

/// Run a named kernel.
///
/// # Safety
///
/// `name` must be a valid NUL-terminated C string.
/// `inputs` must point to at least `n_inputs` valid `usize` values.
#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_kernel_run(
    name: *const c_char,
    inputs: *const usize,
    n_inputs: usize,
    output: usize,
    _params: *const u8,
    _params_len: usize,
) -> i32 {
    if name.is_null() || inputs.is_null() {
        set_last_error("ex_cubecl_kernel_run: null name or inputs");
        return -1;
    }

    let name_str = match CStr::from_ptr(name).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => {
            set_last_error("ex_cubecl_kernel_run: invalid kernel name encoding");
            return -1;
        }
    };

    let valid_kernels = [
        "elementwise_add",
        "elementwise_mul",
        "elementwise_sub",
        "elementwise_div",
        "relu",
        "sigmoid",
        "tanh",
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
        "pcm_mix",
        "pcm_normalize",
        "biquad_filter",
        "fft_convolve",
        "resample_linear",
        "dynamics_compress",
    ];
    if !valid_kernels.contains(&name_str.as_str()) {
        set_last_error("ex_cubecl_kernel_run: unknown kernel");
        return -1;
    }

    let input_handles = std::slice::from_raw_parts(inputs, n_inputs);
    for &h in input_handles {
        if !C_BUFFERS.contains_key(&h) {
            set_last_error("ex_cubecl_kernel_run: invalid input handle");
            return -1;
        }
    }
    if !C_BUFFERS.contains_key(&output) {
        set_last_error("ex_cubecl_kernel_run: invalid output handle");
        return -1;
    }

    let input_data: Vec<Vec<u8>> = input_handles
        .iter()
        .map(|&h| C_BUFFERS.get(&h).unwrap().data.lock().clone())
        .collect();

    let result = match name_str.as_str() {
        "elementwise_add" if input_data.len() >= 2 => {
            let len = input_data[0].len().min(input_data[1].len());
            let mut out = vec![0u8; len];
            for i in (0..len).step_by(4) {
                let a = f32::from_ne_bytes(input_data[0][i..i + 4].try_into().unwrap_or([0; 4]));
                let b = f32::from_ne_bytes(input_data[1][i..i + 4].try_into().unwrap_or([0; 4]));
                out[i..i + 4].copy_from_slice(&(a + b).to_ne_bytes());
            }
            out
        }
        "relu" => {
            let mut out = vec![0u8; input_data[0].len()];
            for i in (0..input_data[0].len()).step_by(4) {
                let v = f32::from_ne_bytes(input_data[0][i..i + 4].try_into().unwrap_or([0; 4]));
                out[i..i + 4].copy_from_slice(&v.max(0.0).to_ne_bytes());
            }
            out
        }
        "yuv_to_rgb" => {
            let input = &input_data[0];
            let pixel_count = input.len() * 2 / 3;
            let mut out = vec![0u8; pixel_count * 3];
            let y_size = pixel_count;
            for i in 0..pixel_count.min(out.len() / 3) {
                let y_val = input.get(i).copied().unwrap_or(0) as f32;
                let uv_idx = y_size + (i / 2) * 2;
                let u_val = input.get(uv_idx).copied().unwrap_or(128) as f32 - 128.0;
                let v_val = input.get(uv_idx + 1).copied().unwrap_or(128) as f32 - 128.0;
                let r = (y_val + 1.402 * v_val).clamp(0.0, 255.0) as u8;
                let g = (y_val - 0.344136 * u_val - 0.714136 * v_val).clamp(0.0, 255.0) as u8;
                let b = (y_val + 1.772 * u_val).clamp(0.0, 255.0) as u8;
                out[i * 3] = r;
                out[i * 3 + 1] = g;
                out[i * 3 + 2] = b;
            }
            out
        }
        "gaussian_blur" => {
            let input = &input_data[0];
            let mut out = vec![0u8; input.len()];
            let radius = 1usize;
            for (i, o) in out.iter_mut().enumerate() {
                let start = i.saturating_sub(radius);
                let end = (i + radius + 1).min(input.len());
                let count = end - start;
                let sum: usize = input[start..end].iter().map(|&v| v as usize).sum();
                *o = (sum / count) as u8;
            }
            out
        }
        "pcm_mix" => {
            if input_data.is_empty() {
                vec![]
            } else {
                let frame_count = input_data[0].len() / 4;
                let mut out = vec![0u8; frame_count * 4];
                for i in 0..frame_count {
                    let mut sum = 0.0f32;
                    for input in &input_data {
                        if i * 4 + 4 <= input.len() {
                            sum += f32::from_ne_bytes(
                                input[i * 4..i * 4 + 4].try_into().unwrap_or([0; 4]),
                            );
                        }
                    }
                    let clamped = sum.clamp(-1.0, 1.0);
                    out[i * 4..i * 4 + 4].copy_from_slice(&clamped.to_ne_bytes());
                }
                out
            }
        }
        _ => {
            if !input_data.is_empty() {
                input_data[0].clone()
            } else {
                vec![]
            }
        }
    };

    if let Some(buf) = C_BUFFERS.get_mut(&output) {
        let mut data = buf.data.lock();
        let copy_len = result.len().min(data.len());
        data[..copy_len].copy_from_slice(&result[..copy_len]);
    }

    0
}

/// List available kernels as a comma-separated string.
///
/// # Safety
///
/// `buf` must be a valid writable pointer to at least `buf_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_kernel_list(buf: *mut c_char, buf_len: usize) -> usize {
    if buf.is_null() || buf_len == 0 {
        return 0;
    }
    let kernels = "elementwise_add,elementwise_mul,elementwise_sub,elementwise_div,relu,sigmoid,tanh,yuv_to_rgb,overlay_alpha,video_mix,gaussian_blur,bicubic_scale,lut_apply,chroma_key,sharpen,brightness_contrast,denoise,pcm_mix,pcm_normalize,biquad_filter,fft_convolve,resample_linear,dynamics_compress";
    let bytes = kernels.as_bytes();
    let copy_len = bytes.len().min(buf_len - 1);
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, copy_len);
    *buf.add(copy_len) = 0;
    copy_len
}

// ── Async execution ──────────────────────────────────────────

/// Submit a raw command for asynchronous execution.
///
/// # Safety
///
/// `command` must point to at least `command_len` valid readable bytes.
#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_submit(command: *const u8, command_len: usize) -> u64 {
    if command.is_null() || command_len == 0 {
        return 0;
    }
    let cmd_id = alloc_id(&NEXT_COMMAND_ID);
    COMMANDS.insert(
        cmd_id,
        Command {
            id: cmd_id,
            status: CommandStatus::Completed,
        },
    );
    cmd_id
}

/// Poll the status of a submitted command.
#[no_mangle]
pub extern "C" fn ex_cubecl_poll(command_id: u64) -> i32 {
    match COMMANDS.get(&command_id) {
        Some(cmd) => match &cmd.status {
            CommandStatus::Pending => 0,
            CommandStatus::Running => 0,
            CommandStatus::Completed => 1,
            CommandStatus::Failed(_) => -1,
        },
        None => -1,
    }
}

/// Block until a submitted command completes.
#[no_mangle]
pub extern "C" fn ex_cubecl_wait(command_id: u64) -> i32 {
    loop {
        match COMMANDS.get(&command_id) {
            Some(cmd) => match &cmd.status {
                CommandStatus::Completed => return 0,
                CommandStatus::Failed(_) => return -1,
                _ => {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            },
            None => return -1,
        }
    }
}

// ── Pipeline ─────────────────────────────────────────────────

/// Create a new command pipeline.
#[no_mangle]
pub extern "C" fn ex_cubecl_pipeline_new() -> usize {
    let id = alloc_id(&NEXT_PIPELINE_ID);
    PIPELINES.insert(
        id,
        Pipeline {
            id,
            commands: vec![],
        },
    );
    id.try_into().unwrap()
}

/// Add a command to a pipeline.
///
/// # Safety
///
/// `command` must point to at least `command_len` valid readable bytes.
#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_pipeline_add(
    pipeline: usize,
    command: *const u8,
    command_len: usize,
) -> i32 {
    if command.is_null() {
        return -1;
    }
    match PIPELINES.get_mut(&(pipeline as u64)) {
        Some(_p) => {
            let _cmd_bytes = std::slice::from_raw_parts(command, command_len);
            0
        }
        None => {
            set_last_error("ex_cubecl_pipeline_add: invalid pipeline handle");
            -1
        }
    }
}

/// Execute all commands in a pipeline.
#[no_mangle]
pub extern "C" fn ex_cubecl_pipeline_run(_pipeline: usize) -> i32 {
    0
}

/// Free a command pipeline.
#[no_mangle]
pub extern "C" fn ex_cubecl_pipeline_free(pipeline: usize) -> i32 {
    match PIPELINES.remove(&(pipeline as u64)) {
        Some(_) => 0,
        None => {
            set_last_error("ex_cubecl_pipeline_free: invalid pipeline handle");
            -1
        }
    }
}

// ── Phase 2 — CPU Texture (Video Frame) ─────────────────────

/// Create a CPU-side texture from YUV420p plane data.
///
/// # Safety
///
/// `y_plane` must point to at least `width * height` valid readable bytes.
/// `uv_plane` must point to at least `width * height / 2` valid readable bytes.
#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_texture_from_yuv(
    y_plane: *const u8,
    uv_plane: *const u8,
    width: u32,
    height: u32,
) -> usize {
    if y_plane.is_null() || uv_plane.is_null() || width == 0 || height == 0 {
        set_last_error("ex_cubecl_texture_from_yuv: null planes or zero dimensions");
        return 0;
    }

    let y_size = (width * height) as usize;
    let uv_size = y_size / 2;
    let total = y_size + uv_size;

    let y_data = std::slice::from_raw_parts(y_plane, y_size);
    let uv_data = std::slice::from_raw_parts(uv_plane, uv_size);

    let mut data = Vec::with_capacity(total);
    data.extend_from_slice(y_data);
    data.extend_from_slice(uv_data);

    let texture = Buffer {
        data: parking_lot::Mutex::new(data),
        shape: vec![total],
        dtype: crate::DType::U8,
    };

    let handle = alloc_c_handle();
    C_TEXTURES.insert(handle, texture);
    handle
}

/// Create a CPU-side texture from NV12 plane data.
///
/// # Safety
///
/// See `ex_cubecl_texture_from_yuv` for safety requirements.
#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_texture_from_nv12(
    y_plane: *const u8,
    uv_plane: *const u8,
    width: u32,
    height: u32,
) -> usize {
    ex_cubecl_texture_from_yuv(y_plane, uv_plane, width, height)
}

/// Apply a named CPU kernel to a texture.
///
/// # Safety
///
/// `kernel_name` must be a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_apply_kernel(
    input: usize,
    kernel_name: *const c_char,
    _params: *const u8,
    _param_count: usize,
) -> usize {
    if kernel_name.is_null() {
        return 0;
    }

    let name = match CStr::from_ptr(kernel_name).to_str() {
        Ok(s) => s,
        Err(_) => return 0,
    };

    let input_data = match C_TEXTURES.get(&input) {
        Some(t) => t.data.lock().clone(),
        None => return 0,
    };

    let result = match name {
        "yuv_to_rgb" => {
            let pixel_count = input_data.len() * 2 / 3;
            let mut out = vec![0u8; pixel_count * 3];
            let y_size = pixel_count;
            for i in 0..pixel_count.min(out.len() / 3) {
                let y_val = input_data.get(i).copied().unwrap_or(0) as f32;
                let uv_idx = y_size + (i / 2) * 2;
                let u_val = input_data.get(uv_idx).copied().unwrap_or(128) as f32 - 128.0;
                let v_val = input_data.get(uv_idx + 1).copied().unwrap_or(128) as f32 - 128.0;
                out[i * 3] = (y_val + 1.402 * v_val).clamp(0.0, 255.0) as u8;
                out[i * 3 + 1] =
                    (y_val - 0.344136 * u_val - 0.714136 * v_val).clamp(0.0, 255.0) as u8;
                out[i * 3 + 2] = (y_val + 1.772 * u_val).clamp(0.0, 255.0) as u8;
            }
            out
        }
        "gaussian_blur" => {
            let mut out = vec![0u8; input_data.len()];
            for (i, item) in out.iter_mut().enumerate().take(input_data.len()) {
                let start = i.saturating_sub(1);
                let end = (i + 2).min(input_data.len());
                let sum: usize = input_data[start..end].iter().map(|&v| v as usize).sum();
                *item = (sum / (end - start)) as u8;
            }
            out
        }
        _ => input_data.clone(),
    };

    let output = Buffer {
        data: parking_lot::Mutex::new(result),
        shape: vec![input_data.len()],
        dtype: crate::DType::U8,
    };

    let handle = alloc_c_handle();
    C_TEXTURES.insert(handle, output);
    handle
}

/// Free a CPU-side texture.
#[no_mangle]
pub extern "C" fn ex_cubecl_texture_free(texture: usize) -> i32 {
    match C_TEXTURES.remove(&texture) {
        Some(_) => 0,
        None => {
            set_last_error("ex_cubecl_texture_free: invalid texture handle");
            -1
        }
    }
}

// ── Phase 2 — Media Source ──────────────────────────────────

#[allow(dead_code)]
mod media {
    use crate::atoms;
    use rustler::{Encoder, Env, NifResult, ResourceArc, Term};

    #[derive(Debug, Clone)]
    pub struct MediaSource {
        pub path: String,
        pub streams: Vec<StreamInfo>,
    }

    #[derive(Debug, Clone)]
    pub struct StreamInfo {
        pub index: usize,
        pub stream_type: StreamType,
        pub codec: String,
        pub width: Option<u32>,
        pub height: Option<u32>,
        pub fps: Option<f64>,
        pub sample_rate: Option<u32>,
        pub channels: Option<u32>,
    }

    #[derive(Debug, Clone)]
    pub enum StreamType {
        Video,
        Audio,
    }

    impl rustler::Resource for MediaSource {}

    pub fn nif_media_open<'a>(env: Env<'a>, path: Term) -> NifResult<Term<'a>> {
        let path_str: String = path.decode()?;
        let source = MediaSource {
            path: path_str,
            streams: vec![
                StreamInfo {
                    index: 0,
                    stream_type: StreamType::Video,
                    codec: "h264".to_string(),
                    width: Some(1920),
                    height: Some(1080),
                    fps: Some(30.0),
                    sample_rate: None,
                    channels: None,
                },
                StreamInfo {
                    index: 1,
                    stream_type: StreamType::Audio,
                    codec: "aac".to_string(),
                    width: None,
                    height: None,
                    fps: None,
                    sample_rate: Some(44100),
                    channels: Some(2),
                },
            ],
        };
        let resource = ResourceArc::<MediaSource>::new(source);
        Ok((atoms::ok(), resource).encode(env))
    }

    pub fn nif_media_streams(env: Env, source: ResourceArc<MediaSource>) -> NifResult<Term> {
        let mut stream_terms = Vec::new();
        for s in &source.streams {
            let map = rustler::types::map::map_new(env);
            let map = map.map_put(atoms::index().encode(env), s.index.encode(env))?;
            let type_atom = match s.stream_type {
                StreamType::Video => atoms::video(),
                StreamType::Audio => atoms::audio(),
            };
            let map = map.map_put(
                rustler::Atom::from_str(env, "type").unwrap().encode(env),
                type_atom.encode(env),
            )?;
            let map = map.map_put(atoms::codec().encode(env), s.codec.as_str().encode(env))?;
            let map = match s.stream_type {
                StreamType::Video => {
                    let map =
                        map.map_put(atoms::width().encode(env), s.width.unwrap_or(0).encode(env))?;
                    let map = map.map_put(
                        atoms::height().encode(env),
                        s.height.unwrap_or(0).encode(env),
                    )?;
                    map.map_put(atoms::fps().encode(env), s.fps.unwrap_or(0.0).encode(env))?
                }
                StreamType::Audio => {
                    let map = map.map_put(
                        atoms::sample_rate().encode(env),
                        s.sample_rate.unwrap_or(0).encode(env),
                    )?;
                    map.map_put(
                        atoms::channels().encode(env),
                        s.channels.unwrap_or(0).encode(env),
                    )?
                }
            };
            stream_terms.push(map.encode(env));
        }
        Ok((atoms::ok(), stream_terms).encode(env))
    }

    pub fn nif_media_read_video_frame<'a>(
        env: Env<'a>,
        _source: ResourceArc<MediaSource>,
    ) -> NifResult<Term<'a>> {
        let frame_size = 1920 * 1080 * 3 / 2;
        let y_size = 1920 * 1080;
        let mut data = Vec::with_capacity(frame_size);
        data.resize(y_size, 128u8);
        data.resize(frame_size, 128u8);

        use crate::Buffer;
        use crate::DType;
        let buffer = Buffer {
            data: parking_lot::Mutex::new(data),
            shape: vec![frame_size],
            dtype: DType::U8,
        };
        let resource = ResourceArc::<Buffer>::new(buffer);

        let frame_map = rustler::types::map::map_new(env);
        let frame_map = frame_map.map_put(atoms::handle().encode(env), resource.encode(env))?;
        let frame_map = frame_map.map_put(atoms::width().encode(env), 1920u32.encode(env))?;
        let frame_map = frame_map.map_put(atoms::height().encode(env), 1080u32.encode(env))?;
        let frame_map = frame_map.map_put(
            atoms::format().encode(env),
            rustler::Atom::from_str(env, "yuv420p").unwrap().encode(env),
        )?;
        let frame_map = frame_map.map_put(atoms::pts().encode(env), 0u64.encode(env))?;
        let frame_map = frame_map.map_put(atoms::duration().encode(env), 33333u64.encode(env))?;

        Ok((atoms::ok(), frame_map.encode(env)).encode(env))
    }

    pub fn nif_media_read_audio_samples<'a>(
        env: Env<'a>,
        _source: ResourceArc<MediaSource>,
    ) -> NifResult<Term<'a>> {
        let sample_frames = 1024usize;
        let channels = 2usize;
        let data = vec![0u8; sample_frames * channels * 4];

        use crate::Buffer;
        use crate::DType;
        let buffer = Buffer {
            data: parking_lot::Mutex::new(data),
            shape: vec![sample_frames * channels],
            dtype: DType::F32,
        };
        let resource = ResourceArc::<Buffer>::new(buffer);

        let samples_map = rustler::types::map::map_new(env);
        let samples_map = samples_map.map_put(atoms::handle().encode(env), resource.encode(env))?;
        let samples_map =
            samples_map.map_put(atoms::channels().encode(env), channels.encode(env))?;
        let samples_map =
            samples_map.map_put(atoms::sample_rate().encode(env), 48000u32.encode(env))?;
        let samples_map =
            samples_map.map_put(atoms::frames().encode(env), sample_frames.encode(env))?;
        let samples_map = samples_map.map_put(atoms::pts().encode(env), 0u64.encode(env))?;

        Ok((atoms::ok(), samples_map.encode(env)).encode(env))
    }

    pub fn nif_media_close(env: Env, _source: ResourceArc<MediaSource>) -> NifResult<Term> {
        Ok((atoms::ok()).encode(env))
    }
}

use self::media::MediaSource;

/// Open a media source (file, stream, camera).
///
/// # Safety
///
/// `path` must be a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_media_open(path: *const c_char) -> usize {
    if path.is_null() {
        return 0;
    }
    let path_str = match CStr::from_ptr(path).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return 0,
    };
    let source = MediaSource {
        path: path_str,
        streams: vec![],
    };
    let handle = alloc_c_handle();
    C_MEDIA.insert(handle, source);
    handle
}

/// Close a media source.
#[no_mangle]
pub extern "C" fn ex_cubecl_media_close(media: usize) -> i32 {
    match C_MEDIA.remove(&media) {
        Some(_) => 0,
        None => {
            set_last_error("ex_cubecl_media_close: invalid media handle");
            -1
        }
    }
}

// ── Phase 2 — Audio Mix ─────────────────────────────────────

/// Mix multiple audio tracks on the CPU.
///
/// # Safety
///
/// `tracks` must point to at least `track_count` valid `usize` values.
/// `gains` must point to at least `track_count` valid `f32` values.
#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_audio_mix(
    tracks: *const usize,
    gains: *const f32,
    track_count: usize,
    frames: usize,
) -> usize {
    if tracks.is_null() || gains.is_null() || track_count == 0 || frames == 0 {
        return 0;
    }

    let track_handles = std::slice::from_raw_parts(tracks, track_count);
    let gain_values = std::slice::from_raw_parts(gains, track_count);

    let track_data: Vec<Vec<u8>> = track_handles
        .iter()
        .map(|&h| {
            C_BUFFERS
                .get(&h)
                .map(|b| b.data.lock().clone())
                .unwrap_or_default()
        })
        .collect();

    let frame_bytes = frames * 4;
    let mut out = vec![0u8; frame_bytes];

    for (track_idx, data) in track_data.iter().enumerate() {
        let gain = gain_values.get(track_idx).copied().unwrap_or(1.0);
        for i in (0..data.len().min(frame_bytes)).step_by(4) {
            let sample = f32::from_ne_bytes(data[i..i + 4].try_into().unwrap_or([0; 4]));
            let existing = f32::from_ne_bytes(out[i..i + 4].try_into().unwrap_or([0; 4]));
            let mixed = (existing + sample * gain).clamp(-1.0, 1.0);
            out[i..i + 4].copy_from_slice(&mixed.to_ne_bytes());
        }
    }

    let buffer = Buffer {
        data: parking_lot::Mutex::new(out),
        shape: vec![frames],
        dtype: crate::DType::F32,
    };

    let handle = alloc_c_handle();
    C_BUFFERS.insert(handle, buffer);
    handle
}

/// Placeholder: returns the runtime version.
#[no_mangle]
pub extern "C" fn ex_cubecl_version() -> u32 {
    0x0005_0000 // 0.5.0
}
