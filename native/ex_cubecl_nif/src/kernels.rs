//! Phase 2 — CubeCL GPU kernel stubs.
//!
//! This module declares the video and audio processing kernels that will run
//! on the GPU via CubeCL. All kernels are currently stubs (CPU-side no-ops)
//! until the CubeCL backend is integrated.
//!
//! When CubeCL is available, each function will:
//!   1. Take GPU buffer handles (ResourceArc<Buffer>).
//!   2. Build a CubeCL kernel launch configuration.
//!   3. Dispatch to the GPU and return a command ID for async tracking.

/// Video: YUV420p → RGB24 color space conversion.
pub fn yuv_to_rgb(_input: &[u8], _width: u32, _height: u32) -> Vec<u8> {
    // TODO: CubeCL kernel dispatch
    vec![]
}

/// Video: Alpha compositing (Porter-Duff Over).
pub fn overlay_alpha(
    _base: &[u8],
    _overlay: &[u8],
    _width: u32,
    _height: u32,
    _x: u32,
    _y: u32,
    _alpha: f32,
) -> Vec<u8> {
    // TODO: CubeCL kernel dispatch
    vec![]
}

/// Video: Frame blending (dissolve, add, multiply).
pub fn video_mix(
    _frame_a: &[u8],
    _frame_b: &[u8],
    _len: usize,
    _mode: &str,
    _ratio: f32,
) -> Vec<u8> {
    // TODO: CubeCL kernel dispatch
    vec![]
}

/// Video: Separable 2-pass Gaussian blur.
pub fn gaussian_blur(_input: &[u8], _width: u32, _height: u32, _radius: u32) -> Vec<u8> {
    // TODO: CubeCL kernel dispatch (horizontal pass + vertical pass)
    vec![]
}

/// Video: Bicubic frame resizing.
pub fn bicubic_scale(
    _input: &[u8],
    _src_width: u32,
    _src_height: u32,
    _dst_width: u32,
    _dst_height: u32,
) -> Vec<u8> {
    // TODO: CubeCL kernel dispatch
    vec![]
}

/// Video: 3D LUT color grade (reads .cube file).
pub fn lut_apply(_input: &[u8], _len: usize, _lut_data: &[f32], _lut_size: u32) -> Vec<u8> {
    // TODO: CubeCL kernel dispatch with LUT texture
    vec![]
}

/// Video: Green/blue screen chroma key.
pub fn chroma_key(
    _input: &[u8],
    _len: usize,
    _key_r: u8,
    _key_g: u8,
    _key_b: u8,
    _threshold: f32,
) -> Vec<u8> {
    // TODO: CubeCL kernel dispatch
    vec![]
}

/// Video: Unsharp mask sharpen.
pub fn sharpen(_input: &[u8], _width: u32, _height: u32, _strength: f32) -> Vec<u8> {
    // TODO: CubeCL kernel dispatch
    vec![]
}

/// Audio: Sum N PCM tracks with per-channel gain.
pub fn pcm_mix(_tracks: &[&[f32]], _frames: usize, _gains: &[f32]) -> Vec<f32> {
    // TODO: CubeCL kernel dispatch
    vec![]
}

/// Audio: Peak/RMS normalize.
pub fn pcm_normalize(_samples: &[f32], _frames: usize, _target_db: f32) -> Vec<f32> {
    // TODO: CubeCL kernel dispatch
    vec![]
}

/// Audio: IIR biquad filter (EQ, HP, LP, shelf).
pub fn biquad_filter(
    _samples: &[f32],
    _frames: usize,
    _b0: f32,
    _b1: f32,
    _b2: f32,
    _a1: f32,
    _a2: f32,
) -> Vec<f32> {
    // TODO: CubeCL kernel dispatch
    vec![]
}

/// Audio: FFT convolution (reverb/impulse response).
pub fn fft_convolve(_samples: &[f32], _ir: &[f32], _frames: usize) -> Vec<f32> {
    // TODO: CubeCL FFT kernel dispatch
    vec![]
}

/// Audio: Linear interpolation resampler.
pub fn resample_linear(_samples: &[f32], _src_frames: usize, _dst_frames: usize) -> Vec<f32> {
    // TODO: CubeCL kernel dispatch
    vec![]
}

/// Audio: Look-ahead dynamics compressor.
pub fn dynamics_compress(
    _samples: &[f32],
    _frames: usize,
    _threshold_db: f32,
    _ratio: f32,
    _attack_ms: f32,
    _release_ms: f32,
) -> Vec<f32> {
    // TODO: CubeCL kernel dispatch
    vec![]
}
