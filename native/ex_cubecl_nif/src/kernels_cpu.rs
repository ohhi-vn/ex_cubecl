//! CPU-side kernel implementations (simulating GPU execution).
//!
//! These run on the host CPU until the CubeCL GPU backend is integrated.

use crate::types::{validate_f32_buffers, Buffer};
use rustler::{Error, NifResult, ResourceArc};

// ══════════════════════════════════════════════════════════════
//  CPU-side kernel implementations (simulating GPU)
// ══════════════════════════════════════════════════════════════

// ── Phase 1: Elementwise compute kernels ─────────────────────

pub fn kernel_elementwise_add(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
) -> NifResult<()> {
    validate_f32_buffers("elementwise_add", inputs, output, 2)?;

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

pub fn kernel_elementwise_mul(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
) -> NifResult<()> {
    validate_f32_buffers("elementwise_mul", inputs, output, 2)?;

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

pub fn kernel_elementwise_sub(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
) -> NifResult<()> {
    validate_f32_buffers("elementwise_sub", inputs, output, 2)?;

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

pub fn kernel_elementwise_div(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
) -> NifResult<()> {
    validate_f32_buffers("elementwise_div", inputs, output, 2)?;

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

pub fn kernel_relu(inputs: &[ResourceArc<Buffer>], output: &ResourceArc<Buffer>) -> NifResult<()> {
    validate_f32_buffers("relu", inputs, output, 1)?;

    let a = inputs[0].data().clone();
    let mut out = output.data();
    let len = a.len().min(out.len());
    for i in (0..len).step_by(4) {
        let v = f32::from_ne_bytes(a[i..i + 4].try_into().unwrap());
        out[i..i + 4].copy_from_slice(&v.max(0.0).to_ne_bytes());
    }
    Ok(())
}

pub fn kernel_sigmoid(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
) -> NifResult<()> {
    validate_f32_buffers("sigmoid", inputs, output, 1)?;

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

pub fn kernel_tanh(inputs: &[ResourceArc<Buffer>], output: &ResourceArc<Buffer>) -> NifResult<()> {
    validate_f32_buffers("tanh", inputs, output, 1)?;

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
pub fn kernel_yuv_to_rgb(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    _params: &serde_json::Value,
) -> NifResult<()> {
    if inputs.is_empty() {
        return Err(Error::RaiseTerm(Box::new("yuv_to_rgb: expected 1 input")));
    }
    let input = inputs[0].data().clone();
    if !input.len().is_multiple_of(3) {
        return Err(Error::RaiseTerm(Box::new(
            "yuv_to_rgb: input length must be a multiple of 3 bytes",
        )));
    }
    let mut out = output.data();
    let total = input.len();
    let pixel_count = total * 2 / 3; // YUV420p: 1.5 bytes per pixel
    let rgb_bytes = pixel_count * 3;

    if out.len() < rgb_bytes {
        return Err(Error::RaiseTerm(Box::new(format!(
            "yuv_to_rgb: output buffer too small, expected {} bytes, got {}",
            rgb_bytes,
            out.len()
        ))));
    }

    let y_plane_size = pixel_count;
    let uv_plane_size = pixel_count / 2;

    for i in 0..pixel_count {
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
        out[out_idx] = r;
        out[out_idx + 1] = g;
        out[out_idx + 2] = b;
    }
    Ok(())
}

/// Alpha compositing: overlay onto base at (x, y) with opacity alpha.
/// Params: {x, y, alpha}
pub fn kernel_overlay_alpha(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    params: &serde_json::Value,
) -> NifResult<()> {
    if inputs.len() < 2 {
        return Ok(());
    }
    let _x: usize = params.get("x").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let _y: usize = params.get("y").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let alpha: f32 = params.get("alpha").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;

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
pub fn kernel_video_mix(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    params: &serde_json::Value,
) -> NifResult<()> {
    if inputs.len() < 2 {
        return Ok(());
    }
    let ratio: f32 = params.get("ratio").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
    let mode: String = params
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("dissolve")
        .to_string();

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
pub fn kernel_gaussian_blur(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    params: &serde_json::Value,
) -> NifResult<()> {
    let input = inputs[0].data().clone();
    let mut out = output.data();
    let radius: usize = params.get("radius").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
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
pub fn kernel_bicubic_scale(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    params: &serde_json::Value,
) -> NifResult<()> {
    let input = inputs[0].data().clone();
    let mut out = output.data();
    if input.is_empty() || out.is_empty() {
        return Ok(());
    }
    let dst_width: usize = params.get("width").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let dst_height: usize = params.get("height").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

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
pub fn kernel_lut_apply(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    _params: &serde_json::Value,
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
pub fn kernel_chroma_key(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    params: &serde_json::Value,
) -> NifResult<()> {
    let input = inputs[0].data().clone();
    let mut out = output.data();
    let threshold: f32 = params
        .get("threshold")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.3) as f32;

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
pub fn kernel_sharpen(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    params: &serde_json::Value,
) -> NifResult<()> {
    let input = inputs[0].data().clone();
    let mut out = output.data();
    let strength: f32 = params
        .get("strength")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;
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
pub fn kernel_brightness_contrast(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    params: &serde_json::Value,
) -> NifResult<()> {
    let input = inputs[0].data().clone();
    let mut out = output.data();
    let brightness: f32 = params
        .get("brightness")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;
    let contrast: f32 = params
        .get("contrast")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;
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
pub fn kernel_denoise(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    params: &serde_json::Value,
) -> NifResult<()> {
    let input = inputs[0].data().clone();
    let mut out = output.data();
    let strength: f32 = params
        .get("strength")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.5) as f32;
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
pub fn kernel_video_crop(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    params: &serde_json::Value,
) -> NifResult<()> {
    let input = inputs[0].data().clone();
    let mut out = output.data();

    let x: usize = params.get("x").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let y: usize = params.get("y").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let crop_w: usize = params.get("width").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let crop_h: usize = params.get("height").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

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
pub fn kernel_pcm_mix(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    params: &serde_json::Value,
) -> NifResult<()> {
    if inputs.is_empty() {
        return Ok(());
    }

    let gains: Vec<f32> = params
        .get("gains")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_f64())
                .map(|f| f as f32)
                .collect()
        })
        .unwrap_or_else(|| vec![1.0; inputs.len()]);

    // Clone all input tracks BEFORE locking output to avoid deadlock when
    // output is the same buffer as one of the inputs (in-place operation).
    let tracks: Vec<Vec<f32>> = inputs
        .iter()
        .map(|input| {
            let data = input.data().clone();
            let mut samples = Vec::with_capacity(data.len() / 4);
            for chunk in data.as_chunks::<4>().0 {
                samples.push(f32::from_ne_bytes(*chunk));
            }
            samples
        })
        .collect();

    let mut out = output.data();
    let out_len = out.len();

    // Mix: sum all tracks with gain, clamp to [-1.0, 1.0]
    let n_samples = out_len / 4;
    for i in 0..n_samples {
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
            let bytes = clamped.to_ne_bytes();
            out[out_idx] = bytes[0];
            out[out_idx + 1] = bytes[1];
            out[out_idx + 2] = bytes[2];
            out[out_idx + 3] = bytes[3];
        }
    }
    Ok(())
}

/// Peak normalize audio to 0 dBFS.
/// Params: none (or optional target_db)
pub fn kernel_pcm_normalize(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    _params: &serde_json::Value,
) -> NifResult<()> {
    let input = inputs[0].data().clone();
    let mut out = output.data();
    let in_len = input.len();
    let out_len = out.len();
    let copy_len = in_len.min(out_len);

    // Find peak
    let mut peak = 0.0f32;
    for chunk in input[..copy_len].as_chunks::<4>().0 {
        let v = f32::from_ne_bytes(*chunk).abs();
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
pub fn kernel_biquad_filter(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    _params: &serde_json::Value,
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
pub fn kernel_fft_convolve(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    params: &serde_json::Value,
) -> NifResult<()> {
    let input = inputs[0].data().clone();
    let mut out = output.data();
    let wet: f32 = params.get("wet").and_then(|v| v.as_f64()).unwrap_or(0.2) as f32;
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
pub fn kernel_resample_linear(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    params: &serde_json::Value,
) -> NifResult<()> {
    let input = inputs[0].data().clone();
    let mut out = output.data();
    let from_rate: f32 = params
        .get("from")
        .and_then(|v| v.as_f64().or_else(|| v.as_i64().map(|i| i as f64)))
        .unwrap_or(48000.0) as f32;
    let to_rate: f32 = params
        .get("to")
        .and_then(|v| v.as_f64().or_else(|| v.as_i64().map(|i| i as f64)))
        .unwrap_or(48000.0) as f32;

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
pub fn kernel_dynamics_compress(
    inputs: &[ResourceArc<Buffer>],
    output: &ResourceArc<Buffer>,
    params: &serde_json::Value,
) -> NifResult<()> {
    let input = inputs[0].data().clone();
    let mut out = output.data();
    let threshold_db: f32 = params
        .get("threshold")
        .and_then(|v| v.as_f64())
        .unwrap_or(-18.0) as f32;
    let ratio: f32 = params.get("ratio").and_then(|v| v.as_f64()).unwrap_or(4.0) as f32;
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
