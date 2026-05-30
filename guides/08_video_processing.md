# Video Processing Guide

Complete reference for GPU-accelerated video operations in ExCubecl.

## VideoFrame Struct

```elixir
%ExCubecl.VideoFrame{
  handle:   #Reference<...>,   # GPU buffer
  width:    1920,
  height:   1080,
  format:   :yuv420p,          # :yuv420p | :rgb24 | :rgba | :nv12
  pts:      0,                 # presentation timestamp (µs)
  duration: 33333              # frame duration (µs)
}
```

## Color Space Conversion

```elixir
# YUV420p → RGB24
{:ok, rgb} = ExCubecl.Video.convert(frame, :yuv420p, :rgb24)

# NV12 → RGB24
{:ok, rgb} = ExCubecl.Video.convert(nv12_frame, :nv12, :rgb24)
```

The `yuv_to_rgb` kernel runs entirely on the GPU. No CPU readback needed.

## Overlay (Alpha Composite)

```elixir
# Basic overlay at origin
{:ok, result} = ExCubecl.Video.overlay(base, overlay)

# Positioned with alpha
{:ok, result} = ExCubecl.Video.overlay(base, overlay, x: 100, y: 50, alpha: 0.8)
```

Uses Porter-Duff Over compositing on the GPU.

## Mix / Blend

```elixir
# Dissolve (cross-fade)
{:ok, result} = ExCubecl.Video.mix(frame_a, frame_b, mode: :dissolve, ratio: 0.5)

# Additive blend
{:ok, result} = ExCubecl.Video.mix(frame_a, frame_b, mode: :add, ratio: 0.3)

# Multiply blend
{:ok, result} = ExCubecl.Video.mix(frame_a, frame_b, mode: :multiply, ratio: 0.5)
```

## Scale / Resize

```elixir
# Downscale to 720p
{:ok, scaled} = ExCubecl.Video.scale(frame, width: 1280, height: 720)

# Upscale
{:ok, scaled} = ExCubecl.Video.scale(frame, width: 3840, height: 2160)
```

Uses bicubic interpolation on the GPU for high-quality results.

## Crop

```elixir
# Extract a region
{:ok, cropped} = ExCubecl.Video.crop(frame, x: 100, y: 100, width: 640, height: 480)
```

## GPU Filters

```elixir
# Gaussian blur
{:ok, blurred} = ExCubecl.Filter.apply(frame, :gaussian_blur, radius: 5)

# Sharpen (unsharp mask)
{:ok, sharp} = ExCubecl.Filter.apply(frame, :sharpen, strength: 1.2)

# 3D LUT color grade
{:ok, graded} = ExCubecl.Filter.apply(frame, :lut, file: "cinematic.cube")

# Chroma key (green screen)
{:ok, keyed} = ExCubecl.Filter.apply(frame, :chroma_key,
  color: {0, 177, 64}, threshold: 0.3)

# Brightness / contrast
{:ok, adjusted} = ExCubecl.Filter.apply(frame, :brightness_contrast,
  brightness: 0.1, contrast: 1.2)

# Denoise
{:ok, clean} = ExCubecl.Filter.apply(frame, :denoise, strength: 0.5)
```

## Filter Chains

```elixir
# Apply multiple filters in sequence
{:ok, result} = ExCubecl.Filter.chain(frame, [
  {:gaussian_blur, [radius: 3]},
  {:sharpen, [strength: 0.8]},
  {:lut, [file: "warm.cube"]}
])
```

## Snapshot

```elixir
# Save frame to PNG (triggers GPU→CPU readback)
:ok = ExCubecl.Video.snapshot(frame, "thumb.png")
```

Note: Snapshots involve a GPU→CPU readback and should be used sparingly in
performance-critical paths.

## Pipeline Integration

```elixir
{:ok, src} = ExCubecl.Media.open("input.mp4")
{:ok, frame} = ExCubecl.Media.read_frame(src, :video)
{:ok, blurred} = ExCubecl.buffer(List.duplicate(0.0, 1920 * 1080), [1920 * 1080], :f32)
{:ok, graded} = ExCubecl.buffer(List.duplicate(0.0, 1920 * 1080), [1920 * 1080], :f32)
{:ok, output} = ExCubecl.buffer(List.duplicate(0.0, 1920 * 1080), [1920 * 1080], :f32)

{:ok, pipeline} = ExCubecl.pipeline()
:ok = ExCubecl.pipeline_add(pipeline, "gaussian_blur", [frame.handle], blurred, %{radius: 3})
:ok = ExCubecl.pipeline_add(pipeline, "lut_apply", [blurred], graded, %{file: "warm.cube"})
:ok = ExCubecl.pipeline_add(pipeline, "overlay_alpha", [graded, watermark.handle], output, %{x: 20, y: 20, alpha: 1.0})
{:ok, _cmd_ids} = ExCubecl.pipeline_run(pipeline)
:ok = ExCubecl.pipeline_free(pipeline)
```
