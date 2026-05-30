# Media Quickstart

ExCubecl Phase 2 adds GPU-accelerated media processing: video/audio I/O,
GPU filters, transcoding, and real-time streaming pipelines.

## Prerequisites

- ExCubecl v0.4.0+
- FFmpeg ≥ 6.x libraries (libavcodec, libavformat, libavfilter, libswresample)
  - macOS: `brew install ffmpeg`
  - Ubuntu: `apt install libavcodec-dev libavformat-dev libavfilter-dev libswresample-dev`

## Opening a Media Source

```elixir
{:ok, src} = ExCubecl.Media.open("input.mp4")

# Inspect streams
{:ok, streams} = ExCubecl.Media.streams(src)
# => [
#   %{index: 0, type: :video, codec: :h264, fps: 30, width: 1920, height: 1080},
#   %{index: 1, type: :audio, codec: :aac, sample_rate: 44100, channels: 2}
# ]
```

## Reading Frames

```elixir
# Read a video frame (returns %VideoFrame{})
{:ok, frame} = ExCubecl.Media.read_frame(src, :video)
frame.width    # 1920
frame.height   # 1080
frame.format   # :yuv420p
frame.pts      # 0 (microseconds)

# Read audio samples (returns %AudioSamples{})
{:ok, samples} = ExCubecl.Media.read_frame(src, :audio)
samples.channels     # 2
samples.sample_rate  # 48000
samples.frames       # 1024
```

## Applying GPU Filters

```elixir
# Single filter
{:ok, blurred} = ExCubecl.Filter.apply(frame, :gaussian_blur, radius: 5)
{:ok, sharp}   = ExCubecl.Filter.apply(frame, :sharpen, strength: 1.2)

# Filter chain
{:ok, result} = ExCubecl.Filter.chain(frame, [
  {:gaussian_blur, [radius: 3]},
  {:lut, [file: "warm.cube"]}
])
```

## Video Operations

```elixir
# Overlay (alpha composite)
{:ok, composited} = ExCubecl.Video.overlay(base, overlay, x: 100, y: 50, alpha: 0.8)

# Mix / blend
{:ok, blended} = ExCubecl.Video.mix(frame_a, frame_b, mode: :dissolve, ratio: 0.5)

# Scale
{:ok, scaled} = ExCubecl.Video.scale(frame, width: 1280, height: 720)

# Color space conversion
{:ok, rgb} = ExCubecl.Video.convert(frame, :yuv420p, :rgb24)

# Crop
{:ok, cropped} = ExCubecl.Video.crop(frame, x: 0, y: 0, width: 640, height: 360)
```

## Audio Operations

```elixir
# Mix tracks with gain
{:ok, mixed} = ExCubecl.Audio.mix([track_a, track_b], gains: [0.7, 0.5])

# Overlay with ducking
{:ok, mixed} = ExCubecl.Audio.overlay(bg, fg, duck_level: -12)

# Resample
{:ok, resampled} = ExCubecl.Audio.resample(samples, from: 44100, to: 48000)

# Channel conversion
{:ok, mono} = ExCubecl.Audio.channels(samples, :stereo, :mono)
```

## Transcoding

```elixir
# File-to-file
ExCubecl.Transcode.run("input.mp4", "output.mp4",
  video: [codec: :h264, bitrate: "4M", fps: 30],
  audio: [codec: :aac, bitrate: "192k", sample_rate: 48000]
)

# Frame-by-frame streaming
{:ok, enc} = ExCubecl.Transcode.start("output.mp4",
  video: [codec: :h265, width: 1280, height: 720],
  audio: [codec: :aac]
)

ExCubecl.Transcode.write_frame(enc, processed_frame)
ExCubecl.Transcode.write_samples(enc, processed_audio)
ExCubecl.Transcode.finish(enc)
```

## Real-time Pipeline (GenServer)

```elixir
defmodule MyLivestream do
  use ExCubecl.MediaPipeline

  def handle_frame(frame, state) do
    frame
    |> ExCubecl.Filter.apply(:gaussian_blur, radius: 2)
    |> ExCubecl.Video.overlay(state.logo, x: 10, y: 10)
    |> ExCubecl.Transcode.write_frame(state.encoder)

    {:ok, state}
  end
end

{:ok, pid} = ExCubecl.MediaPipeline.start_link(MyLivestream, %{logo: logo, encoder: enc})
ExCubecl.MediaPipeline.push_frame(pid, frame)
```

## Architecture

```
FFmpeg (Rust)         →  decode raw frames, encode output
CubeCL GPU kernels    →  all processing in between (filters, mix, effects)
Elixir               →  orchestration, pipeline composition
```

Frames flow: FFmpeg decoder → GPU texture → CubeCL kernel → GPU texture → FFmpeg encoder
without touching Elixir memory.
