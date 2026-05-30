# Video Merge & Picture-in-Picture Guide

Merge two or more video streams into a single output using GPU-accelerated
compositing. This guide covers side-by-side, vertical split, and
picture-in-Picture (PiP) layouts.

## Overview

```
┌──────────────┐
│  Stream A     │──┐
│  (main)      │  │    ┌─────────────┐    ┌──────────────┐
└──────────────┘  ├───▶│ GPU Compose  │───▶│  Encoder     │──▶ output.mp4
┌──────────────┐  │    │ (overlay)    │    │  (FFmpeg)    │
│  Stream B     │──┘    └─────────────┘    └──────────────┘
│  (PiP)       │
└──────────────┘
```

All compositing happens on the GPU via the `overlay_alpha` kernel. No CPU
readback is needed until you optionally save a snapshot.

## Opening Multiple Sources

```elixir
# Open two media sources (files, RTMP streams, cameras, etc.)
{:ok, src_a} = ExCubecl.Media.open("main_speaker.mp4")
{:ok, src_b} = ExCubecl.Media.open("remote_guest.mp4")

# Inspect stream metadata
{:ok, streams_a} = ExCubecl.Media.streams(src_a)
{:ok, streams_b} = ExCubecl.Media.streams(src_b)

# => [
#   %{index: 0, type: :video, codec: :h264, fps: 30,
#     width: 1920, height: 1080},
#   %{index: 1, type: :audio, codec: :aac,
#     sample_rate: 48000, channels: 2}
# ]
```

## Picture-in-Picture (PiP)

The most common video merge pattern: a full-screen main video with a smaller
overlay in the corner.

### Basic PiP

```elixir
defmodule PiPMerger do
  use ExCubecl.MediaPipeline

  @pip_width 320
  @pip_height 240
  @pip_x 1580    # 1920 - 320 - 20 (20px margin)
  @pip_y 820     # 1080 - 240 - 20 (20px margin)

  def start_link(opts) do
    {:ok, src_a} = ExCubecl.Media.open(opts[:main])
    {:ok, src_b} = ExCubecl.Media.open(opts[:overlay])

    {:ok, enc} = ExCubecl.Transcode.start(opts[:output],
      video: [codec: :h264, width: 1920, height: 1080, bitrate: "6M"],
      audio: [codec: :aac, bitrate: "192k", sample_rate: 48000]
    )

    state = %{
      source_a: src_a,
      source_b: src_b,
      encoder: enc,
      main_width: 1920,
      main_height: 1080
    }

    ExCubecl.MediaPipeline.start_link(__MODULE__, state, name: __MODULE__)
  end

  def handle_frame(frame, state) do
    # Read the corresponding frame from the overlay source
    {:ok, overlay_frame} = ExCubecl.Media.read_frame(state.source_b, :video)

    # Scale the overlay to PiP size
    {:ok, scaled_overlay} = ExCubecl.Video.scale(overlay_frame,
      width: @pip_width,
      height: @pip_height
    )

    # Composite the PiP onto the main frame
    {:ok, composited} = ExCubecl.Video.overlay(frame, scaled_overlay,
      x: @pip_x,
      y: @pip_y,
      alpha: 1.0
    )

    # Encode the merged frame
    :ok = ExCubecl.Transcode.write_frame(state.encoder, composited)

    {:ok, state}
  end
end

# Start the PiP merger
{:ok, _pid} = PiPMerger.start_link(
  main: "speaker.mp4",
  overlay: "guest.mp4",
  output: "merged_pip.mp4"
)

# Push frames from the main source
{:ok, src_a} = ExCubecl.Media.open("speaker.mp4")

case ExCubecl.Media.read_frame(src_a, :video) do
  {:ok, frame} ->
    ExCubecl.MediaPipeline.push_frame(PiPMerger, frame)

  {:error, :eof} ->
    IO.puts("All frames processed")
end
```

### PiP with Border and Shadow

For a more polished look, add a subtle border around the PiP window:

```elixir
defmodule StyledPiP do
  use ExCubecl.MediaPipeline

  @pip_width 400
  @pip_height 300
  @pip_x 1480
  @pip_y 760
  @border_size 3

  def handle_frame(frame, state) do
    {:ok, overlay_frame} = ExCubecl.Media.read_frame(state.source_b, :video)

    # Scale overlay to PiP size
    {:ok, scaled} = ExCubecl.Video.scale(overlay_frame,
      width: @pip_width,
      height: @pip_height
    )

    # Step 1: Overlay a dark background (border/shadow)
    # The border frame is slightly larger than the PiP
    border_x = @pip_x - @border_size
    border_y = @pip_y - @border_size
    border_w = @pip_width + @border_size * 2
    border_h = @pip_height + @border_size * 2

    # Crop a dark region from the main frame as border background
    {:ok, border_bg} = ExCubecl.Video.crop(frame,
      x: border_x,
      y: border_y,
      width: border_w,
      height: border_h
    )

    # Darken the border area
    {:ok, darkened} = ExCubecl.Filter.apply(border_bg, :brightness_contrast,
      brightness: -0.4,
      contrast: 1.0
    )

    # Composite border onto main frame
    {:ok, with_border} = ExCubecl.Video.overlay(frame, darkened,
      x: border_x,
      y: border_y,
      alpha: 0.8
    )

    # Step 2: Overlay the scaled PiP on top
    {:ok, composited} = ExCubecl.Video.overlay(with_border, scaled,
      x: @pip_x,
      y: @pip_y,
      alpha: 1.0
    )

    :ok = ExCubecl.Transcode.write_frame(state.encoder, composited)
    {:ok, state}
  end
end
```

## Side-by-Side Merge

Place two videos next to each other horizontally. Useful for comparison views
or interview formats.

```elixir
defmodule SideBySide do
  use ExCubecl.MediaPipeline

  @output_width 1920
  @output_height 540   # half of 1080p
  @half_width 960

  def start_link(opts) do
    {:ok, src_a} = ExCubecl.Media.open(opts[:left])
    {:ok, src_b} = ExCubecl.Media.open(opts[:right])

    {:ok, enc} = ExCubecl.Transcode.start(opts[:output],
      video: [codec: :h264, width: @output_width, height: @output_height,
              bitrate: "8M"],
      audio: [codec: :aac, bitrate: "192k"]
    )

    state = %{
      source_a: src_a,
      source_b: src_b,
      encoder: enc
    }

    ExCubecl.MediaPipeline.start_link(__MODULE__, state)
  end

  def handle_frame(frame, state) do
    # Read the matching frame from source B
    {:ok, frame_b} = ExCubecl.Media.read_frame(state.source_b, :video)

    # Scale both frames to half width
    {:ok, left} = ExCubecl.Video.scale(frame,
      width: @half_width,
      height: @output_height
    )

    {:ok, right} = ExCubecl.Video.scale(frame_b,
      width: @half_width,
      height: @output_height
    )

    # Create a blank output frame (full width)
    # We use the left frame as the base and overlay the right half
    {:ok, merged} = ExCubecl.Video.overlay(left, right,
      x: @half_width,
      y: 0,
      alpha: 1.0
    )

    :ok = ExCubecl.Transcode.write_frame(state.encoder, merged)
    {:ok, state}
  end
end
```

## Vertical Split (Top/Bottom)

Stack two videos vertically. Useful for showing different camera angles of
the same scene.

```elixir
defmodule VerticalSplit do
  use ExCubecl.MediaPipeline

  @output_width 1920
  @output_height 1080
  @half_height 540

  def handle_frame(frame, state) do
    {:ok, frame_b} = ExCubecl.Media.read_frame(state.source_b, :video)

    # Scale both to full width, half height
    {:ok, top} = ExCubecl.Video.scale(frame,
      width: @output_width,
      height: @half_height
    )

    {:ok, bottom} = ExCubecl.Video.scale(frame_b,
      width: @output_width,
      height: @half_height
    )

    # Stack: bottom half overlaid below top half
    {:ok, merged} = ExCubecl.Video.overlay(top, bottom,
      x: 0,
      y: @half_height,
      alpha: 1.0
    )

    :ok = ExCubecl.Transcode.write_frame(state.encoder, merged)
    {:ok, state}
  end
end
```

## 2×2 Grid (Four Streams)

Merge four video streams into a 2×2 grid layout.

```elixir
defmodule GridMerge do
  use ExCubecl.MediaPipeline

  @cell_width 960
  @cell_height 540

  def start_link(opts) do
    sources = Enum.map(opts[:inputs], &ExCubecl.Media.open/1)

    {:ok, enc} = ExCubecl.Transcode.start(opts[:output],
      video: [codec: :h264, width: 1920, height: 1080, bitrate: "10M"],
      audio: [codec: :aac, bitrate: "192k"]
    )

    state = %{
      sources: sources,
      encoder: enc
    }

    ExCubecl.MediaPipeline.start_link(__MODULE__, state)
  end

  def handle_frame(_frame, state) do
    # Read one frame from each source
    frames =
      Enum.map(state.sources, fn src ->
        case ExCubecl.Media.read_frame(src, :video) do
          {:ok, f} -> f
          {:error, :eof} -> nil  # Source exhausted
        end
      end)
      |> Enum.reject(&is_nil/1)

    # Scale all frames to cell size
    scaled = Enum.map(frames, fn f ->
      {:ok, s} = ExCubecl.Video.scale(f,
        width: @cell_width,
        height: @cell_height
      )
      s
    end)

    # Merge in 2×2 grid:
    # [0] [1]
    # [2] [3]
    merged =
      case scaled do
        [top_left, top_right, bottom_left, bottom_right] ->
          {:ok, top} = ExCubecl.Video.overlay(top_left, top_right,
            x: @cell_width, y: 0, alpha: 1.0
          )
          {:ok, bottom} = ExCubecl.Video.overlay(bottom_left, bottom_right,
            x: @cell_width, y: 0, alpha: 1.0
          )
          {:ok, grid} = ExCubecl.Video.overlay(top, bottom,
            x: 0, y: @cell_height, alpha: 1.0
          )
          grid

        [top_left, top_right] ->
          {:ok, row} = ExCubecl.Video.overlay(top_left, top_right,
            x: @cell_width, y: 0, alpha: 1.0
          )
          row

        [single] ->
          single

        [] ->
          # No frames available, send blank
          {:ok, blank} = ExCubecl.buffer(
            List.duplicate(0.0, @cell_width * @cell_height * 3),
            [@cell_width * @cell_height * 3], :f32
          )
          # Wrap in a minimal VideoFrame-like struct for overlay compat
          # In practice you'd handle this with a blank frame source
          raise "No frames available from any source"
      end

    :ok = ExCubecl.Transcode.write_frame(state.encoder, merged)
    {:ok, state}
  end
end
```

## Handling Different Resolutions

When input streams have different resolutions, normalize them before
compositing:

```elixir
defmodule ResolutionNormalizer do
  @target_width 1920
  @target_height 1080

  def normalize(frame, target_width, target_height) do
    cond do
      frame.width == target_width and frame.height == target_height ->
        frame

      true ->
        {:ok, scaled} = ExCubecl.Video.scale(frame,
          width: target_width,
          height: target_height
        )
        scaled
    end
  end

  # Scale while preserving aspect ratio (letterbox)
  def normalize_preserve_aspect(frame, target_width, target_height) do
    aspect = frame.width / frame.height
    target_aspect = target_width / target_height

    {w, h} =
      cond do
        aspect > target_aspect ->
          # Wider than target: fit to width
          {target_width, round(target_width / aspect)}

        true ->
          # Taller than target: fit to height
          {round(target_height * aspect), target_height}
      end

    {:ok, scaled} = ExCubecl.Video.scale(frame, width: w, height: h)

    # Center the scaled frame on a canvas of target size
    # by overlaying at the correct offset
    x_offset = div(target_width - w, 2)
    y_offset = div(target_height - h, 2)

    # Return the scaled frame with offset info for later compositing
    {scaled, x_offset, y_offset}
  end
end
```

## Handling Different Frame Rates

When streams have different frame rates, decide which drives the output:

```elixir
defmodule FrameRateSync do
  # Read frames from the secondary source at the pace of the primary.
  # If the secondary is slower, repeat its last frame.
  # If the secondary is faster, skip frames.

  def handle_frame(primary_frame, state) do
    # Only read a new frame from source_b when the PTS advances
    primary_pts = primary_frame.pts

    frame_b =
      if should_advance_secondary?(primary_pts, state.last_pts_b) do
        case ExCubecl.Media.read_frame(state.source_b, :video) do
          {:ok, f} ->
            Process.put(:last_frame_b, f)
            f

          {:error, :eof} ->
            # Source exhausted — repeat last frame
            Process.get(:last_frame_b, state.fallback_frame)
        end
      else
        # Reuse last frame
        Process.get(:last_frame_b, state.fallback_frame)
      end

    {:ok, composited} = ExCubecl.Video.overlay(primary_frame, frame_b,
      x: 1580, y: 820, alpha: 1.0
    )

    :ok = ExCubecl.Transcode.write_frame(state.encoder, composited)

    {:ok, %{state | last_pts_b: primary_pts}}
  end

  defp should_advance_secondary?(primary_pts, last_pts_b) do
    # Advance if primary has moved forward by at least one frame duration
    # (assuming 30fps ≈ 33333µs per frame)
    primary_pts - last_pts_b >= 33_000
  end
end
```

## Audio Merging

Merge audio from both streams simultaneously:

```elixir
defmodule AudioVideoMerge do
  use ExCubecl.MediaPipeline

  def handle_frame(video_frame, state) do
    # --- Video: PiP composite ---
    {:ok, overlay_v} = ExCubecl.Media.read_frame(state.source_b, :video)
    {:ok, scaled} = ExCubecl.Video.scale(overlay_v, width: 320, height: 240)
    {:ok, composited} = ExCubecl.Video.overlay(video_frame, scaled,
      x: 1580, y: 820, alpha: 1.0
    )

    # --- Audio: mix both tracks ---
    {:ok, audio_a} = ExCubecl.Media.read_frame(state.source_a, :audio)
    {:ok, audio_b} = ExCubecl.Media.read_frame(state.source_b, :audio)

    {:ok, mixed_audio} = ExCubecl.Audio.mix([audio_a, audio_b],
      gains: [1.0, 0.7]  # Slightly lower the overlay audio
    )

    # --- Encode both ---
    :ok = ExCubecl.Transcode.write_frame(state.encoder, composited)
    :ok = ExCubecl.Transcode.write_samples(state.encoder, mixed_audio)

    {:ok, state}
  end
end
```

## Complete End-to-End Example

Merge two RTMP streams into a single PiP output:

```elixir
defmodule ConferencePiP do
  use ExCubecl.MediaPipeline

  @pip_width 480
  @pip_height 270
  @pip_margin 24

  def start_link(opts) do
    {:ok, speaker} = ExCubecl.Media.open(opts[:speaker_url])
    {:ok, guest} = ExCubecl.Media.open(opts[:guest_url])

    {:ok, enc} = ExCubecl.Transcode.start(opts[:output],
      video: [
        codec: :h264,
        width: 1920,
        height: 1080,
        bitrate: "6M",
        fps: 30
      ],
      audio: [
        codec: :aac,
        bitrate: "192k",
        sample_rate: 48000
      ]
    )

    # Pre-create a rounded-corner mask for the PiP window
    # (optional visual polish)

    state = %{
      speaker_src: speaker,
      guest_src: guest,
      encoder: enc,
      frame_count: 0,
      errors: []
    }

    ExCubecl.MediaPipeline.start_link(__MODULE__, state,
      name: __MODULE__
    )
  end

  def handle_frame(frame, state) do
    # Read guest frame (PiP)
    case ExCubecl.Media.read_frame(state.guest_src, :video) do
      {:ok, guest_frame} ->
        pip_frame = compose_pip(frame, guest_frame)
        :ok = ExCubecl.Transcode.write_frame(state.encoder, pip_frame)

        # Mix audio
        with {:ok, spk_audio} <- ExCubecl.Media.read_frame(state.speaker_src, :audio),
             {:ok, gst_audio} <- ExCubecl.Media.read_frame(state.guest_src, :audio),
             {:ok, mixed} <- ExCubecl.Audio.mix([spk_audio, gst_audio],
               gains: [1.0, 0.8]
             ) do
          :ok = ExCubecl.Transcode.write_samples(state.encoder, mixed)
        else
          {:error, reason} ->
            IO.puts("Audio merge error: #{inspect(reason)}")
        end

        {:ok, %{state | frame_count: state.frame_count + 1}}

      {:error, :eof} ->
        # Guest stream ended — continue with speaker only
        :ok = ExCubecl.Transcode.write_frame(state.encoder, frame)
        {:ok, %{state | frame_count: state.frame_count + 1}}

      {:error, reason} ->
        IO.puts("Guest read error: #{inspect(reason)}")
        :ok = ExCubecl.Transcode.write_frame(state.encoder, frame)
        {:ok, %{state | frame_count: state.frame_count + 1}}
    end
  end

  defp compose_pip(main_frame, guest_frame) do
    # Scale guest to PiP size
    {:ok, scaled} = ExCubecl.Video.scale(guest_frame,
      width: @pip_width,
      height: @pip_height
    )

    # Position: bottom-right corner with margin
    x = main_frame.width - @pip_width - @pip_margin
    y = main_frame.height - @pip_height - @pip_margin

    # Apply slight rounded-corner effect via alpha
    {:ok, composited} = ExCubecl.Video.overlay(main_frame, scaled,
      x: x,
      y: y,
      alpha: 0.95
    )

    composited
  end

  def handle_info({:frame, frame}, state) do
    # Entry point for pushed frames
    case handle_frame(frame, state) do
      {:ok, new_state} -> {:noreply, new_state}
      {:error, reason} -> {:stop, reason, state}
    end
  end
end

# Usage:
# {:ok, _pid} = ConferencePiP.start_link(
#   speaker_url: "rtmp://server/speaker",
#   guest_url: "rtmp://server/guest",
#   output: "conference_output.mp4"
# )
```

## Saving Snapshots

Capture a still frame from the merged output at any point:

```elixir
# Save a thumbnail of the current merged frame
:ok = ExCubecl.Video.snapshot(composited_frame, "thumbnail.png")

# Save periodically (e.g., every 300 frames)
if rem(state.frame_count, 300) == 0 do
  filename = "snapshots/frame_#{state.frame_count}.png"
  File.mkdir_p!("snapshots")
  ExCubecl.Video.snapshot(composited_frame, filename)
end
```

> **Note**: `Video.snapshot/2` triggers a GPU→CPU readback. Use it sparingly
> in performance-critical paths. For real-time previews, consider reducing the
> resolution before snapshotting.

## Performance Tips

1. **Scale before overlay**: Always scale the overlay to its final size
   before compositing. Scaling a small region is cheaper than scaling the
   entire frame.

2. **Match resolutions early**: If both sources are the same resolution,
   skip the scale step entirely.

3. **Use filter chains**: Combine scale + color correction in a single
   `Filter.chain/2` call to minimize GPU kernel launches.

4. **Avoid snapshots in the hot path**: GPU→CPU readbacks are expensive.
   Save thumbnails asynchronously or at low frequency.

5. **Pipeline mode for fixed layouts**: If your merge layout doesn't change,
   use `ExCubecl.pipeline()` directly instead of the GenServer for lower
   overhead:

```elixir
{:ok, pipeline} = ExCubecl.pipeline()
:ok = ExCubecl.pipeline_add(pipeline, "bicubic_scale",
  [guest_frame.handle], pip_scaled, %{width: 320, height: 240})
:ok = ExCubecl.pipeline_add(pipeline, "overlay_alpha",
  [main_frame.handle, pip_scaled], output, %{x: 1580, y: 820, alpha: 1.0})
{:ok, _cmd_ids} = ExCubecl.pipeline_run(pipeline)
:ok = ExCubecl.pipeline_free(pipeline)
```

6. **Handle stream endings gracefully**: Always match on `{:error, :eof}`
   when reading from secondary sources. Decide whether to continue with
   the primary only or stop the merge.
