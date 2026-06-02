# GPU Video Processing Example
# Demonstrates video frame operations: overlay, mix, scale, crop, convert, filters
# Run with: mix run examples/video_processing.exs

IO.puts("=== GPU Video Processing Pipeline ===")
IO.puts("")

# Check device
{:ok, info} = ExCubecl.device_info()
IO.puts("Device: #{info.device_name}")
IO.puts("")

# ── Open a media source ──────────────────────────────────────
IO.puts("--- Media Source ---")

{:ok, src} = ExCubecl.Media.open("sample_video.mp4")
{:ok, streams} = ExCubecl.Media.streams(src)

Enum.each(streams, fn stream ->
  case stream.type do
    :video ->
      IO.puts("Video stream: #{stream.codec} #{stream.width}x#{stream.height} @ #{stream.fps}fps")

    :audio ->
      IO.puts("Audio stream: #{stream.codec} #{stream.sample_rate}Hz #{stream.channels}ch")
  end
end)

IO.puts("")

# ── Read video frames ────────────────────────────────────────
IO.puts("--- Reading Frames ---")

{:ok, frame_a} = ExCubecl.Media.read_frame(src, :video)
{:ok, frame_b} = ExCubecl.Media.read_frame(src, :video)

IO.puts("Frame A: #{frame_a.width}x#{frame_a.height} format=#{frame_a.format} pts=#{frame_a.pts}")
IO.puts("Frame B: #{frame_b.width}x#{frame_b.height} format=#{frame_b.format} pts=#{frame_b.pts}")
IO.puts("")

# ── Video operations ─────────────────────────────────────────
IO.puts("--- Video Operations ---")

# Overlay: composite frame_b onto frame_a at position (50, 25) with 80% opacity
{:ok, overlaid} = ExCubecl.Video.overlay(frame_a, frame_b, x: 50, y: 25, alpha: 0.8)
IO.puts("Overlay: #{overlaid.width}x#{overlaid.height}")

# Mix: dissolve blend at 50%
{:ok, blended} = ExCubecl.Video.mix(frame_a, frame_b, mode: :dissolve, ratio: 0.5)
IO.puts("Mix (dissolve, 0.5): #{blended.width}x#{blended.height}")

# Scale: resize to 720p
{:ok, scaled} = ExCubecl.Video.scale(frame_a, width: 1280, height: 720)
IO.puts("Scale: #{scaled.width}x#{scaled.height}")

# Crop: extract a 640x360 region starting at (100, 50)
{:ok, cropped} = ExCubecl.Video.crop(frame_a, x: 100, y: 50, width: 640, height: 360)
IO.puts("Crop: #{cropped.width}x#{cropped.height}")

# Convert: YUV420p → RGB24
{:ok, rgb} = ExCubecl.Video.convert(frame_a, :yuv420p, :rgb24)
IO.puts("Convert: #{frame_a.format} → #{rgb.format}")
IO.puts("")

# ── GPU filters ──────────────────────────────────────────────
IO.puts("--- GPU Filters ---")

# Apply gaussian blur
{:ok, _blurred} = ExCubecl.Filter.apply(frame_a, :gaussian_blur, radius: 5)
IO.puts("Gaussian blur: radius=5")

# Apply sharpen
{:ok, _sharpened} = ExCubecl.Filter.apply(frame_a, :sharpen, strength: 1.5)
IO.puts("Sharpen: strength=1.5")

# Apply chroma key (green screen)
{:ok, _keyed} = ExCubecl.Filter.apply(frame_a, :chroma_key, color: {0, 177, 64}, threshold: 0.3)
IO.puts("Chroma key: green screen")

# Chain multiple filters: blur → sharpen
{:ok, _filtered} =
  ExCubecl.Filter.chain(frame_a, [
    {:gaussian_blur, [radius: 3]},
    {:sharpen, [strength: 1.0]}
  ])

IO.puts("Filter chain: blur(3) → sharpen(1.0)")
IO.puts("")

# ── Pipeline orchestration ───────────────────────────────────
IO.puts("--- Processing Pipeline ---")

# Build a multi-stage video processing pipeline
{:ok, pipeline} = ExCubecl.pipeline()

# Stage 1: Denoise
{:ok, denoised} = ExCubecl.buffer(List.duplicate(0.0, 25), [5, 5], :f32)
ExCubecl.pipeline_add(pipeline, "denoise", [frame_a.handle], denoised)

# Stage 2: Gaussian blur
{:ok, blurred_buf} = ExCubecl.buffer(List.duplicate(0.0, 25), [5, 5], :f32)
ExCubecl.pipeline_add(pipeline, "gaussian_blur", [denoised], blurred_buf)

# Stage 3: Sharpen
{:ok, final_buf} = ExCubecl.buffer(List.duplicate(0.0, 25), [5, 5], :f32)
ExCubecl.pipeline_add(pipeline, "sharpen", [blurred_buf], final_buf)

IO.puts("Pipeline: denoise → gaussian_blur → sharpen (3 stages)")

{:ok, cmd_ids} = ExCubecl.pipeline_run(pipeline)
IO.puts("Pipeline executed: #{length(cmd_ids)} commands")

:ok = ExCubecl.pipeline_free(pipeline)
IO.puts("")

# ── Snapshot ─────────────────────────────────────────────────
IO.puts("--- Snapshot ---")

snapshot_path = "/tmp/video_snapshot_#{System.unique_integer([:positive])}.raw"
:ok = ExCubecl.Video.snapshot(frame_a, snapshot_path)
IO.puts("Snapshot saved: #{snapshot_path}")
File.rm(snapshot_path)
IO.puts("")

# ── Cleanup ──────────────────────────────────────────────────
IO.puts("--- Cleanup ---")

:ok = ExCubecl.Media.close(src)
IO.puts("Media source closed")

# Buffers are automatically freed when GC'd — no manual free needed

IO.puts("")
IO.puts("=== Done ===")
