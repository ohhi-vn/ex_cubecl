# GPU Audio Processing Example
# Demonstrates audio operations: mix, overlay, resample, channel conversion, filters
# Run with: mix run examples/audio_processing.exs

IO.puts("=== GPU Audio Processing Pipeline ===")
IO.puts("")

# Check device
{:ok, info} = ExCubecl.device_info()
IO.puts("Device: #{info.device_name}")
IO.puts("")

# ── Open a media source ──────────────────────────────────────
IO.puts("--- Media Source ---")

{:ok, src} = ExCubecl.Media.open("sample_audio.mp4")
{:ok, streams} = ExCubecl.Media.streams(src)

Enum.each(streams, fn stream ->
  case stream.type do
    :video ->
      IO.puts("Video stream: #{stream.codec} #{stream.width}x#{stream.height}")

    :audio ->
      IO.puts("Audio stream: #{stream.codec} #{stream.sample_rate}Hz #{stream.channels}ch")
  end
end)

IO.puts("")

# ── Read audio samples ───────────────────────────────────────
IO.puts("--- Reading Audio Samples ---")

{:ok, samples_a} = ExCubecl.Media.read_frame(src, :audio)
{:ok, samples_b} = ExCubecl.Media.read_frame(src, :audio)
{:ok, samples_c} = ExCubecl.Media.read_frame(src, :audio)

IO.puts("Track A: #{samples_a.channels}ch #{samples_a.sample_rate}Hz #{samples_a.frames} frames")
IO.puts("Track B: #{samples_b.channels}ch #{samples_b.sample_rate}Hz #{samples_b.frames} frames")
IO.puts("Track C: #{samples_c.channels}ch #{samples_c.sample_rate}Hz #{samples_c.frames} frames")
IO.puts("")

# ── Audio mixing ─────────────────────────────────────────────
IO.puts("--- Audio Mixing ---")

# Mix two tracks with equal gain
{:ok, mixed} = ExCubecl.Audio.mix([samples_a, samples_b])
IO.puts("Mix (2 tracks, equal gain): #{mixed.channels}ch #{mixed.sample_rate}Hz")

# Mix three tracks with custom gains
{:ok, mixed_3} = ExCubecl.Audio.mix([samples_a, samples_b, samples_c], gains: [0.5, 0.3, 0.2])
IO.puts("Mix (3 tracks, gains=[0.5, 0.3, 0.2]): #{mixed_3.channels}ch")
IO.puts("")

# ── Audio overlay with ducking ───────────────────────────────
IO.puts("--- Audio Overlay (Ducking) ---")

# Overlay voiceover on background music with -12dB ducking
{:ok, overlaid} = ExCubecl.Audio.overlay(samples_a, samples_b, duck_level: -12)
IO.puts("Overlay (duck=-12dB): #{overlaid.channels}ch #{overlaid.sample_rate}Hz")

# Overlay with aggressive ducking for voiceover clarity
{:ok, overlaid_deep} = ExCubecl.Audio.overlay(samples_a, samples_b, duck_level: -24)
IO.puts("Overlay (duck=-24dB): #{overlaid_deep.channels}ch #{overlaid_deep.sample_rate}Hz")
IO.puts("")

# ── Resampling ───────────────────────────────────────────────
IO.puts("--- Resampling ---")

# Upsample 44.1kHz → 48kHz (common for video production)
{:ok, upsampled} = ExCubecl.Audio.resample(samples_a, from: 44_100, to: 48_000)
IO.puts("Upsample: 44100Hz → 48000Hz (#{samples_a.frames} → #{upsampled.frames} frames)")

# Downsample 96kHz → 44.1kHz (mastering)
{:ok, downsampled} = ExCubecl.Audio.resample(samples_a, from: 96_000, to: 44_100)
IO.puts("Downsample: 96000Hz → 44100Hz (#{samples_a.frames} → #{downsampled.frames} frames)")

# Same-rate resample (no-op)
{:ok, same_rate} = ExCubecl.Audio.resample(samples_a, from: 48_000, to: 48_000)
IO.puts("Same rate: 48000Hz → 48000Hz (#{same_rate.frames} frames)")
IO.puts("")

# ── Channel conversion ───────────────────────────────────────
IO.puts("--- Channel Conversion ---")

# Stereo → Mono (for voice analysis)
{:ok, mono} = ExCubecl.Audio.channels(samples_a, :stereo, :mono)
IO.puts("Stereo → Mono: #{mono.channels}ch")

# Mono → Stereo (for playback compatibility)
{:ok, stereo} = ExCubecl.Audio.channels(mono, :mono, :stereo)
IO.puts("Mono → Stereo: #{stereo.channels}ch")

# Stereo → 5.1 Surround (for theater mix)
{:ok, surround_51} = ExCubecl.Audio.channels(samples_a, :stereo, :surround_51)
IO.puts("Stereo → 5.1 Surround: #{surround_51.channels}ch")

# Stereo → 7.1 Surround (for IMAX)
{:ok, surround_71} = ExCubecl.Audio.channels(samples_a, :stereo, :surround_71)
IO.puts("Stereo → 7.1 Surround: #{surround_71.channels}ch")
IO.puts("")

# ── GPU audio filters ────────────────────────────────────────
IO.puts("--- GPU Audio Filters ---")

# Normalize audio levels
{:ok, normalized} = ExCubecl.Filter.apply(samples_a, :normalize)
IO.puts("Normalize: #{normalized.channels}ch #{normalized.sample_rate}Hz")

# Apply EQ (biquad filter)
{:ok, _eq_applied} = ExCubecl.Filter.apply(samples_a, :eq, bands: [{:high_pass, 80}])
IO.puts("EQ: high_pass=80Hz")

# Apply dynamics compressor
{:ok, _compressed} = ExCubecl.Filter.apply(samples_a, :compressor, threshold: -18, ratio: 4.0)
IO.puts("Compressor: threshold=-18dB ratio=4:1")

# Apply reverb (FFT convolution)
{:ok, _reverbed} = ExCubecl.Filter.apply(samples_a, :reverb, room_size: 0.5, wet: 0.2)
IO.puts("Reverb: room_size=0.5 wet=0.2")
IO.puts("")

# ── Audio processing pipeline ────────────────────────────────
IO.puts("--- Audio Processing Pipeline ---")

# Build a mastering chain: normalize → compress → EQ
{:ok, pipeline} = ExCubecl.pipeline()

{:ok, stage1_out} = ExCubecl.buffer(List.duplicate(0.0, 8), [8], :f32)
ExCubecl.pipeline_add(pipeline, "pcm_normalize", [samples_a.handle], stage1_out)

{:ok, stage2_out} = ExCubecl.buffer(List.duplicate(0.0, 8), [8], :f32)
ExCubecl.pipeline_add(pipeline, "dynamics_compress", [stage1_out], stage2_out)

{:ok, stage3_out} = ExCubecl.buffer(List.duplicate(0.0, 8), [8], :f32)
ExCubecl.pipeline_add(pipeline, "biquad_filter", [stage2_out], stage3_out)

IO.puts("Pipeline: normalize → compress → EQ (3 stages)")

{:ok, cmd_ids} = ExCubecl.pipeline_run(pipeline)
IO.puts("Pipeline executed: #{length(cmd_ids)} commands")

:ok = ExCubecl.pipeline_free(pipeline)
IO.puts("")

# ── Async batch processing ───────────────────────────────────
IO.puts("--- Async Batch Processing ---")

# Process multiple audio segments concurrently
cmd_ids =
  for i <- 1..5 do
    {:ok, cmd} = ExCubecl.submit("process_segment_#{i}")
    cmd
  end

IO.puts("Submitted #{length(cmd_ids)} async audio processing commands")

# Wait for all to complete
results = Enum.map(cmd_ids, &ExCubecl.wait/1)
completed = Enum.count(results, &(&1 == :ok))
IO.puts("Completed: #{completed}/#{length(cmd_ids)}")

IO.puts("")

# ── Cleanup ──────────────────────────────────────────────────
IO.puts("--- Cleanup ---")

:ok = ExCubecl.Media.close(src)
IO.puts("Media source closed")

# Buffers are automatically freed when GC'd — no manual free needed

IO.puts("")
IO.puts("=== Done ===")
