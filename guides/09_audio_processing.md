# Audio Processing Guide

Complete reference for GPU-accelerated audio operations in ExCubecl.

## AudioSamples Struct

```elixir
%ExCubecl.AudioSamples{
  handle:      #Reference<...>,   # GPU buffer (f32 planar PCM)
  channels:    2,
  sample_rate: 48000,
  frames:      1024,              # samples per channel
  pts:         0                 # presentation timestamp (µs)
}
```

## Mixing

```elixir
# Simple mix (equal gain)
{:ok, mixed} = ExCubecl.Audio.mix([track_a, track_b])

# Mix with per-track gain
{:ok, mixed} = ExCubecl.Audio.mix([track_a, track_b], gains: [0.7, 0.5])

# Mix many tracks
{:ok, mixed} = ExCubecl.Audio.mix(
  [dialogue, music, sfx],
  gains: [1.0, 0.3, 0.6]
)
```

The `pcm_mix` kernel sums N tracks with per-channel gain on the GPU.

## Overlay with Ducking

```elixir
# Overlay foreground on background, ducking background by 12dB
{:ok, mixed} = ExCubecl.Audio.overlay(bg, fg, duck_level: -12)

# Custom duck level
{:ok, mixed} = ExCubecl.Audio.overlay(bg, fg, duck_level: -6)
```

## Resampling

```elixir
# 44.1kHz → 48kHz
{:ok, resampled} = ExCubecl.Audio.resample(samples, from: 44100, to: 48000)

# 48kHz → 44.1kHz
{:ok, resampled} = ExCubecl.Audio.resample(samples, from: 48000, to: 44100)
```

Uses GPU-accelerated linear interpolation. For higher quality, use the
`biquad_filter` kernel for band-limited resampling.

## Channel Conversion

```elixir
# Stereo → Mono (downmix)
{:ok, mono} = ExCubecl.Audio.channels(samples, :stereo, :mono)

# Mono → Stereo (upmix)
{:ok, stereo} = ExCubecl.Audio.channels(mono_samples, :mono, :stereo)

# Supported layouts: :mono, :stereo, :surround_51, :surround_71
```

## GPU Audio Filters

```elixir
# EQ (via biquad filter)
{:ok, eq'd} = ExCubecl.Filter.apply(samples, :eq,
  bands: [{:high_pass, 80}, {:shelf_high, 8000, 3.0}])

# Compressor
{:ok, compressed} = ExCubecl.Filter.apply(samples, :compressor,
  threshold: -18, ratio: 4.0)

# Reverb (FFT convolution)
{:ok, reverbed} = ExCubecl.Filter.apply(samples, :reverb,
  room_size: 0.5, wet: 0.2)

# Normalize
{:ok, normalized} = ExCubecl.Filter.apply(samples, :normalize)
```

## Audio Filter Chains

```elixir
{:ok, result} = ExCubecl.Filter.chain(samples, [
  {:eq, [bands: [{:high_pass, 80}]]},
  {:compressor, [threshold: -18, ratio: 4.0]},
  {:normalize, []}
])
```

## Pipeline Integration

```elixir
ExCubecl.pipeline()
|> ExCubecl.pipeline_add(%{op: :read_frame, source: src})
|> ExCubecl.pipeline_add(%{op: :filter, kernel: :pcm_mix, params: %{gains: [0.7, 0.5]}})
|> ExCubecl.pipeline_add(%{op: :filter, kernel: :biquad_filter, params: %{bands: [{:high_pass, 80}]}})
|> ExCubecl.pipeline_add(%{op: :encode, encoder: enc})
|> ExCubecl.pipeline_run()
```
