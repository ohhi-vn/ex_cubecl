# Transcoding Guide

Complete reference for encoding and muxing media output with ExCubecl.

## Supported Codecs

| Type  | Codecs                  |
|-------|-------------------------|
| Video | h264, h265, vp9, av1, prores |
| Audio | aac, opus, mp3, flac, pcm     |

## Supported Containers

`mp4`, `mkv`, `webm`, `mov`, `ts`

## File-to-File Transcode

```elixir
ExCubecl.Transcode.run("input.mp4", "output.mp4",
  video: [codec: :h264, bitrate: "4M", fps: 30, width: 1280, height: 720],
  audio: [codec: :aac, bitrate: "192k", sample_rate: 48000]
)
```

This is a convenience wrapper that:
1. Opens the input file
2. Creates an encoder for the output
3. Reads frames, encodes, and writes them
4. Finalizes the output

## Frame-by-Frame Streaming Transcode

For real-time processing where you need to apply filters before encoding:

```elixir
# Start encoder
{:ok, enc} = ExCubecl.Transcode.start("output.mp4",
  video: [codec: :h265, width: 1280, height: 720, bitrate: "8M"],
  audio: [codec: :aac, bitrate: "192k"]
)

# Process and write frames
{:ok, frame} = ExCubecl.Media.read_frame(src, :video)
{:ok, processed} = ExCubecl.Filter.apply(frame, :gaussian_blur, radius: 2)
:ok = ExCubecl.Transcode.write_frame(enc, processed)

# Write audio
{:ok, samples} = ExCubecl.Media.read_frame(src, :audio)
:ok = ExCubecl.Transcode.write_samples(enc, samples)

# Finalize
:ok = ExCubecl.Transcode.finish(enc)
```

## H.264 Encoding

```elixir
ExCubecl.Transcode.run("input.mp4", "output.mp4",
  video: [
    codec: :h264,
    bitrate: "4M",
    fps: 30,
    width: 1920,
    height: 1080
  ]
)
```

## H.265 / HEVC Encoding

```elixir
ExCubecl.Transcode.run("input.mp4", "output.mp4",
  video: [
    codec: :h265,
    bitrate: "8M",
    fps: 60,
    width: 3840,
    height: 2160
  ]
)
```

## VP9 Encoding (WebM)

```elixir
ExCubecl.Transcode.run("input.mp4", "output.webm",
  video: [codec: :vp9, bitrate: "4M"],
  audio: [codec: :opus, bitrate: "128k"]
)
```

## AV1 Encoding

```elixir
ExCubecl.Transcode.run("input.mp4", "output.mkv",
  video: [codec: :av1, bitrate: "6M"],
  audio: [codec: :opus, bitrate: "128k"]
)
```

## ProRes Encoding (MOV)

```elixir
ExCubecl.Transcode.run("input.mp4", "output.mov",
  video: [codec: :prores, width: 1920, height: 1080],
  audio: [codec: :pcm, sample_rate: 48000]
)
```

## Audio-Only Transcode

```elixir
ExCubecl.Transcode.run("input.mp4", "output.aac",
  audio: [codec: :aac, bitrate: "192k", sample_rate: 48000]
)
```

## Error Handling

```elixir
case ExCubecl.Transcode.run("input.mp4", "output.mp4", video: [codec: :h264]) do
  :ok ->
    IO.puts("Transcode complete")

  {:error, reason} ->
    IO.puts("Transcode failed: #{inspect(reason)}")
end
```

## Validation

The transcode module validates codecs and containers at the Elixir level:

```elixir
# Raises ArgumentError for unsupported codec
ExCubecl.Transcode.start("out.mp4", video: [codec: :invalid])
# ** (ArgumentError) unsupported video codec: :invalid. Supported: h264, h265, vp9, av1, prores

# Raises ArgumentError for unsupported container
ExCubecl.Transcode.start("out.avi", video: [codec: :h264])
# ** (ArgumentError) unsupported container: avi. Supported: mp4, mkv, webm, mov, ts
```
