# Transcoding Guide

Complete reference for the `ExCubecl.Transcode` API. The current release validates codec/container options and manages the encoder lifecycle via the NIF.

## Supported Codec Names (API Validation Only)

| Type  | Codecs                  |
|-------|-------------------------|
| Video | h264, h265, vp9, av1, prores |
| Audio | aac, opus, mp3, flac, pcm     |

## Supported Container Names (API Validation Only)

`mp4`, `mkv`, `webm`, `mov`, `ts`

## File-to-File Transcode

```elixir
ExCubecl.Transcode.run("input.mp4", "output.mp4",
  video: [codec: :h264, bitrate: "4M", fps: 30, width: 1280, height: 720],
  audio: [codec: :aac, bitrate: "192k", sample_rate: 48000]
)
```

This example shows the file-to-file transcode API:

## Frame-by-Frame Streaming

The example below shows the frame-by-frame streaming API:

```elixir
# Start encoder (API prototype; no real encoder is created)
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
    IO.puts("Transcode API validation complete")

  {:error, reason} ->
    IO.puts("Transcode failed: #{inspect(reason)}")
end
```

## Validation

The transcode module validates codec and container names at the Elixir level before passing to the NIF.

```elixir
# Returns error tuple for unsupported codec
ExCubecl.Transcode.start("out.mp4", video: [codec: :invalid])
# {:error, {:unsupported_codec, :video, :invalid, nil}}

# Returns error tuple for unsupported container
ExCubecl.Transcode.start("out.avi", video: [codec: :h264])
# {:error, {:unsupported_container, "avi", nil}}
```
