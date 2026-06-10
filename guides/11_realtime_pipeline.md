# Real-time Pipeline Guide

Build media processing pipelines using the `MediaPipeline` GenServer
behaviour. The behaviour processes frames through NIF-backed filter and transcode operations.

## MediaPipeline Behaviour

```elixir
defmodule MyPipeline do
  use ExCubecl.MediaPipeline

  @impl true
  def handle_frame(frame, state) do
    # Process the frame
    {:ok, new_state}
  end
end
```

The `handle_frame/2` callback receives each incoming frame and the current
state. Return `{:ok, new_state}` to continue or `{:error, reason}` to stop.

## Basic Example

```elixir
defmodule SimpleFilter do
  use ExCubecl.MediaPipeline

  def handle_frame(frame, state) do
    frame
    |> ExCubecl.Filter.apply(:gaussian_blur, radius: 2)
    |> ExCubecl.Transcode.write_frame(state.encoder)

    {:ok, state}
  end
end

# Start the pipeline
{:ok, enc} = ExCubecl.Transcode.start("output.mp4",
  video: [codec: :h264, width: 1280, height: 720]
)

{:ok, pid} = ExCubecl.MediaPipeline.start_link(SimpleFilter, %{encoder: enc})

# Push frames
ExCubecl.MediaPipeline.push_frame(pid, frame)
```

## Livestream with Overlay

```elixir
defmodule Livestream do
  use ExCubecl.MediaPipeline

  def handle_frame(frame, state) do
    frame
    |> ExCubecl.Filter.apply(:denoise, strength: 0.3)
    |> ExCubecl.Video.overlay(state.logo, x: 10, y: 10, alpha: 0.9)
    |> ExCubecl.Transcode.write_frame(state.encoder)

    {:ok, state}
  end
end
```

## Camera Effects Pipeline

```elixir
defmodule CameraEffects do
  use ExCubecl.MediaPipeline

  def handle_frame(frame, state) do
    frame
    |> ExCubecl.Filter.apply(:brightness_contrast, brightness: 0.05, contrast: 1.1)
    |> ExCubecl.Filter.apply(:lut, file: "portrait.cube")
    |> ExCubecl.Transcode.write_frame(state.encoder)

    {:ok, %{state | frame_count: state.frame_count + 1}}
  end
end

{:ok, pid} = ExCubecl.MediaPipeline.start_link(CameraEffects, %{
  encoder: enc,
  frame_count: 0
})
```

## Multi-Stage Processing

```elixir
defmodule MultiStage do
  use ExCubecl.MediaPipeline

  def handle_frame(frame, state) do
    # Stage 1: Denoise
    {:ok, clean} = ExCubecl.Filter.apply(frame, :denoise, strength: 0.5)

    # Stage 2: Color grade
    {:ok, graded} = ExCubecl.Filter.apply(clean, :lut, file: "cinematic.cube")

    # Stage 3: Composite
    {:ok, composited} = ExCubecl.Video.overlay(graded, state.lower_third,
      x: 50, y: 620, alpha: 1.0)

    # Stage 4: Encode
    :ok = ExCubecl.Transcode.write_frame(state.encoder, composited)

    {:ok, state}
  end
end
```

## Error Handling

```elixir
defmodule RobustPipeline do
  use ExCubecl.MediaPipeline

  def handle_frame(frame, state) do
    case ExCubecl.Filter.apply(frame, :gaussian_blur, radius: 2) do
      {:ok, processed} ->
        case ExCubecl.Transcode.write_frame(state.encoder, processed) do
          :ok ->
            {:ok, state}

          {:error, reason} ->
            IO.puts("Encode error: #{inspect(reason)}")
            {:ok, state}  # Continue with next frame
        end

      {:error, reason} ->
        IO.puts("Filter error: #{inspect(reason)}")
        {:ok, state}  # Continue with next frame
    end
  end
end
```

## Named Pipelines

```elixir
{:ok, _pid} = ExCubecl.MediaPipeline.start_link(MyPipeline, state, name: :livestream)

# Push frame by name
ExCubecl.MediaPipeline.push_frame(:livestream, frame)
```

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Media Source  │────▶│ MediaPipeline │────▶│ Encoder API  │
│ (prototype)   │     │ (GenServer)   │     │ (stub)       │
└──────────────┘     └──────────────┘     └──────────────┘
                           │
                    ┌─────────────┐
                    │ NIF-backed   │
                    │ Operations   │
                    └─────────────┘
```

## Performance Tips

1. **Batch frames**: Process multiple frames in `handle_frame` when possible.
2. **Minimize readbacks**: Avoid `Video.snapshot/2` in hot paths where practical.
3. **Use filter chains**: `Filter.chain/2` groups multiple filters into a single pipeline execution.
4. **Pipeline mode**: For fixed processing graphs, use `ExCubecl.pipeline()` directly
   instead of the GenServer for lower overhead.

## Full Example: RTMP Ingest → Process → HLS Output

```elixir
defmodule RTMPProcessor do
  use ExCubecl.MediaPipeline

  def start_link(opts) do
    {:ok, src} = ExCubecl.Media.open("rtmp://ingest/server")
    {:ok, enc} = ExCubecl.Transcode.start("output/hls/playlist.m3u8",
      video: [codec: :h264, width: 1280, height: 720, bitrate: "3M"],
      audio: [codec: :aac, bitrate: "128k"]
    )

    state = %{
      source: src,
      encoder: enc,
      logo: opts[:logo],
      frame_count: 0
    }

    ExCubecl.MediaPipeline.start_link(__MODULE__, state, name: __MODULE__)
  end

  def handle_frame(frame, state) do
    frame
    |> ExCubecl.Filter.apply(:denoise, strength: 0.3)
    |> ExCubecl.Video.overlay(state.logo, x: 10, y: 10)
    |> ExCubecl.Transcode.write_frame(state.encoder)

    {:ok, %{state | frame_count: state.frame_count + 1}}
  end
end
```
