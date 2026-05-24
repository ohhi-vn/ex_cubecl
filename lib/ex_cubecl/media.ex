defmodule ExCubecl.Media do
  @moduledoc """
  Media I/O — open, inspect, read frames, and close media sources.

  Supports files, RTMP streams, HLS playlists, and camera devices.
  Under the hood, FFmpeg (via Rust NIFs) handles demuxing and decoding.

  ## Examples

      {:ok, src} = ExCubecl.Media.open("input.mp4")
      {:ok, src} = ExCubecl.Media.open("rtmp://live/stream")

      ExCubecl.Media.streams(src)
      # => [%{index: 0, type: :video, codec: :h264, fps: 30, width: 1920, height: 1080},
      #     %{index: 1, type: :audio, codec: :aac, sample_rate: 44100, channels: 2}]

      {:ok, frame} = ExCubecl.Media.read_frame(src, :video)
      {:ok, samples} = ExCubecl.Media.read_frame(src, :audio)

      :ok = ExCubecl.Media.close(src)
  """

  alias ExCubecl.NIF
  alias ExCubecl.VideoFrame
  alias ExCubecl.AudioSamples

  @type source :: reference()
  @type stream_info :: %{
          index: non_neg_integer(),
          type: :video | :audio,
          codec: atom(),
          fps: float(),
          width: non_neg_integer(),
          height: non_neg_integer(),
          sample_rate: non_neg_integer(),
          channels: non_neg_integer()
        }

  @doc """
  Opens a media source for reading.

  Accepts file paths, URLs (RTMP, HLS), or device identifiers.

  ## Examples

      {:ok, src} = ExCubecl.Media.open("input.mp4")
      {:ok, src} = ExCubecl.Media.open("rtmp://live/stream")
  """
  @spec open(String.t()) :: {:ok, source()} | {:error, term()}
  def open(path) when is_binary(path) do
    NIF.media_open(path)
  end

  @doc """
  Returns metadata for all streams in the media source.

  Each stream map contains `:index`, `:type`, `:codec`, and type-specific fields
  (`:width`/`:height`/`:fps` for video, `:sample_rate`/`:channels` for audio).
  """
  @spec streams(source()) :: {:ok, [stream_info()]} | {:error, term()}
  def streams(src) when is_reference(src) do
    NIF.media_streams(src)
  end

  @doc """
  Reads the next decoded frame from the media source.

  Pass `:video` to get a `VideoFrame`, or `:audio` to get `AudioSamples`.

  ## Examples

      {:ok, %VideoFrame{} = frame} = ExCubecl.Media.read_frame(src, :video)
      {:ok, %AudioSamples{} = samples} = ExCubecl.Media.read_frame(src, :audio)
  """
  @spec read_frame(source(), :video | :audio) ::
          {:ok, VideoFrame.t() | AudioSamples.t()} | {:error, term()}
  def read_frame(src, :video) when is_reference(src) do
    with {:ok, frame_map} <- NIF.media_read_video_frame(src) do
      {:ok, VideoFrame.from_map(frame_map)}
    end
  end

  def read_frame(src, :audio) when is_reference(src) do
    with {:ok, samples_map} <- NIF.media_read_audio_samples(src) do
      {:ok, AudioSamples.from_map(samples_map)}
    end
  end

  @doc "Closes the media source and releases all associated resources."
  @spec close(source()) :: :ok | {:error, term()}
  def close(src) when is_reference(src) do
    NIF.media_close(src)
  end
end
