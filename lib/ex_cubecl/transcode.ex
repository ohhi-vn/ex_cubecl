defmodule ExCubecl.Transcode do
  @moduledoc """
  Encode and mux media output.

  Supports file-to-file transcoding and frame-by-frame streaming encoding.

  ## File-to-file transcode

      ExCubecl.Transcode.run("input.mp4", "output.mp4",
        video: [codec: :h264, bitrate: "4M", fps: 30],
        audio: [codec: :aac, bitrate: "192k", sample_rate: 48000]
      )

  ## Frame-by-frame streaming transcode

      {:ok, enc} = ExCubecl.Transcode.start("output.mp4",
        video: [codec: :h265, width: 1280, height: 720],
        audio: [codec: :aac]
      )

      ExCubecl.Transcode.write_frame(enc, processed_video_frame)
      ExCubecl.Transcode.write_samples(enc, processed_audio_samples)
      ExCubecl.Transcode.finish(enc)

  ## Supported codecs

  Video: h264, h265, vp9, av1, prores
  Audio: aac, opus, mp3, flac, pcm

  ## Supported containers

  mp4, mkv, webm, mov, ts
  """

  alias ExCubecl.NIF
  alias ExCubecl.VideoFrame
  alias ExCubecl.AudioSamples

  @type encoder :: reference()

  @video_codecs ~w(h264 h265 vp9 av1 prores)
  @audio_codecs ~w(aac opus mp3 flac pcm)
  @containers ~w(mp4 mkv webm mov ts)

  @doc """
  Transcodes an input file to an output file with the specified options.

  This is a convenience wrapper that opens the input, reads frames, applies
  encoding, and writes the output.

  ## Options

    * `:video` — keyword list with `:codec`, `:bitrate`, `:fps`, `:width`, `:height`
    * `:audio` — keyword list with `:codec`, `:bitrate`, `:sample_rate`
  """
  @spec run(String.t(), String.t(), keyword()) :: :ok | {:error, term()}
  def run(input_path, output_path, opts \\ [])
      when is_binary(input_path) and is_binary(output_path) do
    video_opts = Keyword.get(opts, :video, [])
    audio_opts = Keyword.get(opts, :audio, [])

    with {:ok, src} <- ExCubecl.Media.open(input_path),
         {:ok, enc} <- start(output_path, video: video_opts, audio: audio_opts),
         :ok <- transcode_loop(src, enc),
         :ok <- finish(enc) do
      :ok = ExCubecl.Media.close(src)
      :ok
    end
  end

  @doc """
  Starts a streaming transcoder for frame-by-frame encoding.

  Returns an encoder reference to be used with `write_frame/2`, `write_samples/2`,
  and `finish/1`.

  ## Options

    * `:video` — keyword list with `:codec`, `:width`, `:height`, `:bitrate`, `:fps`
    * `:audio` — keyword list with `:codec`, `:bitrate`, `:sample_rate`
  """
  @spec start(String.t(), keyword()) :: {:ok, encoder()} | {:error, term()}
  def start(output_path, opts \\ []) when is_binary(output_path) do
    video_opts = Keyword.get(opts, :video, [])
    audio_opts = Keyword.get(opts, :audio, [])

    validate_codec!(video_opts[:codec], @video_codecs, :video)
    validate_codec!(audio_opts[:codec], @audio_codecs, :audio)

    validate_container!(output_path)

    opts_map = %{
      "video" => Map.new(video_opts),
      "audio" => Map.new(audio_opts)
    }

    NIF.transcode_start(output_path, opts_map)
  end

  @doc "Writes a video frame to the encoder."
  @spec write_frame(encoder(), VideoFrame.t()) :: :ok | {:error, term()}
  def write_frame(enc, %VideoFrame{} = frame) when is_reference(enc) do
    NIF.transcode_write_video(enc, frame.handle)
  end

  @doc "Writes audio samples to the encoder."
  @spec write_samples(encoder(), AudioSamples.t()) :: :ok | {:error, term()}
  def write_samples(enc, %AudioSamples{} = samples) when is_reference(enc) do
    NIF.transcode_write_audio(enc, samples.handle)
  end

  @doc "Finalizes encoding and closes the output file."
  @spec finish(encoder()) :: :ok | {:error, term()}
  def finish(enc) when is_reference(enc) do
    NIF.transcode_finish(enc)
  end

  # ── Private ─────────────────────────────────────────────────

  defp transcode_loop(src, enc, max_frames \\ 100)
  defp transcode_loop(_src, _enc, 0), do: :ok
  defp transcode_loop(src, enc, remaining) do
    case ExCubecl.Media.read_frame(src, :video) do
      {:ok, frame} ->
        result = write_frame(enc, frame)

        case result do
          :ok -> transcode_loop(src, enc, remaining - 1)
          {:error, reason} -> {:error, reason}
        end

      {:error, :eof} ->
        :ok

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp validate_codec!(nil, _allowed, _type), do: :ok

  defp validate_codec!(codec, allowed, type) when is_atom(codec) do
    codec_str = Atom.to_string(codec)

    if codec_str in allowed do
      :ok
    else
      raise ArgumentError,
            "unsupported #{type} codec: #{codec}. Supported: #{Enum.join(allowed, ", ")}"
    end
  end

  defp validate_container!(path) do
    ext = Path.extname(path) |> String.trim_leading(".")

    if ext in @containers do
      :ok
    else
      raise ArgumentError,
            "unsupported container: .#{ext}. Supported: #{Enum.join(@containers, ", ")}"
    end
  end
end
