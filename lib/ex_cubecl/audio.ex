defmodule ExCubecl.Audio do
  @moduledoc """
  Audio-specific GPU operations: mix, overlay, resample, channel conversion.

  All operations run on GPU-resident sample buffers (f32 planar PCM).

  ## Examples

      result = ExCubecl.Audio.mix([track_a, track_b], gains: [0.7, 0.5])
      result = ExCubecl.Audio.overlay(bg, fg, duck_level: -12)
      resampled = ExCubecl.Audio.resample(samples, from: 44100, to: 48000)
      mono = ExCubecl.Audio.channels(samples, :stereo, :mono)
  """

  alias ExCubecl.AudioSamples

  @type samples :: AudioSamples.t()

  @doc """
  Mixes multiple audio streams by summing with per-channel gain.

  ## Options

    * `:gains` — list of gain values (0.0–1.0+) corresponding to each track
  """
  @spec mix([samples()], keyword()) :: {:ok, samples()} | {:error, term()}
  def mix(tracks, opts \\ []) when is_list(tracks) do
    gains = Keyword.get(opts, :gains, Enum.map(tracks, fn _ -> 1.0 end))
    handles = Enum.map(tracks, & &1.handle)

    params = %{gains: gains}

    case tracks do
      [first | _] ->
        result = ExCubecl.run_kernel("pcm_mix", handles, first.handle, params)

        case result do
          {:ok, _cmd_id} -> {:ok, first}
          {:error, reason} -> {:error, reason}
        end

      [] ->
        {:error, :no_tracks}
    end
  end

  @doc """
  Overlays foreground audio over background, optionally ducking the background.

  ## Options

    * `:duck_level` — dB reduction for background during foreground (default -12)
  """
  @spec overlay(samples(), samples(), keyword()) :: {:ok, samples()} | {:error, term()}
  def overlay(%AudioSamples{} = bg, %AudioSamples{} = fg, opts \\ []) do
    duck_level = Keyword.get(opts, :duck_level, -12)
    duck_gain = :math.pow(10, duck_level / 20)

    params = %{duck_gain: duck_gain}
    result = ExCubecl.run_kernel("pcm_mix", [bg.handle, fg.handle], bg.handle, params)

    case result do
      {:ok, _cmd_id} -> {:ok, bg}
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Resamples audio to a different sample rate using GPU-accelerated linear interpolation.

  ## Options

    * `:from` — source sample rate (Hz)
    * `:to` — target sample rate (Hz)
  """
  @spec resample(samples(), keyword()) :: {:ok, samples()} | {:error, term()}
  def resample(%AudioSamples{} = samples, opts) do
    with {:ok, from, to} <- fetch_resample_rates(opts),
         {:ok, new_frames} <- calculate_resample_frames(samples, from, to),
         {:ok, output_handle} <- allocate_audio_buffer(new_frames, samples.channels) do
      params = %{from: from, to: to}

      case ExCubecl.run_kernel("resample_linear", [samples.handle], output_handle, params) do
        {:ok, _cmd_id} ->
          {:ok,
           %AudioSamples{samples | handle: output_handle, sample_rate: to, frames: new_frames}}

        {:error, reason} ->
          {:error, reason}
      end
    end
  end

  @doc """
  Converts between channel layouts (e.g. stereo → mono, mono → stereo).

  ## Examples

      mono = ExCubecl.Audio.channels(samples, :stereo, :mono)
  """
  @spec channels(samples(), atom(), atom()) :: {:ok, samples()} | {:error, term()}
  def channels(%AudioSamples{} = samples, from_layout, to_layout) do
    with {:ok, from_channels} <- channel_count(from_layout),
         :ok <- validate_current_channels(samples, from_channels),
         {:ok, to_channels} <- channel_count(to_layout),
         {:ok, converted_data} <- convert_channel_data(samples, from_channels, to_channels),
         {:ok, output_handle} <-
           allocate_audio_buffer(converted_data, samples.frames, to_channels) do
      {:ok, %AudioSamples{samples | handle: output_handle, channels: to_channels}}
    end
  end

  @doc "Returns the number of channels for a channel layout atom."
  @spec channel_count(atom()) :: {:ok, pos_integer()} | {:error, term()}
  def channel_count(layout), do: channel_count_from_layout(layout)

  # ── Private ─────────────────────────────────────────────────

  defp fetch_resample_rates(opts) do
    with {:ok, from} <- fetch_option(opts, :from),
         {:ok, to} <- fetch_option(opts, :to),
         :ok <- validate_positive_rate(:from, from),
         :ok <- validate_positive_rate(:to, to) do
      {:ok, from, to}
    end
  end

  defp fetch_option(opts, option) do
    case Keyword.fetch(opts, option) do
      {:ok, value} -> {:ok, value}
      :error -> {:error, {:missing_option, option}}
    end
  end

  defp validate_positive_rate(_option, value) when is_integer(value) and value > 0, do: :ok
  defp validate_positive_rate(_option, value) when is_float(value) and value > 0.0, do: :ok
  defp validate_positive_rate(option, value), do: {:error, {:invalid_rate, option, value}}

  defp calculate_resample_frames(%AudioSamples{frames: frames}, from, to) do
    ratio = to / from
    new_frames = round(frames * ratio)

    if new_frames > 0 do
      {:ok, new_frames}
    else
      {:error, {:invalid_resample_dimensions, frames, from, to}}
    end
  end

  defp validate_current_channels(%AudioSamples{channels: channels}, channels), do: :ok

  defp validate_current_channels(%AudioSamples{channels: channels}, expected_channels) do
    {:error, {:channel_layout_mismatch, channels, expected_channels}}
  end

  defp convert_channel_data(
         %AudioSamples{handle: handle, frames: frames},
         from_channels,
         to_channels
       ) do
    with {:ok, data} <- ExCubecl.read(handle),
         {:ok, samples} <- decode_f32_samples(data, frames, from_channels) do
      converted = convert_samples(samples, from_channels, to_channels)
      data = for sample <- converted, into: <<>>, do: <<sample::float-32-native>>
      {:ok, data}
    end
  end

  defp decode_f32_samples(data, frames, channels) do
    expected_samples = frames * channels

    if byte_size(data) == expected_samples * 4 do
      samples = for <<sample::float-32-native <- data>>, do: sample
      {:ok, samples}
    else
      {:error, {:invalid_audio_buffer_size, byte_size(data), expected_samples * 4}}
    end
  end

  defp convert_samples(samples, from_channels, to_channels)
       when from_channels == to_channels do
    samples
  end

  defp convert_samples(samples, 1, 2) do
    Enum.flat_map(samples, fn sample -> [sample, sample] end)
  end

  defp convert_samples(samples, 2, 1) do
    samples
    |> Enum.chunk_every(2, 2, :discard)
    |> Enum.map(fn [left, right] -> (left + right) / 2.0 end)
  end

  defp convert_samples(samples, 1, 6) do
    Enum.flat_map(samples, fn sample -> [sample, sample, 0.0, 0.0, 0.0, 0.0] end)
  end

  defp convert_samples(samples, 1, 8) do
    Enum.flat_map(samples, fn sample -> [sample, sample, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] end)
  end

  defp convert_samples(samples, 2, 6) do
    samples
    |> Enum.chunk_every(2, 2, :discard)
    |> Enum.flat_map(fn [left, right] -> [left, right, 0.0, 0.0, 0.0, 0.0] end)
  end

  defp convert_samples(samples, 2, 8) do
    samples
    |> Enum.chunk_every(2, 2, :discard)
    |> Enum.flat_map(fn [left, right] -> [left, right, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] end)
  end

  defp convert_samples(samples, 6, 2) do
    samples
    |> Enum.chunk_every(6, 6, :discard)
    |> Enum.map(fn [left, right | _] -> (left + right) / 2.0 end)
  end

  defp convert_samples(samples, 8, 2) do
    samples
    |> Enum.chunk_every(8, 8, :discard)
    |> Enum.map(fn [left, right | _] -> (left + right) / 2.0 end)
  end

  defp convert_samples(samples, 6, 8) do
    samples
    |> Enum.chunk_every(6, 6, :discard)
    |> Enum.flat_map(fn frame -> frame ++ [0.0, 0.0] end)
  end

  defp convert_samples(samples, 8, 6) do
    samples
    |> Enum.chunk_every(8, 8, :discard)
    |> Enum.flat_map(fn frame -> Enum.take(frame, 6) end)
  end

  defp convert_samples(_samples, from_channels, to_channels) do
    raise ArgumentError, "unsupported channel conversion: #{from_channels} to #{to_channels}"
  end

  defp allocate_audio_buffer(samples, channels) do
    data = :binary.copy(<<0::float-32-native>>, samples * channels)
    allocate_audio_buffer(data, samples, channels)
  end

  defp allocate_audio_buffer(data, frames, channels) do
    expected_bytes = frames * channels * 4

    if byte_size(data) == expected_bytes do
      case ExCubecl.buffer(data, [frames * channels], :f32) do
        {:ok, handle} -> {:ok, handle}
        {:error, reason} -> {:error, reason}
      end
    else
      {:error, {:invalid_audio_buffer_size, byte_size(data), expected_bytes}}
    end
  end

  defp channel_count_from_layout(:mono), do: {:ok, 1}
  defp channel_count_from_layout(:stereo), do: {:ok, 2}
  defp channel_count_from_layout(:surround_51), do: {:ok, 6}
  defp channel_count_from_layout(:surround_71), do: {:ok, 8}
  defp channel_count_from_layout(other), do: {:error, {:unsupported_channel_layout, other}}
end
