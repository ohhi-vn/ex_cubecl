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

  alias ExCubecl.NIF
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

    params = [{"gains", gains}]

    case tracks do
      [first | _] ->
        result = NIF.kernel_run("pcm_mix", handles, first.handle, params)

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

    params = [{"duck_gain", duck_gain}]
    result = NIF.kernel_run("pcm_mix", [bg.handle, fg.handle], bg.handle, params)

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
    from = Keyword.fetch!(opts, :from)
    to = Keyword.fetch!(opts, :to)

    params = [{"from", from}, {"to", to}]

    try do
      result = NIF.kernel_run("resample_linear", [samples.handle], samples.handle, params)

      case result do
        {:ok, _cmd_id} ->
          ratio = to / from
          new_frames = round(samples.frames * ratio)
          {:ok, %AudioSamples{samples | sample_rate: to, frames: new_frames}}

        {:error, reason} ->
          {:error, reason}
      end
    rescue
      e in ErlangError -> {:error, Exception.message(e)}
    end
  end

  @doc """
  Converts between channel layouts (e.g. stereo → mono, mono → stereo).

  ## Examples

      mono = ExCubecl.Audio.channels(samples, :stereo, :mono)
  """
  @spec channels(samples(), atom(), atom()) :: {:ok, samples()} | {:error, term()}
  def channels(%AudioSamples{} = samples, from_layout, to_layout) do
    params = [{"from", from_layout}, {"to", to_layout}]
    result = NIF.kernel_run("pcm_mix", [samples.handle], samples.handle, params)

    case result do
      {:ok, _cmd_id} ->
        new_channels = channel_count(to_layout)
        {:ok, %AudioSamples{samples | channels: new_channels}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  # ── Private ─────────────────────────────────────────────────

  defp channel_count(:mono), do: 1
  defp channel_count(:stereo), do: 2
  defp channel_count(:surround_51), do: 6
  defp channel_count(:surround_71), do: 8
  defp channel_count(other), do: raise(ArgumentError, "unsupported channel layout: #{other}")
end
