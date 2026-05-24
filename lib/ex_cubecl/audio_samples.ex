defmodule ExCubecl.AudioSamples do
  @moduledoc """
  An audio sample buffer residing on the GPU (f32 planar PCM).

  ## Fields

    * `handle` — GPU buffer reference (Rustler ResourceArc)
    * `channels` — number of audio channels
    * `sample_rate` — samples per second (e.g. 48000)
    * `frames` — number of samples per channel
    * `pts` — presentation timestamp in microseconds
  """

  @enforce_keys [:handle, :channels, :sample_rate, :frames, :pts]
  defstruct [:handle, :channels, :sample_rate, :frames, :pts]

  @type t :: %__MODULE__{
          handle: reference(),
          channels: non_neg_integer(),
          sample_rate: non_neg_integer(),
          frames: non_neg_integer(),
          pts: non_neg_integer()
        }

  @doc "Creates an AudioSamples from a map returned by the NIF."
  @spec from_map(map()) :: t()
  def from_map(map) when is_map(map) do
    %__MODULE__{
      handle: map[:handle],
      channels: map[:channels],
      sample_rate: map[:sample_rate],
      frames: map[:frames],
      pts: map[:pts]
    }
  end
end
