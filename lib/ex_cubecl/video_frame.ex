defmodule ExCubecl.VideoFrame do
  @moduledoc """
  A video frame buffer residing on the GPU.

  ## Fields

    * `handle` — GPU buffer reference (Rustler ResourceArc)
    * `width` — frame width in pixels
    * `height` — frame height in pixels
    * `format` — pixel format (`:yuv420p`, `:rgb24`, `:rgba`, `:nv12`)
    * `pts` — presentation timestamp in microseconds
    * `duration` — frame duration in microseconds
  """

  @enforce_keys [:handle, :width, :height, :format, :pts, :duration]
  defstruct [:handle, :width, :height, :format, :pts, :duration]

  @type t :: %__MODULE__{
          handle: reference(),
          width: non_neg_integer(),
          height: non_neg_integer(),
          format: :yuv420p | :rgb24 | :rgba | :nv12,
          pts: non_neg_integer(),
          duration: non_neg_integer()
        }

  @doc "Creates a VideoFrame from a map returned by the NIF."
  @spec from_map(map()) :: t()
  def from_map(map) when is_map(map) do
    %__MODULE__{
      handle: map[:handle],
      width: map[:width],
      height: map[:height],
      format: map[:format],
      pts: map[:pts],
      duration: map[:duration]
    }
  end
end
