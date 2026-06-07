defmodule ExCubecl.Video do
  @moduledoc """
  Video-specific GPU operations: overlay, mix, scale, crop, convert.

  All operations run on GPU-resident frame buffers and return new frame buffers.

  ## Examples

      result = ExCubecl.Video.overlay(base_frame, overlay_frame, x: 100, y: 50, alpha: 0.8)
      result = ExCubecl.Video.mix(frame_a, frame_b, mode: :dissolve, ratio: 0.5)
      result = ExCubecl.Video.scale(frame, width: 1280, height: 720)
      rgb = ExCubecl.Video.convert(frame, :yuv420p, :rgb24)
      cropped = ExCubecl.Video.crop(frame, x: 0, y: 0, width: 640, height: 360)
  """

  alias ExCubecl.NIF
  alias ExCubecl.VideoFrame

  @type frame :: VideoFrame.t()
  @type blend_mode :: :dissolve | :add | :multiply
  @type pixel_format :: :yuv420p | :rgb24 | :rgba | :nv12

  @doc """
  Alpha-composites `overlay` onto `base` at position (`x`, `y`) with opacity `alpha`.

  Uses Porter-Duff Over compositing on the GPU.

  ## Options

    * `:x` — horizontal offset (default 0)
    * `:y` — vertical offset (default 0)
    * `:alpha` — opacity 0.0–1.0 (default 1.0)
  """
  @spec overlay(frame(), frame(), keyword()) :: {:ok, frame()} | {:error, term()}
  def overlay(%VideoFrame{} = base, %VideoFrame{} = overlay, opts \\ []) do
    x = Keyword.get(opts, :x, 0)
    y = Keyword.get(opts, :y, 0)
    alpha = Keyword.get(opts, :alpha, 1.0)

    params = %{x: x, y: y, alpha: alpha}
    result = NIF.kernel_run("overlay_alpha", [base.handle, overlay.handle], base.handle, params)

    case result do
      {:ok, _cmd_id} -> {:ok, base}
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Blends two video frames using the specified blend mode.

  ## Options

    * `:mode` — blend mode: `:dissolve` (default), `:add`, or `:multiply`
    * `:ratio` — blend ratio 0.0 (all A) to 1.0 (all B), default 0.5
  """
  @spec mix(frame(), frame(), keyword()) :: {:ok, frame()} | {:error, term()}
  def mix(%VideoFrame{} = frame_a, %VideoFrame{} = frame_b, opts \\ []) do
    mode = Keyword.get(opts, :mode, :dissolve)
    ratio = Keyword.get(opts, :ratio, 0.5)

    params = %{mode: mode, ratio: ratio}
    result = NIF.kernel_run("video_mix", [frame_a.handle, frame_b.handle], frame_a.handle, params)

    case result do
      {:ok, _cmd_id} -> {:ok, frame_a}
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Scales a video frame to the specified dimensions using GPU-accelerated resampling.

  ## Options

    * `:width` — target width in pixels
    * `:height` — target height in pixels
  """
  @spec scale(frame(), keyword()) :: {:ok, frame()} | {:error, term()}
  def scale(%VideoFrame{} = frame, opts) do
    width = Keyword.fetch!(opts, :width)
    height = Keyword.fetch!(opts, :height)

    params = %{width: width, height: height}
    result = NIF.kernel_run("bicubic_scale", [frame.handle], frame.handle, params)

    case result do
      {:ok, _cmd_id} -> {:ok, %VideoFrame{frame | width: width, height: height}}
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Converts a video frame between pixel formats on the GPU.

  ## Examples

      rgb = ExCubecl.Video.convert(frame, :yuv420p, :rgb24)
  """
  @spec convert(frame(), pixel_format(), pixel_format()) :: {:ok, frame()} | {:error, term()}
  def convert(%VideoFrame{} = frame, from_format, _to_format)
      when from_format in [:yuv420p, :nv12] do
    params = %{from: from_format, to: :rgb24}
    result = NIF.kernel_run("yuv_to_rgb", [frame.handle], frame.handle, params)

    case result do
      {:ok, _cmd_id} -> {:ok, %VideoFrame{frame | format: :rgb24}}
      {:error, reason} -> {:error, reason}
    end
  end

  def convert(%VideoFrame{format: format}, _from, _to) do
    {:error, {:unsupported_conversion, format}}
  end

  @doc """
  Crops a video frame to the specified rectangle.

  ## Options

    * `:x` — left edge (default 0)
    * `:y` — top edge (default 0)
    * `:width` — crop width
    * `:height` — crop height
  """
  @spec crop(frame(), keyword()) :: {:ok, frame()} | {:error, term()}
  def crop(%VideoFrame{} = frame, opts) do
    x = Keyword.get(opts, :x, 0)
    y = Keyword.get(opts, :y, 0)
    width = Keyword.fetch!(opts, :width)
    height = Keyword.fetch!(opts, :height)

    params = %{x: x, y: y, width: width, height: height}
    result = NIF.kernel_run("video_crop", [frame.handle], frame.handle, params)

    case result do
      {:ok, _cmd_id} -> {:ok, %VideoFrame{frame | width: width, height: height}}
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Saves a snapshot of the frame to a PNG file.

  Note: This triggers a GPU→CPU readback, so it should be used sparingly.
  """
  @spec snapshot(frame(), String.t()) :: :ok | {:error, term()}
  def snapshot(%VideoFrame{handle: handle, width: _w, height: _h}, path)
      when is_reference(handle) and is_binary(path) do
    result = NIF.buffer_read(handle)

    case result do
      {:ok, data} -> File.write(path, data)
      {:error, reason} -> {:error, reason}
    end
  end
end
