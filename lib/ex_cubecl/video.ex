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
    with {:ok, width, height} <- fetch_dimensions(opts),
         {:ok, output_handle} <- allocate_video_buffer(width, height, frame.format) do
      params = %{width: width, height: height}

      case ExCubecl.run_kernel("bicubic_scale", [frame.handle], output_handle, params) do
        {:ok, _cmd_id} ->
          {:ok, %VideoFrame{frame | handle: output_handle, width: width, height: height}}

        {:error, reason} ->
          {:error, reason}
      end
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
    # YUV420p uses 1.5 bytes/pixel; RGB24 uses 3 bytes/pixel.
    # Allocate a new output buffer with the correct size for RGB.
    rgb_bytes = frame.width * frame.height * 3
    output_data = :binary.copy(<<0::unsigned-8>>, rgb_bytes)

    with {:ok, output_handle} <- ExCubecl.buffer(output_data, [rgb_bytes], :u8) do
      params = %{from: from_format, to: :rgb24}

      case NIF.kernel_run("yuv_to_rgb", [frame.handle], output_handle, params) do
        {:ok, _cmd_id} ->
          {:ok, %VideoFrame{frame | handle: output_handle, format: :rgb24}}

        {:error, reason} ->
          {:error, reason}
      end
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
    with {:ok, x, y, width, height} <- fetch_crop_rect(opts),
         :ok <- validate_crop_rect(frame, x, y, width, height),
         {:ok, output_handle} <- allocate_video_buffer(width, height, frame.format) do
      params = %{x: x, y: y, width: width, height: height}

      case ExCubecl.run_kernel("video_crop", [frame.handle], output_handle, params) do
        {:ok, _cmd_id} ->
          {:ok, %VideoFrame{frame | handle: output_handle, width: width, height: height}}

        {:error, reason} ->
          {:error, reason}
      end
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

  # ── Private ─────────────────────────────────────────────────

  defp fetch_dimensions(opts) do
    with {:ok, width} <- fetch_option(opts, :width),
         {:ok, height} <- fetch_option(opts, :height),
         :ok <- validate_positive_dimension(:width, width),
         :ok <- validate_positive_dimension(:height, height) do
      {:ok, width, height}
    end
  end

  defp fetch_crop_rect(opts) do
    x = Keyword.get(opts, :x, 0)
    y = Keyword.get(opts, :y, 0)

    with {:ok, width} <- fetch_option(opts, :width),
         {:ok, height} <- fetch_option(opts, :height),
         :ok <- validate_non_negative_offset(:x, x),
         :ok <- validate_non_negative_offset(:y, y),
         :ok <- validate_positive_dimension(:width, width),
         :ok <- validate_positive_dimension(:height, height) do
      {:ok, x, y, width, height}
    end
  end

  defp fetch_option(opts, option) do
    case Keyword.fetch(opts, option) do
      {:ok, value} -> {:ok, value}
      :error -> {:error, {:missing_option, option}}
    end
  end

  defp validate_positive_dimension(_option, value) when is_integer(value) and value > 0, do: :ok

  defp validate_positive_dimension(option, value),
    do: {:error, {:invalid_dimension, option, value}}

  defp validate_non_negative_offset(_option, value) when is_integer(value) and value >= 0, do: :ok

  defp validate_non_negative_offset(option, value),
    do: {:error, {:invalid_offset, option, value}}

  defp validate_crop_rect(%VideoFrame{width: src_width, height: src_height}, x, y, width, height) do
    if x + width <= src_width and y + height <= src_height do
      :ok
    else
      {:error, {:crop_out_of_bounds, x, y, width, height, src_width, src_height}}
    end
  end

  defp allocate_video_buffer(width, height, :yuv420p), do: allocate_yuv420p_buffer(width, height)
  defp allocate_video_buffer(width, height, :rgb24), do: allocate_rgb_buffer(width, height, 3)
  defp allocate_video_buffer(width, height, :rgba), do: allocate_rgb_buffer(width, height, 4)
  defp allocate_video_buffer(width, height, :nv12), do: allocate_yuv420p_buffer(width, height)

  defp allocate_yuv420p_buffer(width, height) do
    bytes = div(width * height * 3, 2)
    data = :binary.copy(<<0::unsigned-8>>, bytes)
    ExCubecl.buffer(data, [bytes], :u8)
  end

  defp allocate_rgb_buffer(width, height, channels) do
    bytes = width * height * channels
    data = :binary.copy(<<0::unsigned-8>>, bytes)
    ExCubecl.buffer(data, [bytes], :u8)
  end
end
