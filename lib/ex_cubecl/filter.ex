defmodule ExCubecl.Filter do
  @moduledoc """
  GPU-accelerated filters for video and audio.

  Supports single-filter application and chainable filter pipelines.

  ## Video filters

      filtered = ExCubecl.Filter.apply(frame, :gaussian_blur, radius: 5)
      filtered = ExCubecl.Filter.apply(frame, :sharpen, strength: 1.2)
      filtered = ExCubecl.Filter.apply(frame, :lut, file: "cinematic.cube")
      filtered = ExCubecl.Filter.apply(frame, :chroma_key, color: {0, 177, 64}, threshold: 0.3)
      filtered = ExCubecl.Filter.apply(frame, :brightness_contrast, brightness: 0.1, contrast: 1.2)
      filtered = ExCubecl.Filter.apply(frame, :denoise, strength: 0.5)

  ## Audio filters

      filtered = ExCubecl.Filter.apply(samples, :eq, bands: [{:high_pass, 80}, {:shelf_high, 8000, 3.0}])
      filtered = ExCubecl.Filter.apply(samples, :compressor, threshold: -18, ratio: 4.0)
      filtered = ExCubecl.Filter.apply(samples, :reverb, room_size: 0.5, wet: 0.2)
      filtered = ExCubecl.Filter.apply(samples, :normalize)

  ## Filter chains via pipeline

      ExCubecl.pipeline()
      |> ExCubecl.pipeline_add(%{op: :filter, kernel: :gaussian_blur, input: frame, params: %{radius: 3}})
      |> ExCubecl.pipeline_add(%{op: :filter, kernel: :lut, input: :prev, params: %{file: "warm.cube"}})
      |> ExCubecl.pipeline_run()
  """

  alias ExCubecl.NIF
  alias ExCubecl.VideoFrame
  alias ExCubecl.AudioSamples

  @type frame_or_samples :: VideoFrame.t() | AudioSamples.t()

  @video_kernels [:gaussian_blur, :sharpen, :lut, :chroma_key, :brightness_contrast, :denoise]
  @audio_kernels [:eq, :compressor, :reverb, :normalize]

  @doc """
  Applies a named GPU filter to a frame or audio buffer.

  Returns the filtered result (same type as input).
  """
  @spec apply(frame_or_samples(), atom(), keyword()) ::
          {:ok, frame_or_samples()} | {:error, term()}
  def apply(input, kernel, params \\ [])

  def apply(%VideoFrame{handle: handle} = frame, kernel, params)
      when kernel in @video_kernels do
    result = apply_kernel(handle, kernel, params)

    case result do
      {:ok, _cmd_id} -> {:ok, frame}
      {:error, reason} -> {:error, reason}
    end
  end

  def apply(%AudioSamples{handle: handle} = samples, kernel, params)
      when kernel in @audio_kernels do
    result = apply_kernel(handle, kernel, params)

    case result do
      {:ok, _cmd_id} -> {:ok, samples}
      {:error, reason} -> {:error, reason}
    end
  end

  def apply(_input, kernel, _params) do
    {:error, {:unknown_filter, kernel}}
  end

  @doc """
  Applies a chain of filters in sequence using a pipeline.

  Each filter is a `{kernel_name, params}` tuple.

  ## Examples

      ExCubecl.Filter.chain(frame, [
        {:gaussian_blur, [radius: 3]},
        {:lut, [file: "warm.cube"]}
      ])
  """
  @spec chain(frame_or_samples(), [{atom(), keyword()}]) ::
          {:ok, frame_or_samples()} | {:error, term()}
  def chain(input, filters) when is_list(filters) do
    case ExCubecl.pipeline() do
      {:ok, pipeline_id} ->
        result =
          Enum.reduce_while(filters, {:ok, pipeline_id}, fn {kernel, params}, {:ok, pid} ->
            cmd =
              ExCubecl.Command.run_kernel(
                Atom.to_string(kernel),
                [input.handle],
                input.handle,
                Map.new(params)
              )

            case ExCubecl.pipeline_add_struct(pid, cmd) do
              :ok -> {:cont, {:ok, pid}}
              {:error, reason} -> {:halt, {:error, reason}}
            end
          end)

        case result do
          {:ok, pid} ->
            case ExCubecl.pipeline_run(pid) do
              {:ok, _} ->
                :ok = ExCubecl.pipeline_free(pid)
                {:ok, input}

              {:error, reason} ->
                :ok = ExCubecl.pipeline_free(pid)
                {:error, reason}
            end

          {:error, reason} ->
            {:error, reason}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  # ── Private ─────────────────────────────────────────────────

  defp apply_kernel(handle, kernel, params) do
    kernel_str = Atom.to_string(kernel)
    params_binary = :erlang.term_to_binary(Map.new(params))
    NIF.kernel_run(kernel_str, [handle], handle, params_binary)
  end
end
