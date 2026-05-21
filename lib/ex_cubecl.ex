# Copyright 2026 ExCubecl Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

defmodule ExCubecl do
  @moduledoc """
  ExCubecl — GPU compute runtime for Elixir.

  Provides GPU buffer management, kernel execution, async command submission,
  and pipeline orchestration via CubeCL (Rust NIFs).

  ## Quick start

      # Check availability
      ExCubecl.available?()

      # Create a buffer from a list
      {:ok, buf} = ExCubecl.buffer([1.0, 2.0, 3.0], [3], :f32)

      # Inspect
      {:ok, shape} = ExCubecl.shape(buf)   # [3]
      {:ok, dtype} = ExCubecl.dtype(buf)   # "f32"
      {:ok, size}  = ExCubecl.size(buf)    # 12  (bytes)

      # Read back
      {:ok, binary} = ExCubecl.read(buf)

      # Free when done
      ExCubecl.free(buf)

  ## Kernel execution

      {:ok, out} = ExCubecl.buffer([0.0, 0.0, 0.0], [3], :f32)
      ExCubecl.run_kernel("elementwise_add", [buf_a, buf_b], out)

  ## Async commands

      {:ok, cmd} = ExCubecl.submit("some_command")
      {:ok, :completed} = ExCubecl.poll(cmd)
      :ok = ExCubecl.wait(cmd)

  ## Pipelines

      {:ok, p} = ExCubecl.pipeline()
      :ok = ExCubecl.pipeline_add(p, "elementwise_add:1,2:3")
      :ok = ExCubecl.pipeline_run(p)
      :ok = ExCubecl.pipeline_free(p)
  """

  alias ExCubecl.NIF

  @version "0.2.0"

  @dtypes ~w(f32 f64 s32 s64 u32 u8)

  # ── Device ──────────────────────────────────────────────────

  @doc "Returns information about the GPU compute device."
  @spec device_info() :: {:ok, map()} | {:error, term()}
  def device_info, do: NIF.device_info()

  @doc "Returns the number of available GPU devices."
  @spec device_count() :: {:ok, non_neg_integer()} | {:error, term()}
  def device_count, do: NIF.device_count()

  # ── Buffers ─────────────────────────────────────────────────

  @doc """
  Creates a GPU buffer from a list of values.

  ## Parameters

    * `data` — a flat list of numbers
    * `shape` — the tensor shape (e.g. `[3]` for a vector, `[2, 3]` for a matrix)
    * `type` — element type, one of: `:f32`, `:f64`, `:s32`, `:s64`, `:u32`, `:u8` (default `:f32`)

  ## Returns

    `{:ok, buffer_id}` on success, `{:error, reason}` on failure.
  """
  @spec buffer(list(), [non_neg_integer()], atom()) :: {:ok, non_neg_integer()} | {:error, term()}
  def buffer(data, shape, type \\ :f32) when is_list(data) and is_list(shape) do
    dtype_str = dtype_to_string(type)
    binary = list_to_binary(data, dtype_str)
    NIF.buffer_new(binary, shape, dtype_str)
  end

  @doc """
  Creates a GPU buffer, raising on error.

  See `buffer/3` for parameters.
  """
  @spec buffer!(list(), [non_neg_integer()], atom()) :: non_neg_integer()
  def buffer!(data, shape, type \\ :f32) do
    case apply(NIF, :buffer_new, [
           list_to_binary(data, dtype_to_string(type)),
           shape,
           dtype_to_string(type)
         ]) do
      {:ok, id} -> id
      {:error, reason} -> raise "ExCubecl.buffer!/3 failed: #{inspect(reason)}"
    end
  end

  @doc """
  Reads buffer data back from the GPU as a binary.
  """
  @spec read(non_neg_integer()) :: {:ok, binary()} | {:error, term()}
  def read(id) when is_integer(id), do: NIF.buffer_read(id)

  @doc """
  Reads buffer data back, raising on error.
  """
  @spec read!(non_neg_integer()) :: binary()
  def read!(id) when is_integer(id) do
    case apply(NIF, :buffer_read, [id]) do
      {:ok, data} -> data
      {:error, reason} -> raise "ExCubecl.read!/1 failed: #{inspect(reason)}"
    end
  end

  @doc "Returns the byte size of a buffer."
  @spec size(non_neg_integer()) :: {:ok, non_neg_integer()} | {:error, term()}
  def size(id) when is_integer(id), do: NIF.buffer_size(id)

  @doc "Returns the shape of a buffer."
  @spec shape(non_neg_integer()) :: {:ok, [non_neg_integer()]} | {:error, term()}
  def shape(id) when is_integer(id), do: NIF.buffer_shape(id)

  @doc "Returns the dtype string of a buffer (e.g. `\"f32\"`)."
  @spec dtype(non_neg_integer()) :: {:ok, String.t()} | {:error, term()}
  def dtype(id) when is_integer(id), do: NIF.buffer_dtype(id)

  @doc "Frees a GPU buffer."
  @spec free(non_neg_integer()) :: :ok | {:error, term()}
  def free(id) when is_integer(id), do: NIF.buffer_free(id)

  # ── Kernels ─────────────────────────────────────────────────

  @doc """
  Runs a kernel on the GPU.

  ## Parameters

    * `name` — kernel name string (see `kernels/0`)
    * `inputs` — list of input buffer IDs
    * `output` — output buffer ID
    * `params` — optional map of kernel parameters (default `%{}`)
  """
  @spec run_kernel(String.t(), [non_neg_integer()], non_neg_integer(), map()) ::
          {:ok, non_neg_integer()} | {:error, term()}
  def run_kernel(name, inputs, output, params \\ %{}) when is_binary(name) do
    params_binary = :erlang.term_to_binary(params)
    NIF.kernel_run(name, inputs, output, params_binary)
  end

  @doc "Returns the list of available kernel names."
  @spec kernels() :: {:ok, [String.t()]} | {:error, term()}
  def kernels, do: NIF.kernel_list()

  # ── Async commands ──────────────────────────────────────────

  @doc """
  Submits a command for asynchronous execution.

  Returns `{:ok, command_id}` which can be used with `poll/1` and `wait/1`.
  """
  @spec submit(String.t()) :: {:ok, non_neg_integer()} | {:error, term()}
  def submit(command) when is_binary(command), do: NIF.submit(command)

  @doc """
  Polls the status of an async command.

  Returns `{:ok, :pending | :running | :completed | :failed}` or `{:error, reason}`.
  """
  @spec poll(non_neg_integer()) ::
          {:ok, :pending | :running | :completed | :failed} | {:error, term()}
  def poll(command_id) when is_integer(command_id), do: NIF.poll(command_id)

  @doc "Blocks until an async command completes."
  @spec wait(non_neg_integer()) :: :ok | {:error, term()}
  def wait(command_id) when is_integer(command_id), do: NIF.wait(command_id)

  # ── Pipelines ───────────────────────────────────────────────

  @doc "Creates a new empty pipeline."
  @spec pipeline() :: {:ok, non_neg_integer()} | {:error, term()}
  def pipeline, do: NIF.pipeline_new()

  @doc """
  Adds a command to a pipeline.

  Command format: `\"kernel_name:input_id,input_id,...:output_id\"`
  Example: `\"elementwise_add:1,2:3\"`
  """
  @spec pipeline_add(non_neg_integer(), String.t()) :: :ok | {:error, term()}
  def pipeline_add(pipeline_id, command) when is_integer(pipeline_id) and is_binary(command) do
    NIF.pipeline_add(pipeline_id, command)
  end

  @doc "Runs all commands in a pipeline sequentially."
  @spec pipeline_run(non_neg_integer()) :: {:ok, [non_neg_integer()]} | {:error, term()}
  def pipeline_run(pipeline_id) when is_integer(pipeline_id) do
    NIF.pipeline_run(pipeline_id)
  end

  @doc "Frees a pipeline and its resources."
  @spec pipeline_free(non_neg_integer()) :: :ok | {:error, term()}
  def pipeline_free(pipeline_id) when is_integer(pipeline_id) do
    NIF.pipeline_free(pipeline_id)
  end

  # ── Convenience ─────────────────────────────────────────────

  @doc """
  Checks if the NIF library is loaded and available.

  Returns `true` if the NIF can be loaded, `false` otherwise.
  """
  @spec available?() :: boolean()
  def available? do
    case apply(NIF, :device_count, []) do
      {:ok, _} -> true
      {:error, _} -> false
    end
  end

  @doc "Returns the version of ExCubecl."
  @spec version() :: String.t()
  def version, do: @version

  @doc "Returns the list of supported dtype atoms."
  @spec supported_dtypes() :: [atom()]
  def supported_dtypes, do: [:f32, :f64, :s32, :s64, :u32, :u8]

  # ── Private ─────────────────────────────────────────────────

  defp dtype_to_string(:f32), do: "f32"
  defp dtype_to_string(:f64), do: "f64"
  defp dtype_to_string(:s32), do: "s32"
  defp dtype_to_string(:s64), do: "s64"
  defp dtype_to_string(:u32), do: "u32"
  defp dtype_to_string(:u8), do: "u8"

  defp dtype_to_string(other) do
    raise ArgumentError, "unsupported dtype: #{inspect(other)}. Supported: #{inspect(@dtypes)}"
  end

  defp list_to_binary(data, "f32") do
    for x <- data, into: <<>>, do: <<x::float-32-native>>
  end

  defp list_to_binary(data, "f64") do
    for x <- data, into: <<>>, do: <<x::float-64-native>>
  end

  defp list_to_binary(data, "s32") do
    for x <- data, into: <<>>, do: <<x::signed-32-native>>
  end

  defp list_to_binary(data, "s64") do
    for x <- data, into: <<>>, do: <<x::signed-64-native>>
  end

  defp list_to_binary(data, "u32") do
    for x <- data, into: <<>>, do: <<x::unsigned-32-native>>
  end

  defp list_to_binary(data, "u8") do
    for x <- data, into: <<>>, do: <<x::unsigned-8>>
  end
end
