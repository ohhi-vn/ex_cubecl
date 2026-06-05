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

      # Buffers are automatically freed when the Elixir term is garbage collected.
      # No manual free is needed.

  ## Kernel execution

      {:ok, out} = ExCubecl.buffer([0.0, 0.0, 0.0], [3], :f32)
      ExCubecl.run_kernel("elementwise_add", [buf_a, buf_b], out)

  ## Async commands

      {:ok, cmd} = ExCubecl.submit("some_command")
      {:ok, :completed} = ExCubecl.poll(cmd)
      :ok = ExCubecl.wait(cmd)

  ## Pipelines

      {:ok, p} = ExCubecl.pipeline()
      :ok = ExCubecl.pipeline_add(p, "elementwise_add", [buf_a, buf_b], buf_out)
      :ok = ExCubecl.pipeline_run(p)
      :ok = ExCubecl.pipeline_free(p)
  """

  alias ExCubecl.NIF
  alias ExCubecl.Command

  @version "0.4.1"

  @dtypes ~w(f32 f64 s32 s64 u32 u8)

  @type buffer_ref :: reference()

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

    `{:ok, buffer}` on success where `buffer` is a Rustler resource reference.
    The buffer is automatically freed when the Elixir term is garbage collected.

  ## Examples

      {:ok, buf} = ExCubecl.buffer([1.0, 2.0, 3.0], [3], :f32)
  """
  @spec buffer(list() | binary(), [non_neg_integer()], atom()) ::
          {:ok, reference()} | {:error, term()}
  def buffer(data, shape, type \\ :f32)

  def buffer(data, shape, type) when is_list(data) and is_list(shape) do
    dtype_str = dtype_to_string(type)
    binary = list_to_binary(data, dtype_str)
    NIF.buffer_new(binary, shape, dtype_str)
  end

  def buffer(data, shape, type) when is_binary(data) and is_list(shape) do
    dtype_str = dtype_to_string(type)
    NIF.buffer_new(data, shape, dtype_str)
  end

  @doc """
  Creates a GPU buffer, raising on error.

  See `buffer/3` for parameters.
  """
  @spec buffer!(list() | binary(), [non_neg_integer()], atom()) :: reference()
  def buffer!(data, shape, type \\ :f32) do
    dtype_str = dtype_to_string(type)

    binary = if is_list(data), do: list_to_binary(data, dtype_str), else: data

    case NIF.buffer_new(binary, shape, dtype_str) do
      {:ok, buf} -> buf
      {:error, reason} -> raise "ExCubecl.buffer!/3 failed: #{inspect(reason)}"
    end
  end

  @doc """
  Reads buffer data back from the GPU as a binary.
  """
  @spec read(reference()) :: {:ok, binary()} | {:error, term()}
  def read(buf) when is_reference(buf), do: NIF.buffer_read(buf)

  @doc """
  Reads buffer data back, raising on error.
  """
  @spec read!(reference()) :: binary()
  def read!(buf) when is_reference(buf) do
    case NIF.buffer_read(buf) do
      {:ok, data} -> data
      {:error, reason} -> raise "ExCubecl.read!/1 failed: #{inspect(reason)}"
    end
  end

  @doc "Returns the byte size of a buffer."
  @spec size(reference()) :: {:ok, non_neg_integer()} | {:error, term()}
  def size(buf) when is_reference(buf), do: NIF.buffer_size(buf)

  @doc "Returns the shape of a buffer."
  @spec shape(reference()) :: {:ok, [non_neg_integer()]} | {:error, term()}
  def shape(buf) when is_reference(buf), do: NIF.buffer_shape(buf)

  @doc "Returns the dtype string of a buffer (e.g. `\"f32\"`)."
  @spec dtype(reference()) :: {:ok, String.t()} | {:error, term()}
  def dtype(buf) when is_reference(buf), do: NIF.buffer_dtype(buf)

  # ── Kernels ─────────────────────────────────────────────────

  @doc """
  Runs a kernel on the GPU.

  ## Parameters

    * `name` — kernel name string (see `kernels/0`)
    * `inputs` — list of input buffer references
    * `output` — output buffer reference
    * `params` — optional map of kernel parameters (default `%{}`)
  """
  @spec run_kernel(String.t(), [reference()], reference(), [{atom(), term()}]) ::
          {:ok, non_neg_integer()} | {:error, term()}
  def run_kernel(name, inputs, output, params \\ %{}) when is_binary(name) do
    NIF.kernel_run(name, inputs, output, params)
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
  Adds a kernel command to a pipeline.

  ## Parameters

    * `pipeline_id` — pipeline reference returned by `pipeline/0`
    * `kernel` — kernel name string (e.g. `"elementwise_add"`)
    * `inputs` — list of input buffer references
    * `output` — output buffer reference
    * `params` — optional map of kernel parameters (default `%{}`)

  ## Examples

      {:ok, pipeline} = ExCubecl.pipeline()
      ExCubecl.pipeline_add(pipeline, "elementwise_add", [buf_a, buf_b], buf_out)
  """
  @spec pipeline_add(non_neg_integer(), String.t(), [reference()], reference(), [{atom(), term()}]) ::
          :ok | {:error, term()}
  def pipeline_add(pipeline_id, kernel, inputs, output, params \\ %{})
      when is_integer(pipeline_id) and is_binary(kernel) do
    NIF.pipeline_add(pipeline_id, kernel, inputs, output, params)
  end

  @doc """
  Adds a command to a pipeline using a `Command` struct.

  ## Examples

      cmd = ExCubecl.Command.run_kernel("elementwise_add", [buf_a, buf_b], buf_out)
      ExCubecl.pipeline_add_struct(pipeline, cmd)
  """
  @spec pipeline_add_struct(non_neg_integer(), Command.t()) :: :ok | {:error, term()}
  def pipeline_add_struct(pipeline_id, %Command{} = command) when is_integer(pipeline_id) do
    pipeline_add(pipeline_id, command.kernel, command.inputs, command.output, command.params)
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
    case NIF.device_count() do
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
