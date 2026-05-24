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

defmodule ExCubecl.Command do
  @moduledoc """
  A typed struct representing a pipeline command.

  ## Fields

    * `:op` — the operation type (currently only `:run_kernel`)
    * `:kernel` — the kernel name (e.g. `"elementwise_add"`)
    * `:inputs` — list of input buffer references
    * `:output` — output buffer reference
    * `:params` — optional map of kernel parameters (default `%{}`)
  """

  @enforce_keys [:op, :kernel, :inputs, :output]
  defstruct [:op, :kernel, :inputs, :output, params: %{}]

  @type t :: %__MODULE__{
          op: :run_kernel | :filter | :overlay | :encode | :read_frame,
          kernel: String.t(),
          inputs: [reference()],
          output: reference(),
          params: map()
        }

  @doc """
  Creates a new Command struct for running a kernel.

  ## Examples

      iex> cmd = ExCubecl.Command.run_kernel("elementwise_add", [input_buf], output_buf)
      iex> cmd.op
      :run_kernel
      iex> cmd.kernel
      "elementwise_add"
  """
  @spec run_kernel(String.t(), [reference()], reference(), map()) :: t()
  def run_kernel(name, inputs, output, params \\ %{}) do
    %__MODULE__{
      op: :run_kernel,
      kernel: name,
      inputs: inputs,
      output: output,
      params: params
    }
  end

  @doc """
  Returns a human-readable string representation of the command.
  """
  @spec to_string(t()) :: String.t()
  def to_string(%__MODULE__{
        op: :run_kernel,
        kernel: name,
        inputs: inputs,
        output: _output,
        params: params
      }) do
    input_count = length(inputs)
    "Command(run_kernel: #{name}, inputs: #{input_count}, params: #{inspect(params)})"
  end
end
