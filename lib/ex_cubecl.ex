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
  ExCubecl - An Nx backend powered by CubeCL via Rust NIFs.

  Provides tensor operations using a Rust NIF layer for compute.
  Operations not yet implemented in the Nif fall back to Nx.BinaryBackend.

  ## Usage

      iex> t = Nx.tensor([1.0, 2.0, 3.0], backend: ExCubecl.Backend)
      iex> Nx.add(t, t)
      #Nx.Tensor<
        f32[3]
        [2.0, 4.0, 6.0]
      >

  ## Supported types

    * `{:f, 32}` - 32-bit float
    * `{:f, 64}` - 64-bit float
    * `{:s, 32}` - 32-bit signed integer
    * `{:s, 64}` - 64-bit signed integer
    * `{:u, 32}` - 32-bit unsigned integer
    * `{:u, 8}`  - 8-bit unsigned integer

  ## Mobile (iOS / Android) support

  ExCubecl provides a C FFI layer (`ex_cubecl.h`) for integration with
  iOS (via Objective-C / Swift bridging) and Android (via JNI).
  See the header at `native/ex_cubecl_nif/include/ex_cubecl.h`.
  """

  alias ExCubecl.Backend, as: B

  @version "0.1.0"

  @doc "Returns the version of ExCubecl."
  @spec version() :: String.t()
  def version, do: @version

  @doc """
  Checks if the NIF library is loaded and available.

  Returns `true` if the NIF can be loaded, `false` otherwise.
  Useful for graceful fallback when running on platforms without NIF support.
  """
  @spec available?() :: boolean()
  def available? do
    case ExCubecl.NIF.tensor_shape({:ok, make_ref()}) do
      {:error, _} -> true
      _ -> true
    end
  rescue
    _ -> false
  end

  @doc """
  Returns information about the compute device/backend.

  Currently returns CPU-based backend info. When CubeCL GPU support is enabled,
  this will return GPU device information.
  """
  @spec device_info() :: map()
  def device_info do
    %{
      backend: :cpu,
      name: "ExCubecl CPU (Rust NIF)",
      version: @version,
      gpu: false,
      mobile_ffi: true
    }
  end

  @doc "Returns the list of supported tensor types."
  @spec supported_types() :: [tuple()]
  def supported_types do
    [{:f, 32}, {:f, 64}, {:s, 32}, {:s, 64}, {:u, 32}, {:u, 8}]
  end

  @doc "Create a tensor from a binary with the given shape and type."
  @spec from_binary(binary(), tuple(), tuple()) :: Nx.Tensor.t()
  def from_binary(binary, shape, type) do
    Nx.from_binary(binary, type, backend: B)
  end

  @doc "Read a tensor's data into a binary."
  @spec to_binary(Nx.Tensor.t()) :: binary()
  def to_binary(tensor) do
    Nx.to_binary(tensor)
  end

  @doc "Return the size in bytes of a single element for the given Nx type."
  @spec type_size(tuple()) :: pos_integer()
  def type_size({_, size}), do: div(size, 8)

  @doc "Return the total byte size for a tensor of the given shape and type."
  @spec byte_size(tuple(), tuple()) :: non_neg_integer()
  def byte_size(shape, type) do
    Tuple.product(shape) * type_size(type)
  end
end
