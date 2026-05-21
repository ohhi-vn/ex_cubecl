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

defmodule ExCubecl.NIF do
  @moduledoc """
  NIF stubs for the CubeCL Rust backend.

  Each function delegates to the Rust NIF implementation.
  If the NIF is not loaded, returns `{:error, :nif_not_loaded}`.
  """

  use Rustler, otp_app: :ex_cubecl, crate: "ex_cubecl_nif"

  # ── Device ──────────────────────────────────────────────────

  def device_info(), do: nif_error()
  def device_count(), do: nif_error()

  # ── Buffer lifecycle ────────────────────────────────────────

  def buffer_new(_data, _shape, _dtype), do: nif_error()
  def buffer_read(_id), do: nif_error()
  def buffer_size(_id), do: nif_error()
  def buffer_shape(_id), do: nif_error()
  def buffer_dtype(_id), do: nif_error()
  def buffer_free(_id), do: nif_error()

  # ── Kernel execution ────────────────────────────────────────

  def kernel_run(_name, _inputs, _output, _params), do: nif_error()
  def kernel_list(), do: nif_error()

  # ── Async commands ──────────────────────────────────────────

  def submit(_command), do: nif_error()
  def poll(_command_id), do: nif_error()
  def wait(_command_id), do: nif_error()

  # ── Pipelines ───────────────────────────────────────────────

  def pipeline_new(), do: nif_error()
  def pipeline_add(_pipeline_id, _command), do: nif_error()
  def pipeline_run(_pipeline_id), do: nif_error()
  def pipeline_free(_pipeline_id), do: nif_error()

  # ── Helpers ─────────────────────────────────────────────────

  defp nif_error, do: {:error, :nif_not_loaded}
end
