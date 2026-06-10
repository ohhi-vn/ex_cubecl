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

  @spec device_info() :: dynamic()
  def device_info(), do: nif_error()
  @spec device_count() :: dynamic()
  def device_count(), do: nif_error()

  # ── Buffer lifecycle ────────────────────────────────────────

  @spec buffer_new(binary(), [non_neg_integer()], String.t()) :: dynamic()
  def buffer_new(_data, _shape, _dtype), do: nif_error()
  @spec buffer_read(reference()) :: dynamic()
  def buffer_read(_buffer), do: nif_error()
  @spec buffer_size(reference()) :: dynamic()
  def buffer_size(_buffer), do: nif_error()
  @spec buffer_shape(reference()) :: dynamic()
  def buffer_shape(_buffer), do: nif_error()
  @spec buffer_dtype(reference()) :: dynamic()
  def buffer_dtype(_buffer), do: nif_error()

  # ── Kernel execution ────────────────────────────────────────

  @spec kernel_run(String.t(), [reference()], reference(), map()) :: dynamic()
  def kernel_run(_name, _inputs, _output, _params), do: nif_error()
  @spec kernel_list() :: dynamic()
  def kernel_list(), do: nif_error()

  # ── Async commands ──────────────────────────────────────────

  @spec submit(String.t()) :: dynamic()
  def submit(_command), do: nif_error()

  @spec poll(non_neg_integer()) :: dynamic()
  def poll(_command_id), do: nif_error()
  @spec wait(non_neg_integer()) :: dynamic()
  def wait(_command_id), do: nif_error()

  # ── Pipelines ───────────────────────────────────────────────

  @spec pipeline_new() :: dynamic()
  def pipeline_new(), do: nif_error()

  @spec pipeline_add(non_neg_integer(), String.t(), [reference()], reference(), map()) ::
          dynamic()
  def pipeline_add(_pipeline_id, _name, _inputs, _output, _params), do: nif_error()
  @spec pipeline_run(non_neg_integer()) :: dynamic()
  def pipeline_run(_pipeline_id), do: nif_error()
  @spec pipeline_free(non_neg_integer()) :: dynamic()
  def pipeline_free(_pipeline_id), do: nif_error()

  # ── Media I/O ──────────────────────────────────────────────

  @spec media_open(String.t()) :: dynamic()
  def media_open(_path), do: nif_error()
  @spec media_streams(reference()) :: dynamic()
  def media_streams(_source), do: nif_error()
  @spec media_read_video_frame(reference()) :: dynamic()
  def media_read_video_frame(_source), do: nif_error()
  @spec media_read_audio_samples(reference()) :: dynamic()
  def media_read_audio_samples(_source), do: nif_error()
  @spec media_close(reference()) :: dynamic()
  def media_close(_source), do: nif_error()

  # ── Transcode ──────────────────────────────────────────────

  @spec transcode_start(String.t(), map()) :: dynamic()
  def transcode_start(_path, _opts), do: nif_error()
  @spec transcode_write_video(reference(), reference()) :: dynamic()
  def transcode_write_video(_encoder, _frame), do: nif_error()
  @spec transcode_write_audio(reference(), reference()) :: dynamic()
  def transcode_write_audio(_encoder, _samples), do: nif_error()
  @spec transcode_finish(reference()) :: dynamic()
  def transcode_finish(_encoder), do: nif_error()

  # ── Helpers ─────────────────────────────────────────────────

  defp nif_error do
    :erlang.nif_error(:nif_not_loaded)
  end
end
