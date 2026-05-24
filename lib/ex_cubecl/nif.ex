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

  Buffers are managed via Rustler `ResourceArc` — they are automatically
  freed when the Elixir term is garbage collected. No manual free is needed.
  """

  use Rustler, otp_app: :ex_cubecl, crate: "ex_cubecl_nif"

  # ── Device ──────────────────────────────────────────────────

  @spec device_info() :: {:ok, map()} | {:error, term()}
  def device_info(), do: nif_error()
  @spec device_count() :: {:ok, non_neg_integer()} | {:error, term()}
  def device_count(), do: nif_error()

  # ── Buffer lifecycle ────────────────────────────────────────

  @spec buffer_new(binary(), [non_neg_integer()], String.t()) ::
          {:ok, reference()} | {:error, term()}
  def buffer_new(_data, _shape, _dtype), do: nif_error()
  @spec buffer_read(reference()) :: {:ok, binary()} | {:error, term()}
  def buffer_read(_buffer), do: nif_error()
  @spec buffer_size(reference()) :: {:ok, non_neg_integer()} | {:error, term()}
  def buffer_size(_buffer), do: nif_error()
  @spec buffer_shape(reference()) :: {:ok, [non_neg_integer()]} | {:error, term()}
  def buffer_shape(_buffer), do: nif_error()
  @spec buffer_dtype(reference()) :: {:ok, String.t()} | {:error, term()}
  def buffer_dtype(_buffer), do: nif_error()

  # ── Kernel execution ────────────────────────────────────────

  @spec kernel_run(String.t(), [reference()], reference(), binary()) ::
          {:ok, non_neg_integer()} | {:error, term()}
  def kernel_run(_name, _inputs, _output, _params), do: nif_error()
  @spec kernel_list() :: {:ok, [String.t()]} | {:error, term()}
  def kernel_list(), do: nif_error()

  # ── Async commands ──────────────────────────────────────────

  @spec submit(String.t()) :: {:ok, non_neg_integer()} | {:error, term()}
  def submit(_command), do: nif_error()

  @spec poll(non_neg_integer()) ::
          {:ok, :pending | :running | :completed | :failed} | {:error, term()}
  def poll(_command_id), do: nif_error()
  @spec wait(non_neg_integer()) :: :ok | {:error, term()}
  def wait(_command_id), do: nif_error()

  # ── Pipelines ───────────────────────────────────────────────

  @spec pipeline_new() :: {:ok, non_neg_integer()} | {:error, term()}
  def pipeline_new(), do: nif_error()

  @spec pipeline_add(non_neg_integer(), String.t(), [reference()], reference(), binary()) ::
          :ok | {:error, term()}
  def pipeline_add(_pipeline_id, _name, _inputs, _output, _params), do: nif_error()
  @spec pipeline_run(non_neg_integer()) :: {:ok, [non_neg_integer()]} | {:error, term()}
  def pipeline_run(_pipeline_id), do: nif_error()
  @spec pipeline_free(non_neg_integer()) :: :ok | {:error, term()}
  def pipeline_free(_pipeline_id), do: nif_error()

  # ── Media I/O ──────────────────────────────────────────────

  @spec media_open(String.t()) :: {:ok, reference()} | {:error, term()}
  def media_open(_path), do: nif_error()
  @spec media_streams(reference()) :: {:ok, [map()]} | {:error, term()}
  def media_streams(_source), do: nif_error()
  @spec media_read_video_frame(reference()) :: {:ok, map()} | {:error, term()}
  def media_read_video_frame(_source), do: nif_error()
  @spec media_read_audio_samples(reference()) :: {:ok, map()} | {:error, term()}
  def media_read_audio_samples(_source), do: nif_error()
  @spec media_close(reference()) :: :ok | {:error, term()}
  def media_close(_source), do: nif_error()

  # ── Transcode ──────────────────────────────────────────────

  @spec transcode_start(String.t(), map()) :: {:ok, reference()} | {:error, term()}
  def transcode_start(_path, _opts), do: nif_error()
  @spec transcode_write_video(reference(), reference()) :: :ok | {:error, term()}
  def transcode_write_video(_encoder, _frame), do: nif_error()
  @spec transcode_write_audio(reference(), reference()) :: :ok | {:error, term()}
  def transcode_write_audio(_encoder, _samples), do: nif_error()
  @spec transcode_finish(reference()) :: :ok | {:error, term()}
  def transcode_finish(_encoder), do: nif_error()

  # ── Helpers ─────────────────────────────────────────────────

  defp nif_error, do: :erlang.nif_error(:nif_not_loaded)
end
