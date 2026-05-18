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
  If the NIF is not loaded, returns `:erlang.nif_error(:nif_not_loaded)`.
  """

  @on_load :load_nifs

  def load_nifs do
    path = :filename.join(:code.priv_dir(:ex_cubecl), ~c"libex_cubecl_nif")
    :erlang.load_nif(path, 0)
  end

  # ── Buffer lifecycle ────────────────────────────────────────

  def new_tensor(_data, _shape, _dtype), do: nif_error()
  def read_tensor(_ref), do: nif_error()
  def deallocate_tensor(_ref), do: nif_error()
  def tensor_shape(_ref), do: nif_error()
  def tensor_dtype(_ref), do: nif_error()

  # ── Binary ops ──────────────────────────────────────────────

  def add(_a, _b), do: nif_error()
  def subtract(_a, _b), do: nif_error()
  def multiply(_a, _b), do: nif_error()
  def divide(_a, _b), do: nif_error()
  def pow(_a, _b), do: nif_error()
  def remainder(_a, _b), do: nif_error()
  def atan2(_a, _b), do: nif_error()
  def min_tensor(_a, _b), do: nif_error()
  def max_tensor(_a, _b), do: nif_error()
  def quotient(_a, _b), do: nif_error()
  def bitwise_and(_a, _b), do: nif_error()
  def bitwise_or(_a, _b), do: nif_error()
  def bitwise_xor(_a, _b), do: nif_error()
  def left_shift(_a, _b), do: nif_error()
  def right_shift(_a, _b), do: nif_error()
  def equal(_a, _b), do: nif_error()
  def not_equal(_a, _b), do: nif_error()
  def greater(_a, _b), do: nif_error()
  def less(_a, _b), do: nif_error()
  def greater_equal(_a, _b), do: nif_error()
  def less_equal(_a, _b), do: nif_error()
  def logical_and(_a, _b), do: nif_error()
  def logical_or(_a, _b), do: nif_error()
  def logical_xor(_a, _b), do: nif_error()

  # ── Unary ops ───────────────────────────────────────────────

  def negate(_a), do: nif_error()
  def abs_tensor(_a), do: nif_error()
  def exp(_a), do: nif_error()
  def log(_a), do: nif_error()
  def sqrt(_a), do: nif_error()
  def sin(_a), do: nif_error()
  def cos(_a), do: nif_error()
  def tan(_a), do: nif_error()
  def sigmoid(_a), do: nif_error()
  def relu(_a), do: nif_error()
  def expm1(_a), do: nif_error()
  def log1p(_a), do: nif_error()
  def cosh(_a), do: nif_error()
  def sinh(_a), do: nif_error()
  def tanh(_a), do: nif_error()
  def acos(_a), do: nif_error()
  def asin(_a), do: nif_error()
  def atan(_a), do: nif_error()
  def acosh(_a), do: nif_error()
  def asinh(_a), do: nif_error()
  def atanh(_a), do: nif_error()
  def rsqrt(_a), do: nif_error()
  def cbrt(_a), do: nif_error()
  def erf(_a), do: nif_error()
  def erfc(_a), do: nif_error()
  def erf_inv(_a), do: nif_error()
  def bitwise_not(_a), do: nif_error()
  def ceil_tensor(_a), do: nif_error()
  def floor_tensor(_a), do: nif_error()
  def round_tensor(_a), do: nif_error()
  def sign_tensor(_a), do: nif_error()
  def conjugate(_a), do: nif_error()
  def count_leading_zeros(_a), do: nif_error()
  def population_count(_a), do: nif_error()
  def real(_a), do: nif_error()
  def imag(_a), do: nif_error()
  def is_nan(_a), do: nif_error()
  def is_infinity(_a), do: nif_error()

  # ── Shape ops ───────────────────────────────────────────────

  def reshape_tensor(_buf, _new_shape), do: nif_error()
  def squeeze_tensor(_buf, _axes), do: nif_error()
  def broadcast_tensor(_buf, _target_shape, _axes), do: nif_error()
  def transpose_tensor(_buf, _axes), do: nif_error()
  def pad_tensor(_buf, _pad_value, _padding_config), do: nif_error()
  def reverse_tensor(_buf, _axes), do: nif_error()

  # ── Slice / concat / select ─────────────────────────────────

  def slice_tensor(_buf, _starts, _lengths, _strides), do: nif_error()
  def concatenate_tensors(_bufs, _axis), do: nif_error()
  def stack_tensors(_bufs, _axis), do: nif_error()
  def select_tensor(_pred, _on_true, _on_false), do: nif_error()

  # ── Reductions ──────────────────────────────────────────────

  def sum_tensor(_buf, _opts), do: nif_error()
  def product_tensor(_buf, _opts), do: nif_error()
  def reduce_max(_buf, _opts), do: nif_error()
  def reduce_min(_buf, _opts), do: nif_error()
  def all_tensor(_buf, _opts), do: nif_error()
  def any_tensor(_buf, _opts), do: nif_error()
  def argmax_tensor(_buf, _opts), do: nif_error()
  def argmin_tensor(_buf, _opts), do: nif_error()

  # ── Window ops ──────────────────────────────────────────────

  def window_sum(_buf, _shape, _opts), do: nif_error()
  def window_max(_buf, _shape, _opts), do: nif_error()
  def window_min(_buf, _shape, _opts), do: nif_error()

  # ── LinAlg / clip / type ────────────────────────────────────

  def dot_tensor(_a, _c1, _b1, _b, _c2, _b2), do: nif_error()
  def clip_tensor(_buf, _min, _max), do: nif_error()
  def as_type_tensor(_buf, _dtype_str), do: nif_error()
  def constant_tensor(_shape, _dtype_str, _value), do: nif_error()
  def eye_tensor(_shape, _dtype_str), do: nif_error()
  def iota_tensor(_shape, _dtype_str, _axis), do: nif_error()

  # ── Stubs for complex ops (fallback to BinaryBackend) ───────

  def triangular_solve(_a, _b, _opts), do: nif_error()
  def conv(_input, _kernel, _opts), do: nif_error()
  def sort_tensor(_buf, _opts), do: nif_error()
  def argsort_tensor(_buf, _opts), do: nif_error()
  def fft_tensor(_buf, _opts), do: nif_error()
  def ifft_tensor(_buf, _opts), do: nif_error()
  def indexed_add(_t, _idx, _upd, _opts), do: nif_error()
  def indexed_put(_t, _idx, _upd, _opts), do: nif_error()
  def window_scatter_max(_t, _src, _init, _shape, _opts), do: nif_error()
  def window_scatter_min(_t, _src, _init, _shape, _opts), do: nif_error()
  def gather(_input, _indices, _opts), do: nif_error()
  def put_slice(_t, _starts, _slice), do: nif_error()
  def bitcast_tensor(_buf, _dtype_str), do: nif_error()

  defp nif_error, do: :erlang.nif_error(:nif_not_loaded)
end
