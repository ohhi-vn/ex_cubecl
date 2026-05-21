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

  ## ── Buffer lifecycle ───────────────────────────────────────

  @spec new_tensor(binary(), [non_neg_integer()], String.t()) :: {:ok, reference()} | {:error, term()}
  def new_tensor(_data, _shape, _dtype), do: nif_error()

  @spec read_tensor(reference()) :: {:ok, binary()} | {:error, term()}
  def read_tensor(_ref), do: nif_error()

  @spec deallocate_tensor(reference()) :: :ok | {:error, term()}
  def deallocate_tensor(_ref), do: nif_error()

  @spec tensor_shape(reference()) :: {:ok, [non_neg_integer()]} | {:error, term()}
  def tensor_shape(_ref), do: nif_error()

  @spec tensor_dtype(reference()) :: {:ok, non_neg_integer()} | {:error, term()}
  def tensor_dtype(_ref), do: nif_error()

  ## ── Binary ops ─────────────────────────────────────────────

  @spec add(reference(), reference()) :: {:ok, reference()} | {:error, term()}
  def add(_a, _b), do: nif_error()

  @spec subtract(reference(), reference()) :: {:ok, reference()} | {:error, term()}
  def subtract(_a, _b), do: nif_error()

  @spec multiply(reference(), reference()) :: {:ok, reference()} | {:error, term()}
  def multiply(_a, _b), do: nif_error()

  @spec divide(reference(), reference()) :: {:ok, reference()} | {:error, term()}
  def divide(_a, _b), do: nif_error()

  @spec pow(reference(), reference()) :: {:ok, reference()} | {:error, term()}
  def pow(_a, _b), do: nif_error()

  @spec remainder(reference(), reference()) :: {:ok, reference()} | {:error, term()}
  def remainder(_a, _b), do: nif_error()

  @spec atan2(reference(), reference()) :: {:ok, reference()} | {:error, term()}
  def atan2(_a, _b), do: nif_error()

  @spec min_tensor(reference(), reference()) :: {:ok, reference()} | {:error, term()}
  def min_tensor(_a, _b), do: nif_error()

  @spec max_tensor(reference(), reference()) :: {:ok, reference()} | {:error, term()}
  def max_tensor(_a, _b), do: nif_error()

  @spec quotient(reference(), reference()) :: {:ok, reference()} | {:error, term()}
  def quotient(_a, _b), do: nif_error()

  @spec bitwise_and(reference(), reference()) :: {:ok, reference()} | {:error, term()}
  def bitwise_and(_a, _b), do: nif_error()

  @spec bitwise_or(reference(), reference()) :: {:ok, reference()} | {:error, term()}
  def bitwise_or(_a, _b), do: nif_error()

  @spec bitwise_xor(reference(), reference()) :: {:ok, reference()} | {:error, term()}
  def bitwise_xor(_a, _b), do: nif_error()

  @spec left_shift(reference(), reference()) :: {:ok, reference()} | {:error, term()}
  def left_shift(_a, _b), do: nif_error()

  @spec right_shift(reference(), reference()) :: {:ok, reference()} | {:error, term()}
  def right_shift(_a, _b), do: nif_error()

  @spec equal(reference(), reference()) :: {:ok, reference()} | {:error, term()}
  def equal(_a, _b), do: nif_error()

  @spec not_equal(reference(), reference()) :: {:ok, reference()} | {:error, term()}
  def not_equal(_a, _b), do: nif_error()

  @spec greater(reference(), reference()) :: {:ok, reference()} | {:error, term()}
  def greater(_a, _b), do: nif_error()

  @spec less(reference(), reference()) :: {:ok, reference()} | {:error, term()}
  def less(_a, _b), do: nif_error()

  @spec greater_equal(reference(), reference()) :: {:ok, reference()} | {:error, term()}
  def greater_equal(_a, _b), do: nif_error()

  @spec less_equal(reference(), reference()) :: {:ok, reference()} | {:error, term()}
  def less_equal(_a, _b), do: nif_error()

  @spec logical_and(reference(), reference()) :: {:ok, reference()} | {:error, term()}
  def logical_and(_a, _b), do: nif_error()

  @spec logical_or(reference(), reference()) :: {:ok, reference()} | {:error, term()}
  def logical_or(_a, _b), do: nif_error()

  @spec logical_xor(reference(), reference()) :: {:ok, reference()} | {:error, term()}
  def logical_xor(_a, _b), do: nif_error()

  ## ── Unary ops ──────────────────────────────────────────────

  @spec negate(reference()) :: {:ok, reference()} | {:error, term()}
  def negate(_a), do: nif_error()

  @spec abs_tensor(reference()) :: {:ok, reference()} | {:error, term()}
  def abs_tensor(_a), do: nif_error()

  @spec exp(reference()) :: {:ok, reference()} | {:error, term()}
  def exp(_a), do: nif_error()

  @spec log(reference()) :: {:ok, reference()} | {:error, term()}
  def log(_a), do: nif_error()

  @spec sqrt(reference()) :: {:ok, reference()} | {:error, term()}
  def sqrt(_a), do: nif_error()

  @spec sin(reference()) :: {:ok, reference()} | {:error, term()}
  def sin(_a), do: nif_error()

  @spec cos(reference()) :: {:ok, reference()} | {:error, term()}
  def cos(_a), do: nif_error()

  @spec tan(reference()) :: {:ok, reference()} | {:error, term()}
  def tan(_a), do: nif_error()

  @spec sigmoid(reference()) :: {:ok, reference()} | {:error, term()}
  def sigmoid(_a), do: nif_error()

  @spec relu(reference()) :: {:ok, reference()} | {:error, term()}
  def relu(_a), do: nif_error()

  @spec expm1(reference()) :: {:ok, reference()} | {:error, term()}
  def expm1(_a), do: nif_error()

  @spec log1p(reference()) :: {:ok, reference()} | {:error, term()}
  def log1p(_a), do: nif_error()

  @spec cosh(reference()) :: {:ok, reference()} | {:error, term()}
  def cosh(_a), do: nif_error()

  @spec sinh(reference()) :: {:ok, reference()} | {:error, term()}
  def sinh(_a), do: nif_error()

  @spec tanh(reference()) :: {:ok, reference()} | {:error, term()}
  def tanh(_a), do: nif_error()

  @spec acos(reference()) :: {:ok, reference()} | {:error, term()}
  def acos(_a), do: nif_error()

  @spec asin(reference()) :: {:ok, reference()} | {:error, term()}
  def asin(_a), do: nif_error()

  @spec atan(reference()) :: {:ok, reference()} | {:error, term()}
  def atan(_a), do: nif_error()

  @spec acosh(reference()) :: {:ok, reference()} | {:error, term()}
  def acosh(_a), do: nif_error()

  @spec asinh(reference()) :: {:ok, reference()} | {:error, term()}
  def asinh(_a), do: nif_error()

  @spec atanh(reference()) :: {:ok, reference()} | {:error, term()}
  def atanh(_a), do: nif_error()

  @spec rsqrt(reference()) :: {:ok, reference()} | {:error, term()}
  def rsqrt(_a), do: nif_error()

  @spec cbrt(reference()) :: {:ok, reference()} | {:error, term()}
  def cbrt(_a), do: nif_error()

  @spec erf(reference()) :: {:ok, reference()} | {:error, term()}
  def erf(_a), do: nif_error()

  @spec erfc(reference()) :: {:ok, reference()} | {:error, term()}
  def erfc(_a), do: nif_error()

  @spec erf_inv(reference()) :: {:ok, reference()} | {:error, term()}
  def erf_inv(_a), do: nif_error()

  @spec bitwise_not(reference()) :: {:ok, reference()} | {:error, term()}
  def bitwise_not(_a), do: nif_error()

  @spec ceil_tensor(reference()) :: {:ok, reference()} | {:error, term()}
  def ceil_tensor(_a), do: nif_error()

  @spec floor_tensor(reference()) :: {:ok, reference()} | {:error, term()}
  def floor_tensor(_a), do: nif_error()

  @spec round_tensor(reference()) :: {:ok, reference()} | {:error, term()}
  def round_tensor(_a), do: nif_error()

  @spec sign_tensor(reference()) :: {:ok, reference()} | {:error, term()}
  def sign_tensor(_a), do: nif_error()

  @spec conjugate(reference()) :: {:ok, reference()} | {:error, term()}
  def conjugate(_a), do: nif_error()

  @spec count_leading_zeros(reference()) :: {:ok, reference()} | {:error, term()}
  def count_leading_zeros(_a), do: nif_error()

  @spec population_count(reference()) :: {:ok, reference()} | {:error, term()}
  def population_count(_a), do: nif_error()

  @spec real(reference()) :: {:ok, reference()} | {:error, term()}
  def real(_a), do: nif_error()

  @spec imag(reference()) :: {:ok, reference()} | {:error, term()}
  def imag(_a), do: nif_error()

  @spec is_nan(reference()) :: {:ok, reference()} | {:error, term()}
  def is_nan(_a), do: nif_error()

  @spec is_infinity(reference()) :: {:ok, reference()} | {:error, term()}
  def is_infinity(_a), do: nif_error()

  ## ── Shape ops ──────────────────────────────────────────────

  @spec reshape_tensor(reference(), [non_neg_integer()]) :: {:ok, reference()} | {:error, term()}
  def reshape_tensor(_buf, _new_shape), do: nif_error()

  @spec squeeze_tensor(reference(), [integer()]) :: {:ok, reference()} | {:error, term()}
  def squeeze_tensor(_buf, _axes), do: nif_error()

  @spec broadcast_tensor(reference(), [non_neg_integer()], [non_neg_integer()]) :: {:ok, reference()} | {:error, term()}
  def broadcast_tensor(_buf, _target_shape, _axes), do: nif_error()

  @spec transpose_tensor(reference(), [non_neg_integer()]) :: {:ok, reference()} | {:error, term()}
  def transpose_tensor(_buf, _axes), do: nif_error()

  @spec pad_tensor(reference(), reference(), [{integer(), integer(), integer()}]) :: {:ok, reference()} | {:error, term()}
  def pad_tensor(_buf, _pad_value, _padding_config), do: nif_error()

  @spec reverse_tensor(reference(), [non_neg_integer()]) :: {:ok, reference()} | {:error, term()}
  def reverse_tensor(_buf, _axes), do: nif_error()

  ## ── Slice / concat / select ────────────────────────────────

  @spec slice_tensor(reference(), [non_neg_integer()], [non_neg_integer()], [non_neg_integer()]) :: {:ok, reference()} | {:error, term()}
  def slice_tensor(_buf, _starts, _lengths, _strides), do: nif_error()

  @spec concatenate_tensors([reference()], non_neg_integer()) :: {:ok, reference()} | {:error, term()}
  def concatenate_tensors(_bufs, _axis), do: nif_error()

  @spec stack_tensors([reference()], non_neg_integer()) :: {:ok, reference()} | {:error, term()}
  def stack_tensors(_bufs, _axis), do: nif_error()

  @spec select_tensor(reference(), reference(), reference()) :: {:ok, reference()} | {:error, term()}
  def select_tensor(_pred, _on_true, _on_false), do: nif_error()

  ## ── Reductions ─────────────────────────────────────────────

  @spec sum_tensor(reference(), keyword()) :: {:ok, reference()} | {:error, term()}
  def sum_tensor(_buf, _opts), do: nif_error()

  @spec product_tensor(reference(), keyword()) :: {:ok, reference()} | {:error, term()}
  def product_tensor(_buf, _opts), do: nif_error()

  @spec reduce_max(reference(), keyword()) :: {:ok, reference()} | {:error, term()}
  def reduce_max(_buf, _opts), do: nif_error()

  @spec reduce_min(reference(), keyword()) :: {:ok, reference()} | {:error, term()}
  def reduce_min(_buf, _opts), do: nif_error()

  @spec all_tensor(reference(), keyword()) :: {:ok, reference()} | {:error, term()}
  def all_tensor(_buf, _opts), do: nif_error()

  @spec any_tensor(reference(), keyword()) :: {:ok, reference()} | {:error, term()}
  def any_tensor(_buf, _opts), do: nif_error()

  @spec argmax_tensor(reference(), keyword()) :: {:ok, reference()} | {:error, term()}
  def argmax_tensor(_buf, _opts), do: nif_error()

  @spec argmin_tensor(reference(), keyword()) :: {:ok, reference()} | {:error, term()}
  def argmin_tensor(_buf, _opts), do: nif_error()

  ## ── Window ops ─────────────────────────────────────────────

  @spec window_sum(reference(), [non_neg_integer()], keyword()) :: {:ok, reference()} | {:error, term()}
  def window_sum(_buf, _shape, _opts), do: nif_error()

  @spec window_max(reference(), [non_neg_integer()], keyword()) :: {:ok, reference()} | {:error, term()}
  def window_max(_buf, _shape, _opts), do: nif_error()

  @spec window_min(reference(), [non_neg_integer()], keyword()) :: {:ok, reference()} | {:error, term()}
  def window_min(_buf, _shape, _opts), do: nif_error()

  ## ── LinAlg / clip / type ───────────────────────────────────

  @spec dot_tensor(reference(), non_neg_integer(), non_neg_integer(), reference(), non_neg_integer(), non_neg_integer()) :: {:ok, reference()} | {:error, term()}
  def dot_tensor(_a, _c1, _b1, _b, _c2, _b2), do: nif_error()

  @spec clip_tensor(reference(), reference(), reference()) :: {:ok, reference()} | {:error, term()}
  def clip_tensor(_buf, _min, _max), do: nif_error()

  @spec as_type_tensor(reference(), String.t()) :: {:ok, reference()} | {:error, term()}
  def as_type_tensor(_buf, _dtype_str), do: nif_error()

  @spec constant_tensor([non_neg_integer()], String.t(), float()) :: {:ok, reference()} | {:error, term()}
  def constant_tensor(_shape, _dtype_str, _value), do: nif_error()

  @spec eye_tensor([non_neg_integer()], String.t()) :: {:ok, reference()} | {:error, term()}
  def eye_tensor(_shape, _dtype_str), do: nif_error()

  @spec iota_tensor([non_neg_integer()], String.t(), non_neg_integer()) :: {:ok, reference()} | {:error, term()}
  def iota_tensor(_shape, _dtype_str, _axis), do: nif_error()

  ## ── Sorting ────────────────────────────────────────────────

  @spec sort_tensor(reference(), keyword()) :: {:ok, reference()} | {:error, term()}
  def sort_tensor(_buf, _opts), do: nif_error()

  @spec argsort_tensor(reference(), keyword()) :: {:ok, reference()} | {:error, term()}
  def argsort_tensor(_buf, _opts), do: nif_error()

  ## ── Additional ops ─────────────────────────────────────────

  @spec gather(reference(), reference(), keyword()) :: {:ok, reference()} | {:error, term()}
  def gather(_input, _indices, _opts), do: nif_error()

  @spec put_slice(reference(), [non_neg_integer()], reference()) :: {:ok, reference()} | {:error, term()}
  def put_slice(_t, _starts, _slice), do: nif_error()

  @spec bitcast_tensor(reference(), String.t()) :: {:ok, reference()} | {:error, term()}
  def bitcast_tensor(_buf, _dtype_str), do: nif_error()

  @spec conv(reference(), reference(), keyword()) :: {:ok, reference()} | {:error, term()}
  def conv(_input, _kernel, _opts), do: nif_error()

  @spec indexed_add(reference(), reference(), reference(), keyword()) :: {:ok, reference()} | {:error, term()}
  def indexed_add(_t, _idx, _upd, _opts), do: nif_error()

  @spec indexed_put(reference(), reference(), reference(), keyword()) :: {:ok, reference()} | {:error, term()}
  def indexed_put(_t, _idx, _upd, _opts), do: nif_error()

  ## ── Stubs for complex ops (fallback to BinaryBackend) ──────

  @spec triangular_solve(reference(), reference(), keyword()) :: {:ok, reference()} | {:error, term()}
  def triangular_solve(_a, _b, _opts), do: nif_error()

  @spec fft_tensor(reference(), keyword()) :: {:ok, reference()} | {:error, term()}
  def fft_tensor(_buf, _opts), do: nif_error()

  @spec ifft_tensor(reference(), keyword()) :: {:ok, reference()} | {:error, term()}
  def ifft_tensor(_buf, _opts), do: nif_error()

  @spec window_scatter_max(reference(), reference(), reference(), [non_neg_integer()], keyword()) :: {:ok, reference()} | {:error, term()}
  def window_scatter_max(_t, _src, _init, _shape, _opts), do: nif_error()

  @spec window_scatter_min(reference(), reference(), reference(), [non_neg_integer()], keyword()) :: {:ok, reference()} | {:error, term()}
  def window_scatter_min(_t, _src, _init, _shape, _opts), do: nif_error()

  @spec reduce(reference(), reference(), keyword(), term()) :: {:ok, reference()} | {:error, term()}
  def reduce(_t, _acc, _opts, _fun), do: nif_error()

  @spec window_reduce(reference(), reference(), [non_neg_integer()], keyword(), term()) :: {:ok, reference()} | {:error, term()}
  def window_reduce(_t, _acc, _shape, _opts, _fun), do: nif_error()

  @spec window_product(reference(), [non_neg_integer()], keyword()) :: {:ok, reference()} | {:error, term()}
  def window_product(_buf, _shape, _opts), do: nif_error()

  defp nif_error, do: :erlang.nif_error(:nif_not_loaded)
end
