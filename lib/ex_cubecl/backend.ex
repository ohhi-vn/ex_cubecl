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

defmodule ExCubecl.Backend do
  @moduledoc """
  Nx backend using CubeCL Rust NIFs for tensor compute.
  """

  @behaviour Nx.Backend
  @enforce_keys [:ref]
  defstruct [:ref]

  alias Nx.Tensor, as: T
  alias ExCubecl.Backend, as: EB

  ## Type / shape helpers

  def type_to_string({:f, 32}), do: "f32"
  def type_to_string({:f, 64}), do: "f64"
  def type_to_string({:s, 32}), do: "s32"
  def type_to_string({:s, 64}), do: "s64"
  def type_to_string({:u, 32}), do: "u32"
  def type_to_string({:u, 8}), do: "u8"
  def type_to_string({:f, 16}), do: "f32"
  def type_to_string({:bf, 16}), do: "f32"
  def type_to_string(other), do: raise("unsupported type: #{inspect(other)}")

  def type_to_tuple("f32"), do: {:f, 32}
  def type_to_tuple("f64"), do: {:f, 64}
  def type_to_tuple("s32"), do: {:s, 32}
  def type_to_tuple("s64"), do: {:s, 64}
  def type_to_tuple("u32"), do: {:u, 32}
  def type_to_tuple("u8"), do: {:u, 8}

  defp shape_to_list(shape), do: Tuple.to_list(shape)

  defp unwrap!({:ok, ref}), do: ref
  defp unwrap!({:error, reason}), do: raise("NIF error: #{inspect(reason)}")

  defp wrap_tensor(ref, shape, type, names \\ nil) do
    names = names || List.duplicate(nil, tuple_size(shape))
    %T{data: %EB{ref: ref}, shape: shape, type: type, names: names}
  end

  ## to_ref helper

  def to_ref(%T{data: %EB{ref: ref}}), do: ref

  def to_ref(%T{} = tensor) do
    tensor = Nx.backend_transfer(tensor, __MODULE__)
    to_ref(tensor)
  end

  ## read binary from ref

  defp read_binary(ref), do: unwrap!(ExCubecl.NIF.read_tensor(ref))

  ## keep_dims detection

  defp keep_dims?(out_shape, in_shape), do: tuple_size(out_shape) == tuple_size(in_shape)

  ## broadcast axes

  defp broadcast_axes(from_shape, to_shape) do
    from_dims = Tuple.to_list(from_shape)
    to_dims = Tuple.to_list(to_shape)
    n = length(to_dims) - length(from_dims)
    from_dims = if n > 0, do: List.duplicate(1, n) ++ from_dims, else: from_dims

    # Build a mapping: for each input axis, which output axis does it correspond to?
    # When from_dims was padded with 1s, the non-padded dims align with the last
    # len(from_dims_original) dims of to_dims.
    from_rank = length(from_dims) - n
    to_rank = length(to_dims)

    if from_rank == 0 do
      []
    else
      # The input axes map to the last `from_rank` output axes
      Enum.map(0..(from_rank - 1), &(&1 + to_rank - from_rank))
    end
  end

  ## maybe broadcast

  defp maybe_broadcast_to(out_shape, %T{shape: shape} = t) when shape == out_shape, do: to_ref(t)

  defp maybe_broadcast_to(out_shape, %T{} = t) do
    ref = to_ref(t)
    axes = broadcast_axes(t.shape, out_shape)
    unwrap!(ExCubecl.NIF.broadcast_tensor(ref, shape_to_list(out_shape), axes))
  end

  ## reduction opts builder

  defp reduction_opts(axes, keep_dims) do
    keys = []
    keys = if axes != [], do: [{"axes", axes} | keys], else: keys
    keys = if keep_dims, do: [{"keep_axes", true} | keys], else: keys
    keys
  end

  # When axes is empty and output shape differs from input, reduce all dims.
  defp compute_axes(out_shape, in_shape, []), do: Enum.to_list(0..(tuple_size(in_shape) - 1))
  defp compute_axes(_out_shape, _in_shape, axes), do: axes

  ## axis opts builder

  defp axis_opts(axis, keep_dims) do
    keys = [{"axis", axis}]
    keys = if keep_dims, do: [{"keep_axes", true} | keys], else: keys
    keys
  end

  defp window_opts(window_shape, axes, keep_dims) do
    keys = [{"shape", window_shape}, {"axes", axes}]
    keys = if keep_dims, do: [{"keep_axes", true} | keys], else: keys
    keys
  end

  ## Fallback helper

  defp with_fallback(out, fun) do
    fun.()
  rescue
    _ ->
      binary = Nx.to_binary(out)
      tmp = Nx.from_binary(binary, out.type, backend: Nx.BinaryBackend)
      tmp_shape = Nx.shape(tmp)
      tmp_names = Nx.names(tmp)
      %T{data: %EB{ref: ref}} = Nx.backend_transfer(tmp, __MODULE__)
      wrap_tensor(ref, tmp_shape, out.type, tmp_names)
  end

  ################################################################
  # Nx.Backend callbacks
  ################################################################

  @impl true
  def init(opts), do: Keyword.validate!(opts, [])

  ## Allocation / deallocation / transfer

  @impl true
  def constant(out, constant, _opts) do
    shape = shape_to_list(out.shape)
    type_str = type_to_string(out.type)
    ref = unwrap!(ExCubecl.NIF.constant_tensor(shape, type_str, constant))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def from_binary(%T{shape: shape, type: type} = tensor, binary, _opts) do
    ref = unwrap!(ExCubecl.NIF.new_tensor(binary, shape_to_list(shape), type_to_string(type)))
    wrap_tensor(ref, shape, type, tensor.names)
  end

  @impl true
  def backend_copy(%T{data: %EB{ref: ref}} = tensor, EB, _opts) do
    binary = read_binary(ref)

    new_ref =
      unwrap!(
        ExCubecl.NIF.new_tensor(binary, shape_to_list(tensor.shape), type_to_string(tensor.type))
      )

    wrap_tensor(new_ref, tensor.shape, tensor.type, tensor.names)
  end

  def backend_copy(%T{data: %EB{ref: ref}} = tensor, backend, opts) do
    binary = read_binary(ref)
    backend.from_binary(tensor, binary, opts)
  end

  @impl true
  def backend_transfer(%T{data: %EB{ref: ref}} = tensor, backend, opts) do
    try do
      backend_copy(tensor, backend, opts)
    after
      ExCubecl.NIF.deallocate_tensor(ref)
    end
  end

  @impl true
  def backend_deallocate(%T{data: %EB{ref: ref}}) do
    ExCubecl.NIF.deallocate_tensor(ref)
    :ok
  end

  @impl true
  def to_binary(%T{data: %EB{ref: ref}, type: {_, size}} = t, limit) do
    binary = read_binary(ref)
    byte_limit = min(limit, Nx.size(t)) * div(size, 8)
    binary_part(binary, 0, min(byte_limit, byte_size(binary)))
  end

  @impl true
  def inspect(%T{} = tensor, inspect_opts) do
    limit = if inspect_opts.limit == :infinity, do: :infinity, else: inspect_opts.limit + 1
    binary = to_binary(tensor, min(limit, Nx.size(tensor)))
    Nx.Backend.inspect(tensor, binary, inspect_opts)
  end

  ## Creation helpers

  @impl true
  def eye(%T{shape: shape, type: type} = out, _opts) do
    ref = unwrap!(ExCubecl.NIF.eye_tensor(shape_to_list(shape), type_to_string(type)))
    wrap_tensor(ref, shape, type, out.names)
  end

  @impl true
  def iota(%T{shape: shape, type: type} = out, axis, _opts) do
    axis = axis || 0
    ref = unwrap!(ExCubecl.NIF.iota_tensor(shape_to_list(shape), type_to_string(type), axis))
    wrap_tensor(ref, shape, type, out.names)
  end

  ## Type conversion

  @impl true
  def as_type(%T{type: type} = out, %T{} = t) do
    t_ref = to_ref(t)
    ref = unwrap!(ExCubecl.NIF.as_type_tensor(t_ref, type_to_string(type)))
    wrap_tensor(ref, out.shape, type, out.names)
  end

  @impl true
  def bitcast(%T{type: type} = out, %T{} = t) do
    t_ref = to_ref(t)
    ref = unwrap!(ExCubecl.NIF.bitcast_tensor(t_ref, type_to_string(type)))
    wrap_tensor(ref, out.shape, type, out.names)
  end

  ## Shape operations

  @impl true
  def reshape(%T{shape: shape} = out, %T{} = t) do
    t_ref = to_ref(t)
    ref = unwrap!(ExCubecl.NIF.reshape_tensor(t_ref, shape_to_list(shape)))
    wrap_tensor(ref, shape, out.type, out.names)
  end

  @impl true
  def squeeze(out, %T{} = t, axes) do
    t_ref = to_ref(t)
    axes_i = Enum.map(axes, &if(&1 < 0, do: tuple_size(t.shape) + &1, else: &1))
    ref = unwrap!(ExCubecl.NIF.squeeze_tensor(t_ref, axes_i))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def broadcast(out, %T{} = t, shape, axes) do
    t_ref = to_ref(t)
    ref = unwrap!(ExCubecl.NIF.broadcast_tensor(t_ref, shape_to_list(shape), axes))
    wrap_tensor(ref, shape, out.type, out.names)
  end

  @impl true
  def transpose(out, %T{} = t, axes) do
    t_ref = to_ref(t)
    ref = unwrap!(ExCubecl.NIF.transpose_tensor(t_ref, axes))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def pad(out, %T{} = t, pad_value, padding_config) do
    t_ref = to_ref(t)
    pad_ref = to_ref(pad_value)
    config = Enum.map(padding_config, fn {lo, hi, interior} -> {lo, hi, interior} end)
    ref = unwrap!(ExCubecl.NIF.pad_tensor(t_ref, pad_ref, config))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def reverse(out, %T{} = t, axes) do
    t_ref = to_ref(t)
    ref = unwrap!(ExCubecl.NIF.reverse_tensor(t_ref, axes))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  ## Slice / concat / indexing

  @impl true
  def slice(out, %T{} = t, starts, lengths, strides) do
    t_ref = to_ref(t)
    ref = unwrap!(ExCubecl.NIF.slice_tensor(t_ref, starts, lengths, strides))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def put_slice(out, %T{} = t, start_indices, %T{} = slice) do
    t_ref = to_ref(t)
    slice_ref = to_ref(slice)
    ref = unwrap!(ExCubecl.NIF.put_slice(t_ref, start_indices, slice_ref))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def gather(out, %T{} = input, %T{} = indices, opts) do
    input_ref = to_ref(input)
    indices_ref = to_ref(indices)
    ref = unwrap!(ExCubecl.NIF.gather(input_ref, indices_ref, opts))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def concatenate(out, tensors, axis) do
    refs = Enum.map(tensors, &to_ref/1)
    ref = unwrap!(ExCubecl.NIF.concatenate_tensors(refs, axis))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def stack(out, tensors, axis) do
    refs = Enum.map(tensors, &to_ref/1)
    ref = unwrap!(ExCubecl.NIF.stack_tensors(refs, axis))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def select(out, %T{} = pred, %T{} = on_true, %T{} = on_false) do
    pred_ref = to_ref(pred)
    on_true_ref = to_ref(on_true)
    on_false_ref = to_ref(on_false)
    ref = unwrap!(ExCubecl.NIF.select_tensor(pred_ref, on_true_ref, on_false_ref))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  ## Reductions

  @impl true
  def all(out, %T{} = t, opts) do
    t_ref = to_ref(t)
    axes = compute_axes(out.shape, t.shape, opts[:axes] || [])
    kd = keep_dims?(out.shape, t.shape)
    ref = unwrap!(ExCubecl.NIF.all_tensor(t_ref, reduction_opts(axes, kd)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def any(out, %T{} = t, opts) do
    t_ref = to_ref(t)
    axes = compute_axes(out.shape, t.shape, opts[:axes] || [])
    kd = keep_dims?(out.shape, t.shape)
    ref = unwrap!(ExCubecl.NIF.any_tensor(t_ref, reduction_opts(axes, kd)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def sum(out, %T{} = t, opts) do
    t_ref = to_ref(t)
    axes = compute_axes(out.shape, t.shape, opts[:axes] || [])
    kd = keep_dims?(out.shape, t.shape)
    ref = unwrap!(ExCubecl.NIF.sum_tensor(t_ref, reduction_opts(axes, kd)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def product(out, %T{} = t, opts) do
    t_ref = to_ref(t)
    axes = compute_axes(out.shape, t.shape, opts[:axes] || [])
    kd = keep_dims?(out.shape, t.shape)
    ref = unwrap!(ExCubecl.NIF.product_tensor(t_ref, reduction_opts(axes, kd)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def reduce_max(out, %T{} = t, opts) do
    t_ref = to_ref(t)
    axes = compute_axes(out.shape, t.shape, opts[:axes] || [])
    kd = keep_dims?(out.shape, t.shape)
    ref = unwrap!(ExCubecl.NIF.reduce_max(t_ref, reduction_opts(axes, kd)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def reduce_min(out, %T{} = t, opts) do
    t_ref = to_ref(t)
    axes = compute_axes(out.shape, t.shape, opts[:axes] || [])
    kd = keep_dims?(out.shape, t.shape)
    ref = unwrap!(ExCubecl.NIF.reduce_min(t_ref, reduction_opts(axes, kd)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def argmax(out, %T{} = t, opts) do
    t_ref = to_ref(t)
    axis = opts[:axis] || 0
    kd = keep_dims?(out.shape, t.shape)
    ref = unwrap!(ExCubecl.NIF.argmax_tensor(t_ref, axis_opts(axis, kd)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def argmin(out, %T{} = t, opts) do
    t_ref = to_ref(t)
    axis = opts[:axis] || 0
    kd = keep_dims?(out.shape, t.shape)
    ref = unwrap!(ExCubecl.NIF.argmin_tensor(t_ref, axis_opts(axis, kd)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def reduce(out, %T{} = t, %T{} = acc, opts, fun) do
    t_ref = to_ref(t)
    acc_ref = to_ref(acc)
    ref = unwrap!(ExCubecl.NIF.reduce(t_ref, acc_ref, opts, fun))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def window_reduce(out, %T{} = t, %T{} = acc, shape, opts, fun) do
    t_ref = to_ref(t)
    acc_ref = to_ref(acc)
    ref = unwrap!(ExCubecl.NIF.window_reduce(t_ref, acc_ref, shape, opts, fun))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  ## Window operations

  @impl true
  def window_sum(out, %T{} = t, window_shape, opts) do
    t_ref = to_ref(t)
    ws = shape_to_list(window_shape)
    axes = opts[:axes] || Enum.to_list(0..(tuple_size(t.shape) - 1))
    kd = keep_dims?(out.shape, t.shape)

    ref =
      unwrap!(ExCubecl.NIF.window_sum(t_ref, ws, window_opts(ws, axes, kd)))

    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def window_product(out, %T{} = t, window_shape, opts) do
    with_fallback(out, fn ->
      t_ref = to_ref(t)
      ws = shape_to_list(window_shape)
      axes = opts[:axes] || Enum.to_list(0..(tuple_size(t.shape) - 1))
      kd = keep_dims?(out.shape, t.shape)

      ref =
        unwrap!(ExCubecl.NIF.window_product(t_ref, ws, window_opts(ws, axes, kd)))

      wrap_tensor(ref, out.shape, out.type, out.names)
    end)
  end

  @impl true
  def window_max(out, %T{} = t, window_shape, opts) do
    t_ref = to_ref(t)
    ws = shape_to_list(window_shape)
    axes = opts[:axes] || Enum.to_list(0..(tuple_size(t.shape) - 1))
    kd = keep_dims?(out.shape, t.shape)

    ref =
      unwrap!(ExCubecl.NIF.window_max(t_ref, ws, window_opts(ws, axes, kd)))

    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def window_min(out, %T{} = t, window_shape, opts) do
    t_ref = to_ref(t)
    ws = shape_to_list(window_shape)
    axes = opts[:axes] || Enum.to_list(0..(tuple_size(t.shape) - 1))
    kd = keep_dims?(out.shape, t.shape)

    ref =
      unwrap!(ExCubecl.NIF.window_min(t_ref, ws, window_opts(ws, axes, kd)))

    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  ## Sorting

  @impl true
  def sort(out, %T{} = t, opts) do
    t_ref = to_ref(t)
    ref = unwrap!(ExCubecl.NIF.sort_tensor(t_ref, opts))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def argsort(out, %T{} = t, opts) do
    t_ref = to_ref(t)
    ref = unwrap!(ExCubecl.NIF.argsort_tensor(t_ref, opts))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  ## Scatter / indexed

  @impl true
  def window_scatter_max(out, %T{} = t, %T{} = source, %T{} = init, shape, opts) do
    ref =
      unwrap!(
        ExCubecl.NIF.window_scatter_max(to_ref(t), to_ref(source), to_ref(init), shape, opts)
      )

    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def window_scatter_min(out, %T{} = t, %T{} = source, %T{} = init, shape, opts) do
    ref =
      unwrap!(
        ExCubecl.NIF.window_scatter_min(to_ref(t), to_ref(source), to_ref(init), shape, opts)
      )

    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def indexed_add(out, %T{} = t, %T{} = indices, %T{} = updates, opts) do
    ref = unwrap!(ExCubecl.NIF.indexed_add(to_ref(t), to_ref(indices), to_ref(updates), opts))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def indexed_put(out, %T{} = t, %T{} = indices, %T{} = updates, opts) do
    ref = unwrap!(ExCubecl.NIF.indexed_put(to_ref(t), to_ref(indices), to_ref(updates), opts))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  ## Linear algebra

  @impl true
  def dot(out, %T{} = left, c1, b1, %T{} = right, c2, b2) do
    left_ref = to_ref(left)
    right_ref = to_ref(right)
    ref = unwrap!(ExCubecl.NIF.dot_tensor(left_ref, c1, b1, right_ref, c2, b2))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def triangular_solve(out, %T{} = a, %T{} = b, opts) do
    with_fallback(out, fn ->
      ref = unwrap!(ExCubecl.NIF.triangular_solve(to_ref(a), to_ref(b), opts))
      wrap_tensor(ref, out.shape, out.type, out.names)
    end)
  end

  ## Convolution

  @impl true
  def conv(out, %T{} = t, %T{} = kernel, opts) do
    ref = unwrap!(ExCubecl.NIF.conv(to_ref(t), to_ref(kernel), opts))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  ## FFT

  @impl true
  def fft(out, %T{} = t, opts) do
    with_fallback(out, fn ->
      ref = unwrap!(ExCubecl.NIF.fft_tensor(to_ref(t), opts))
      wrap_tensor(ref, out.shape, out.type, out.names)
    end)
  end

  @impl true
  def ifft(out, %T{} = t, opts) do
    with_fallback(out, fn ->
      ref = unwrap!(ExCubecl.NIF.ifft_tensor(to_ref(t), opts))
      wrap_tensor(ref, out.shape, out.type, out.names)
    end)
  end

  ## Clip

  @impl true
  def clip(out, %T{} = t, %T{} = min_t, %T{} = max_t) do
    ref = unwrap!(ExCubecl.NIF.clip_tensor(to_ref(t), to_ref(min_t), to_ref(max_t)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  ## Batched output

  @impl true
  def to_batched(out, tensor, opts) do
    leftover = opts[:leftover]
    batch_size = elem(out.shape, 0)
    axis_size = elem(tensor.shape, 0)
    remainder = rem(axis_size, batch_size)
    num_full = div(axis_size, batch_size)

    range = if remainder != 0 and leftover == :repeat, do: 0..num_full, else: 0..(num_full - 1)

    Stream.map(range, fn
      ^num_full ->
        Nx.concatenate([
          Nx.slice_along_axis(tensor, num_full * batch_size, remainder),
          Nx.slice_along_axis(tensor, 0, batch_size - remainder)
        ])

      i ->
        Nx.slice_along_axis(tensor, i * batch_size, batch_size)
    end)
  end

  ## Block

  @impl true
  def block(struct, output, args, fun) do
    apply(fun, [struct | args])
  end

  ## Pointer operations

  @impl true
  def to_pointer(%T{data: %EB{ref: ref}}, _opts), do: ref

  @impl true
  def from_pointer(opaque_pointer, type, shape, _backend_opts, _opts) do
    wrap_tensor(opaque_pointer, shape, type)
  end

  ################################################################
  # Binary operations (NIF-backed)
  ################################################################

  @impl true
  def add(out, left, right) do
    l = maybe_broadcast_to(out.shape, left)
    r = maybe_broadcast_to(out.shape, right)
    ref = unwrap!(ExCubecl.NIF.add(l, r))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def subtract(out, left, right) do
    l = maybe_broadcast_to(out.shape, left)
    r = maybe_broadcast_to(out.shape, right)
    ref = unwrap!(ExCubecl.NIF.subtract(l, r))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def multiply(out, left, right) do
    l = maybe_broadcast_to(out.shape, left)
    r = maybe_broadcast_to(out.shape, right)
    ref = unwrap!(ExCubecl.NIF.multiply(l, r))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def divide(out, left, right) do
    l = maybe_broadcast_to(out.shape, left)
    r = maybe_broadcast_to(out.shape, right)
    ref = unwrap!(ExCubecl.NIF.divide(l, r))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def pow(out, left, right) do
    l = maybe_broadcast_to(out.shape, left)
    r = maybe_broadcast_to(out.shape, right)
    ref = unwrap!(ExCubecl.NIF.pow(l, r))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def remainder(out, left, right) do
    l = maybe_broadcast_to(out.shape, left)
    r = maybe_broadcast_to(out.shape, right)
    ref = unwrap!(ExCubecl.NIF.remainder(l, r))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def atan2(out, left, right) do
    l = maybe_broadcast_to(out.shape, left)
    r = maybe_broadcast_to(out.shape, right)
    ref = unwrap!(ExCubecl.NIF.atan2(l, r))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def min(out, left, right) do
    l = maybe_broadcast_to(out.shape, left)
    r = maybe_broadcast_to(out.shape, right)
    ref = unwrap!(ExCubecl.NIF.min_tensor(l, r))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def max(out, left, right) do
    l = maybe_broadcast_to(out.shape, left)
    r = maybe_broadcast_to(out.shape, right)
    ref = unwrap!(ExCubecl.NIF.max_tensor(l, r))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def quotient(out, left, right) do
    l = maybe_broadcast_to(out.shape, left)
    r = maybe_broadcast_to(out.shape, right)
    ref = unwrap!(ExCubecl.NIF.quotient(l, r))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def bitwise_and(out, left, right) do
    l = maybe_broadcast_to(out.shape, left)
    r = maybe_broadcast_to(out.shape, right)
    ref = unwrap!(ExCubecl.NIF.bitwise_and(l, r))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def bitwise_or(out, left, right) do
    l = maybe_broadcast_to(out.shape, left)
    r = maybe_broadcast_to(out.shape, right)
    ref = unwrap!(ExCubecl.NIF.bitwise_or(l, r))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def bitwise_xor(out, left, right) do
    l = maybe_broadcast_to(out.shape, left)
    r = maybe_broadcast_to(out.shape, right)
    ref = unwrap!(ExCubecl.NIF.bitwise_xor(l, r))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def left_shift(out, left, right) do
    l = maybe_broadcast_to(out.shape, left)
    r = maybe_broadcast_to(out.shape, right)
    ref = unwrap!(ExCubecl.NIF.left_shift(l, r))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def right_shift(out, left, right) do
    l = maybe_broadcast_to(out.shape, left)
    r = maybe_broadcast_to(out.shape, right)
    ref = unwrap!(ExCubecl.NIF.right_shift(l, r))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def equal(out, left, right) do
    l = maybe_broadcast_to(out.shape, left)
    r = maybe_broadcast_to(out.shape, right)
    ref = unwrap!(ExCubecl.NIF.equal(l, r))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def not_equal(out, left, right) do
    l = maybe_broadcast_to(out.shape, left)
    r = maybe_broadcast_to(out.shape, right)
    ref = unwrap!(ExCubecl.NIF.not_equal(l, r))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def greater(out, left, right) do
    l = maybe_broadcast_to(out.shape, left)
    r = maybe_broadcast_to(out.shape, right)
    ref = unwrap!(ExCubecl.NIF.greater(l, r))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def less(out, left, right) do
    l = maybe_broadcast_to(out.shape, left)
    r = maybe_broadcast_to(out.shape, right)
    ref = unwrap!(ExCubecl.NIF.less(l, r))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def greater_equal(out, left, right) do
    l = maybe_broadcast_to(out.shape, left)
    r = maybe_broadcast_to(out.shape, right)
    ref = unwrap!(ExCubecl.NIF.greater_equal(l, r))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def less_equal(out, left, right) do
    l = maybe_broadcast_to(out.shape, left)
    r = maybe_broadcast_to(out.shape, right)
    ref = unwrap!(ExCubecl.NIF.less_equal(l, r))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def logical_and(out, left, right) do
    l = maybe_broadcast_to(out.shape, left)
    r = maybe_broadcast_to(out.shape, right)
    ref = unwrap!(ExCubecl.NIF.logical_and(l, r))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def logical_or(out, left, right) do
    l = maybe_broadcast_to(out.shape, left)
    r = maybe_broadcast_to(out.shape, right)
    ref = unwrap!(ExCubecl.NIF.logical_or(l, r))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def logical_xor(out, left, right) do
    l = maybe_broadcast_to(out.shape, left)
    r = maybe_broadcast_to(out.shape, right)
    ref = unwrap!(ExCubecl.NIF.logical_xor(l, r))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  ################################################################
  # Unary operations (NIF-backed)
  ################################################################

  @impl true
  def exp(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.exp(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def expm1(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.expm1(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def log(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.log(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def log1p(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.log1p(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def sigmoid(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.sigmoid(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def cos(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.cos(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def sin(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.sin(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def tan(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.tan(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def cosh(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.cosh(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def sinh(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.sinh(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def tanh(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.tanh(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def acos(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.acos(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def asin(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.asin(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def atan(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.atan(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def acosh(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.acosh(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def asinh(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.asinh(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def atanh(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.atanh(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def sqrt(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.sqrt(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def rsqrt(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.rsqrt(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def cbrt(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.cbrt(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def erf(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.erf(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def erfc(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.erfc(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def erf_inv(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.erf_inv(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def abs(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.abs_tensor(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def bitwise_not(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.bitwise_not(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def ceil(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.ceil_tensor(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def conjugate(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.conjugate(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def floor(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.floor_tensor(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def negate(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.negate(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def round(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.round_tensor(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def sign(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.sign_tensor(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def count_leading_zeros(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.count_leading_zeros(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def population_count(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.population_count(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def real(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.real(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def imag(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.imag(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def is_nan(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.is_nan(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end

  @impl true
  def is_infinity(out, tensor) do
    ref = unwrap!(ExCubecl.NIF.is_infinity(to_ref(tensor)))
    wrap_tensor(ref, out.shape, out.type, out.names)
  end
end
