//! C FFI layer for ex_cubecl tensor operations.
//!
//! This module exposes all tensor operations as plain C functions using `usize` handles
//! to reference tensors. It is designed for iOS (via Objective-C/Swift bridging) and
//! Android (via JNI) interoperability.

use crate::{binary_op, from_f64, strides_for, to_f64, unary_op, BufResource, DType};
use std::cell::RefCell;
use std::collections::HashMap;
use std::os::raw::c_char;
use std::slice;

// ---------------------------------------------------------------------------
// Handle table
// ---------------------------------------------------------------------------

/// Opaque handle type — pointer-sized, passed across the FFI boundary.
pub type TensorHandle = usize;

thread_local! {
    static TENSOR_STORE: RefCell<HashMap<TensorHandle, BufResource>> = RefCell::new(HashMap::new());
    static NEXT_ID: RefCell<TensorHandle> = RefCell::new(1);
    static LAST_ERROR: RefCell<String> = RefCell::new(String::new());
}

fn set_error(msg: impl Into<String>) {
    LAST_ERROR.with(|e| *e.borrow_mut() = msg.into());
}

fn alloc_handle(resource: BufResource) -> TensorHandle {
    NEXT_ID.with(|cell| {
        let id = *cell.borrow();
        *cell.borrow_mut() = id.wrapping_add(1);
        TENSOR_STORE.with(|store| {
            store.borrow_mut().insert(id, resource);
        });
        id
    })
}

fn get_tensor(handle: TensorHandle) -> Result<BufResource, String> {
    TENSOR_STORE.with(|store| {
        store
            .borrow()
            .get(&handle)
            .cloned()
            .ok_or_else(|| format!("invalid tensor handle: {}", handle))
    })
}

fn remove_tensor(handle: TensorHandle) -> Result<BufResource, String> {
    TENSOR_STORE.with(|store| {
        store
            .borrow_mut()
            .remove(&handle)
            .ok_or_else(|| format!("invalid tensor handle: {}", handle))
    })
}

// ---------------------------------------------------------------------------
// DType helpers
// ---------------------------------------------------------------------------

/// C-facing dtype enum.
#[repr(C)]
pub enum CDType {
    F32 = 0,
    F64 = 1,
    S32 = 2,
    S64 = 3,
    U32 = 4,
    U8 = 5,
}

fn dtype_from_c(cdt: CDType) -> DType {
    match cdt {
        CDType::F32 => DType::F32,
        CDType::F64 => DType::F64,
        CDType::S32 => DType::S32,
        CDType::S64 => DType::S64,
        CDType::U32 => DType::U32,
        CDType::U8 => DType::U8,
    }
}

fn dtype_to_c(dt: DType) -> CDType {
    match dt {
        DType::F32 => CDType::F32,
        DType::F64 => CDType::F64,
        DType::S32 => CDType::S32,
        DType::S64 => CDType::S64,
        DType::U32 => CDType::U32,
        DType::U8 => CDType::U8,
    }
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

/// Copy the last error message into `buf` (up to `len` bytes, including NUL).
/// Returns the number of bytes written (excluding NUL terminator).
#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_last_error(buf: *mut c_char, len: usize) -> usize {
    if buf.is_null() || len == 0 {
        return 0;
    }
    let msg = LAST_ERROR.with(|e| e.borrow().clone());
    if msg.is_empty() {
        return 0;
    }
    let bytes = msg.as_bytes();
    let to_copy = (len - 1).min(bytes.len());
    let out = slice::from_raw_parts_mut(buf as *mut u8, to_copy);
    out.copy_from_slice(&bytes[..to_copy]);
    *buf.add(to_copy) = 0; // NUL terminator
    to_copy
}

// ---------------------------------------------------------------------------
// Tensor lifecycle
// ---------------------------------------------------------------------------

/// Create a tensor from raw data.
///
/// `data` must point to `shape[0] * shape[1] * ... * shape[ndim-1] * dtype_size` bytes.
/// Returns a handle, or 0 on error.
#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_new_tensor(
    data: *const u8,
    shape: *const usize,
    ndim: usize,
    dtype: CDType,
) -> TensorHandle {
    if data.is_null() || shape.is_null() {
        set_error("null pointer passed to ex_cubecl_new_tensor");
        return 0;
    }
    let shape_vec = slice::from_raw_parts(shape, ndim).to_vec();
    let dt = dtype_from_c(dtype);
    let n: usize = shape_vec.iter().product();
    let nbytes = n * dt.size_in_bytes();
    let data_vec = slice::from_raw_parts(data, nbytes).to_vec();
    alloc_handle(BufResource {
        data: data_vec,
        shape: shape_vec,
        dtype: dt,
    })
}

/// Read tensor data into `out_data`. `out_len` is the capacity in bytes.
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_read_tensor(
    handle: TensorHandle,
    out_data: *mut u8,
    out_len: usize,
) -> i32 {
    if out_data.is_null() {
        set_error("null pointer passed to ex_cubecl_read_tensor");
        return -1;
    }
    match get_tensor(handle) {
        Ok(res) => {
            if out_len < res.data.len() {
                set_error("output buffer too small");
                return -1;
            }
            let out = slice::from_raw_parts_mut(out_data, res.data.len());
            out.copy_from_slice(&res.data);
            0
        }
        Err(e) => {
            set_error(e);
            -1
        }
    }
}

/// Deallocate a tensor by handle. Returns 0 on success, -1 on error.
#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_deallocate_tensor(handle: TensorHandle) -> i32 {
    match remove_tensor(handle) {
        Ok(_) => 0,
        Err(e) => {
            set_error(e);
            -1
        }
    }
}

/// Get the shape of a tensor.
/// `out_shape` must have room for `out_ndim` entries.
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_tensor_shape(
    handle: TensorHandle,
    out_shape: *mut usize,
    out_ndim: usize,
) -> i32 {
    if out_shape.is_null() {
        set_error("null pointer passed to ex_cubecl_tensor_shape");
        return -1;
    }
    match get_tensor(handle) {
        Ok(res) => {
            if out_ndim < res.shape.len() {
                set_error("output buffer too small for shape");
                return -1;
            }
            let out = slice::from_raw_parts_mut(out_shape, res.shape.len());
            out.copy_from_slice(&res.shape);
            0
        }
        Err(e) => {
            set_error(e);
            -1
        }
    }
}

/// Get the dtype of a tensor. Returns the CDType value, or -1 on error.
#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_tensor_dtype(handle: TensorHandle, out_dtype: *mut i32) -> i32 {
    if out_dtype.is_null() {
        set_error("null pointer passed to ex_cubecl_tensor_dtype");
        return -1;
    }
    match get_tensor(handle) {
        Ok(res) => {
            *out_dtype = dtype_to_c(res.dtype) as i32;
            0
        }
        Err(e) => {
            set_error(e);
            -1
        }
    }
}

/// Get the size of tensor data in bytes.
#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_tensor_nbytes(handle: TensorHandle) -> usize {
    match get_tensor(handle) {
        Ok(res) => res.data.len(),
        Err(e) => {
            set_error(e);
            0
        }
    }
}

// ---------------------------------------------------------------------------
// Binary ops
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_add(a: TensorHandle, b: TensorHandle) -> TensorHandle {
    match (get_tensor(a), get_tensor(b)) {
        (Ok(a), Ok(b)) => alloc_handle(BufResource {
            data: binary_op(&a.data, a.dtype, &b.data, b.dtype, |x, y| x + y),
            shape: a.shape,
            dtype: a.dtype,
        }),
        (Err(e), _) | (_, Err(e)) => {
            set_error(e);
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_subtract(a: TensorHandle, b: TensorHandle) -> TensorHandle {
    match (get_tensor(a), get_tensor(b)) {
        (Ok(a), Ok(b)) => alloc_handle(BufResource {
            data: binary_op(&a.data, a.dtype, &b.data, b.dtype, |x, y| x - y),
            shape: a.shape,
            dtype: a.dtype,
        }),
        (Err(e), _) | (_, Err(e)) => {
            set_error(e);
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_multiply(a: TensorHandle, b: TensorHandle) -> TensorHandle {
    match (get_tensor(a), get_tensor(b)) {
        (Ok(a), Ok(b)) => alloc_handle(BufResource {
            data: binary_op(&a.data, a.dtype, &b.data, b.dtype, |x, y| x * y),
            shape: a.shape,
            dtype: a.dtype,
        }),
        (Err(e), _) | (_, Err(e)) => {
            set_error(e);
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_divide(a: TensorHandle, b: TensorHandle) -> TensorHandle {
    match (get_tensor(a), get_tensor(b)) {
        (Ok(a), Ok(b)) => alloc_handle(BufResource {
            data: binary_op(&a.data, a.dtype, &b.data, b.dtype, |x, y| {
                if y == 0.0 {
                    f64::NAN
                } else {
                    x / y
                }
            }),
            shape: a.shape,
            dtype: a.dtype,
        }),
        (Err(e), _) | (_, Err(e)) => {
            set_error(e);
            0
        }
    }
}

// ---------------------------------------------------------------------------
// Unary ops
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_negate(a: TensorHandle) -> TensorHandle {
    match get_tensor(a) {
        Ok(a) => alloc_handle(BufResource {
            data: unary_op(&a.data, a.dtype, |x| -x),
            shape: a.shape,
            dtype: a.dtype,
        }),
        Err(e) => {
            set_error(e);
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_abs(a: TensorHandle) -> TensorHandle {
    match get_tensor(a) {
        Ok(a) => alloc_handle(BufResource {
            data: unary_op(&a.data, a.dtype, |x| x.abs()),
            shape: a.shape,
            dtype: a.dtype,
        }),
        Err(e) => {
            set_error(e);
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_exp(a: TensorHandle) -> TensorHandle {
    match get_tensor(a) {
        Ok(a) => alloc_handle(BufResource {
            data: unary_op(&a.data, a.dtype, |x| x.exp()),
            shape: a.shape,
            dtype: a.dtype,
        }),
        Err(e) => {
            set_error(e);
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_log(a: TensorHandle) -> TensorHandle {
    match get_tensor(a) {
        Ok(a) => alloc_handle(BufResource {
            data: unary_op(&a.data, a.dtype, |x| x.ln()),
            shape: a.shape,
            dtype: a.dtype,
        }),
        Err(e) => {
            set_error(e);
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_sqrt(a: TensorHandle) -> TensorHandle {
    match get_tensor(a) {
        Ok(a) => alloc_handle(BufResource {
            data: unary_op(&a.data, a.dtype, |x| x.sqrt()),
            shape: a.shape,
            dtype: a.dtype,
        }),
        Err(e) => {
            set_error(e);
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_sigmoid(a: TensorHandle) -> TensorHandle {
    match get_tensor(a) {
        Ok(a) => alloc_handle(BufResource {
            data: unary_op(&a.data, a.dtype, |x| 1.0 / (1.0 + (-x).exp())),
            shape: a.shape,
            dtype: a.dtype,
        }),
        Err(e) => {
            set_error(e);
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_relu(a: TensorHandle) -> TensorHandle {
    match get_tensor(a) {
        Ok(a) => alloc_handle(BufResource {
            data: unary_op(&a.data, a.dtype, |x| if x > 0.0 { x } else { 0.0 }),
            shape: a.shape,
            dtype: a.dtype,
        }),
        Err(e) => {
            set_error(e);
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_sin(a: TensorHandle) -> TensorHandle {
    match get_tensor(a) {
        Ok(a) => alloc_handle(BufResource {
            data: unary_op(&a.data, a.dtype, |x| x.sin()),
            shape: a.shape,
            dtype: a.dtype,
        }),
        Err(e) => {
            set_error(e);
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_cos(a: TensorHandle) -> TensorHandle {
    match get_tensor(a) {
        Ok(a) => alloc_handle(BufResource {
            data: unary_op(&a.data, a.dtype, |x| x.cos()),
            shape: a.shape,
            dtype: a.dtype,
        }),
        Err(e) => {
            set_error(e);
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_tanh(a: TensorHandle) -> TensorHandle {
    match get_tensor(a) {
        Ok(a) => alloc_handle(BufResource {
            data: unary_op(&a.data, a.dtype, |x| x.tanh()),
            shape: a.shape,
            dtype: a.dtype,
        }),
        Err(e) => {
            set_error(e);
            0
        }
    }
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

fn reduce_along_axes(
    data: &[u8],
    dtype: DType,
    shape: &[usize],
    axes: &[usize],
    keep_dims: bool,
    init: f64,
    op: impl Fn(f64, f64) -> f64,
) -> (Vec<u8>, Vec<usize>) {
    if shape.is_empty() {
        return (data.to_vec(), vec![]);
    }
    let v = to_f64(data, dtype);
    let total_in: usize = shape.iter().product();
    if total_in == 0 {
        return (from_f64(vec![init], dtype), vec![1]);
    }
    let in_strides = strides_for(shape);
    let mut out_shape = Vec::new();
    for (i, &s) in shape.iter().enumerate() {
        if !axes.contains(&i) {
            out_shape.push(s);
        } else if keep_dims {
            out_shape.push(1);
        }
    }
    if out_shape.is_empty() {
        out_shape = vec![1];
    }
    let out_total: usize = out_shape.iter().product();
    let out_strides = strides_for(&out_shape);
    let mut result = vec![init; out_total];
    for idx in 0..total_in {
        let mut remainder = idx;
        let mut out_idx = 0usize;
        for d in 0..shape.len() {
            let coord = remainder / in_strides[d];
            remainder %= in_strides[d];
            if !axes.contains(&d) {
                let out_d = if keep_dims {
                    d
                } else {
                    shape[..d]
                        .iter()
                        .enumerate()
                        .filter(|&(dd, _)| !axes.contains(&dd))
                        .count()
                };
                if out_d < out_strides.len() {
                    out_idx += coord * out_strides[out_d];
                }
            }
        }
        if out_idx < result.len() {
            result[out_idx] = op(result[out_idx], v[idx]);
        }
    }
    (from_f64(result, dtype), out_shape)
}

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_sum(
    handle: TensorHandle,
    axes: *const usize,
    naxes: usize,
    keep_dims: bool,
) -> TensorHandle {
    match get_tensor(handle) {
        Ok(res) => {
            let axes_vec = if axes.is_null() || naxes == 0 {
                vec![]
            } else {
                slice::from_raw_parts(axes, naxes).to_vec()
            };
            let (data, shape) = reduce_along_axes(
                &res.data,
                res.dtype,
                &res.shape,
                &axes_vec,
                keep_dims,
                0.0,
                |a, b| a + b,
            );
            alloc_handle(BufResource {
                data,
                shape,
                dtype: res.dtype,
            })
        }
        Err(e) => {
            set_error(e);
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_reduce_max(
    handle: TensorHandle,
    axes: *const usize,
    naxes: usize,
    keep_dims: bool,
) -> TensorHandle {
    match get_tensor(handle) {
        Ok(res) => {
            let axes_vec = if axes.is_null() || naxes == 0 {
                vec![]
            } else {
                slice::from_raw_parts(axes, naxes).to_vec()
            };
            let (data, shape) = reduce_along_axes(
                &res.data,
                res.dtype,
                &res.shape,
                &axes_vec,
                keep_dims,
                f64::MIN,
                |a, b| a.max(b),
            );
            alloc_handle(BufResource {
                data,
                shape,
                dtype: res.dtype,
            })
        }
        Err(e) => {
            set_error(e);
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_reduce_min(
    handle: TensorHandle,
    axes: *const usize,
    naxes: usize,
    keep_dims: bool,
) -> TensorHandle {
    match get_tensor(handle) {
        Ok(res) => {
            let axes_vec = if axes.is_null() || naxes == 0 {
                vec![]
            } else {
                slice::from_raw_parts(axes, naxes).to_vec()
            };
            let (data, shape) = reduce_along_axes(
                &res.data,
                res.dtype,
                &res.shape,
                &axes_vec,
                keep_dims,
                f64::MAX,
                |a, b| a.min(b),
            );
            alloc_handle(BufResource {
                data,
                shape,
                dtype: res.dtype,
            })
        }
        Err(e) => {
            set_error(e);
            0
        }
    }
}

// ---------------------------------------------------------------------------
// Shape ops
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_reshape(
    handle: TensorHandle,
    new_shape: *const usize,
    ndim: usize,
) -> TensorHandle {
    match get_tensor(handle) {
        Ok(res) => {
            let ns = slice::from_raw_parts(new_shape, ndim).to_vec();
            let old_n: usize = res.shape.iter().product();
            let new_n: usize = ns.iter().product();
            if old_n != new_n {
                set_error(format!("reshape: {} vs {} elements", old_n, new_n));
                return 0;
            }
            alloc_handle(BufResource {
                data: res.data,
                shape: ns,
                dtype: res.dtype,
            })
        }
        Err(e) => {
            set_error(e);
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_transpose(
    handle: TensorHandle,
    axes: *const usize,
    ndim: usize,
) -> TensorHandle {
    match get_tensor(handle) {
        Ok(res) => {
            let rank = res.shape.len();
            let axes_vec = if axes.is_null() || ndim == 0 {
                // Default: reverse all axes
                (0..rank).rev().collect::<Vec<_>>()
            } else {
                slice::from_raw_parts(axes, ndim).to_vec()
            };
            if axes_vec.len() != rank {
                set_error("transpose: axes length mismatch");
                return 0;
            }
            let new_shape: Vec<usize> = axes_vec.iter().map(|&a| res.shape[a]).collect();
            let total: usize = res.shape.iter().product();
            let in_strides = strides_for(&res.shape);
            let out_strides = strides_for(&new_shape);
            let in_f64 = to_f64(&res.data, res.dtype);
            let mut out_vals = vec![0.0f64; total];
            for i in 0..total {
                let mut in_coords = vec![0usize; rank];
                let mut rem = i;
                for d in 0..rank {
                    in_coords[axes_vec[d]] = rem / out_strides[d];
                    rem %= out_strides[d];
                }
                let flat_in: usize = in_coords
                    .iter()
                    .zip(in_strides.iter())
                    .map(|(c, s)| c * s)
                    .sum();
                out_vals[i] = in_f64[flat_in];
            }
            alloc_handle(BufResource {
                data: from_f64(out_vals, res.dtype),
                shape: new_shape,
                dtype: res.dtype,
            })
        }
        Err(e) => {
            set_error(e);
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_broadcast(
    handle: TensorHandle,
    target_shape: *const usize,
    ndim: usize,
    axes: *const usize,
    naxes: usize,
) -> TensorHandle {
    match get_tensor(handle) {
        Ok(res) => {
            let ts = slice::from_raw_parts(target_shape, ndim).to_vec();
            let ax = slice::from_raw_parts(axes, naxes).to_vec();
            let in_shape = &res.shape;
            let in_rank = in_shape.len();
            let out_rank = ts.len();
            if ax.len() != in_rank {
                set_error("broadcast: axes length mismatch");
                return 0;
            }
            let total_out: usize = ts.iter().product();
            let out_strides = strides_for(&ts);
            let in_strides = strides_for(in_shape);
            let in_f64 = to_f64(&res.data, res.dtype);
            let mut out_vals = vec![0.0f64; total_out];
            for i in 0..total_out {
                let mut coords = vec![0usize; out_rank];
                let mut rem = i;
                for d in 0..out_rank {
                    coords[d] = rem / out_strides[d];
                    rem %= out_strides[d];
                }
                let mut in_coords = vec![0usize; in_rank];
                for (di, &ax) in ax.iter().enumerate() {
                    let ax = ax as usize;
                    in_coords[di] = if ax < out_rank && in_shape[di] == ts[ax] {
                        coords[ax]
                    } else {
                        0
                    };
                }
                let flat_in: usize = in_coords
                    .iter()
                    .zip(in_strides.iter())
                    .map(|(c, s)| c * s)
                    .sum();
                out_vals[i] = in_f64[flat_in];
            }
            alloc_handle(BufResource {
                data: from_f64(out_vals, res.dtype),
                shape: ts,
                dtype: res.dtype,
            })
        }
        Err(e) => {
            set_error(e);
            0
        }
    }
}

// ---------------------------------------------------------------------------
// Dot / Matmul
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_dot(a: TensorHandle, b: TensorHandle) -> TensorHandle {
    match (get_tensor(a), get_tensor(b)) {
        (Ok(a), Ok(b)) => {
            let a_s = &a.shape;
            let b_s = &b.shape;
            if a_s.len() < 2 || b_s.len() < 2 {
                set_error("dot: need 2D");
                return 0;
            }
            let m = a_s[a_s.len() - 2];
            let k = a_s[a_s.len() - 1];
            let k2 = b_s[b_s.len() - 2];
            let n = b_s[b_s.len() - 1];
            if k != k2 {
                set_error(format!("dot: inner mismatch {} vs {}", k, k2));
                return 0;
            }
            let total_out = m * n;
            let out_dtype = a.dtype;
            let a_f = to_f64(&a.data, a.dtype);
            let b_f = to_f64(&b.data, b.dtype);
            let mut out_vals = vec![0.0f64; total_out];
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for l in 0..k {
                        sum += a_f[i * k + l] * b_f[l * n + j];
                    }
                    out_vals[i * n + j] = sum;
                }
            }
            alloc_handle(BufResource {
                data: from_f64(out_vals, out_dtype),
                shape: vec![m, n],
                dtype: out_dtype,
            })
        }
        (Err(e), _) | (_, Err(e)) => {
            set_error(e);
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_matmul(a: TensorHandle, b: TensorHandle) -> TensorHandle {
    // Alias for dot
    ex_cubecl_dot(a, b)
}

// ---------------------------------------------------------------------------
// Concatenate
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_concatenate(
    handles: *const TensorHandle,
    n: usize,
    axis: usize,
) -> TensorHandle {
    if handles.is_null() || n == 0 {
        set_error("concatenate: empty handles");
        return 0;
    }
    let handles_slice = slice::from_raw_parts(handles, n);
    let mut tensors = Vec::with_capacity(n);
    for &h in handles_slice {
        match get_tensor(h) {
            Ok(t) => tensors.push(t),
            Err(e) => {
                set_error(e);
                return 0;
            }
        }
    }
    let dtype = tensors[0].dtype;
    let rank = tensors[0].shape.len();
    for b in &tensors {
        if b.shape.len() != rank {
            set_error("concat: rank mismatch");
            return 0;
        }
    }
    let out_axis: usize = tensors.iter().map(|b| b.shape[axis]).sum();
    let mut out_shape = tensors[0].shape.clone();
    out_shape[axis] = out_axis;
    let total_out: usize = out_shape.iter().product();
    let out_strides = strides_for(&out_shape);
    let mut out_vals = vec![0.0f64; total_out];
    let mut offset = 0usize;
    for buf in &tensors {
        let in_strides = strides_for(&buf.shape);
        let in_f64 = to_f64(&buf.data, buf.dtype);
        let total_in: usize = buf.shape.iter().product();
        for i in 0..total_in {
            let mut in_coords = vec![0usize; rank];
            let mut rem = i;
            for d in 0..rank {
                in_coords[d] = rem / in_strides[d];
                rem %= in_strides[d];
            }
            let mut out_coords = in_coords.clone();
            out_coords[axis] += offset;
            let flat_out: usize = out_coords
                .iter()
                .zip(out_strides.iter())
                .map(|(c, s)| c * s)
                .sum();
            out_vals[flat_out] = in_f64[i];
        }
        offset += buf.shape[axis];
    }
    alloc_handle(BufResource {
        data: from_f64(out_vals, dtype),
        shape: out_shape,
        dtype,
    })
}

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_slice(
    handle: TensorHandle,
    starts: *const usize,
    lengths: *const usize,
    strides: *const usize,
    ndim: usize,
) -> TensorHandle {
    match get_tensor(handle) {
        Ok(res) => {
            let rank = res.shape.len();
            let s_starts = slice::from_raw_parts(starts, ndim);
            let s_lengths = slice::from_raw_parts(lengths, ndim);
            let s_strides = slice::from_raw_parts(strides, ndim);
            if s_starts.len() != rank || s_lengths.len() != rank || s_strides.len() != rank {
                set_error("slice: parameter length mismatch");
                return 0;
            }
            let out_shape = s_lengths.to_vec();
            let total_out: usize = out_shape.iter().product();
            let out_strides = strides_for(&out_shape);
            let in_strides = strides_for(&res.shape);
            let in_f64 = to_f64(&res.data, res.dtype);
            let mut out_vals = vec![0.0f64; total_out];
            for i in 0..total_out {
                let mut out_coords = vec![0usize; rank];
                let mut rem = i;
                for d in 0..rank {
                    out_coords[d] = rem / out_strides[d];
                    rem %= out_strides[d];
                }
                let mut in_coords = vec![0usize; rank];
                for d in 0..rank {
                    in_coords[d] = s_starts[d] + out_coords[d] * s_strides[d];
                }
                let flat_in: usize = in_coords
                    .iter()
                    .zip(in_strides.iter())
                    .map(|(c, s)| c * s)
                    .sum();
                out_vals[i] = in_f64[flat_in];
            }
            alloc_handle(BufResource {
                data: from_f64(out_vals, res.dtype),
                shape: out_shape,
                dtype: res.dtype,
            })
        }
        Err(e) => {
            set_error(e);
            0
        }
    }
}

// ---------------------------------------------------------------------------
// Pad
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_pad(
    handle: TensorHandle,
    pad_value: f64,
    padding_config: *const i64,
    nconfig: usize,
) -> TensorHandle {
    match get_tensor(handle) {
        Ok(res) => {
            let rank = res.shape.len();
            if nconfig != rank * 3 {
                set_error("pad: config length must be rank * 3 (lo, hi, interior per dim)");
                return 0;
            }
            let cfg_raw = slice::from_raw_parts(padding_config, nconfig);
            let mut cfg = Vec::with_capacity(rank);
            for d in 0..rank {
                cfg.push((cfg_raw[d * 3], cfg_raw[d * 3 + 1], cfg_raw[d * 3 + 2]));
            }
            let new_shape: Vec<usize> = res
                .shape
                .iter()
                .zip(cfg.iter())
                .map(|(&s, &(lo, hi, interior))| {
                    let lo = if lo < 0 { 0 } else { lo as usize };
                    let hi = if hi < 0 { 0 } else { hi as usize };
                    s + lo
                        + hi
                        + s.saturating_sub(1) * (if interior < 0 { 0 } else { interior as usize })
                })
                .collect();
            let total_out: usize = new_shape.iter().product();
            let out_strides = strides_for(&new_shape);
            let in_strides = strides_for(&res.shape);
            let in_f64 = to_f64(&res.data, res.dtype);
            let mut out_vals = vec![pad_value; total_out];
            let total_in: usize = res.shape.iter().product();
            for i in 0..total_in {
                let mut in_coords = vec![0usize; rank];
                let mut rem = i;
                for d in 0..rank {
                    in_coords[d] = rem / in_strides[d];
                    rem %= in_strides[d];
                }
                let mut out_coords = vec![0usize; rank];
                let mut acc = 0usize;
                for d in 0..rank {
                    let lo = if cfg[d].0 < 0 { 0 } else { cfg[d].0 as usize };
                    out_coords[d] = in_coords[d] + lo + acc;
                    acc += in_coords[d] * (if cfg[d].2 < 0 { 0 } else { cfg[d].2 as usize });
                }
                let flat_out: usize = out_coords
                    .iter()
                    .zip(out_strides.iter())
                    .map(|(c, s)| c * s)
                    .sum();
                if flat_out < total_out {
                    out_vals[flat_out] = in_f64[i];
                }
            }
            alloc_handle(BufResource {
                data: from_f64(out_vals, res.dtype),
                shape: new_shape,
                dtype: res.dtype,
            })
        }
        Err(e) => {
            set_error(e);
            0
        }
    }
}

// ---------------------------------------------------------------------------
// Reverse
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_reverse(
    handle: TensorHandle,
    axes: *const usize,
    naxes: usize,
) -> TensorHandle {
    match get_tensor(handle) {
        Ok(res) => {
            let rank = res.shape.len();
            let axes_vec = slice::from_raw_parts(axes, naxes).to_vec();
            let total: usize = res.shape.iter().product();
            let in_strides = strides_for(&res.shape);
            let in_f64 = to_f64(&res.data, res.dtype);
            let mut out_vals = vec![0.0f64; total];
            for i in 0..total {
                let mut coords = vec![0usize; rank];
                let mut rem = i;
                for d in 0..rank {
                    coords[d] = rem / in_strides[d];
                    rem %= in_strides[d];
                }
                let mut in_coords = coords.clone();
                for &ax in &axes_vec {
                    let ax = ax as usize;
                    if ax < rank {
                        in_coords[ax] = res.shape[ax] - 1 - coords[ax];
                    }
                }
                let flat_in: usize = in_coords
                    .iter()
                    .zip(in_strides.iter())
                    .map(|(c, s)| c * s)
                    .sum();
                out_vals[i] = in_f64[flat_in];
            }
            alloc_handle(BufResource {
                data: from_f64(out_vals, res.dtype),
                shape: res.shape,
                dtype: res.dtype,
            })
        }
        Err(e) => {
            set_error(e);
            0
        }
    }
}

// ---------------------------------------------------------------------------
// Type conversion
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_as_type(handle: TensorHandle, dtype: CDType) -> TensorHandle {
    match get_tensor(handle) {
        Ok(res) => {
            let new_dtype = dtype_from_c(dtype);
            if res.dtype == new_dtype {
                alloc_handle(BufResource {
                    data: res.data,
                    shape: res.shape,
                    dtype: res.dtype,
                })
            } else {
                let vals = to_f64(&res.data, res.dtype);
                alloc_handle(BufResource {
                    data: from_f64(vals, new_dtype),
                    shape: res.shape,
                    dtype: new_dtype,
                })
            }
        }
        Err(e) => {
            set_error(e);
            0
        }
    }
}

// ---------------------------------------------------------------------------
// Constant / Eye / Iota
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_constant(
    shape: *const usize,
    ndim: usize,
    dtype: CDType,
    value: f64,
) -> TensorHandle {
    let shape_vec = slice::from_raw_parts(shape, ndim).to_vec();
    let dt = dtype_from_c(dtype);
    let n: usize = shape_vec.iter().product();
    alloc_handle(BufResource {
        data: from_f64(vec![value; n], dt),
        shape: shape_vec,
        dtype: dt,
    })
}

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_eye(
    shape: *const usize,
    ndim: usize,
    dtype: CDType,
) -> TensorHandle {
    let shape_vec = slice::from_raw_parts(shape, ndim).to_vec();
    if shape_vec.len() < 2 {
        set_error("eye: need >= 2D");
        return 0;
    }
    let dt = dtype_from_c(dtype);
    let rows = shape_vec[shape_vec.len() - 2];
    let cols = shape_vec[shape_vec.len() - 1];
    let batch: usize = shape_vec[..shape_vec.len() - 2].iter().product();
    let n = batch * rows * cols;
    let mut vals = vec![0.0f64; n];
    for b in 0..batch {
        for i in 0..rows {
            for j in 0..cols {
                let idx = b * rows * cols + i * cols + j;
                vals[idx] = if i == j { 1.0 } else { 0.0 };
            }
        }
    }
    alloc_handle(BufResource {
        data: from_f64(vals, dt),
        shape: shape_vec,
        dtype: dt,
    })
}

#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_iota(
    shape: *const usize,
    ndim: usize,
    dtype: CDType,
    axis: usize,
) -> TensorHandle {
    let shape_vec = slice::from_raw_parts(shape, ndim).to_vec();
    let dt = dtype_from_c(dtype);
    let n: usize = shape_vec.iter().product();
    let strides = strides_for(&shape_vec);
    let mut vals = vec![0.0f64; n];
    for i in 0..n {
        let mut coords = vec![0usize; shape_vec.len()];
        let mut rem = i;
        for d in 0..shape_vec.len() {
            coords[d] = rem / strides[d];
            rem %= strides[d];
        }
        vals[i] = coords[axis] as f64;
    }
    alloc_handle(BufResource {
        data: from_f64(vals, dt),
        shape: shape_vec,
        dtype: dt,
    })
}
