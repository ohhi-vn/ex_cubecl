/**
 * @file ex_cubecl.h
 * @brief C FFI for ex_cubecl tensor operations.
 *
 * This header provides a C-compatible interface to the ex_cubecl tensor library.
 * It is designed for use on iOS (via Objective-C/Swift bridging) and Android
 * (via JNI).
 *
 * All functions use opaque `usize` handles to reference tensors. A handle of 0
 * indicates an error. Call ex_cubecl_last_error() to retrieve error details.
 *
 * Thread safety: The internal tensor store is thread-local. Handles are only
 * valid on the thread that created them.
 */

#ifndef EX_CUBECL_H
#define EX_CUBECL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Opaque handle
 * --------------------------------------------------------------------------- */

/** Opaque handle to a tensor. 0 = invalid/null. */
typedef size_t ex_cubecl_tensor_handle_t;

/* ---------------------------------------------------------------------------
 * Data types
 * --------------------------------------------------------------------------- */

/** Tensor element data types. */
typedef enum {
    EX_CUBECL_DTYPE_F32 = 0,
    EX_CUBECL_DTYPE_F64 = 1,
    EX_CUBECL_DTYPE_S32 = 2,
    EX_CUBECL_DTYPE_S64 = 3,
    EX_CUBECL_DTYPE_U32 = 4,
    EX_CUBECL_DTYPE_U8  = 5,
} ex_cubecl_dtype_t;

/* ---------------------------------------------------------------------------
 * Error handling
 * --------------------------------------------------------------------------- */

/**
 * Retrieve the last error message.
 *
 * @param buf   Buffer to write the NUL-terminated error string into.
 * @param len   Capacity of `buf` in bytes (including NUL terminator).
 * @return      Number of bytes written (excluding NUL), or 0 if no error.
 */
size_t ex_cubecl_last_error(char *buf, size_t len);

/* ---------------------------------------------------------------------------
 * Tensor lifecycle
 * --------------------------------------------------------------------------- */

/**
 * Create a tensor from raw data.
 *
 * @param data  Pointer to raw element data (row-major).
 * @param shape Array of dimension sizes.
 * @param ndim  Number of dimensions.
 * @param dtype Element data type.
 * @return      Tensor handle, or 0 on error.
 */
ex_cubecl_tensor_handle_t ex_cubecl_new_tensor(
    const uint8_t *data,
    const size_t *shape,
    size_t ndim,
    ex_cubecl_dtype_t dtype);

/**
 * Read tensor data into a caller-provided buffer.
 *
 * @param handle   Tensor handle.
 * @param out_data Output buffer (must be at least ex_cubecl_tensor_nbytes() bytes).
 * @param out_len  Capacity of out_data in bytes.
 * @return         0 on success, -1 on error.
 */
int ex_cubecl_read_tensor(
    ex_cubecl_tensor_handle_t handle,
    uint8_t *out_data,
    size_t out_len);

/**
 * Deallocate a tensor.
 *
 * @param handle Tensor handle.
 * @return       0 on success, -1 on error.
 */
int ex_cubecl_deallocate_tensor(ex_cubecl_tensor_handle_t handle);

/**
 * Get the shape of a tensor.
 *
 * @param handle    Tensor handle.
 * @param out_shape Output array (must hold at least out_ndim entries).
 * @param out_ndim  Capacity of out_shape.
 * @return          0 on success, -1 on error.
 */
int ex_cubecl_tensor_shape(
    ex_cubecl_tensor_handle_t handle,
    size_t *out_shape,
    size_t out_ndim);

/**
 * Get the data type of a tensor.
 *
 * @param handle   Tensor handle.
 * @param out_dtype Output dtype value.
 * @return         0 on success, -1 on error.
 */
int ex_cubecl_tensor_dtype(
    ex_cubecl_tensor_handle_t handle,
    int *out_dtype);

/**
 * Get the size of tensor data in bytes.
 *
 * @param handle Tensor handle.
 * @return       Data size in bytes, or 0 on error.
 */
size_t ex_cubecl_tensor_nbytes(ex_cubecl_tensor_handle_t handle);

/* ---------------------------------------------------------------------------
 * Binary element-wise operations
 * --------------------------------------------------------------------------- */

/** Element-wise add: result = a + b */
ex_cubecl_tensor_handle_t ex_cubecl_add(
    ex_cubecl_tensor_handle_t a,
    ex_cubecl_tensor_handle_t b);

/** Element-wise subtract: result = a - b */
ex_cubecl_tensor_handle_t ex_cubecl_subtract(
    ex_cubecl_tensor_handle_t a,
    ex_cubecl_tensor_handle_t b);

/** Element-wise multiply: result = a * b */
ex_cubecl_tensor_handle_t ex_cubecl_multiply(
    ex_cubecl_tensor_handle_t a,
    ex_cubecl_tensor_handle_t b);

/** Element-wise divide: result = a / b (NaN if b == 0) */
ex_cubecl_tensor_handle_t ex_cubecl_divide(
    ex_cubecl_tensor_handle_t a,
    ex_cubecl_tensor_handle_t b);

/* ---------------------------------------------------------------------------
 * Unary element-wise operations
 * --------------------------------------------------------------------------- */

/** Element-wise negate: result = -a */
ex_cubecl_tensor_handle_t ex_cubecl_negate(ex_cubecl_tensor_handle_t a);

/** Element-wise absolute value: result = |a| */
ex_cubecl_tensor_handle_t ex_cubecl_abs(ex_cubecl_tensor_handle_t a);

/** Element-wise exponential: result = e^a */
ex_cubecl_tensor_handle_t ex_cubecl_exp(ex_cubecl_tensor_handle_t a);

/** Element-wise natural logarithm: result = ln(a) */
ex_cubecl_tensor_handle_t ex_cubecl_log(ex_cubecl_tensor_handle_t a);

/** Element-wise square root: result = sqrt(a) */
ex_cubecl_tensor_handle_t ex_cubecl_sqrt(ex_cubecl_tensor_handle_t a);

/** Element-wise sigmoid: result = 1 / (1 + e^(-a)) */
ex_cubecl_tensor_handle_t ex_cubecl_sigmoid(ex_cubecl_tensor_handle_t a);

/** Element-wise ReLU: result = max(0, a) */
ex_cubecl_tensor_handle_t ex_cubecl_relu(ex_cubecl_tensor_handle_t a);

/** Element-wise sine */
ex_cubecl_tensor_handle_t ex_cubecl_sin(ex_cubecl_tensor_handle_t a);

/** Element-wise cosine */
ex_cubecl_tensor_handle_t ex_cubecl_cos(ex_cubecl_tensor_handle_t a);

/** Element-wise hyperbolic tangent */
ex_cubecl_tensor_handle_t ex_cubecl_tanh(ex_cubecl_tensor_handle_t a);

/* ---------------------------------------------------------------------------
 * Reductions
 * --------------------------------------------------------------------------- */

/**
 * Sum reduction along axes.
 *
 * @param handle    Tensor handle.
 * @param axes      Array of axis indices to reduce.
 * @param naxes     Number of axes (0 = reduce all).
 * @param keep_dims If non-zero, reduced axes are kept as size 1.
 * @return          Result tensor handle, or 0 on error.
 */
ex_cubecl_tensor_handle_t ex_cubecl_sum(
    ex_cubecl_tensor_handle_t handle,
    const size_t *axes,
    size_t naxes,
    int keep_dims);

/**
 * Max reduction along axes. See ex_cubecl_sum for parameters.
 */
ex_cubecl_tensor_handle_t ex_cubecl_reduce_max(
    ex_cubecl_tensor_handle_t handle,
    const size_t *axes,
    size_t naxes,
    int keep_dims);

/**
 * Min reduction along axes. See ex_cubecl_sum for parameters.
 */
ex_cubecl_tensor_handle_t ex_cubecl_reduce_min(
    ex_cubecl_tensor_handle_t handle,
    const size_t *axes,
    size_t naxes,
    int keep_dims);

/* ---------------------------------------------------------------------------
 * Shape operations
 * --------------------------------------------------------------------------- */

/**
 * Reshape a tensor (total element count must match).
 *
 * @param handle   Tensor handle.
 * @param new_shape New dimension sizes.
 * @param ndim     Number of new dimensions.
 * @return         Result tensor handle, or 0 on error.
 */
ex_cubecl_tensor_handle_t ex_cubecl_reshape(
    ex_cubecl_tensor_handle_t handle,
    const size_t *new_shape,
    size_t ndim);

/**
 * Transpose a tensor.
 *
 * @param handle Tensor handle.
 * @param axes   Permutation of dimensions. If NULL or ndim==0, reverses all axes.
 * @param ndim   Length of axes array.
 * @return       Result tensor handle, or 0 on error.
 */
ex_cubecl_tensor_handle_t ex_cubecl_transpose(
    ex_cubecl_tensor_handle_t handle,
    const size_t *axes,
    size_t ndim);

/**
 * Broadcast a tensor to a target shape.
 *
 * @param handle       Tensor handle.
 * @param target_shape Desired output shape.
 * @param ndim         Length of target_shape.
 * @param axes         Mapping of input dims to output dims.
 * @param naxes        Length of axes (must match input rank).
 * @return             Result tensor handle, or 0 on error.
 */
ex_cubecl_tensor_handle_t ex_cubecl_broadcast(
    ex_cubecl_tensor_handle_t handle,
    const size_t *target_shape,
    size_t ndim,
    const size_t *axes,
    size_t naxes);

/* ---------------------------------------------------------------------------
 * Linear algebra
 * --------------------------------------------------------------------------- */

/**
 * Dot product (matrix multiply for 2D tensors).
 * For 2D tensors: result = a @ b where a is (m, k) and b is (k, n).
 *
 * @param a Tensor handle (m, k).
 * @param b Tensor handle (k, n).
 * @return  Result tensor handle (m, n), or 0 on error.
 */
ex_cubecl_tensor_handle_t ex_cubecl_dot(
    ex_cubecl_tensor_handle_t a,
    ex_cubecl_tensor_handle_t b);

/**
 * Matrix multiplication (alias for ex_cubecl_dot).
 */
ex_cubecl_tensor_handle_t ex_cubecl_matmul(
    ex_cubecl_tensor_handle_t a,
    ex_cubecl_tensor_handle_t b);

/* ---------------------------------------------------------------------------
 * Concatenate
 * --------------------------------------------------------------------------- */

/**
 * Concatenate tensors along an axis.
 *
 * @param handles Array of tensor handles.
 * @param n       Number of tensors.
 * @param axis    Axis along which to concatenate.
 * @return        Result tensor handle, or 0 on error.
 */
ex_cubecl_tensor_handle_t ex_cubecl_concatenate(
    const ex_cubecl_tensor_handle_t *handles,
    size_t n,
    size_t axis);

/* ---------------------------------------------------------------------------
 * Slice
 * --------------------------------------------------------------------------- */

/**
 * Extract a slice from a tensor.
 *
 * @param handle  Tensor handle.
 * @param starts  Start index per dimension.
 * @param lengths Output size per dimension.
 * @param strides Step per dimension.
 * @param ndim    Number of dimensions.
 * @return        Result tensor handle, or 0 on error.
 */
ex_cubecl_tensor_handle_t ex_cubecl_slice(
    ex_cubecl_tensor_handle_t handle,
    const size_t *starts,
    const size_t *lengths,
    const size_t *strides,
    size_t ndim);

/* ---------------------------------------------------------------------------
 * Pad
 * --------------------------------------------------------------------------- */

/**
 * Pad a tensor with a constant value.
 *
 * @param handle        Tensor handle.
 * @param pad_value     Value to pad with.
 * @param padding_config Array of (lo, hi, interior) per dimension, length = rank * 3.
 * @param nconfig       Length of padding_config (must be rank * 3).
 * @return              Result tensor handle, or 0 on error.
 */
ex_cubecl_tensor_handle_t ex_cubecl_pad(
    ex_cubecl_tensor_handle_t handle,
    double pad_value,
    const int64_t *padding_config,
    size_t nconfig);

/* ---------------------------------------------------------------------------
 * Reverse
 * --------------------------------------------------------------------------- */

/**
 * Reverse elements along specified axes.
 *
 * @param handle Tensor handle.
 * @param axes   Axes to reverse.
 * @param naxes  Number of axes.
 * @return       Result tensor handle, or 0 on error.
 */
ex_cubecl_tensor_handle_t ex_cubecl_reverse(
    ex_cubecl_tensor_handle_t handle,
    const size_t *axes,
    size_t naxes);

/* ---------------------------------------------------------------------------
 * Type conversion
 * --------------------------------------------------------------------------- */

/**
 * Cast tensor to a different data type.
 *
 * @param handle Tensor handle.
 * @param dtype  Target data type.
 * @return       Result tensor handle, or 0 on error.
 */
ex_cubecl_tensor_handle_t ex_cubecl_as_type(
    ex_cubecl_tensor_handle_t handle,
    ex_cubecl_dtype_t dtype);

/* ---------------------------------------------------------------------------
 * Constant / Eye / Iota
 * --------------------------------------------------------------------------- */

/**
 * Create a tensor filled with a constant value.
 *
 * @param shape Dimension sizes.
 * @param ndim  Number of dimensions.
 * @param dtype Data type.
 * @param value Fill value.
 * @return      Tensor handle, or 0 on error.
 */
ex_cubecl_tensor_handle_t ex_cubecl_constant(
    const size_t *shape,
    size_t ndim,
    ex_cubecl_dtype_t dtype,
    double value);

/**
 * Create an identity-like tensor (1 on diagonal, 0 elsewhere).
 * Requires ndim >= 2. The last two dimensions form the matrix.
 *
 * @param shape Dimension sizes.
 * @param ndim  Number of dimensions (>= 2).
 * @param dtype Data type.
 * @return      Tensor handle, or 0 on error.
 */
ex_cubecl_tensor_handle_t ex_cubecl_eye(
    const size_t *shape,
    size_t ndim,
    ex_cubecl_dtype_t dtype);

/**
 * Create a tensor with values equal to the index along `axis`.
 *
 * @param shape Dimension sizes.
 * @param ndim  Number of dimensions.
 * @param dtype Data type.
 * @param axis  Axis along which to generate indices.
 * @return      Tensor handle, or 0 on error.
 */
ex_cubecl_tensor_handle_t ex_cubecl_iota(
    const size_t *shape,
    size_t ndim,
    ex_cubecl_dtype_t dtype,
    size_t axis);

#ifdef __cplusplus
}
#endif

#endif /* EX_CUBECL_H */
