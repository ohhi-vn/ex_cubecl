/**
 * @file ex_cubecl.h
 * @brief C FFI for ex_cubecl GPU compute runtime.
 *
 * This header provides a C-compatible interface to the ex_cubecl GPU compute
 * library. It is designed for use on iOS (via Objective-C/Swift bridging) and
 * Android (via JNI).
 *
 * All functions use opaque handles to reference buffers, pipelines, and commands.
 * A buffer handle of 0 indicates an error. Call ex_cubecl_last_error() to
 * retrieve error details.
 *
 * Thread safety: The internal context is thread-local. Handles are only valid
 * on the thread that created them.
 */

#ifndef EX_CUBECL_H
#define EX_CUBECL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Opaque handles
 * --------------------------------------------------------------------------- */

/** Opaque handle to a GPU buffer. 0 = invalid/null. */
typedef size_t ex_cubecl_buffer_handle_t;

/** Opaque handle to a command pipeline. 0 = invalid/null. */
typedef size_t ex_cubecl_pipeline_handle_t;

/* ---------------------------------------------------------------------------
 * Data types
 * --------------------------------------------------------------------------- */

/** Buffer element data types. */
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
 * Device management
 * --------------------------------------------------------------------------- */

/**
 * Get a human-readable device info string.
 *
 * @param buf   Buffer to write the NUL-terminated info string into.
 * @param len   Capacity of `buf` in bytes (including NUL terminator).
 * @return      Number of bytes written (excluding NUL), or 0 on error.
 */
size_t ex_cubecl_device_info(char *buf, size_t len);

/**
 * Get the number of available GPU devices.
 *
 * @return Number of GPU devices.
 */
int ex_cubecl_device_count(void);

/* ---------------------------------------------------------------------------
 * Buffer management
 * --------------------------------------------------------------------------- */

/**
 * Create a GPU buffer from raw data.
 *
 * @param data  Pointer to raw element data (row-major).
 * @param shape Array of dimension sizes.
 * @param ndim  Number of dimensions.
 * @param dtype Element data type.
 * @return      Buffer handle, or 0 on error.
 */
ex_cubecl_buffer_handle_t ex_cubecl_buffer_new(
    const uint8_t *data,
    const size_t *shape,
    size_t ndim,
    ex_cubecl_dtype_t dtype);

/**
 * Read buffer data into a caller-provided buffer.
 *
 * @param handle Buffer handle.
 * @param out    Output buffer (must be at least ex_cubecl_buffer_size() bytes).
 * @param len    Capacity of out in bytes.
 * @return       0 on success, -1 on error.
 */
int ex_cubecl_buffer_read(
    ex_cubecl_buffer_handle_t handle,
    uint8_t *out,
    size_t len);

/**
 * Get the size of buffer data in bytes.
 *
 * @param handle Buffer handle.
 * @return       Data size in bytes, or 0 on error.
 */
size_t ex_cubecl_buffer_size(ex_cubecl_buffer_handle_t handle);

/**
 * Get the shape of a buffer.
 *
 * @param handle    Buffer handle.
 * @param out_shape Output array (must hold at least out_ndim entries).
 * @param out_ndim  Capacity of out_shape.
 * @return          0 on success, -1 on error.
 */
int ex_cubecl_buffer_shape(
    ex_cubecl_buffer_handle_t handle,
    size_t *out_shape,
    size_t out_ndim);

/**
 * Get the data type of a buffer.
 *
 * @param handle    Buffer handle.
 * @param out_dtype Output dtype value.
 * @return          0 on success, -1 on error.
 */
int ex_cubecl_buffer_dtype(
    ex_cubecl_buffer_handle_t handle,
    int *out_dtype);

/**
 * Free a GPU buffer.
 *
 * @param handle Buffer handle.
 * @return       0 on success, -1 on error.
 */
int ex_cubecl_buffer_free(ex_cubecl_buffer_handle_t handle);

/* ---------------------------------------------------------------------------
 * Kernel execution
 * --------------------------------------------------------------------------- */

/**
 * Run a named kernel.
 *
 * @param name       Kernel name.
 * @param inputs     Array of input buffer handles.
 * @param n_inputs   Number of input buffers.
 * @param output     Output buffer handle.
 * @param params     Pointer to kernel parameters (can be NULL if params_len == 0).
 * @param params_len Size of params in bytes.
 * @return           0 on success, -1 on error.
 */
int ex_cubecl_kernel_run(
    const char *name,
    const ex_cubecl_buffer_handle_t *inputs,
    size_t n_inputs,
    ex_cubecl_buffer_handle_t output,
    const uint8_t *params,
    size_t params_len);

/**
 * List available kernels as a comma-separated string.
 *
 * @param buf      Buffer to write the NUL-terminated list into.
 * @param buf_len  Capacity of buf in bytes (including NUL terminator).
 * @return         Number of bytes written (excluding NUL), or 0 on error.
 */
size_t ex_cubecl_kernel_list(char *buf, size_t buf_len);

/* ---------------------------------------------------------------------------
 * Async execution
 * --------------------------------------------------------------------------- */

/**
 * Submit a raw command for asynchronous execution.
 *
 * @param command      Pointer to serialized command data.
 * @param command_len  Size of command data in bytes.
 * @return             Command ID (non-zero), or 0 on error.
 */
uint64_t ex_cubecl_submit(
    const uint8_t *command,
    size_t command_len);

/**
 * Poll the status of a submitted command.
 *
 * @param command_id Command ID returned by ex_cubecl_submit().
 * @return           0 = pending, 1 = complete, -1 on error.
 */
int ex_cubecl_poll(uint64_t command_id);

/**
 * Block until a submitted command completes.
 *
 * @param command_id Command ID returned by ex_cubecl_submit().
 * @return           0 on success, -1 on error.
 */
int ex_cubecl_wait(uint64_t command_id);

/* ---------------------------------------------------------------------------
 * Pipeline
 * --------------------------------------------------------------------------- */

/**
 * Create a new command pipeline.
 *
 * @return Pipeline handle, or 0 on error.
 */
ex_cubecl_pipeline_handle_t ex_cubecl_pipeline_new(void);

/**
 * Add a command to a pipeline.
 *
 * @param pipeline     Pipeline handle.
 * @param command      Pointer to serialized command data.
 * @param command_len  Size of command data in bytes.
 * @return             0 on success, -1 on error.
 */
int ex_cubecl_pipeline_add(
    ex_cubecl_pipeline_handle_t pipeline,
    const uint8_t *command,
    size_t command_len);

/**
 * Execute all commands in a pipeline.
 *
 * @param pipeline Pipeline handle.
 * @return         0 on success, -1 on error.
 */
int ex_cubecl_pipeline_run(ex_cubecl_pipeline_handle_t pipeline);

/**
 * Free a command pipeline.
 *
 * @param pipeline Pipeline handle.
 * @return         0 on success, -1 on error.
 */
int ex_cubecl_pipeline_free(ex_cubecl_pipeline_handle_t pipeline);

#ifdef __cplusplus
}
#endif

#endif /* EX_CUBECL_H */
