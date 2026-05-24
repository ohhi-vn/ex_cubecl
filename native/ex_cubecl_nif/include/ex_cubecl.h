/**
 * @file ex_cubecl.h
 * @brief C FFI for ex_cubecl GPU compute runtime.
 *
 * This header provides a C-compatible interface to the ex_cubecl GPU compute
 * library. It is designed for use on iOS (via Objective-C/Swift bridging) and
 * Android (via JNI).
 *
 * All functions use opaque handles to reference buffers, pipelines, commands,
 * media sources, and encoders. A handle of 0 indicates an error. Call
 * ex_cubecl_last_error() to retrieve error details.
 *
 * Thread safety: The internal context is thread-local. Handles are only valid
 * on the thread that created them.
 *
 * Phase 2 extensions: GPU texture upload from YUV planes, kernel application
 * to textures, and GPU audio mix.
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

/** Opaque handle to a GPU texture (video frame). 0 = invalid/null. */
typedef size_t ex_cubecl_texture_handle_t;

/** Opaque handle to a media source. 0 = invalid/null. */
typedef size_t ex_cubecl_media_handle_t;

/** Opaque handle to a transcoder/encoder. 0 = invalid/null. */
typedef size_t ex_cubecl_encoder_handle_t;

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

/** Pixel format for video textures. */
typedef enum {
    EX_CUBECL_FMT_YUV420P = 0,
    EX_CUBECL_FMT_NV12    = 1,
    EX_CUBECL_FMT_RGB24   = 2,
    EX_CUBECL_FMT_RGBA    = 3,
} ex_cubecl_pixel_format_t;

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

/* ---------------------------------------------------------------------------
 * Phase 2 — GPU Texture (Video Frame)
 * --------------------------------------------------------------------------- */

/**
 * Create a GPU texture from YUV420p plane data.
 *
 * Uploads Y and UV planes to a GPU texture suitable for video processing
 * kernels. The texture can be used as input to filter kernels.
 *
 * @param y_plane   Pointer to Y plane data (width * height bytes).
 * @param uv_plane  Pointer to UV plane data (width * height / 2 bytes).
 * @param width     Texture width in pixels.
 * @param height    Texture height in pixels.
 * @return          Texture handle, or 0 on error.
 */
ex_cubecl_texture_handle_t ex_cubecl_texture_from_yuv(
    const uint8_t *y_plane,
    const uint8_t *uv_plane,
    uint32_t width,
    uint32_t height);

/**
 * Create a GPU texture from NV12 plane data.
 *
 * @param y_plane   Pointer to Y plane data.
 * @param uv_plane  Pointer to interleaved UV plane data.
 * @param width     Texture width in pixels.
 * @param height    Texture height in pixels.
 * @return          Texture handle, or 0 on error.
 */
ex_cubecl_texture_handle_t ex_cubecl_texture_from_nv12(
    const uint8_t *y_plane,
    const uint8_t *uv_plane,
    uint32_t width,
    uint32_t height);

/**
 * Apply a named GPU kernel to a texture (filter chain).
 *
 * This is the primary entry point for video filter operations on mobile.
 * The kernel is applied in-place or to the output texture.
 *
 * @param input         Input texture handle.
 * @param kernel_name   Kernel name (e.g. "gaussian_blur", "lut_apply").
 * @param params        Array of kernel parameters.
 * @param param_count   Number of parameters.
 * @return              Output texture handle, or 0 on error.
 */
ex_cubecl_texture_handle_t ex_cubecl_apply_kernel(
    ex_cubecl_texture_handle_t input,
    const char *kernel_name,
    const void *params,
    size_t param_count);

/**
 * Free a GPU texture.
 *
 * @param texture Texture handle.
 * @return        0 on success, -1 on error.
 */
int ex_cubecl_texture_free(ex_cubecl_texture_handle_t texture);

/* ---------------------------------------------------------------------------
 * Phase 2 — Media Source
 * --------------------------------------------------------------------------- */

/**
 * Open a media source (file, stream, camera).
 *
 * @param path  File path or URL (rtmp://, http://, etc.).
 * @return      Media handle, or 0 on error.
 */
ex_cubecl_media_handle_t ex_cubecl_media_open(const char *path);

/**
 * Close a media source.
 *
 * @param media Media handle.
 * @return      0 on success, -1 on error.
 */
int ex_cubecl_media_close(ex_cubecl_media_handle_t media);

/* ---------------------------------------------------------------------------
 * Phase 2 — Audio Mix
 * --------------------------------------------------------------------------- */

/**
 * Mix multiple audio tracks on the GPU.
 *
 * Sums N audio tracks with per-channel gain into a single output buffer.
 * All tracks must have the same sample count.
 *
 * @param tracks       Array of input buffer handles.
 * @param gains        Array of gain values (one per track).
 * @param track_count  Number of tracks.
 * @param frames       Number of samples per channel.
 * @return             Output buffer handle, or 0 on error.
 */
ex_cubecl_buffer_handle_t ex_cubecl_audio_mix(
    const ex_cubecl_buffer_handle_t *tracks,
    const float *gains,
    size_t track_count,
    size_t frames);

#ifdef __cplusplus
}
#endif

#endif /* EX_CUBECL_H */
