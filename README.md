# ExCubecl

[![Hex.pm](https://img.shields.io/hexpm/v/ex_cubecl.svg)](https://hex.pm/packages/ex_cubecl)

**ExCubecl** is a GPU compute runtime for Elixir, powered by [CubeCL](https://github.com/tracel-ai/cubecl) via Rust NIFs.

It provides GPU buffer management, kernel execution, async command submission, and pipeline orchestration — designed for AI inference, media processing, and realtime GPU effects on mobile and desktop.

## Architecture

```
┌─────────────────────────────────────────────┐
│              Elixir / BEAM                   │
│  ExCubecl.buffer(...)                       │
│  ExCubecl.run_kernel("elementwise_add", ...) │
│  ExCubecl.pipeline() |> pipeline_run()      │
├─────────────────────────────────────────────┤
│           ExCubecl.NIF (Elixir)              │
│  - NIF function stubs                        │
├─────────────────────────────────────────────┤
│           Rust NIF (lib.rs)                  │
│  - GPU device management                     │
│  - Buffer pool / Texture pool                │
│  - Kernel cache                              │
│  - Async command queue                       │
│  - Stream scheduler                          │
├─────────────────────────────────────────────┤
│           CubeCL Runtime                     │
│  - GPU kernel compilation                    │
│  - Buffer management                         │
│  - Dispatch execution                        │
│  - Synchronization                           │
├─────────────────────────────────────────────┤
│           C FFI (ex_cubecl.h)                │
│  - Mobile platform interface                 │
│  - iOS / Android interop                     │
└─────────────────────────────────────────────┘
```

## Installation

Add `ex_cubecl` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:ex_cubecl, "~> 0.4.1"}
  ]
end
```

## Quick Start

```elixir
# Check device
{:ok, info} = ExCubecl.device_info()
# %{device_name: "CubeCL GPU (Phase 1 — CPU simulation)", ...}
# Note: Currently runs on CPU; GPU dispatch coming in a future release

# Create GPU buffers (returns resource references, not integer IDs)
a = ExCubecl.buffer!([1.0, 2.0, 3.0], [3], :f32)
b = ExCubecl.buffer!([4.0, 5.0, 6.0], [3], :f32)

# Inspect
{:ok, [3]} = ExCubecl.shape(a)
{:ok, "f32"} = ExCubecl.dtype(a)
{:ok, 12} = ExCubecl.size(a)    # bytes

# Read data back
{:ok, data} = ExCubecl.read(a)

# Run a kernel
output = ExCubecl.buffer!([0.0, 0.0, 0.0], [3], :f32)
{:ok, _cmd} = ExCubecl.run_kernel("elementwise_add", [a, b], output)

# Async execution
{:ok, cmd_id} = ExCubecl.submit("some_command")
{:ok, :completed} = ExCubecl.poll(cmd_id)
:ok = ExCubecl.wait(cmd_id)

# Pipeline orchestration
{:ok, pipeline} = ExCubecl.pipeline()
:ok = ExCubecl.pipeline_add(pipeline, "elementwise_add", [a, b], output)
:ok = ExCubecl.pipeline_add(pipeline, "relu", [output], output)
{:ok, _cmd_ids} = ExCubecl.pipeline_run(pipeline)
:ok = ExCubecl.pipeline_free(pipeline)

# Buffers are automatically freed when GC'd — no manual free needed
```

## Supported Types

| Type  | Description            |
|-------|------------------------|
| `:f32`| 32-bit float           |
| `:f64`| 64-bit float           |
| `:s32`| 32-bit signed integer  |
| `:s64`| 64-bit signed integer  |
| `:u32`| 32-bit unsigned integer|
| `:u8` | 8-bit unsigned integer |

## Mobile Integration (iOS / Android)

ExCubecl includes a C FFI layer for mobile platform integration.

### iOS (Objective-C / Swift)

```objc
#include "ex_cubecl.h"

float data[] = {1.0f, 2.0f, 3.0f};
size_t shape[] = {3};
ex_cubecl_buffer_handle_t buf = ex_cubecl_buffer_new(
    (const uint8_t*)data, shape, 1, EX_CUBECL_DTYPE_F32
);

float out[3];
ex_cubecl_buffer_read(buf, (uint8_t*)out, sizeof(out));

ex_cubecl_buffer_free(buf);
```

### Android (JNI)

```c
#include "ex_cubecl.h"
#include <jni.h>

JNIEXPORT jlong JNICALL
Java_com_example_excubecl_ExCubeclBuffer_create(
    JNIEnv *env, jobject thiz, jbyteArray data, jlongArray shape, jint dtype) {
    jsize data_len = (*env)->GetArrayLength(env, data);
    jbyte *data_ptr = (*env)->GetByteArrayElements(env, data, NULL);
    jlong *shape_ptr = (*env)->GetLongArrayElements(env, shape, NULL);
    jsize ndim = (*env)->GetArrayLength(env, shape);

    ex_cubecl_buffer_handle_t handle = ex_cubecl_buffer_new(
        (const uint8_t*)data_ptr, (const size_t*)shape_ptr, ndim, dtype
    );

    (*env)->ReleaseByteArrayElements(env, data, data_ptr, 0);
    (*env)->ReleaseLongArrayElements(env, shape, shape_ptr, 0);

    return (jlong)handle;
}
```

### Phase 2 — Video Texture & Audio Mix (C FFI)

```c
#include "ex_cubecl.h"

// Upload YUV420p camera frame to GPU texture
uint8_t y_plane[1920*1080];
uint8_t uv_plane[1920*1080/2];
ex_cubecl_texture_handle_t tex = ex_cubecl_texture_from_yuv(
    y_plane, uv_plane, 1920, 1080
);

// Apply gaussian blur filter
ex_cubecl_texture_handle_t blurred = ex_cubecl_apply_kernel(
    tex, "gaussian_blur", NULL, 0
);

// Mix two audio tracks with gain
float gains[] = {0.7f, 0.5f};
ex_cubecl_buffer_handle_t tracks[] = {track_a, track_b};
ex_cubecl_buffer_handle_t mixed = ex_cubecl_audio_mix(
    tracks, gains, 2, 48000
);

// Cleanup
ex_cubecl_texture_free(tex);
ex_cubecl_texture_free(blurred);
ex_cubecl_buffer_free(mixed);
```

See `native/ex_cubecl_nif/include/ex_cubecl.h` for the full API reference.

## Use Cases

### GPU Image Processing
```
camera frame → GPU texture → CubeCL kernel → screen render
```
Blur, sharpen, denoise, beauty filters, LUT filters — all without CPU copies.

### AI Inference
```
tensor → CubeCL kernels → prediction
```
Segmentation, face landmarks, pose detection, embeddings — realtime camera AI.

### Video Processing
```
video texture → GPU kernels → encoder
```
Compositing, transitions, overlays, subtitles, color grading.

### Livestream Effects
```
camera → AI segmentation → background replacement → stream encoder
```
Virtual background, AR effects, realtime filters — all GPU-native.

## Evolution Path

| Phase | Focus                          | Status        |
|-------|--------------------------------|---------------|
| 1     | GPU compute runtime            | ✅ Complete   |
| 2     | Media runtime (video/camera)   | ✅ Complete   |

### Phase 1 — GPU Compute Runtime
- Buffer management with automatic GC-based cleanup (Rustler ResourceArc)
- Kernel execution (`elementwise_add`, `relu`, and extensible kernel list)
- Async command submission with submit/poll/wait
- Pipeline orchestration for chaining GPU operations
- C FFI layer for mobile platform integration (iOS/Android)

### Phase 2 — Media Runtime (current)
- Media I/O: open, inspect streams, read video frames & audio samples, close — CPU-side implementation with synthetic data for testing
- Video GPU operations: overlay (alpha compositing), mix (dissolve/add/multiply), scale, crop, pixel format conversion (YUV420p→RGB24) — all CPU-side implementations
- Audio GPU operations: mix (multi-track with gain), overlay with ducking, resample (linear interpolation), channel conversion — all CPU-side implementations
- GPU-accelerated filters: gaussian blur, sharpen, LUT color grading, chroma key, brightness/contrast, denoise, EQ (biquad), compressor, reverb (delay-based), normalize — all CPU-side implementations
- Transcoding: encode & mux to mp4/mkv/webm/mov/ts with h264/h265/vp9/av1/prores video and aac/opus/mp3/flac/pcm audio — API stubs with validation
- Real-time media pipeline (GenServer-based) for livestreaming and camera effects — behaviour with `__using__` macro
- C FFI extensions: GPU texture upload (YUV420p/NV12), kernel application to textures, audio mix — full C header implementation with buffer management, kernel dispatch, and error handling

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.
