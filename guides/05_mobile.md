# Mobile Integration

ExCubecl includes a C FFI layer for iOS and Android integration.

## iOS (Objective-C / Swift)

```objc
#include "ex_cubecl.h"

// Create a buffer
float data[] = {1.0f, 2.0f, 3.0f};
size_t shape[] = {3};
ex_cubecl_buffer_handle_t buf = ex_cubecl_buffer_new(
    (const uint8_t*)data, shape, 1, EX_CUBECL_DTYPE_F32
);

// Read data back
float out[3];
ex_cubecl_buffer_read(buf, (uint8_t*)out, sizeof(out));

// Cleanup
ex_cubecl_buffer_free(buf);
```

### Swift Bridge

```swift
import Foundation

let data: [Float] = [1.0, 2.0, 3.0]
let shape: [UInt] = [3]

data.withUnsafeBytes { ptr in
    let buf = ex_cubecl_buffer_new(
        ptr.baseAddress?.assumingMemoryBound(to: UInt8.self),
        shape,
        1,
        EX_CUBECL_DTYPE_F32
    )
    // Use buffer...
    ex_cubecl_buffer_free(buf)
}
```

## Android (JNI)

```c
#include "ex_cubecl.h"
#include <jni.h>

JNIEXPORT jlong JNICALL
Java_com_example_excubecl_ExCubeclBuffer_create(
    JNIEnv *env, jobject thiz,
    jbyteArray data, jlongArray shape, jint dtype) {

    jbyte *data_ptr = (*env)->GetByteArrayElements(env, data, NULL);
    jlong *shape_ptr = (*env)->GetLongArrayElements(env, shape, NULL);
    jsize ndim = (*env)->GetArrayLength(env, shape);

    ex_cubecl_buffer_handle_t handle = ex_cubecl_buffer_new(
        (const uint8_t*)data_ptr,
        (const size_t*)shape_ptr,
        ndim,
        (ex_cubecl_dtype_t)dtype
    );

    (*env)->ReleaseByteArrayElements(env, data, data_ptr, 0);
    (*env)->ReleaseLongArrayElements(env, shape, shape_ptr, 0);

    return (jlong)handle;
}
```

## Mobile GPU Pipeline

```
Camera Frame
    ↓
GPU Texture (via C FFI)
    ↓
CubeCL Kernel (blur / edge detect / AI)
    ↓
Screen Render / Encoder
```

All without CPU copies — critical for realtime mobile performance.

## Thread Safety

The C FFI uses thread-local storage. Buffers are only valid on the thread that created them. For multi-threaded mobile apps, create buffers on the thread that will use them.
