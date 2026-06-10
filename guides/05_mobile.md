# Mobile Integration

ExCubecl includes a C header that sketches an iOS and Android integration API. The header is an interface prototype for native mobile integration.

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

## Mobile Pipeline Prototype

```
Camera Frame
    ↓
Buffer/texture (via C header prototype)
    ↓
Kernel/filter
    ↓
Screen Render / Encoder
```

This guide shows how a mobile app could integrate with the API shape.

## Thread Safety

The C header prototype is not currently used by the Elixir runtime. Treat it as documentation for a future native integration boundary; current buffers are Rust NIF-managed resources.
