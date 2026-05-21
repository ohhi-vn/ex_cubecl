# Mobile Integration (iOS / Android)

ExCubecl provides a C FFI layer for integrating tensor operations into iOS and Android apps.

## Architecture

```
┌─────────────────────────────────────────┐
│           iOS / Android App             │
├─────────────────────────────────────────┤
│  Objective-C / Swift  │  Java / Kotlin  │
├─────────────────────────────────────────┤
│         C FFI (ex_cubecl.h)             │
├─────────────────────────────────────────┤
│     Rust Static Library (ffi.rs)        │
│     - Handle-based tensor management    │
│     - Thread-local storage              │
└─────────────────────────────────────────┘
```

## Building the Static Library

The `staticlib` crate type in `Cargo.toml` produces a `.a` library:

```bash
cd native/ex_cubecl_nif
cargo build --release
# Output: target/release/libex_cubecl_nif.a
```

## iOS Integration

### Swift Bridge

```objc
// Bridging header: ex_cubecl_bridge.h
#include "ex_cubecl.h"

// Create tensors
float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
size_t shape[] = {2, 2};
ex_cubecl_tensor_handle_t a = ex_cubecl_new_tensor(
    (const uint8_t*)data, shape, 2, EX_CUBECL_DTYPE_F32
);

float data2[] = {5.0f, 6.0f, 7.0f, 8.0f};
ex_cubecl_tensor_handle_t b = ex_cubecl_new_tensor(
    (const uint8_t*)data2, shape, 2, EX_CUBECL_DTYPE_F32
);

// Add
ex_cubecl_tensor_handle_t result = ex_cubecl_add(a, b);

// Read result
float out[4];
ex_cubecl_read_tensor(result, (uint8_t*)out, sizeof(out));
// out = [6.0, 8.0, 10.0, 12.0]

// Cleanup
ex_cubecl_deallocate_tensor(a);
ex_cubecl_deallocate_tensor(b);
ex_cubecl_deallocate_tensor(result);
```

### Error Handling

```objc
ex_cubecl_tensor_handle_t result = ex_cubecl_add(invalid_handle, b);
if (result == 0) {
    char error[256];
    ex_cubecl_last_error(error, sizeof(error));
    printf("Error: %s\n", error);
}
```

## Android Integration

### JNI Bridge

```c
// jni_bridge.c
#include "ex_cubecl.h"
#include <jni.h>

JNIEXPORT jlong JNICALL
Java_com_example_excubecl_ExCubeclTensor_create(
    JNIEnv *env, jobject thiz, jbyteArray data, jintArray shape, jint dtype) {
    jsize data_len = (*env)->GetArrayLength(env, data);
    jsize shape_len = (*env)->GetArrayLength(env, shape);

    jbyte *data_ptr = (*env)->GetByteArrayElements(env, data, NULL);
    jint *shape_ptr = (*env)->GetIntArrayElements(env, shape, NULL);

    ex_cubecl_tensor_handle_t handle = ex_cubecl_new_tensor(
        (const uint8_t*)data_ptr, (const size_t*)shape_ptr,
        shape_len, (ex_cubecl_dtype_t)dtype
    );

    (*env)->ReleaseByteArrayElements(env, data, data_ptr, JNI_ABORT);
    (*env)->ReleaseIntArrayElements(env, shape, shape_ptr, JNI_ABORT);

    return (jlong)handle;
}

JNIEXPORT jlong JNICALL
Java_com_example_excubecl_ExCubeclTensor_add(
    JNIEnv *env, jobject thiz, jlong a_handle, jlong b_handle) {
    return (jlong)ex_cubecl_add(
        (ex_cubecl_tensor_handle_t)a_handle,
        (ex_cubecl_tensor_handle_t)b_handle
    );
}

JNIEXPORT jbyteArray JNICALL
Java_com_example_excubecl_ExCubeclTensor_read(
    JNIEnv *env, jobject thiz, jlong handle) {
    size_t nbytes = ex_cubecl_tensor_nbytes((ex_cubecl_tensor_handle_t)handle);
    jbyteArray result = (*env)->NewByteArray(env, nbytes);
    jbyte *result_ptr = (*env)->GetByteArrayElements(env, result, NULL);

    ex_cubecl_read_tensor((ex_cubecl_tensor_handle_t)handle,
                          (uint8_t*)result_ptr, nbytes);

    (*env)->ReleaseByteArrayElements(env, result, result_ptr, 0);
    return result;
}

JNIEXPORT void JNICALL
Java_com_example_excubecl_ExCubeclTensor_destroy(
    JNIEnv *env, jobject thiz, jlong handle) {
    ex_cubecl_deallocate_tensor((ex_cubecl_tensor_handle_t)handle);
}
```

### Kotlin Wrapper

```kotlin
// ExCubeclTensor.kt
class ExCubeclTensor(private var handle: Long) {
    companion object {
        init { System.loadLibrary("excubecl_nif") }

        fun create(data: ByteArray, shape: IntArray, dtype: Int): ExCubeclTensor {
            val handle = create(data, shape, dtype)
            return ExCubeclTensor(handle)
        }

        @JvmStatic private external fun create(data: ByteArray, shape: IntArray, dtype: Int): Long
        @JvmStatic private external fun add(a: Long, b: Long): Long
        @JvmStatic private external fun read(handle: Long): ByteArray
        @JvmStatic private external fun destroy(handle: Long)
    }

    fun add(other: ExCubeclTensor): ExCubeclTensor {
        return ExCubeclTensor(add(this.handle, other.handle))
    }

    fun read(): ByteArray = read(handle)

    fun destroy() {
        if (handle != 0L) {
            destroy(handle)
            handle = 0L
        }
    }

    protected fun finalize() = destroy()
}
```

## Thread Safety

The C FFI uses thread-local storage for the tensor store. Handles are only valid on the thread that created them. For multi-threaded apps:

```objc
// Each thread must manage its own handles
dispatch_queue_t queue = dispatch_queue_create("tensor_queue", NULL);
dispatch_async(queue, ^{
    // Create and use tensors on this thread
    ex_cubecl_tensor_handle_t t = ex_cubecl_new_tensor(...);
    // ... operations ...
    ex_cubecl_deallocate_tensor(t);
});
```

## Supported C FFI Operations

| Category | Functions |
|----------|-----------|
| Lifecycle | `new_tensor`, `read_tensor`, `deallocate_tensor`, `tensor_shape`, `tensor_dtype`, `tensor_nbytes` |
| Binary | `add`, `subtract`, `multiply`, `divide` |
| Unary | `negate`, `abs`, `exp`, `log`, `sqrt`, `sigmoid`, `relu`, `sin`, `cos`, `tanh` |
| Reductions | `sum`, `reduce_max`, `reduce_min` |
| Shape | `reshape`, `transpose`, `broadcast` |
| LinAlg | `dot`, `matmul` |
| Other | `concatenate`, `slice`, `pad`, `reverse`, `as_type`, `constant`, `eye`, `iota` |
| Error | `last_error` |
