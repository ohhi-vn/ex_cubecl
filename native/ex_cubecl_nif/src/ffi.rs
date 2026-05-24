//! Minimal C FFI stub for ex_cubecl GPU compute runtime.
//!
//! This module is intentionally kept minimal. All GPU state lives in Rust
//! (see `lib.rs` and `media.rs`). The C header (`include/ex_cubecl.h`) is
//! the authoritative reference for the FFI surface used by iOS/Android.
//!
//! Phase 2 additions: texture handles, media handles, encoder handles,
//! and audio mix functions are declared in the C header but implemented
//! as stubs here until the native mobile bridge is built.

/// Placeholder: returns the runtime version string.
#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_version() -> u32 {
    0x0003_0000 // 0.3.0
}
