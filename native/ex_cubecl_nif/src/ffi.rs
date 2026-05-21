//! Minimal C FFI stub for ex_cubecl GPU compute runtime.
//!
//! This module previously exposed a full CPU tensor operation layer via C handles.
//! It is intentionally kept minimal for Phase 1 (GPU compute runtime) — the C
//! header will be rewritten separately to match the new NIF-based API.
//!
//! All GPU state now lives in Rust (see `lib.rs`). No tensor handles cross the
//! FFI boundary.

/// Placeholder: returns the runtime version string.
#[no_mangle]
pub unsafe extern "C" fn ex_cubecl_version() -> u32 {
    0x0002_0000 // 0.2.0
}
