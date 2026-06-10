# Changelog

## [0.5.0] - 2026-06-10

### Added
- Pipeline execution now actually runs stored commands (previously only validated kernel names)
- Rust unit tests for DType, kernel list, command status, and ID allocation
- `# Safety` documentation on all unsafe C FFI functions
- C FFI media source storage with proper handle validation

### Fixed
- C FFI handle allocation bug: `ex_cubecl_buffer_new` now uses `alloc_c_handle()` instead of `NEXT_COMMAND_ID`
- `ExCubecl.NIF` not-loaded behavior: returns `{:error, :nif_not_loaded}` instead of raising
- Kernel execution errors now return `{:error, msg}` instead of raising
- `Video.convert` now allocates a properly-sized output buffer for RGB conversion
- `Audio.channels` now properly converts sample data between channel layouts
- `Transcode.start/2` returns errors instead of raising for invalid codecs/containers
- `Video.scale/2`, `Video.crop/2`, `Audio.resample/2` return errors for missing/invalid options
- Removed unimplemented kernels from `kernel_list()` (matmul, softmax, etc.)
- Clippy warnings fixed across FFI and media modules

### Changed
- Version bumped to 0.5.0 across all files
- Documentation updated to accurately describe NIF-backed runtime status
- `pipeline_add` now stores params as JSON bytes for proper pipeline execution
- `execute_kernel` helper extracted for reuse by both `kernel_run` and `pipeline_run`

## [0.4.0] - 2026-05-30

### Fixed
- Version mismatch: aligned `@version` in `lib/ex_cubecl.ex` and `Cargo.toml` with `mix.exs` (0.4.0)
- `ExCubecl.buffer!/3` and `read!/1`: replaced indirect `apply(NIF, ...)` calls with direct `NIF` calls
- `ExCubecl.NIF`: removed duplicate private `nif_error/0` that shadowed Rustler's generated function
- `ExCubecl.Transcode.write_frame/2`: removed duplicate clause with swapped argument order
- `ExCubecl.Transcode.write_samples/2`: removed duplicate clause with swapped argument order
- Rust NIF `poll`: `CommandStatus::Failed` now returns `{:ok, :failed}` instead of `{:error, msg}` for consistency
- Rust NIF `pipeline_run`: removed dangling no-op `let _copy_len` statement

### Changed
- Switched from manual `ExCubecl.free/1` to Rustler resource-based automatic memory management
- Loosened Elixir version requirement from `~> 1.19` to `~> 1.15`
- Removed misleading Nx-backend claims from documentation

## [0.2.2] - 2026-05-22

### Added
- Initial GPU compute runtime with CubeCL via Rust NIFs
- Buffer management, kernel execution, async commands, pipeline orchestration
- C FFI layer for mobile platform integration (iOS/Android)
