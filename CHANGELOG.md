# Changelog

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
