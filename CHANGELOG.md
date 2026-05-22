# Changelog

## [Unreleased]

### Changed
- Switched from manual `ExCubecl.free/1` to Rustler resource-based automatic memory management
- Loosened Elixir version requirement from `~> 1.19` to `~> 1.15`
- Removed misleading Nx-backend claims from documentation

## [0.2.2] - 2026-05-22

### Added
- Initial GPU compute runtime with CubeCL via Rust NIFs
- Buffer management, kernel execution, async commands, pipeline orchestration
- C FFI layer for mobile platform integration (iOS/Android)
