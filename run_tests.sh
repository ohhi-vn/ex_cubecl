#!/bin/bash
cd /Users/manhvu/ohhi/OSS_Lib/ex_cubecl
cargo build --release --manifest-path native/ex_cubecl_nif/Cargo.toml 2>&1 > /tmp/build.log
mix compile 2>&1 >> /tmp/build.log
mix test 2>&1 > /tmp/test_results.log
echo "DONE" >> /tmp/test_results.log
