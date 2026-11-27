# Kernel Authoring Skill (`gfx1201`)

1. Add or edit `extern "C" __global__` kernels in `crates/kernels/hip/*.hip`.
2. Ensure `crates/kernels/build.rs` compiles with `/opt/rocm/lib/llvm/bin/clang++ -x hip --offload-device-only --no-gpu-bundle-output --offload-arch=gfx1201 -O3 --rocm-path=/opt/rocm`. Never use `hipcc --genco`.
3. Disassemble with `/opt/rocm/lib/llvm/bin/llvm-objdump -d --mcpu=gfx1201` and verify expected vector/WMMA/cache-hint mnemonics are emitted (no silent scalar fallback).
4. Add an ISA assertion in `crates/kernels/src/lib.rs` tests and run `./scripts/check.sh`.
5. Microbenchmark on GPU 0 via `scoria-cli` and record observed numbers in `harness/baselines/`.
