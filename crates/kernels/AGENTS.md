# Kernel Crate Rules (`gfx1201`)

- Every kernel entrypoint must be `extern "C" __global__`.
- Compile only with `/opt/rocm/lib/llvm/bin/clang++ -x hip --offload-device-only --no-gpu-bundle-output --offload-arch=gfx1201 -O3 --rocm-path=/opt/rocm`.
- Banned mnemonics: `ds_load_tr*`, `v_bitop3_b32`, `v_cvt_pk_f16_fp8`, `v_wmma_scale_*`.
- Every new kernel instruction family requires an `llvm-objdump` assertion test in `src/lib.rs`.
