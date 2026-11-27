# Verified gfx1201 (RDNA 4 / RX 9070 XT) ISA Reference

All entries verified against `/opt/rocm/lib/llvm/bin/llvm-mc -mcpu=gfx1201` and `clang++` 24.0.0git (ROCm 7.16.0).

## Hardware Baseline
- Architecture: Navi 48 (`gfx1201`), 64 CUs, Wave32, 16 GB GDDR6 (~636 GB/s ceiling).
- Caches: 32 KB L0v/CU, 128 KB LDS/WGP (32 banks), 8 MB high-bandwidth L2 (no shared array L1), 64 MB L3 Infinity Cache, 256-byte cachelines.

## Verified Matrix & Permutation Builtins
- **FP16 WMMA (16x16x16, no A/B lane duplication — 8 elements/lane):**
  `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(half8_t a, half8_t b, float8_t c)` -> `v_wmma_f32_16x16x16_f16`
- **FP8 WMMA (16x16x16, 2 VGPRs/operand):**
  `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(int2_t a, int2_t b, float8_t c)` -> `v_wmma_f32_16x16x16_fp8_fp8`
- **INT4 WMMA (16x16x32, 2 VGPRs/operand):**
  `__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(bool signA, int2_t a, bool signB, int2_t b, int8_t_v c, bool clamp)` -> `v_wmma_i32_16x16x32_iu4`
- **Sparse 2:4 SWMMAC:**
  - `__builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(half8_t a, half16_t b, float8_t c, int idx)` -> `v_swmmac_f32_16x16x32_f16`
  - `__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(int2_t a, int4_t_v b, float8_t c, int idx)` -> `v_swmmac_f32_16x16x32_fp8_fp8`
- **Register-File 16-entry LUT (`v_permlane16_var_b32`):**
  `__builtin_amdgcn_permlane16_var(unsigned old, unsigned src, unsigned lane_sel, bool fi, bool bound_ctrl)`

## Memory & Cache Policy Rules
- **Double-K Fusing:** Single FP8/INT8 WMMA loads 64b/lane; fuse 2 K-slices (`K=32` FP8, `K=64` INT4) so every global/LDS access uses 128-bit `global_load_b128`.
- **Temporal Hints (`TH`):**
  - Streamed dense/cold-expert weights: `global_load_b128 ... th:TH_LOAD_NT` (avoid evicting 8 MB L2 / 64 MB L3).
  - Hot KV cache, activations, and looped-block dense weights: `th:TH_LOAD_HT`.

## Banned on gfx1201 (Do Not Emit)
- `ds_load_tr*` (no hardware LDS transpose load; pre-swizzle weights offline instead)
- `v_bitop3_b32`
- `v_cvt_pk_f16_fp8` (hardware converts FP8 <-> FP32 only; feed FP8 straight to `v_wmma_*_fp8_fp8`)
- `v_wmma_scale_*` (`gfx1250`/CDNA4 only; no native FP4 WMMA on `gfx1201`)

## CDNA 3 / CDNA 4 Portability (`gfx942` MI300X, `gfx950` MI350X)
| Feature | RDNA 4 (`gfx1201`) | CDNA 3 (`gfx942` MI300X) | CDNA 4 (`gfx950` MI350X) |
|---|---|---|---|
| Wave size | Wave32 | Wave64 | Wave64 |
| Matrix MMA | `v_wmma_*` (`__builtin_amdgcn_wmma_*`) | `v_mfma_*` (`__builtin_amdgcn_mfma_*`) | `v_mfma_*` + `v_mfma_scale_*_f8f6f4` |
| 16-entry Register LUT | `v_permlane16_var_b32` | `ds_bpermute_b32((nib & 15) << 2, val)` | `ds_bpermute_b32` |
| Non-temporal Load | `global_load_b128 ... th:TH_LOAD_NT` | `global_load_dwordx4 ... nt` | `global_load_dwordx4 ... nt` |
| Wait Counter | `s_wait_loadcnt 0` | `s_waitcnt vmcnt(0)` | `s_waitcnt vmcnt(0)` |
| FP8 Encoding Note | OCP E4M3 (`bias=7`, max `448`) | FNUZ E4M3 (`bias=8`, max `240` -> 2x scale adjust on `gfx942`) | OCP E4M3 |
