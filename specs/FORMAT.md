# Scoria AMD-Native Tensor & Storage Format (Design Spec)

## Physical Layout Principles
1. **256-byte cacheline alignment:** Every tile group and SoA scale array aligns to 256 bytes (matching `gfx1201` L2/L3 cacheline size).
2. **Arch-neutral 16x64 on-disk tile + target-specific register swizzle:** On-disk tiles stay target-independent (`16x64` sub-tiles aligned to 256B); load-time repacking (or pre-baked `.scoria` cache) swizzles into Wave32 `v_wmma` (`gfx1201`) or Wave64 `v_mfma` (`gfx942`/`gfx950`) fragment order without LDS transpose. On `gfx942`, `H4F8` FP8 E4M3 FNUZ applies a `2.0x` scale compensation (`bias=8` vs OCP `bias=7`).
3. **SoA Metadata Separation:** Group scales, zero-points, and codebooks live in contiguous aligned arrays fetched via `TH_LOAD_HT`, separate from `TH_LOAD_NT` weight payloads.

## Target Codecs
- **`H4F8` (W4A8 Hadamard + 16-entry FP8 LUT):** 4-bit indices unpacked via `v_permlane16_var_b32` directly to FP8, paired with Wave32 FWHT-rotated FP8 activations on `v_wmma_f32_16x16x16_fp8_fp8`.
- **`Q4W32` (Symmetric INT4 + QServe progressive scales):** Native `v_wmma_i32_16x16x32_iu4` (prefill/batch) and `v_dot8_i32_iu4` (decode).
- **`BITCOS` (Ternary 1.58-bit / sub-1.5-bit for PrismML Bonsai):** 1-bit non-zero bitmap + compacted 1-bit sign stream unpacked via hardware popcount (`v_bcnt_u32_b32`) + group-128 scale.
- **`QTIPW32` (2.0–3.5 bpw EXL3-style bitshift trellis):** Tail-biting 16x16 Wave32 tiles with procedural lookup-free codebooks + Hadamard rotation.

## Strata-Style NVMe Extents (Phase 3+)
- Profile-guided contiguous expert bundles aligned to 4 KiB / 256 KiB boundaries for Linux `io_uring` + `O_DIRECT` prefetching across dual Gen3 x4 NVMe drives (`nvme1n1` P31 + `nvme0n1` 980).
