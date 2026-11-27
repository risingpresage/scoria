# Scoria Roofline & Throughput Ceilings (`gfx1201` RX 9070 XT)

Derived from hardware measurements recorded in `harness/baselines/gfx1201_gate.json` following the `sybil-solutions/inference-optimisation-skill` ceilings-first methodology.

## 1. Measured Hardware Ceilings (`gfx1201`, Navi 48, 64 CUs, 16 GB GDDR6)

| Resource | Measured Ceiling (`scoria gate`) | Notes |
|---|---|---|
| VRAM Bandwidth | **561.66 GB/s** (stream), **2275.64 GB/s** (32 MiB L3 HT) | ~88% of 636 GB/s theoretical GDDR6 |
| FP16 Dense WMMA | **207.53 TFLOPS** | Practical prefill ceiling (~50% MFU): **~103.8 TFLOPS** |
| FP8 Dense WMMA | **383.76 TFLOPS** | Practical prefill ceiling (~50% MFU): **~191.9 TFLOPS** |
| IU4 Dense WMMA | **763.95 TOPS** | Practical prefill ceiling (~50% MFU): **~382.0 TOPS** |
| 2:4 Sparse SWMMAC | **385.68 TFLOPS** (FP16) / **734.63 TFLOPS** (FP8) | 1.858x / 1.914x dense |
| Register LUT (`v_permlane16_var_b32`) | **1824.35 G-lookups/s** | Beats LDS LUT (1709.56 G-lookups/s) |
| Ternary BITCOS Unpack (`v_bcnt_u32_b32`) | **4321.57 G-weights/s** | Popcount bitmap + sign decode |
| PCIe Pinned DMA | **6.96 GB/s** H2D / **6.88 GB/s** D2H | Gen3/Gen4 host link bound |
| NVMe `io_uring` `O_DIRECT` | **3.558 GB/s** (P31) / **2.544 GB/s** (980) / **3.494 GB/s** (Dual) | Cold expert paging floor |

## 2. Model Decode & Prefill Ceilings

### A. `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` (0.398 GB on disk)
- Active decode bytes/tok: ~0.398 GB (entire model fits in 0.4 GB; note `token_embd` is `Q8_0` at 144 MB, shared as tied LM head; active per-token embedding lookup reads 1 row, while LM head reads full 144 MB).
- **Decode ceiling (`561.66 / 0.398`)**: **~1,411 tok/s** (kernel-launch & host-overhead bound at small layer dims `n_embd=896`).
- **Prefill ceiling (50% FP16 WMMA, ~1.0 GFLOP/tok)**: **~100,000 tok/s** (compute-underutilized at small `M`; bound by kernel dispatch & `K`-reduction overhead).

### B. `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` (4.683 GB on disk)
- File size `4.683 GB` includes `token_embd.weight` (`Q4_K`, `3584 x 152064` = ~306 MB, only 1 row touched per token), `output.weight` (`Q6_K`, `3584 x 152064` = ~447 MB, full matrix read every token), plus 28 transformer layers (~3.93 GB).
- Active decode bytes/tok (short context): `~4.683 - 0.306 = ~4.38 GB/tok` (+ negligible KV at short ctx).
- **Decode ceiling (`561.66 / 4.38`)**: **~128.2 tok/s** (compare `llama.cpp` Vulkan RADV observed: **123.32 tok/s** = 96% of VRAM bandwidth!).
- **Prefill FLOPs/tok**: `~2 * 7.0B = 14.0 GFLOP/tok`.
  - **Prefill ceiling (50% FP16 WMMA = 103.8 TFLOPS)**: **~7,414 tok/s** (at `M=128` tiled GEMM; compare Phase 1 `scoria` 55.41 tok/s and `llama.cpp` Vulkan 521.05 tok/s).

### C. `Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf` (6.233 GB on disk, Phase 2B/3 target)
- Base 64 hybrid blocks + LM head (`~5.99 GB` without MTP draft head).
- **No-spec decode ceiling (`561.66 / 5.99`)**: **~93.8 tok/s**.
- **With 1-step MTP (`a ≈ 1.65`, draft cost ~0.24 GB)**: **~145–155 tok/s** on a 27B model resident in 16 GB VRAM.

### D. `Inkling-2-Mini-NVFP4.gguf` (17.628 GB on disk, Phase 3 target)
- Exceeds 16 GB VRAM (16.42 GiB file vs ~15 GiB usable VRAM -> requires VRAM + Pinned RAM / NVMe `io_uring` tiering).
- 24 physical layers (`6..18` looped 3x = 48 logical evals), 128 experts top-8 (~3.4B active physical, ~5.4B effective).
- **VRAM-resident Q3_K_XL (`13.63 GB`) active bytes/tok**: `~2.3 GB` -> **~244 tok/s** ceiling if hot experts stay cached.
- **Cold expert NVMe paging**: at `3.558 GB/s`, 1 cold 2-expert layer miss (~10 MB) costs ~2.8 ms if not overlapped via `io_uring` prefetch.
