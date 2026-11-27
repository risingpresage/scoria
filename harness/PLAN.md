# Scoria Optimization Plan & Experiment Log

Methodology: ceilings-first + interleaved Control-Candidate-Control (`C-A-C`) via `scripts/bench_cac.py` -> `harness/ledger.jsonl` (3 trials, median).

## Target Reference (`llama.cpp` Vulkan RADV on RX 9070 XT)
- `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M`: **521.05 tok/s** prefill, **123.32 tok/s** decode (theoretical decode ceiling: **~128.2 tok/s** at 561.66 GB/s).
- `Qwen2.5-0.5B-Instruct-Q4_K_M`: **622.72 tok/s** decode.

## Experiment Log (Observed on RX 9070 XT `gfx1201`)
- **Exp 0 (`phase1_baseline`)**: Unoptimized reference HIP kernels + per-launch `hipModuleGetFunction` + host logits argmax.
  - 0.5B: decode `249.13` tok/s, prefill(26) `1133.69` tok/s, prefill(128) `1452.91` tok/s.
  - 7B: decode `39.49` tok/s, prefill(20) `57.38` tok/s, prefill(128) `66.51` tok/s.
- **Exp 1 (`step2_handle_argmax`)**: Pre-resolved `KernelHandle` cache + 2-stage GPU `argmax_stage{1,2}_f32` (4B D2H readback).
  - 0.5B: decode `272.15` tok/s; 7B: decode `39.64` tok/s.
- **Exp 2 (`step3_gemv`, `step3b_vec_splitk_gemv`)**: Cooperative GEMV + Split-K `gemv_q6_k_f32` (unaligned `u8/i8` eliminated; `lm_head` reached `597 GB/s`).
  - 0.5B: decode `249.41` tok/s; 7B: decode `89.60` tok/s.
- **Exp 3 (`step3c_warp_attn_fused`)**: Warp `gqa_attn_f32` + fused `add_rmsnorm_f32`.
  - 0.5B: decode `280.57` tok/s; 7B: decode `86.51` tok/s (flagged `attn` regression `1.654 -> 2.844 ms`).
- **Exp 4 (`step3d_attn_q4k_b128`)**: 4-wave split-T `gqa_attn_f32` (`float4` coalesced KV loads) + 2-row/WG `uint4` `global_load_b128` Split-K `gemv_q4_k_f32` + `float4` RMSNorm/SiLU + declarative Rust `define_infer_kernels!` / `hip_launch!`.
  - 0.5B: decode `386.69` tok/s; 7B: decode `107.75` tok/s (`attn` dropped `2.844 -> 0.605 ms`).
- **Exp 6 (`phase2b_qwen3`)**: `qwen3` arch (per-head `qk_rmsnorm_f32`, `key_length` head_dim, `embed_lookup_q6_k`), Q4_K GEMV row scaling, `hipGraph` decode capture, 64x128 prefill tile.
  - 4B: decode **`154.35` tok/s**, prefill(13) **`645.86`**, prefill(128) **`3615.97`**.
  - 8B: decode **`99.90` tok/s**, prefill(13) **`470.58`**, prefill(128) **`2230.71`**.
  - 7B: decode **`112.52` tok/s**, prefill(20) **`647.49`**, prefill(128) **`2141.72`**.
  - 0.5B: decode **`401.56` tok/s**, prefill(26) **`4265.60`**, prefill(128) **`14853.13`**.
  - Note: `bench_cac.py` now runs one in-process `--warmup` pass so lazy HSACO binding (~25-80 ms) is not folded into the prefill timer, matching `llama-bench`.
  - Remaining: 4B/8B decode and pp128 are still below the `llama.cpp` Vulkan targets. Decode is at ~88% of the measured stream ceiling; prefill has a ~25 ms forward constant from Q4_K superblock-header re-reads (`BK=32` re-reads the 16 B header 8x per 256 weights) plus a second weight pass at `BM=64`/`m=128`.
- **Exp 5 (`step4_wmma_prefill` / `phase2a_final`)**: Silicon-probed `v_wmma_f32_16x16x16_f16` tiled prefill GEMM (`BM=32, BN=64, BK=32`) across `F16`, `Q5_0`, `Q8_0`, `Q4_K`, `Q6_K`.
  - 0.5B: decode **`386.22` tok/s**, prefill(26) **`4356.28` tok/s**, prefill(128) **`15534.62` tok/s**.
  - 7B: decode **`111.94` tok/s** (`gpu_sum=10.854 ms`), prefill(20) **`673.11` tok/s** (`1.29x` `llama.cpp` Vulkan), prefill(128) **`2079.89` tok/s** (`31.27x` Phase 1).
