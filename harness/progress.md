# Scoria Session Progress

## 2026-11-26 — Phase 0 Complete (Verified on RX 9070 XT `gfx1201`)
- Built 2026 AI-native repository harness (`AGENTS.md`, `CLAUDE.md` bridge, `specs/`, `.agents/skills/`, `scripts/{init,lint-arch,check}.sh`, `harness/`).
- Implemented `scoria-hip-rt` (lazy `dlopen` of `libamdhip64.so` + `alloc_guard` 32B `__libc_memalign` interposition), `scoria-kernels` (cross-compiled `gfx1201` + `gfx942` `.hsaco` with `llvm-objdump` ISA tests), `scoria-gguf` / `scoria-engine` architectural seams, and `scoria-cli` (`gate` + raw Linux `io_uring` `O_DIRECT` reader).
- Root-cause discoveries recorded in `AGENTS.md`:
  - `hipcc --genco` silently targets `gfx906`; must invoke `clang++ -x hip --offload-device-only --no-gpu-bundle-output --offload-arch=<arch>`.
  - Inline `global_load_b128` requires `"=&v"` earlyclobber + `s_wait_loadcnt 0` (`s_waitcnt vmcnt(0)` on CDNA).
  - `libamdhip64.so.7` (`0x74d5b8`) emits 32-byte `vmovdqa %ymm1` against `malloc` allocations; `alloc_guard.rs` + `.cargo/config.toml` `--export-dynamic-symbol` eliminates `#GP` crashes across multi-threaded tests and standalone runs (5/5 quick + 100/100 soak).
  - Btrfs `compress=zstd` deflated constant test payloads; switching to PRNG bytes yielded true physical NVMe `O_DIRECT` speeds.
- Verified `gfx1201` gate baseline (`harness/baselines/gfx1201_gate.json`):
  - Soak: `100/100` (`[1, 4, 8]` GiB, `15307.14 ms`)
  - VRAM stream: `561.66` GB/s default, `560.09` GB/s `TH_LOAD_NT`; L3 probe dropped from `2275.64` GB/s to `433.30` / `423.88` GB/s after 1 GiB stream
  - WMMA: FP16 `207.53` TFLOPS, FP8 `383.76` TFLOPS, IU4 `763.95` TOPS
  - SWMMAC (2:4 sparse): FP16 `385.68` TFLOPS (`1.858x`), FP8 `734.63` TFLOPS (`1.914x`)
  - Register LUT (`v_permlane16_var_b32`) vs LDS LUT: `1824.35` vs `1709.56` G-lookups/s; BITCOS popcount: `4321.57` G-weights/s
  - PCIe pinned DMA: `6.96` H2D / `6.88` D2H GB/s
  - Incompressible `io_uring` `O_DIRECT` (256 MiB): P31 `3.558` GB/s, 980 `2.544` GB/s, Dual `3.494` GB/s

## 2026-11-26 — Phase 1 Complete (GGUF Parser, BPE Tokenizer, Reference HIP Kernels, Parity Verified)
- Implemented full pure-Rust mmap GGUF v2/v3 parser (`scoria-gguf`), CPU reference dequantizers (`F32`, `F16`, `Q5_0`, `Q8_0`, `Q4_K`, `Q6_K`), and deterministic Qwen2/GPT-2 byte-level BPE tokenizer (`12/12` exact token ID matches against `llama-tokenize` across `Qwen2.5-0.5B` and `DeepSeek-R1-Distill-Qwen-7B`).
- Added `infer_kernels.hip` (`embed_lookup_*`, `rmsnorm_f32`, `linear_{f32,f16,q5_0,q8_0,q4_k,q6_k}`, `rope_neox_f32`, `kv_cache_append_f32`, `gqa_attn_f32`, `silu_mul_f32`, `add_inplace_f32`) cross-compiled for both `gfx1201` and `gfx942` via `clang++ --offload-device-only`, plus `Qwen2Model` wired onto the `ExecutionGraph`, `TensorStore`, and `LinearKernel` seams.
- Verified parity against `libllama.so` (`b11050` CPU reference) and recorded `harness/baselines/phase1_parity.json`:
  - `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`: `cos = 0.999774`, `max_abs_diff = 0.4590`, `top1_match = true`, `32/32` exact greedy tokens; `1114.52` tok/s prefill, `247.79` tok/s decode.
  - `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf`: `cos = 0.999913`, `max_abs_diff = 0.8123`, `top5_match = true`, `21/32` exact greedy prefix (divergence from `Q8_K` activation quantization in `llama.cpp`); `55.41` tok/s prefill, `40.02` tok/s decode (vs `llama.cpp` Vulkan RADV `521.05` / `123.32` tok/s).
  - `Qwen2.5-3B-Instruct-f16.gguf` (no activation quant): `cos = 0.999999503`, `max_abs_diff = 0.0273`, `mean_abs_diff = 0.003887`, `16/16` exact greedy tokens.
- Downloaded Phase 2/3 target GGUFs to `/mnt/nuclearmissilesilo/models/{bonsai,inkling}` (symlinked into `~/models/{bonsai,inkling}`): `Ternary-Bonsai-2-2B-{PTQ1_0,Q2_0}`, `Ternary-Bonsai-2.5-27B-{PTQ1_0,Q2_0}-mtp` (`qwen35` hybrid SSM+attn + `prism.hadamard.*`), `dspark-Inkling-2-Mini-Q8_0` (`dflash`), `Inkling-2-Mini-UD-Q3_K_XL` (`inkling2_loop`, 24 layers looped 3x on 6..18, 128-expert MoE).

## 2026-11-26 — Phase 2A Complete (`gfx1201` Decode & WMMA Prefill Optimization)
- Recorded ceilings & C-A-C experiment ledger in `harness/THEORY.md`, `harness/PLAN.md`, `harness/ledger.jsonl`, and `harness/baselines/phase2a_gfx1201.json` via `scripts/bench_cac.py` (`--trials 3`).
- Eliminated host bottlenecks and added declarative Rust metaprogramming: `KernelHandle` cache + `define_infer_kernels!` in `crates/engine/src/lib.rs`, `KernelArg` + `hip_launch!` in `crates/engine/src/qwen2.rs`, 2-stage GPU `argmax_stage{1,2}_f32`, and `--profile` instrumentation in `scoria run`.
- Rewrote hot decode kernels: fused `add_rmsnorm_f32` (`float4`), 4-wave split-T `gqa_attn_f32` (`float4`/`float2` coalesced KV loads; 7B attention dropped `2.844 -> 0.602 ms`), and 2-row/WG Split-K `gemv_q4_k_f32` (`12` `global_load_b128` loads, `0` byte loads) + `gemv_q6_k_f32` (`lm_head` at `597 GB/s`).
- Verified exact `v_wmma_f32_16x16x16_f16` fragment layout on `gfx1201` (`wmma_probe_f16`: `err_a=0.000000`) and shipped shared templated `gemm_wmma_{f16,q5_0,q8_0,q4_k,q6_k}` (`BM=32, BN=64, BK=32`) with `gpu_wmma_prefill_parity` unit tests (`M=20, 64`) and `gfx942` portability preserved.
- Phase 2A 3-trial medians on RX 9070 XT (`harness/baselines/phase2a_gfx1201.json`):
  - `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M`: decode `111.94 tok/s` (`2.83x` Phase 1 `39.49`), prefill(20) `673.11 tok/s` (`1.29x` `llama.cpp` Vulkan `521.05`), prefill(128) `2079.89 tok/s` (`31.27x` Phase 1 `66.51`). Parity vs `libllama.so` CPU: `cos=0.999913`, `top1=true`, `top5=true`, `21/32` greedy prefix.
  - `Qwen2.5-0.5B-Instruct-Q4_K_M`: decode `386.22 tok/s` (`1.55x` Phase 1), prefill(26) `4356.28 tok/s` (`3.84x`), prefill(128) `15534.62 tok/s` (`10.69x`). Parity vs `libllama.so` CPU: `cos=0.999413`, `top1=true`, `top5=true`, `32/32` greedy tokens.

## 2026-11-26 — Phase 2B (Qwen3 Baselines, hipGraph Decode, Wider Prefill Tile)
- Added `qwen3` support (`crates/gguf` already read `{arch}.attention.key_length`; `crates/engine/src/qwen2.rs` now accepts `qwen2 | qwen3`, loads optional `blk.{l}.attn_{q,k}_norm.weight`, and applies the new `qk_rmsnorm_f32` kernel per head before RoPE). Added `embed_lookup_q6_k` because Qwen3-4B ties `token_embd`/`output` as Q6_K.
- Parity vs `libllama.so` b11050 CPU (`scripts/parity_oracle.py`, tokens `785,6722,315,9625,374`, 24 greedy): 4B `cos=0.999657`, 8B `cos=0.999335`, 7B `cos=0.999913` (identical to the Phase 2A figure), 0.5B `cos=0.999345`; every model `top1_match=true` and `24/24` exact greedy.
- Decode: `gemv_q4_k_f32` is now templated on rows-per-workgroup (`gemv_q4_k_r4/r8_f32`) so all 16 superblock groups stay busy when `k/256 < 16` (Qwen3-4B `nb=10` was leaving 62.5% of the block idle: 342 -> ~370 GB/s).
- Decode: `hipGraph` capture. `rope_neox_f32`, `kv_cache_append_f32` and `gqa_attn_f32` now read the position from a device slot (`*pos`), and `decode_advance_u32` closes the loop on-device (`token_ids[0] <- argmax[0]; pos += 1`), so one 36-layer decode step is captured once and replayed per token. `scoria graph-soak` is the mandatory leak guard: 1000 replays on Qwen3-8B drift `-6 MiB`.
- Prefill: `gemm_wmma_*` is now a 64x128 macro-tile with four 16x16 accumulators per wave (2x2 wave-tile pairs, ~43 flop/byte vs ~21), dispatched only when it yields >=128 blocks; the 32x64 tile is retained as `gemm_wmma_narrow_*` for small matrices (the wide tile cost the 4B ~8%).
- Benchmarks (`scripts/bench_cac.py --trials 3`, now with an in-process `--warmup` so lazy HSACO binding does not land in the prefill timer, matching llama-bench): 4B decode `154.35` / pp128 `3615.97`; 8B decode `99.90` / pp128 `2230.71`; 7B decode `112.52` / pp128 `2141.72`; 0.5B decode `401.56` / pp128 `14853.13`. Recorded in `harness/baselines/phase2b_gfx1201.json`.
- Remaining gap (documented, not met): targets are 4B `>189.32` decode / `>5826.04` pp128 and 8B `>111.73` decode / `>3902.77` pp128. Decode streams weights at ~492 GB/s on the 8B against a `561.66 GB/s` measured stream ceiling while `lm_head` alone reaches ~600 GB/s, so the loss is per-layer launch/fill work. Prefill carries a ~25 ms per-forward constant because the Q4_K fill re-reads the 16 B superblock header once per `BK=32` step and every weight is read twice at `BM=64` with `m=128`; the marginal rate past that constant is ~4165 tok/s on the 8B.

## Next Step
- Phase 2B / Phase 3: `hipGraph` capture + `Q8_1` `v_dot4_i32_i8` / Wave32 repacked GEMV to surpass `123.32 tok/s` decode, then Bonsai 2.5 (`qwen35` + Hadamard + `PTQ1_0` ternary popcount) & Inkling 2 Mini (`inkling2_loop` + `io_uring` `O_DIRECT` MoE), plus Remotion ASCII/audio demo video from `phase2a_gfx1201.json`.
