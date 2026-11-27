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
- Remaining gap (documented, not met): targets are 4B `>189.32` decode / `>5826.04` pp128 and 8B `>111.73` decode / `>3902.77` pp128. Decode streams weights at ~492 GB/s on the 8B against a `561.66 GB/s` measured stream ceiling while `lm_head` alone reaches ~600 GB/s, so the loss is per-layer launch/fill work (hipGraph removed the dispatch chain but only bought ~3%). Prefill is **not** weight-traffic bound: a 128x64 tile that streams each weight once at `m=128` instead of twice ran at exactly the same speed (57.29 vs 57.29 ms on 8B pp128). It is bound by LDS fill throughput serialised against the WMMA compute by `__syncthreads`, since every loader fuses the global load with the LDS store and exposes the load latency; double-buffered LDS with split fetch/store loaders is the next step.

## 2026-11-27 — Phase 2D: Q8_1 decode GEMV retired from the default path
- A/B'd the Q8_1 integer-dot decode GEMV with a switch (`SCORIA_Q4K_Q8`) instead of trusting the phase2c medians, which were taken under GPU contention: 8B `99.73 -> 101.64` tok/s (+1.9%), 7B `113.10 -> 113.68`, 0.5B `427.4 -> 430.5`, 4B a wash.
- Reason it loses: quantising `x` to int8 removes activation traffic that was already L1-resident, while adding a quantise launch per GEMV group (3-4 per layer) to a chain that is already dispatch-limited. Q4_K weight bytes are the entire decode budget.
- Default is now **off** (`SCORIA_Q4K_Q8=1` opts in); the kernels stay because they are ISA-tested and may win on other shapes. `phase2d_q8_off` 3-trial medians: 0.5B `414.90` / `15877.16`, 7B `115.20` / `2758.20`, 4B `158.41` / `3973.01`, 8B `100.07` / `2486.28` — at or above the phase2c record on every model.
- Also published the repo at `github.com/risingpresage/scoria` with the branding set in `assets/`.

## 2026-11-27 — Inkling 2 Mini budgeting, KV planner, and measured Strata paging
- `inkling2_loop` checkpoints now parse. The architecture stores `attention.head_count`, `head_count_kv` and `sliding_window_pattern` as **arrays** (one entry per block, the pattern as booleans), plus `expert_count`, `expert_used_count` and `loop.{count,start,end}`; `ModelConfig` tolerates scalar-or-array and gained `kv_budget()`.
- New `scoria plan <gguf> [--ctx N] [--vram-mib N]`: KV slots after loop expansion, per-token cost, a context table at FP32/FP16/FP8, and the largest context that fits the free VRAM. New `scoria strata <gguf>`: replays the MoE routing read pattern and reports the paged decode ceiling.
- Inkling-2-Mini-UD-Q3_K_XL (12.70 GiB) measured: 48 KV slots = **8 full-history + 40 windowed** (window 1024; layers 5/11/17/23 keep history, and 11 and 17 sit inside the 3x loop so they cost three caches each). 200k context = 12.52 GiB FP32 / 6.26 FP16 / 3.13 FP8, so with 15.69 GiB free the ceiling is 43963 / 93047 / **191214** tokens. Uniform attention at 200k would have needed 56 GiB: the window is the whole reason long context is possible here.
- Strata prototype measured: 11.36 GiB of the checkpoint is experts (48 tensors, 1.42-2.85 MiB slabs); top-8 of 128 routes **727.1 MiB per token (6.2%)**. `O_DIRECT` is rejected on this model store because the volume is btrfs `compress=zstd:1`; the buffered fallback with `FADV_DONTNEED` runs at 1.57 GB/s, i.e. **486 ms/token = 2 tok/s** with zero compute. Paging is the fallback, not a substitute for residency.
- Blocked on running Inkling at all: the model needs relative attention (16 buckets), per-layer head counts and window masks, 4-tap short convs on q/k/v, attention gates and sinks, block attention residuals, `SiTU`, and a 128-expert top-8 MoE. llama.cpp b11050 cannot load `inkling2_loop`, so it cannot serve as the parity oracle either.

## Parity gate calibration (cosine > 0.999 is prompt-dependent, not an implementation defect)
- Measured the control that settles it: **llama.cpp's own Vulkan backend vs llama.cpp's own CPU backend** scores `cos = 0.997985` (`maxdiff 1.152`) on the Qwen3-4B harness chat prompt (`151644,872,198,785,…,77091,198`, `m=13`), where scoria scores `0.997652` (`maxdiff 1.103`). Both pick top1 `785`. No GPU implementation reaches 0.999 on that prompt.
- Forcing prefill onto the scalar FP32 GEMM (`SCORIA_PREFILL_FP32=1`) changes nothing: 4B `0.997614` vs `0.997652` for FP16 WMMA, 8B `0.996415` vs `0.996404`. So the deficit is **not** FP16 WMMA rounding.
- Cosine is not monotonic in prompt length for the same tokens (`m=2 → 0.9847`, `m=4..8 → 0.9992–0.9997`, `m=10 → 0.9961`, `m=11 → 0.9994`, `m=12/13 → 0.9977`); it tracks how peaked the distribution is (`max_logit` 39.0 at `m=13`), not error accumulation.
- Conclusion: gate on `top1_match` plus greedy-prefix agreement, and treat `cos > 0.999` as a per-prompt sanity check rather than an absolute bar. Current status: every model/prompt tested has `top1_match=true`.

## Next Step
- Phase 2B / Phase 3: `hipGraph` capture + `Q8_1` `v_dot4_i32_i8` / Wave32 repacked GEMV to surpass `123.32 tok/s` decode, then Bonsai 2.5 (`qwen35` + Hadamard + `PTQ1_0` ternary popcount) & Inkling 2 Mini (`inkling2_loop` + `io_uring` `O_DIRECT` MoE), plus Remotion ASCII/audio demo video from `phase2a_gfx1201.json`.
