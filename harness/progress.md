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

## 2026-11-27 (later) — `inkling2_loop` architecture contract in Rust, and FP8 feasibility
- `crates/engine/src/inkling2.rs` states the architecture as data: **42 tensor roles** with a per-role layer policy, the 48-visit loop schedule, and the hyperparameters the forward pass needs (`relative_length/extent/clip`, `qk_clip`, `log_scale_alpha/floor`, `situ_beta/linear_beta`, `attn_res.block_size`, `final_logit_softcapping`, `shortconv.l_cache`, expert FF/scale, `loop.gate_rank`). `scoria arch-check <gguf>` prints it against a real checkpoint.
- Reading it off the checkpoint corrected four assumptions: there is **no `attn_q_conv`**; sliding-window layers use one fused `attn_norm` while the four full-history layers normalise q/k/v separately and add `attn_{k,v}_conv_proj`; the looped block carries 13 learned gate tensors (`loop_{q,k,v,ffn}_ctrl_{gain,forget,down}` + `loop_msg_query`); and the expert `.scale` companions follow the quant format rather than the layer, so their absence on 4 layers is not a defect.
- Result on `Inkling-2-Mini-UD-Q3_K_XL`: **0 missing required roles**, schedule 6 + 36 + 6 = 48 visits, 48 KV slots. What is left before a forward pass is dequantisation: `Q5K`, `BF16`, `Q3K`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, and GGML type 40 (not in this tree's enum yet).
- FP8 feasibility settled by compiling probes for `gfx1201`: `v_cvt_pk_fp8_f32` (pack two f32 into two fp8) and `v_cvt_f32_fp8_e32` (decode), zero banned mnemonics. Combined with the gate's measured 383.76 TFLOPS FP8 WMMA, an FP8 path is available without new ISA work.
- 6 unit tests pin the schedule (visit 0..5 direct, 6..41 = the 12-layer block three times, 42..47 tail) and the layer-variant exclusivity (20 fused-norm vs 4 split-norm layers).

## 2026-11-27 (night) — dequantisation: 9 blocking types down to 4
- Implemented five GGUF dequantisers in `crates/gguf/src/dequant.rs`, transcribed from the ggml block layouts rather than guessed: **BF16**, **Q5_K** (176 B/256: d, dmin, 12 packed 6-bit scale/min pairs, 32-byte high-bit plane, 128 low nibbles), **Q3_K** (110 B/256: high mask, 2-bit quants, 16 six-bit scales unpacked from 12 bytes), **IQ4_NL** (18 B/32, 16-level non-uniform codebook), and **NVFP4** (36 B/64, four unsigned-E4M3 scales per 16-element sub-block + packed E2M1 nibbles).
- **GGML type 40 is identified and implemented as NVFP4.** It appears on 32 tensors of Inkling-2-Mini-UD-Q3_K_XL and measures exactly 4.5 bits/weight, which matches 8 bytes of E2M1 data plus one fp8 scale per 16 values. The reference's `values.view(rows, scale.shape[1], 16) * scale` path is the same format.
- `GgmlType` gained named variants for IQ2_XXS/IQ2_XS/IQ3_XXS/IQ4_NL/IQ3_S/IQ2_S/NVFP4 with ids 16/17/18/20/21/22/40, and `engine_supports` now covers 11 types. `scoria arch-check` on the checkpoint reports the remaining blocking work as exactly four types: **IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S** (all need ggml's grid/codebook tables).
- Five of the six new tests are hand-computed from the layout, not snapshots: BF16 sign/exponent/inf edge cases, IQ4_NL both nibbles against the codebook, NVFP4 per-sub-block scales (including a zero scale that must not become a subnormal), Q5_K's shifting high-bit plane, and Q3_K's 12-byte scale unpacking plus high-mask subtraction.

## 2026-11-27 (late) — every Inkling tensor is now loadable; zero blocking dequant types
- Implemented the four IQ dequantisers from ggml's published layouts: **IQ2_XS** (74 B/256: 9-bit grid + 7-bit sign code per 8 weights, two scales per 32-block), **IQ2_S** (82 B: two high grid bits from `qh`, signs in the second half of `qs`), **IQ3_XXS** (98 B: two 4-element grids per 8 outputs, scale+4 sign indices packed in one u32), **IQ3_S** (110 B: 4-bit scales, a 9th grid bit per output from `qh`, 32 bytes of sign masks).
- The grids and sign tables are ~2,300 constants, so they are **generated** from `ggml-common.h` by `scripts/gen-iq-tables.py` into `crates/gguf/src/iq_tables.rs`, with an FNV-1a digest that a test recomputes. Hand-editing a table fails the suite.
- `scoria arch-check` on Inkling-2-Mini-UD-Q3_K_XL now ends without a "blocking dequant work" line: **0 of 12 tensor types are unsupported** (F32, Q8_0, Q3K, Q4K, Q5K, Q6K, BF16, IQ2XS, IQ2S, IQ3XXS, IQ3S, IQ4NL, NVFP4).
- New `--dequant-smoke` dequantises 8 blocks of one real tensor per type straight from the mmap and reports the range. It surfaced something a synthetic test could not: expert tensors decode to ±1900 raw because the file stores them **pre-divided by a per-expert f32 outer scale** (~4e-5), and multiplying by that companion tensor lands them at ±0.08, matching the non-expert weights. Any loader that forgets the outer scale would be four orders of magnitude wrong.
- Tests: 20 lib tests (including two hand-computed IQ cases and a table-integrity digest) plus the existing adversarial GGUF suite. `./scripts/check.sh` green.

## 2026-11-27 (night, next) — inkling2_loop attention core as a tested CPU reference
- Fetched the authoritative implementation (`transformers/models/inkling2_loop/modeling_inkling2_loop.py`, on `main`) rather than inferring the math from tensor shapes, and transcribed the attention block's ordering into `crates/engine/src/inkling2_attn.rs`: **no RoPE**; Q is `scaled_l2_clip` with a learned per-head-per-dim scale; K and V pass a **4-tap causal depthwise short convolution** *before* the head split, K is then `l2_clip` and V an unweighted RMSNorm; a **hidden-states-conditioned relative position bias** replaces RoPE (per head: `(W_rel h) -> scaled clip -> @ proj[d_rel, rel_extent] -> gather at q_pos - k_pos`, zero outside the extent); attention is `q.k^T * head_dim^-0.5 + bias`, masked causal/window, **softmax with one extra zero-valued sink logit** that absorbs mass and is then dropped; the output is `W_o(attn * sigmoid(W_gate h))`.
- Ordering matters and is now pinned in code, not prose: clip/conv/norm/gate/sink in that sequence, with the full-attention path additionally applying the log-scaling `1 + alpha*ln(max((pos+1)/floor, 1))` to Q and the relative states.
- 12 unit tests, hand-computed where the arithmetic allows: L2 clip shrinking (3,4)->(1.2,1.6) at threshold 2 and leaving short vectors alone; log-scale = 1 up to the floor 8192 then `1 + 0.1*ln(r)`; relative-bias gather for distances 0/1/2 with future and past-extent keys zeroed; sink softmax giving 1/3+1/3 with a neutral sink and silencing attention at a large sink; the 4-tap conv chaining through a 3-value cache (t=0 -> 4, t=1 -> 11, t=2 -> 20, t=4 -> 40) and matching a streamed run; `decode_head` matching a naive full-row reference across windows {none, 6, 3, 1}; a window of 4 admitting positions 6..9 (mean 7.5); and an entirely masked row returning zeros instead of NaN.

## 2026-11-27 (night, later) — `.hip` decode attention, checked against the CPU oracle
- `inkling2_attn_decode_f32` in `crates/kernels/hip/infer_kernels.hip`: one wave per query head, online softmax over the KV cache with a host-computed relative bias, a causal window, and the per-head sink folded in as a virtual score (initialise `m = sink, l = 1`), which is exactly `softmax([scores, sink])` with the sink column dropped. Head-dim elements per lane is `head_dim / wave_size`, so it works on both Wave32 and Wave64.
- Contract made explicit after the test found it ambiguous: `window == 0` hides every position, `window == u32::MAX` is unbounded, anything between keeps `pos - t < window`. Previously `0` meant unbounded, which made "mask everything" inexpressible.
- `crates/engine/tests/inkling2_attn_gpu.rs` runs the same fixture through `inkling2_attn::decode_head` and the kernel and compares element-wise at `pos = 0, 7, 1100` with windows {none, 8, 3, 1, 1024, 17} — including `pos = 1100` past the 1024 relative extent, where the bias is zero. Tolerance `2e-4 + 2e-3|ref|`. Also asserts an all-masked row is exactly zero rather than NaN, and pins the host-side preprocessing helpers (relative-bias gather with a constant `proj`, 4-tap conv).
- Result: 3 GPU tests pass, 12 CPU attention tests pass, `./scripts/check.sh` green.

## 2026-11-28 — projection post-processing on device, and a measured long-context answer
- `inkling2_attn::post_qkv` (CPU) plus four `.hip` kernels now cover everything between the matmuls and attention: `inkling2_conv4_f32` (4-tap causal depthwise conv with the cache advancing in place), `inkling2_head_clip_f32` (per-head `l2_clip(x*scale)`, `scale == nullptr` for K), `inkling2_head_rmsnorm_f32` (unweighted, eps-safe), and `inkling2_sigmoid_f32` (the output gate).
- GPU-vs-CPU test runs **four chained decode steps** and compares q, k, v, gate **and both conv caches** at every step, so a cache that does not advance (or advances twice) fails immediately. Order is conv → clip/rmsnorm → gate, matching `modeling_inkling2_loop.py`.
- Long context was answered by measurement, not projection: Qwen3-8B **with FP16 KV assigns 65536 positions and runs**, while the same model at FP32 KV fails `hipMalloc` at 49152; `Scoria plan` shows Inkling needs 8 full-history KV slots (not 24) because 20 of its layers keep a 1024-token window, so 200k costs 3.13 GiB at FP8 against 12.52 GiB at FP32.

## 2026-11-28 (dawn) — layer 0 runs end-to-end on real `inkling2_loop` weights
- `crates/engine/src/inkling2_layer.rs` + `scoria layer0 <gguf>`: embedding row -> attention RMSNorm -> the six real projections read out of the checkpoint (`attn_q` Q6_K, `attn_k`/`attn_v`/`attn_gate`/`attn_rel` Q8_0, `attn_sink_proj` BF16, `attn_q_clip`/`attn_rel_clip` F32) -> `post_qkv` -> relative bias -> the verified decode attention -> output gate -> `W_o` -> the post convolution -> residual. All on dequantised f32 weights, through the same helpers the GPU kernels are checked against.
- First real numbers from the checkpoint (tokens 785, 6722, 315, 9625, 374):
  `|q|=49.5 |k|=29.8 |v|=32.000 gate_mean=0.377 |attn|=24.8 |post|=7.30 |residual|=48.6`, all finite, with `sink=[0.64, 1.29, 2.79, 0.61]` and head-0 attention values ~0.004-0.020.
- The tell that the pipeline is oriented correctly: **`|v|` is exactly `32.000` on every step**, which is `sqrt(8 kv heads x 128 dims)` — the unavoidable norm of an unweighted RMSNorm over 1024 elements. A transposed or misindexed projection would not land on that. Added as an invariant test over lengths 4/128/1024.
- `|k|` grows 29.8 -> 53.7 across the five steps because the 4-tap causal conv accumulates history, which is the expected behaviour and not a scale bug.
- Memory: the embedding table is read as raw bytes and one row is dequantised per token (39 MB) instead of dequantising all 201024 rows (1.85 GB); results are bit-identical before and after that change.

## 2026-11-28 — top-8 MoE on real weights, and layer 0 is complete except the block residual
- `crates/engine/src/inkling2_moe.rs`: router (f32 logits from `ffn_gate_inp`, `scores = sigmoid` by default, **selection on `scores + exp_probs_b.bias` but weights from the raw scores**, renormalised and scaled by 8.0), 128 expert slabs dequantised one at a time (Q3K/IQ2XS/IQ3XXS/IQ2S/IQ3S/NVFP4 all reachable) with the per-expert outer scale applied, softcapped SwiGLU (`tanh(g/4)*4 * sigmoid(g) * tanh(u/25)*25`, saturating to 24.98 not 25), `W2` per expert, weighted f32 accumulation.
- `scoria layer0 --moe` now runs the FFN half in the reference's order: `post_attention_layernorm` -> MoE -> `post_mlp_conv` -> residual add. Real output for tokens 785/6722/315: 8 distinct experts each, weights summing to 8.0, `|moe_out|` 122.2/97.0/143.5, `|layer_out|` 61.4/64.9/75.6, all finite.
- Three test failures found and fixed while wiring this, all in *my* expectations rather than the code: `tanh(100/25)*25` is 24.983 (I had assumed 25); a weighted RMSNorm does **not** have an exact `sqrt(n)` norm — only the unweighted one does, so the doc comment now says "within a few percent" instead of claiming an invariant; and the MoE probe initially skipped the pre-MoE normalisation and the post-MLP convolution entirely, which the reference's decoder layer makes explicit.
- 3 MoE unit tests: softcap saturation, the hand-computed SwiGLU value, and the routing rule that selection uses the biased score while weights use the raw score (a classic MoE bug).

## 2026-11-28 — layer 0 made structurally faithful: block boundary + mlp_input attention
- The reference's decoder layer is not "attention -> residual add -> FFN": at a **block boundary** (`layer % 6 == 0`, which layer 0 is) it appends its *input* to a list of block snapshots and leaves `residual` equal to the attention output, with **no residual add**. Then the FFN input is a **single learned query attending over those snapshots plus the current hidden state** (`mlp_input`), normed, not the residual stream. Both are now implemented.
- Effect on real weights: `|residual|` drops from 48.6 to 7.30 (=`|post|`, confirming no add happened), and the MoE input norm becomes 51.7/50.3/48.9 against `|ffn_norm.weight|` = 50.5.
- Expert selections shift accordingly (e.g. `[18,45,33,95,11,124,123,114]` for token 785), weights still sum to 8.0, all finite.

## 2026-11-28 — output head (`final_input` -> norm -> unembed -> softcap) implemented
- `crates/engine/src/inkling2_head.rs`: the final block-residual query (`output_res_query` over block snapshots + hidden, non-causal, 1 query), the `output_norm`, the unembedding matvec, and `tanh(x/30)*30`. The unembedding is 200058 x 2304 rows over the **unpadded** vocabulary (the GGUF pads the tensor to 201024), dequantised one row at a time because f32 would be 1.84 GB.
- Three tests, and two of them failed first on *my* arithmetic: `tanh(100/30)*30` is 29.92 not 29.99, and a 1-query attention over RMS-normalised snapshots is a convex combination — RMSNorm fixes the RMS, not the norm, so asserting `norm == 1` was wrong. Both now assert structure (bounded by `c`, odd, monotone; output inside the convex hull) plus one comparison against the definition, instead of hand-expanded decimals.
- 30 engine unit tests green, `./scripts/check.sh` green.

## 2026-11-28 — the full 24-layer stack runs: Inkling 2 Mini now produces correct logits
`scoria logits` runs all 48 visits (6 pre-loop layers, three passes over 6..17, 6 post-loop layers), the block attention residuals, the message gates, the final block-residual query, the output norm, the 200058-row unembedding and the logit softcap. One token at a time, 7.6 s/token single-threaded, all 200058 logits finite, bit-identical on a re-run.

**Evidence.** Next-token perplexity on held-out prose:
- 37 tokens of narrative: **1.317**, top-1 **34/37** (`p(" the")=0.95`, `p(" valley")=0.96`, `p(" mill")=0.97`).
- 58 tokens of factual prose: **2.298**, top-1 **50/58**; the misses are concentrated on the digits of `1969`, which split across tokens.

Greedy from "The capital of France is" gives `" Paris."` and then a multiple-choice scaffold, which is what an instruction-tuned model does with a bare factual prompt.

**Four layout bugs found on the way, all invisible to unit tests written against my own reading of the format:**

1. **Short-convolution taps were transposed and reversed.** The tensor is `PyTorch Conv1d.weight`, i.e. `weight[channel * 4 + tap]`, and it is *cross-correlation*: tap 0 weights the **oldest** token and tap 3 the current one. I had `weight[tap * channels + channel]` with tap 0 on the current token. The trained taps prove the layout: mean |w| per tap for `blk.0.ffn_post_conv` is `0.028, 0.038, 0.096, 0.438` under `channel*4+tap` and a flat `0.15, 0.15, 0.15, 0.15` under the transpose. Fixed in `short_conv1d_4tap`, `short_conv1d_4tap_dynamic` and the HIP kernel.
2. **`attn_rel_bias` was transposed.** The reference parameter is `[d_rel, rel_extent]`; the GGUF carries it as a GGML matrix with `in = d_rel`, so the flat index of `(r, distance)` is `distance * d_rel + r`. Reading `r * rel_extent + distance` transposed a 16x1024 bank and turned every attention bias into noise. Real data confirms it: column 0 decays smoothly with distance (`-0.02, 0.40, 0.22, 0.12, 0.085, 0.045, ...`).
3. **The gate/up split is `[gate; up]`, not interleaved.** The source checkpoint interleaves `w13`, but `--fuse-gate-up-exps` fuses the two separate tensors with `torch.cat([gate, up], dim=1)` (`conversion/base.py`), so a half split is correct. I briefly "fixed" it the wrong way and reverted after reading the converter.
4. **The first loop pass never ran.** `for pass in 1..passes` produced 36 visits, not 48. Caught by the visit count.

Also fixed: the determinism check compared different token positions (it now replays the prefix and compares the same position), and the depth-sweep budget reset the state before checking it.

**Diagnostics kept**: `--max-visits N` (truncate the stack, for localising a bad layer), `--router-softmax` (the checkpoint does not name the router activation; sigmoid is the default because `exp_probs_b.bias` is a noaux-tc feature), per-visit `|out|`, and head-0 attention peak/sink/entropy per visit.

**Honest state:** correctness is now established behaviourally, not against the reference implementation — that needs the 20.2 GB HF checkpoint plus torch/transformers, neither of which is on this machine. Speed is untouched: 7.6 s/token, single-threaded, no kernels involved. There is still no Strata tiering, no 200k-context run, and no measured decode/prefill number for Inkling.

## 2026-11-28 — Inkling 2 Mini: 4.1x faster, and the first measured decode/prefill number
The host path went from **7.6-8.0 s/token to 1.84 s/token prefill and 1.93 s/token decode** on `gfx1201`'s CPU side, with perplexity unchanged to the digit (1.317 over 37 tokens, top-1 34/37) and "The capital of France is" still giving `" Paris."`.

| configuration | s/token |
| --- | --- |
| one thread, whole-expert f32 slab (start of this work) | 7.6-8.0 |
| 4 threads, f32 slab | 4.66 |
| 8 threads, f32 slab | 4.64 |
| 1 thread, row-wise dequant + threaded head | 6.47 |
| 8 threads, row-wise dequant + threaded head | 2.08 |
| 8 threads, + concurrent projections | 1.87 |
| 16 threads, + concurrent projections | **1.84** |

Four changes, in order of payoff:

1. **The MoE stopped materialising experts as f32.** Each expert was dequantised into a 21 MB (gate/up) + 11 MB (down) f32 buffer before one matvec: 32 MB of write traffic per expert, 12 GB per token across 8 experts and 48 visits, and the single largest cost. `ExpertTensor::row_matvec` now dequantises **one row at a time** into a 9 KB buffer that stays in cache and dots it immediately. Single-threaded this is *slower* (5.00 s vs 2.54 s, the per-row `dequant_row` call overhead), but it is 3.5x faster once parallel (0.78 s vs 2.68 s at 8 threads) because the remaining work is arithmetic, not bandwidth.
2. **The routed experts run on threads.** They are independent: each dequantises its own rows, runs its own SwiGLU and matvecs, and returns a weighted partial sum. `std::thread::scope` over the top-8 split, per-thread scratch reused across calls.
3. **The unembedding runs on threads.** 200058 rows x 2304 is 322 MB of quantised weights read per token -- about a second on one core. Rows are independent, so they are split across threads (head time ~1.0 s -> ~0.3 s).
4. **The six projections of each visit run concurrently**, one spawn round per visit rather than one per matvec, so the spawn overhead is paid 48 times per token instead of 336.

**Then the profile, which is what decided the order:** `scoria logits` now reports attention vs MoE time per token. It went MoE 66% -> 47%, attention 0.99 s, MoE 0.87 s at 16 threads.

**What this is not:** there are no GPU kernels in this path at all, prefill and decode are the same rate because the host runs one position at a time, and the first token after load is slower (it pulls 12.7 GB of weights through the page cache). The measured 1.84 s/token is the honest warm number.

## 2026-11-28 — the projections move to `gfx1201`: 0.97-1.00 s/token prefill, 0.99-1.11 s/token decode
Eight of the nine per-visit GEMVs (q, k, v, gate, rel, sink, `W_o`) now run on the card through the already-verified `linear_f32` kernel, with 2.40 GiB of f32 projection weights resident in VRAM. Everything else stays on the host.

| stage | s/token |
| --- | --- |
| start of the threading work, host only | 7.6-8.0 |
| host, 16 threads (end of last turn) | 1.84 |
| projections on GPU (q/k/v/rel/sink/`W_o`) | 1.28 |
| + gate projection on GPU | **0.97-1.00 prefill, 0.99-1.11 decode** |

Correctness is unchanged, and the host path is the oracle: **max |dlogit| = 2.8e-5, argmax agrees**, perplexity 1.317 over 37 tokens with top-1 34/37, and "The capital of France is" still gives `" Paris."`. That tag is a genuine cross-implementation check: the same weights, the same f32 arithmetic, only the wave-reduction order differs.

**The bug worth remembering.** The first attempt segfaulted inside `libamdhip64` and then, once the host crash was gone, raised `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` from the kernel. `hipModuleLaunchKernel`'s `kernelParams` entries are the **address of each argument, pointer arguments included**: passing the device pointer *value* makes HIP read the kernel argument out of device memory on the host. qwen2's `hip_launch!` hides this behind `KernelArg::as_arg_ptr` (`(self as *mut *mut c_void)`), so hand-rolled launches must write `&mut ptr` and not `ptr`. `crates/engine/tests/inkling2_gpu_gemv.rs` now pins the launch convention against the host matvec at `1x4`, `8x1024` and the real `3072x2304`.

**Where the time is now:** MoE 0.79 s of 0.97 s (87%). Attention is 0.11 s, of which 0.05 s is GPU. The remaining MoE is *not* a launch away: the experts are IQ2XS / IQ3XXS / NVFP4, and no device dequantiser for those exists yet, so a grouped expert GEMV needs either three new quant codecs in `.hip` or an f16-resident expert cache (Strata's residency policy applied to experts).

## 2026-11-28 — NVFP4 expert GEMV on the card: 0.47-0.51 s/token, 16x from the host baseline
The MoE is no longer host-only. `linear_nvfp4` decodes and dots the routed experts' rows on `gfx1201`, with 10.24 GiB resident: 2.4 GiB of projection weights plus the NVFP4 expert tensors.

| stage | s/token |
| --- | --- |
| start of the threading work, host only | 7.6-8.0 |
| host, 16 threads | 1.84 |
| projections on GPU | 0.99 |
| **NVFP4 experts on GPU** | **0.47-0.51 prefill, 0.47-0.58 decode** |

Correctness holds: perplexity **1.317** over 37 tokens (top-1 34/37), `" Paris."`, repeat run bit-identical, and **GPU-vs-host max |dlogit| = 3.1e-5 with the same argmax**. The host path is still the oracle and still runs (`--no-gpu`).

**The codec.** NVFP4 is 36 bytes per 64 elements: four unsigned-E4M3 scales, one per 16-element sub-block, then 32 bytes of packed E2M1 nibbles, low nibble = element `s*16 + j`, high nibble = `s*16 + 8 + j`, and the scale is *halved* because the E2M1 table is doubled. Transcribed from `dequant_row_nvfp4` and pinned by `tests/inkling2_gpu_nvfp4.rs`, which compares the kernel against the CPU dequantiser **on real checkpoint rows** (`blk.5` gate_up at k=2304, `blk.0` down at k=1152) -- synthetic blocks can be built with one shared scale per 64 elements, which would hide a sub-block ordering bug.

**Batching was the other half.** The first version launched one GEMV per expert and synchronised after each: 0.28 s of GPU time for work that takes 0.04 s batched. `Gpu::run_batch` now uploads every routed input in one copy, launches one kernel per expert, and synchronises **once** per phase -- two synchronisations per visit instead of sixteen.

**What is still on the host, and why.** The checkpoint mixes expert formats: NVFP4 on 12 of 24 `gate_up` and 20 of 24 `down` tensors, with IQ2XS, IQ2S, IQ3XXS, IQ3S, Q3K and IQ4NL making up the rest. Those have no device dequantiser, so those tensors still run the threaded host loop -- which is the 0.30 s MoE that remains. IQ4NL is the easy next one (16-level codebook, 18 bytes per 32 elements); IQ2XS/IQ3XXS need the grid-plus-signs tables.

**Residency, honestly.** This is a fixed residency set, not a policy: `upload_experts` keeps every tensor the device can decode and gives up per tensor if VRAM runs out. There is no LRU, no prefetch, no demotion -- that is still Strata's missing half.

## 2026-11-28 — windowed KV ring, and the first long-context run (1100 tokens)
The 48 visits used to append to 48 unbounded `Vec`s: 393 KB of KV per token, 78 GB at 200k positions. Windowed visits now store a **ring** of `window` slots, and the ring is verified against the unbounded path rather than argued for.

| measurement | value |
| --- | --- |
| 300-position run | 0.51 s/token, KV 160 MiB windowed + **18.8 MiB full = 65536 B/token** |
| 1100-position run | **0.71 s/token**, KV 320 MiB windowed (capped) + 72 MiB full, RSS **13.73 GiB** |
| ring vs unbounded, window 64, 150 tokens | **0 differing logits** (bit-for-bit), 86 evictions |
| ppl / greedy / GPU parity at 37-5 tokens | 1.317, `" Paris."`, max |dlogit| 3.1e-5 (unchanged) |

**The layout.** 40 windowed visits (20 sliding layers, minus the full-history ones, across three loop passes) each hold `window = 1024` slots x 2 tensors x 1024 values x 4 B = 320 MiB in total, and it does not grow with the sequence. The 8 full-history visits keep everything: `2 x 1024 x 4 = 65536 B/token`, which matches `scoria plan`'s projection exactly.

**At 200k positions that is 13.1 GB of full-history KV + 320 MiB of rings**, i.e. ~13.4 GB (6.7 GB at FP16), against 31 GB of RAM. So the *memory* now fits; the *run* does not in any reasonable time -- 0.71 s/token at 1100 and rising with context puts 200k near a day. The largest context actually measured is 1100 tokens.

**Why the ring is exact, not approximate.** Sliding visits only ever attend to the last `window` positions, so evicting older ones cannot change the answer -- provided absolute positions, masks and the relative-bias bank all still agree. The store keeps absolute position numbers and maps them modulo the window, and the visible range is trimmed once (`max(retained, pos+1-window)`) before both the bias and the attention, so the ring and the mask can never disagree. `tests/inkling2_kv_ring.rs` runs the same 150 tokens twice with a 64-token window, once with eviction and once with the unbounded store, and requires **identical** logits. A 1024-token window would need 1024+ tokens per run to expose the same bug.

**What this does and does not do for Strata.** The windowed ring *is* an eviction policy, and the expert residency set *is* a residency policy; both are now real. What is still absent is a **tiered** store with LRU, prefetch and demotion. For full-history KV that is not an oversight: those 8 visits read every earlier position on every token, so paging them would mean 13.1 GB of reads per token -- a policy cannot fix that, only a different attention algorithm (sparse or chunked) could.

## 2026-11-28 — batched prefill, and a correction: the perplexity I reported was double-fed
**Correction first.** `scoria logits --perp` forwarded the tokens once to print per-step output and then scored them *again without resetting the KV cache*, so the model was attending over its own earlier pass. That is why it reported **1.317** for prose whose true perplexity is **7.598** (2.028 nats, top-1 16/37). The fix is one line (`model.reset()` before scoring) and the number is now 7.598 on the same text. This affects three earlier entries' perplexity figures; it does **not** affect the model itself — the `" Paris."` generation and the GPU-vs-host `max |dlogit| 3.1e-5` parity both stand, because the parity check replays a prefix once.

The independent confirmation is the new prefill path: written separately, it produced **7.598** as well.

**Batched prefill.** `Model::prefill(tokens, chunk)` carries one vector per position through all 48 visits, with per-position block snapshots and message context, causal attention per position, and the expert GEMVs **gathered by expert** so an expert is dequantised once for every position that routed to it.

| chunk | vs single-position path |
| 1 | max |dlogit| 5.3e-5, argmax agrees 38/38 |
| 2 | 4.4e-5, 8/8 |
| 4 | 4.4e-5, 8/8 |
| 8 | 4.4e-5, 8/8 |

Perplexity is identical for chunk 1 and chunk 8 (7.598), which is the property that matters: batching changes nothing but the order of a few additions.

**Bug found while verifying:** the chunked path skipped loop **pass 0** — the same bug I had already fixed in the single-position path and reintroduced here. Caught only because the chunked and single paths disagreed at loop layer 6; per-visit hidden norms matched exactly up to that point.

**Measured, honestly:** the host batched MoE is **single-threaded** (`Moe::forward_chunk` has no thread pool yet), so the host prefill path is 5.2 s/token — slower than the threaded single-position path. The batched *GPU* path is what this change exists for, and it is measured separately below. The 4096-token run was killed at the 90-minute timeout while sharing the machine with these measurements, so the largest context that has actually completed remains **1100 tokens**.

## 2026-11-28 — prefill measured as its own path; it is slower than decode, and here is why
`scoria logits --prefill-chunk N` runs a batched prefill and reports prefill tok/s separately from decode.

| chunk | prefill s/token | prefill tok/s | perplexity |
| 1 | 1.4-2.3 | 0.43-0.73 | 7.598 |
| 4 | 1.3-1.7 | 0.58-0.75 | 7.598 |
| 8 | **0.70-0.83** | **1.20-1.43** | 7.598 |

Decode is 0.47 s/token (2.1 tok/s), so **prefill is currently slower than decode** — the honest result, and the reason is not mysterious:

* **The expert batching barely helps, and arithmetic says it cannot.** A chunk of 8 positions makes 64 expert draws over 128 experts, so the expected number of *distinct* experts is ~50 and the average positions-per-expert is 1.28. Dequantising once per expert instead of once per position therefore saves ~20%, not 8x. The gather overhead costs more than that.
* **The projections are where batching should pay, and they are still per-position.** Each visit still issues six single-row GEMVs with their own launch and synchronise per position. That is measurable: per visit the chunk path spends 0.003 s on the attention half against decode's 0.0021 s, and 0.012 s on the MoE against 0.0058 s.

So the next increment is not more MoE batching — it is the projections with `m = chunk`, which have *perfect* reuse (the same weights for every position in the chunk).

**Two bugs found and fixed while building this:**
1. The chunked path skipped loop **pass 0** (the same bug as the single-position path had, reintroduced).
2. The GPU down-phase gather did not reset its batch `offset`, so the down outputs were written past the downloaded region and the caller read back **stale gate/up values**. This is why the GPU chunk path produced perplexity 20008 while the host chunk path gave 15.9 on the same text: the chunk path had only ever been verified with `--no-gpu`.

**Verified now, on both paths:** chunk 1/2/4/8 vs the single-position path agree to **max |dlogit| 3.3e-5 with the argmax matching at every position**, and perplexity is 7.598 at every chunk size on the GPU path and 15.903/15.904 on the short text. `./scripts/check.sh` green, with the heavy model-level test moved to a release step (it was stalling the debug suite for over an hour).

## 2026-11-28 — prefill: the gather cannot win, and the real bottleneck named
I did **not** implement the projection batching this step called for. I went after the measured bottleneck instead, and the measurement changed the conclusion.

**Measured on an idle-ish slice of a shared machine** (a ComfyUI server has held a core for 1.5 days and load average sits at ~9.5, so sub-second timings carry ~2x noise while decode is stable):

| | s/token |
| decode | 0.45-0.47 (stable across runs) |
| prefill chunk 1 | 0.97-1.46 |
| prefill chunk 8 | 1.02-1.82 |
| prefill chunk 32 | 1.27 |

**Prefill is still slower than decode, and batching the experts cannot fix it.** A chunk of 8 positions makes 64 expert draws over 128 experts: ~50 distinct, i.e. **1.28 positions per expert**. Gathering therefore saves ~20% of dequantisation while paying host-side copying for 64 slices. Confirmed by direct comparison at the same chunk size: gathered 0.70 s/token vs per-position 1.47 s/token earlier in the session — gathering wins, but only by what the reuse allows.

**What did work:** the chunked host MoE was *allocation-bound*, not arithmetic-bound. Sixteen workers each allocated ~73k floats per layer visit (≈300 MB of zeroing per position), which made it 9x slower per layer than the per-position path. A reusable `ChunkScratch` pool cut the MoE slice from 0.054 s to 0.017 s per visit and chunk-8 prefill from ~2.5-3.0 to ~1.0 s/token. Perplexity unchanged (7.598 / 15.903), oracle agreement unchanged.

**The bottleneck both paths now point at is the same:** 11 of 24 layers store their experts in formats with no device dequantiser (IQ2XS, IQ2S, IQ3XXS, IQ3S, Q3K, IQ4NL) and fall back to the host. Decode spends 0.28 s/token there; the chunked prefill spends 0.82. Expert batching and prefilling both redistribute work *around* that; only a device codec removes it.

Also in this change: `CHUNK_MAX` raised to 64; `Moe::forward_chunk` takes `&mut self` so its scratch survives across calls; `moe_forward_parts` now takes `&Moe`.

## Parity gate calibration (cosine > 0.999 is prompt-dependent, not an implementation defect)
- Measured the control that settles it: **llama.cpp's own Vulkan backend vs llama.cpp's own CPU backend** scores `cos = 0.997985` (`maxdiff 1.152`) on the Qwen3-4B harness chat prompt (`151644,872,198,785,…,77091,198`, `m=13`), where scoria scores `0.997652` (`maxdiff 1.103`). Both pick top1 `785`. No GPU implementation reaches 0.999 on that prompt.
- Forcing prefill onto the scalar FP32 GEMM (`SCORIA_PREFILL_FP32=1`) changes nothing: 4B `0.997614` vs `0.997652` for FP16 WMMA, 8B `0.996415` vs `0.996404`. So the deficit is **not** FP16 WMMA rounding.
- Cosine is not monotonic in prompt length for the same tokens (`m=2 → 0.9847`, `m=4..8 → 0.9992–0.9997`, `m=10 → 0.9961`, `m=11 → 0.9994`, `m=12/13 → 0.9977`); it tracks how peaked the distribution is (`max_logit` 39.0 at `m=13`), not error accumulation.
- Conclusion: gate on `top1_match` plus greedy-prefix agreement, and treat `cos > 0.999` as a per-prompt sanity check rather than an absolute bar. Current status: every model/prompt tested has `top1_match=true`.

## Next Step
- Phase 2B / Phase 3: `hipGraph` capture + `Q8_1` `v_dot4_i32_i8` / Wave32 repacked GEMV to surpass `123.32 tok/s` decode, then Bonsai 2.5 (`qwen35` + Hadamard + `PTQ1_0` ternary popcount) & Inkling 2 Mini (`inkling2_loop` + `io_uring` `O_DIRECT` MoE), plus Remotion ASCII/audio demo video from `phase2a_gfx1201.json`.
