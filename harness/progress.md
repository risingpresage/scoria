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

## Next Step
- Phase 2: Load-time Wave32 repacking (`H4F8` / `Q4W32` / `BITCOS` / `PTQ1_0`), `v_wmma` prefill GEMM, `v_permlane16_var_b32` / popcount decode, and `hipGraph` capture targeting `> llama.cpp` Vulkan on RX 9070 XT.
