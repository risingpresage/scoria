# Phase 2B Handoff Brief — Modern Qwen3 Baselines & Surpassing `llama.cpp` Vulkan on `gfx1201`

This document gives the next session/model everything needed to execute Phase 2B without re-discovering repository constraints, hardware quirks, or target baselines.

## 1. Non-Negotiable Constraints & Environment

- **Repo**: `/home/noorm/projects/scoria` (commit all work on branch **`dev`**, never `main`).
- **Host / Kernel Split**: Host and runtime are pure Rust (`crates/{hip-rt,kernels,gguf,engine,cli}`); GPU compute lives strictly in standalone `.hip` files under `crates/kernels/hip/`. Zero `.cpp`/`.cc`/`.cxx` files under `crates/`.
- **Device Compilation**: Never call `hipcc --genco` (silently emits `gfx906`). Kernels are compiled in `crates/kernels/build.rs` via:
  `/opt/rocm/lib/llvm/bin/clang++ -x hip --offload-device-only --no-gpu-bundle-output --offload-arch=<arch> -O3 --rocm-path=/opt/rocm`
  Targeting both **`gfx1201`** (RDNA 4, Wave32, `#ifdef SCORIA_RDNA4`) and **`gfx942`** (CDNA 3, Wave64).
- **Banned `gfx1201` Mnemonics** (enforced by tests + `scripts/lint-arch.sh`): `ds_load_tr*`, `v_bitop3_b32`, `v_cvt_pk_f16_fp8`, `v_wmma_scale_*`.
- **ROCm `malloc` Alignment Bug**: `libamdhip64.so.7` emits 32B `vmovdqa %ymm1` against glibc `malloc`; solved in `crates/hip-rt/src/{alloc_guard.rs,ffi.rs}` + `.cargo/config.toml`.
- **Crate DAG**: `scoria-hip-rt` and `scoria-gguf` have zero `scoria-*` deps; `scoria-kernels` never depends on `scoria-engine`/`scoria-cli`.
- **Disk Hygiene**: `/home` has ~82 GB free; `/mnt/nuclearmissilesilo/models` has ~613 GB free. Download any new GGUFs to `/mnt/nuclearmissilesilo/models/` and symlink into `/home/noorm/models/`.
- **Hardware Gate Reference (`gfx1201` RX 9070 XT, 16 GiB GDDR6, 64 CUs, Wave32)**:
  - Observed DRAM stream: `561.66 GB/s` (baseline), `560.09 GB/s` (`TH_LOAD_NT`)
  - L3 probe (`TH_LOAD_HT`): `2275.64 GB/s`
  - Dense `v_wmma`: FP16 `207.53 TFLOPS`, FP8 `383.76 TFLOPS`, IU4 `763.95 TOPS`
  - Register LUT (`v_permlane16_var_b32`): `1824.35 G-lookups/s`; BITCOS popcount (`v_bcnt_u32_b32`): `4321.57 G-weights/s`
- **Verification Gate**: Every edit must pass `./scripts/check.sh` (`cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `scripts/lint-arch.sh`, and `cargo test --workspace`).

## 2. Current Verified Baseline (Phase 2A, Commit `3eeecc9`) vs. Modern Targets

All models below are already on disk. `llama-bench` (`llama-b11050`, Vulkan RADV, `ngl=99`, `r=3`) establishes the exact target to beat on RX 9070 XT:

| Model | Path | Arch | Size | `llama-bench` `pp128` | `llama-bench` `tg64` | `scoria` Phase 2A (`pp128` / `tg64`) |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| **Qwen3-4B-Instruct-2507-Q4_K_M** | `/home/noorm/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | `qwen3` | 2.50 GB | **5826.04** tok/s | **189.32** tok/s | Target (Step 2B.1) |
| **Qwen3-8B-Q4_K_M** | `/home/noorm/models/spec-lab/Qwen3-8B-Q4_K_M.gguf` | `qwen3` | 5.03 GB | **3902.77** tok/s | **111.73** tok/s | Target (Step 2B.1) |
| **DeepSeek-R1-Distill-Qwen-7B-Q4_K_M** | `/home/noorm/models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | `qwen2` | 4.68 GB | **4151.29** tok/s | **121.97** tok/s | `2079.89` / `111.94` tok/s |

Key hyperparameters already inspected from disk:
- **Qwen3-4B-Instruct-2507-Q4_K_M**: `36` layers, `n_embd=2560`, `n_ff=9728`, `n_heads=32`, `n_kv_heads=8`, `key_length=value_length=128` (`q_dim = 32*128 = 4096 != n_embd`), `rope.freq_base=5000000.0`, tensor types `{F32: 145, Q4_K: 216, Q6_K: 37}`.
- **Qwen3-8B-Q4_K_M**: `36` layers, `n_embd=4096`, `n_ff=12288`, `n_heads=32`, `n_kv_heads=8`, `key_length=value_length=128`, `rope.freq_base=1000000.0`, tensor types `{F32: 145, Q4_K: 217, Q6_K: 37}`.

Additional modern models optionally available later (download to `/mnt/nuclearmissilesilo/models/`): `Qwen3-14B-Q4_K_M` (~9 GB, 16 GB VRAM fit) or `Qwen3-30B-A3B-Q4_K_M`.

## 3. Verified Kernel Contracts Already Built

- **`v_wmma_f32_16x16x16_f16` C Fragment Layout (`gfx1201`, silicon-verified `err_a = 0.000000`)**:
  - `a_frag[i] = A[m = lane & 15, k = (lane >> 4) * 8 + i]`
  - `b_frag[i] = B[n = lane & 15, k = (lane >> 4) * 8 + i]`
  - `c_frag[i] = C[m = (lane >> 4) * 8 + i, n = lane & 15]` (Map A)
- **Alignment Traps**:
  - `Q6_K`: `210` bytes/block -> only **2-byte aligned** (never cast to `uint2*` or `uint4*`; use `unsigned short`).
  - `Q4_K`: `144` bytes/block -> strictly **16-byte aligned** (`uint4` safe at `blk + 0` and `blk + 16 + lane * 16`).
  - `Q5_0` (`22B`) and `Q8_0` (`34B`) -> **2-byte aligned**.

## 4. Phase 2B Optimization Plan (C-A-C Hill-Climb)

Follow `harness/THEORY.md` + `scripts/bench_cac.py` -> `harness/ledger.jsonl` (keep a change only when 3-trial median improves and `scripts/parity_oracle.py` holds `cosine_sim > 0.999` and `top1_match == true`).

### Step 2B.1 — Add `qwen3` Architecture (~60 LOC delta over `qwen2`)
1. `crates/gguf/src/gguf.rs`: accept `"qwen3"`, read `{arch}.attention.key_length` for `head_dim` (`n_embd / n_heads` is **wrong** for Qwen3-4B where `2560 / 32 = 80` vs `key_length = 128`).
2. `crates/kernels/hip/infer_kernels.hip`: add per-head `qk_rmsnorm_f32` (or fused `qk_rmsnorm_rope_neox_f32` using `float4` on Wave32 for `head_dim = 128` across `attn_q_norm.weight` and `attn_k_norm.weight`), plus register in `crates/kernels/src/lib.rs`, `crates/engine/src/lib.rs` (`define_infer_kernels!`).
3. `crates/engine/src/qwen2.rs`: load optional `blk.{l}.attn_q_norm.weight` / `blk.{l}.attn_k_norm.weight`, allow `qwen2 | qwen3`, apply per-head Q/K RMSNorm before RoPE.
4. Update `scripts/bench_cac.py` `DEFAULT_MODELS` to include `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` and `Qwen3-8B-Q4_K_M.gguf` (keep 7B optional or `--legacy`).
5. **Gate**: verify `parity_oracle.py` (`cos > 0.999`, `top1_match == true`, greedy prefix) on both Qwen3 models; log `phase2b_qwen3_baseline` in `harness/ledger.jsonl`.

### Step 2B.2 — Decode Throughput (`> 112 tok/s` Qwen3-8B, `> 190 tok/s` Qwen3-4B)
Phase 2A 7B 1-step `--profile` breakdown:
`gpu_sum = 10.854 ms` (`gemv_q4k/other = 6.655 ms`, `q6k_layers = 1.581 ms`, `lm_head = 0.746 ms` at 597 GB/s, `rmsnorm = 0.581 ms`, `attn = 0.602 ms`, `other = 0.689 ms`).
1. **Kernel Fusion (`Step 2B.2a`)**:
   - Fuse `qk_rmsnorm + rope_neox + kv_cache_append` into one launch per layer (cuts 72 launches/step on 36-layer Qwen3).
   - Fuse `ffn_gate + ffn_up + silu_mul` (reads `normed` once from VRAM/L2, cuts 72 launches/step).
2. **Wave32 Repacking (`Q4W32`) + `Q8_1` Dot GEMV (`Step 2B.2b`)**:
   - Currently `gemv_q4k/other = 6.655 ms` achieves ~435 GB/s vs `561.66 GB/s` GDDR6 peak because `Q4_K` has mixed `d`/`dmin`/`scales[12]` headers and `Q5_0`/`Q8_0` are unaligned (`22B`/`34B`).
   - Add a const-generic CPU loader/repacker in `crates/engine` converting GGUF matrices at model load into 256B-aligned Wave32 SoA strips (separate contiguous scale/min buffer loaded via `TH_LOAD_HT` + contiguous coalesced weight payload) and quantize activations per-token into `Q8_1` (`v_dot4_i32_i8` / `v_dot8_i32_iu4` on RDNA 4) so GEMV runs at `>= 530 GB/s`.
   - Keep `Q6_K` `gemv_q6_k_f32` or repack `lm_head` for extra speedup on 151k/248k vocabs.
3. **`hipGraph` Decode Capture (`Step 2B.2c`)**:
   - Expose `hipStreamBeginCapture`, `hipStreamEndCapture`, `hipGraphInstantiate`, `hipGraphLaunch`, `hipGraphDestroy` in `crates/hip-rt/src/{ffi,lib}.rs`.
   - Put decode inputs (`u32* d_token`, `u32* d_pos`, `argmax_out`) in fixed device slots and capture one 36-layer decode graph per sequence (attention reads `*d_pos` directly from device memory, tiny epilogue kernel copies `argmax_out -> d_token` and increments `*d_pos`).
   - **Mandatory leak guard**: run a 1000-replay `hipMemGetInfo` check (ROCm graph instantiation/exec leak check).

### Step 2B.3 — Prefill WMMA (`> 3903 tok/s` Qwen3-8B, `> 5826 tok/s` Qwen3-4B on `pp128`)
Why Phase 2A `gemm_wmma_*` hit `2079.89 tok/s` vs `llama.cpp` `4151.29 tok/s` on 7B:
- Small macro-tile (`BM=32, BN=64, BK=32`) gives only `1x1` WMMA per wave (low arithmetic intensity and frequent LDS barriers every `K=32`).
- `WmmaLoaderQ4_K` re-reads the 16B block header 8 times across `K=256`.
Fixes:
1. Move to **`BM=64, BN=128, BK=64`** (or `128x64x64`) with **`2x2` (4 WMMA accumulators/wave)** or `2x4`, reading one `Q4_K` superblock (`256` weights) every 4 `K=64` tiles with **double-buffered LDS** (`s_x[2][64][68]`, `s_w[2][128][68]`) so global loads overlap WMMA math.
2. Load both low and high nibbles from a single `uint4` into consecutive `s_w` halves.
3. Add a `pp128` profiler (`--profile-prefill`): if `gqa_attn_f32` becomes >10% of `pp128`, replace scalar prefill attention with tiled `v_wmma` FlashAttention-2.

### Step 2B.4 — Verify & Commit
- Parity (`cosine_sim > 0.999`, `top1_match == true`) on `Qwen3-4B`, `Qwen3-8B`, and `DeepSeek-R1-Distill-Qwen-7B`.
- Save `harness/baselines/phase2b_gfx1201.json`, update `harness/{PLAN.md,features.json,progress.md}`.
- Run `cargo fmt --all && ./scripts/check.sh` and commit on `dev`.

## 5. Key File Map

- `crates/hip-rt/src/{ffi,lib,alloc_guard}.rs` — `dlopen` HIP FFI, `KernelHandle`, 32B `malloc` guard
- `crates/kernels/hip/{arch_compat.h,gate_kernels.hip,infer_kernels.hip}` — standalone `.hip` kernels (`gfx1201` + `gfx942`)
- `crates/kernels/{build.rs,src/lib.rs}` — `clang++` cross-compiler + `llvm-objdump` ISA tests
- `crates/gguf/src/{gguf,dequant,tokenizer}.rs` — zero-dep mmap GGUF parser + CPU ref dequant + BPE
- `crates/engine/src/{lib,qwen2}.rs` — `define_infer_kernels!`, `hip_launch!`, `ExecutionGraph`, `VramTensorStore`, `Qwen2Model`, `gpu_wmma_prefill_parity`
- `crates/cli/src/{main,gate,uring}.rs` — `scoria {gate,inspect,tokenize,run}` CLI + `io_uring` `O_DIRECT`
- `scripts/{init,lint-arch,check}.sh`, `scripts/{bench_cac,parity_oracle}.py`
