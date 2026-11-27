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

## Next Step
- Phase 1: Full GGUF tensor/metadata/tokenizer loader + reference HIP inference kernels (`Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` -> `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf`) with greedy logit parity against `llama.cpp`.
