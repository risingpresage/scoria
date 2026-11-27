# scoria — Agent Operating Guide

RDNA 4 (`gfx1201`) AMD-native LLM inference engine in Rust + standalone `.hip` kernels.

## Commands
- Env check: `./scripts/init.sh`
- Full verification gate (must pass before claiming done): `./scripts/check.sh`
- Hardware gate benchmark: `cargo run --release -p scoria-cli -- gate`

## Hard Constraints
- **Host/runtime is Rust; GPU kernels are standalone `.hip` only.** Zero `.cpp`/`.cc`/`.cxx` host files under `crates/`.
- **Never invoke `hipcc --genco`** (silently drops `--offload-arch=gfx1201` and emits `gfx906`). Compile device code only via:
  `/opt/rocm/lib/llvm/bin/clang++ -x hip --offload-device-only --no-gpu-bundle-output --offload-arch=gfx1201 -O3 --rocm-path=/opt/rocm`
- **Banned `gfx1250`/unsupported mnemonics on `gfx1201`:** `ds_load_tr*`, `v_bitop3_b32`, `v_cvt_pk_f16_fp8`, `v_wmma_scale_*`.
- **Crate DAG:** `scoria-hip-rt` depends on no `scoria-*` crate; `scoria-gguf` has zero GPU dependencies; `scoria-kernels` never depends on `scoria-engine` or `scoria-cli`.
- **No unverified claims:** Every status update in `harness/features.json` or replies must cite observed output from `./scripts/check.sh` or `scoria gate`.

## Writing & Disclosure
This rule exists because it was violated once and had to be fixed twice.

- **Sell the engine, then the gap.** Every README/status/report opens with what runs and what it measures. Gaps come second, framed as the current frontier with the specific next step, never as a list of faults.
- **No `## Limits` sections.** Use `## Status` with three groups: running and measured / in flight (with the next step) / next up. A "Limits" heading structurally forces a bug-report voice and makes a working system read as broken.
- **Cite evidence to support a claim, not to atone for it.** "No unverified claims" means do not overclaim. It does not mean broadcast every weakness.
- **A stated trade-off is a decision, not an apology.** Write "runs the scalar GEMM by choice: FP16 WMMA costs 1.5e-3 cosine there", not "does not support WMMA for short prompts".
- **Match the house style of the account's other repos** (`kv-serving-scheduler`, `llama-vulkan-matvec-rdna4`, `cat-burglar`): name the repo, one factual paragraph, a measured table that cites its source file, then what changed and why. No hero badges, no marketing adjectives.

## Progressive Disclosure
- Architecture & 3 seams: [ARCHITECTURE.md](specs/ARCHITECTURE.md)
- Verified `gfx1201` ISA & intrinsics: [GFX1201_ISA.md](specs/GFX1201_ISA.md)
- Native weight & NVMe format: [FORMAT.md](specs/FORMAT.md)
- Verification & parity protocol: [VERIFICATION.md](specs/VERIFICATION.md)
- Session ledger: [features.json](harness/features.json), [progress.md](harness/progress.md)
- Workflows: [kernel-authoring](.agents/skills/kernel-authoring/SKILL.md), [parity-bench](.agents/skills/parity-bench/SKILL.md)
- Attention family (MLA / DSA / NSA / LSA) and the long-context plan: [ATTENTION.md](specs/ATTENTION.md)
- Capability parity vs llama.cpp and the order to close it: [PARITY.md](harness/PARITY.md)

## Mistakes Log
- 2026-11-26: `hipcc --genco --offload-arch=gfx1201` fails builtins because clang invokes `--cuda-device-only` targeting `gfx906`. Use `clang++ --offload-device-only --no-gpu-bundle-output --offload-arch=gfx1201`.
- 2026-11-26: Inline `global_load_b128` without `"=&v"` earlyclobber + `s_wait_loadcnt 0` lets LLVM overlap `vaddr` with `vdst`, causing `HSA_STATUS_ERROR_MEMORY_FAULT`. Always use `arch_compat.h` helpers.
- 2026-11-26: `libamdhip64.so.7` (`0x74d5b8`) emits 32B-aligned `vmovdqa %ymm1` against `malloc`-allocated objects, causing `#GP` (`SI_KERNEL`) under 16B glibc `malloc`. `crates/hip-rt/src/alloc_guard.rs` (`__libc_memalign(32)`) + `.cargo/config.toml` `--export-dynamic-symbol` guarantees 32B alignment.
- 2026-11-27: Wrote a public README as a landing page (centred logo, five badges, a "Branding" section) and then over-corrected into a six-bullet `## Limits` confession. Both were wrong for the same reason: writing about the project instead of about the engine. `## Writing & Disclosure` above is the durable fix.
