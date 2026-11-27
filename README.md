<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo-on-dark.svg">
  <img src="assets/logo.svg" alt="scoria — AMD-native LLM inference engine" width="380">
</picture>

**AMD-native LLM inference, written in Rust with standalone HIP kernels.**

[![Rust](https://img.shields.io/badge/host_%26_runtime-Rust-dea584?logo=rust&logoColor=white)](Cargo.toml)
[![Kernels](https://img.shields.io/badge/kernels-HIP-ED1C24?logo=amd&logoColor=white)](crates/kernels/hip)
[![Target](https://img.shields.io/badge/gfx1201-RDNA_4-ff5a1f)](specs/GFX1201_ISA.md)
[![Portable](https://img.shields.io/badge/gfx942-CDMA_3-8a94a6)](crates/kernels/hip/arch_compat.h)
[![License](https://img.shields.io/badge/license-MIT-14a800)](Cargo.toml)

</div>

---

scoria runs GGUF language models directly on AMD GPUs. The host and runtime are pure
Rust; every byte of GPU compute lives in standalone `.hip` files that are cross-compiled
for `gfx1201` (RDNA 4) and `gfx942` (CDMA 3) — no `.cpp` anywhere in the tree.

It is a research engine, built in public: every optimisation is a Control-Candidate-Control
experiment recorded in [`harness/ledger.jsonl`](harness/ledger.jsonl), and nothing is called
"done" until [`./scripts/check.sh`](scripts/check.sh) and a parity run against llama.cpp agree.

## Quick start

```bash
./scripts/init.sh          # check toolchain, ROCm, and the target GPU
cargo build --release

# hardware gate: stream bandwidth, WMMA, L2/L3, PCIe, NVMe ceilings
./target/release/scoria gate

# generate
./target/release/scoria run ~/models/Qwen3-8B-Q4_K_M.gguf \
    --prompt "The capital of France is" --max-tokens 64
```

```
timing: prefill_ms=... (2230.71 tok/s), decode_ms=... (99.19 tok/s)
```

## Supported today

| | |
|---|---|
| Architectures | `qwen2`, `qwen3` (per-head QK RMSNorm + `key_length` head dim) |
| Quantisation | `F32`, `F16`, `Q5_0`, `Q8_0`, `Q4_K`, `Q6_K` |
| Prefill | tiled `v_wmma` FP16 GEMM (64×128 and 32×64 tiles, double-buffered LDS) |
| Decode | split-K GEMV, Q8_1 integer-dot Q4_K, `hipGraph` capture, GPU argmax |
| GPU | `gfx1201` (RX 9070 XT, Wave32) primary, `gfx942` (Wave64) portable |

## Measured on RX 9070 XT (`gfx1201`, 16 GiB)

3-trial medians from [`harness/ledger.jsonl`](harness/ledger.jsonl), 64 decode tokens,
128-token prefill. Full method and baselines: [`harness/baselines/`](harness/baselines).

| model | decode (tok/s) | prefill 128 (tok/s) |
|---|---:|---:|
| Qwen2.5-0.5B-Instruct Q4_K_M | 413.24 | 15 619 |
| DeepSeek-R1-Distill-Qwen-7B Q4_K_M | 113.52 | 2 768 |
| Qwen3-4B-Instruct-2507 Q4_K_M | 156.02 | 3 946 |
| Qwen3-8B Q4_K_M | 99.19 | 2 328 |

Decode streams weights at ~490 GB/s against a measured `561.66 GB/s` DRAM ceiling.
We are **not** yet ahead of llama.cpp Vulkan on every shape; the outstanding gap is
written down honestly in
[`phase2b_gfx1201.json → remaining_gap`](harness/baselines/phase2b_gfx1201.json).

## How it is put together

```
crates/hip-rt     dlopen'd HIP FFI, KernelHandle cache, 32 B malloc guard
crates/kernels    standalone .hip sources + clang++ cross-compile + ISA assertions
crates/gguf       zero-dependency mmap GGUF parser, CPU dequant, BPE tokenizer
crates/engine     ExecutionGraph, VramTensorStore, Qwen model wiring, hipGraph capture
crates/cli        scoria {gate,inspect,tokenize,run,graph-soak}
```

Three seams keep it honest: a `TensorStore` (VRAM / pinned / NVMe), a `LinearKernel`
trait (codec + shape dispatch), and an `ExecutionGraph` (dense and looped layer
schedules). Kernels are compiled only with

```bash
clang++ -x hip --offload-device-only --no-gpu-bundle-output \
        --offload-arch=gfx1201 -O3 --rocm-path=/opt/rocm
```

because `hipcc --genco` silently emits `gfx906` — one of several hardware traps recorded
in [`AGENTS.md`](AGENTS.md).

## Verification

```bash
./scripts/check.sh                              # fmt, clippy -D warnings, ISA lint, tests
python3 scripts/parity_oracle.py --model ...    # llama.cpp CPU reference logits
python3 scripts/bench_cac.py --tag <name> --trials 3
./target/release/scoria graph-soak <model> --replays 1000   # VRAM leak guard
```

Parity is judged on `top1_match` plus greedy-prefix agreement. The cosine bar is a
per-prompt sanity check, not an absolute: on peaked prompts llama.cpp's own Vulkan
backend scores `0.997985` against its own CPU backend, so see
[`parity_gate_calibration`](harness/baselines/phase2b_gfx1201.json) before treating a
`0.996–0.998` figure as a regression.

## Branding

The mark is a faceted volcanic shard — scoria is the cooled, vesicular rock left after
heat, which is roughly what a weight tensor is after the GPU is done with it. Assets,
palette and usage rules live in [`assets/`](assets/README.md).

## License

MIT. See [`Cargo.toml`](Cargo.toml).
