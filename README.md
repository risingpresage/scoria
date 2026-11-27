<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-on-dark.svg">
    <img src="assets/logo.svg" alt="scoria" width="360">
  </picture>
</div>

An inference engine for AMD GPUs, built for speed on real weights. The host and
runtime are Rust; all device code is standalone HIP kernels cross-compiled for
`gfx1201` (RDNA4) and `gfx942` (CDMA3). It loads Qwen2 and Qwen3 GGUF checkpoints
and runs prefill and decode with tiled `v_wmma` GEMMs, split-K GEMV, and a captured
`hipGraph` decode loop.

Speed here is measured, not asserted: every optimisation is a Control-Candidate-
Control experiment recorded in `harness/ledger.jsonl`, and `./scripts/check.sh`
gates what lands.

## Quick start

```bash
./scripts/init.sh                 # toolchain, ROCm, target GPU
cargo build --release

./target/release/scoria gate      # hardware ceilings: stream BW, WMMA, L2, PCIe, NVMe
./target/release/scoria inspect ~/models/Qwen3-8B-Q4_K_M.gguf

./target/release/scoria run ~/models/Qwen3-8B-Q4_K_M.gguf \
  --prompt "The capital of France is" --max-tokens 64
```

`run` reports prefill and decode separately and prints the decoded text.
`--profile` adds a per-kernel breakdown of one decode step. `--warmup N` runs N
untimed forwards first: the first forward in a fresh process pays lazy HSACO
binding and clock ramp, which otherwise lands entirely in the prefill number.

Supported: `qwen2`, `qwen3` (per-head QK RMSNorm and `key_length` head dim), and
the `F32`, `F16`, `Q5_0`, `Q8_0`, `Q4_K`, `Q6_K` tensor types.

## Results

RX 9070 XT (16 GiB, 64 CUs, Wave32), Q4_K_M weights, 64 decode tokens after a
short prompt, 128-token prefill, warmed up, median of 3 runs. Source:
`harness/ledger.jsonl` tag `phase2d_q8_off`.

| Model | Decode tok/s | Prefill 128 tok/s |
|---|---:|---:|
| Qwen2.5-0.5B-Instruct | 414.9 | 15877 |
| DeepSeek-R1-Distill-Qwen-7B | 115.2 | 2758 |
| Qwen3-4B-Instruct-2507 | 158.4 | 3973 |
| Qwen3-8B | 100.1 | 2486 |

Decode reads every weight once per token. The 8B moves about 4.67 GB per step,
which at 100 tok/s is 467 GB/s against the 561.7 GB/s that `scoria gate` measures
for a plain DRAM stream on this card (a hand-tuned read pattern has been measured
higher, so treat that as a floor rather than the peak). The 4B is lower, 394 GB/s,
because its matrices are small enough that kernel dispatch rather than bandwidth
dominates.

Against llama.cpp `b11050` (Vulkan, RADV, `-ngl 99`) on the same card, this is
84-94% of its decode rate and 64-68% of its prefill rate. Prefill is the open
gap.

## Status

Running and measured on one card (RX 9070 XT, ROCm 7.16, `gfx1201`): dense
Qwen2/Qwen3 from 0.5B to 8B in `Q4_K_M` through `F16`, greedy decode, at 84-94%
of llama.cpp Vulkan's decode rate. `gfx942` compiles and passes the ISA
assertions; it has not been benchmarked.

In flight:

- **Prefill**, at 64-68% of llama.cpp. The tile is bound by LDS fill serialised
  against the WMMA compute, not by weight traffic: a 128x64 tile that halves the
  weight re-reads measured identical, 57.29 ms both ways on Qwen3-8B pp128.
  Double-buffered loaders landed; splitting the loaders into fetch and store
  phases is next.
- **Decode** sits about 17% off the streaming floor on the 8B, and the residue is
  dispatch rather than bandwidth. `hipGraph` took the entire launch chain out and
  returned about 3%, which says the rest was already overlapped.

Next up:

- **Inkling 2 Mini**: 24 layers with layers 6-18 executed three times, and 128 MoE
  experts with top-8 routing. The schedule expands in `ExecutionGraph` already
  (48 steps, covered by `expand_inkling2_mini_looped_schedule`); the expert GEMM
  and the looped KV policy are unwritten.
- **Ternary and paged weights**: a `PTQ1_0` popcount path for the Bonsai ternary
  models, and `io_uring`/`O_DIRECT` paging so checkpoints larger than 16 GiB
  stream from NVMe instead of having to fit.

Outside the current scope: sampling beyond greedy argmax, continuous batching, and
an HTTP server. Prompts of 2-7 tokens run the scalar GEMM by choice: FP16 WMMA
costs about 1.5e-3 cosine there, and the greedy tokens match either way.

## How it works

Five crates:

- `scoria-hip-rt`: `dlopen`'d HIP FFI, pre-resolved kernel handles, stream and
  hipGraph wrappers, and the 32-byte `malloc` guard for the ROCm alignment bug.
- `scoria-kernels`: the `.hip` sources, the `clang++` cross-compile in `build.rs`,
  and `llvm-objdump` assertions that fail the build on a banned mnemonic.
- `scoria-gguf`: mmap GGUF parser, CPU reference dequantisation, byte-level BPE.
- `scoria-engine`: tensor upload, KV cache, layer schedule, per-codec dispatch.
- `scoria-cli`: `gate`, `inspect`, `tokenize`, `run`, `graph-soak`, `plan`, `strata`.

Device code is compiled only with

```bash
clang++ -x hip --offload-device-only --no-gpu-bundle-output \
        --offload-arch=gfx1201 -O3 --rocm-path=/opt/rocm
```

because `hipcc --genco` silently emits `gfx906`. `AGENTS.md` has the rest of the
hardware traps, including the inline `global_load_b128` earlyclobber rule.

## Correctness

Parity is checked against llama.cpp `b11050`'s CPU path: cosine and max absolute
difference over the full logit vector at the last prompt position, plus the
greedy continuation.

| Model | cos | max abs diff | top 1 | greedy |
|---|---:|---:|---|---|
| Qwen3-4B-Instruct-2507 | 0.999657 | 0.85 | yes | 24/24 |
| Qwen3-8B | 0.999335 | 0.76 | yes | 24/24 |
| DeepSeek-R1-Distill-Qwen-7B | 0.999913 | 0.81 | yes | 24/24 |
| Qwen2.5-0.5B-Instruct | 0.999345 | 0.74 | yes | 24/24 |

Those are 5-token prompts (`785,6722,315,9625,374`), which take the FP32 scalar
path. Longer prompts use the FP16 WMMA prefill and land at 0.996-0.998 on sharply
peaked prompts, where llama.cpp's own Vulkan backend scores 0.9980 against its own
CPU backend. The 0.999 bar is not reachable by any GPU path on those prompts;
top-1 and greedy tokens still match. Details in
`harness/baselines/phase2b_gfx1201.json` under `parity_gate_calibration`.

## Verification

```bash
./scripts/check.sh                    # fmt, clippy -D warnings, ISA lint, tests
python3 scripts/parity_oracle.py --model <gguf> --tokens <ids>
python3 scripts/bench_cac.py --tag <name> --trials 3
./target/release/scoria graph-soak <gguf> --replays 1000   # VRAM leak guard
```

Per-experiment history, negative results and the optimisation method are in
`harness/PLAN.md`, `harness/THEORY.md`, `harness/progress.md` and
`harness/baselines/`.

## License

MIT. See `LICENSE`.
