# Capability parity

Where scoria stands against llama.cpp `b11050` on the same card, and the order to
close the difference. Scoria's column is read from the code, so it is authoritative.
The llama.cpp column is the local build's `--help` plus its documented feature set,
kept qualitative where a count would rot.

## What scoria already does that matters

- Native Q4_K / Q6_K GEMV and `v_wmma` prefill written for `gfx1201`, not ported
  from a CUDA kernel: split-K GEMV with `global_load_b128`, integer-dot Q4_K,
  double-buffered WMMA tiles.
- `hipGraph` decode capture with a device-resident position and an on-device
  argmax-to-token epilogue, so a decode step is one graph replay.
- A parity harness against llama.cpp's CPU path with per-model cos / top-1 / greedy
  agreement, a hardware gate that measures the card's real ceilings, and a
  C-A-C ledger that every optimisation has to beat.
- Zero C++ host code: the host and runtime are Rust, device code is standalone
  `.hip` compiled for both `gfx1201` and `gfx942`.

## Matrix

| Capability | scoria | llama.cpp b11050 | Blocks |
|---|---|---|---|
| Architectures | `qwen2`, `qwen3` | dozens (llama, gemma, mistral, phi, deepseek2/3, qwen2/3, …) | most GGUFs cannot load |
| Tensor types | `F32`, `F16`, `Q5_0`, `Q8_0`, `Q4_K`, `Q6_K` | K-quants, i-quants (IQ1-IQ4), `TQ1_0`/`TQ2_0`, `MXFP4`, … | most quantised GGUFs cannot load |
| Tokenizer | GPT-2-style byte BPE | BPE or SentencePiece | llama/gemma checkpoints cannot load |
| Attention | full causal GQA, FP32 KV | full causal, `--flash-attn`, KV quant | long context is bandwidth bound |
| KV cache | FP32, dense, this-run only | FP16/quantised, defrag, context shift, checkpoints | 2x VRAM and 2x KV traffic |
| RoPE | NeoX only | NeoX + linear/YaRN scaling | cannot exceed trained context sanely |
| MoE | `MoeSpec` in the schedule, no kernels | yes | Inkling 2 Mini, DeepSeek |
| Looped layers | `Stage::Looped` in the schedule, no kernels | no | Inkling 2 Mini |
| Sliding window / MLA / sparse | none | MLA for DeepSeek-V2/V3; sliding window | attention family, see `specs/ATTENTION.md` |
| Sampling | greedy argmax | top-k, top-p, min-p, typical, XTC, DRY, penalties, temp | `temperature` is ignored |
| Structured output | none | GBNF grammar, JSON schema | |
| Batching / server | none | continuous batching, `llama-server` HTTP + metrics | |
| Speculative decoding | none | `--draft*`, draft-model and n-gram | |
| Multimodal | none | image input | |
| LoRA / control vectors | none | adapters | |
| Backends | ROCm only (`gfx1201`, `gfx942`) | CUDA, Metal, Vulkan, CPU, … | |

## Order to close it

Ordered by what unblocks the most, not by effort.

1. **Tensor types beyond `Q4_K`/`Q6_K`.** `IQ4_XS`, `Q3_K`, `Q5_K` and `Q8_0` K-quant
   layouts cover most of the GGUF ecosystem. Each is a loader plus a GEMV; the
   dequant reference already exists in `crates/gguf/src/dequant.rs`, so parity is
   cheap to prove per type.
2. **SentencePiece tokenizer.** A second tokenizer path unlocks llama/gemma-class
   checkpoints. Purely host-side, no kernels, and the existing tokenizer tests give
   the pattern to copy.
3. **Sampling.** `temperature`, `top-k`, `top-p`, `min-p`, repetition penalty, plus
   seeded RNG. Small, entirely on the host plus one kernel, and it makes the engine
   usable for anything except benchmarks.
4. **FP16 KV cache, grouped split-K flash decoding, WMMA FlashAttention-2.** See
   `specs/ATTENTION.md`; this is the long-context workstream and the only one with
   a 2x-class number attached.
5. **MoE + looped-layer kernels.** The schedule and `MoeSpec` already exist and are
   tested by `expand_inkling2_mini_looped_schedule`. Inkling 2 Mini now runs the whole
   48-visit stack on the host at **1.84 s/token prefill and 1.93 s/token decode** (16
   threads; 7.6-8.0 before the threading work) with perplexity 1.317 on held-out prose,
   so there is a correctness oracle and a baseline to beat. What is missing is the
   grouped expert GEMV on `gfx1201` and a looped KV policy.
6. **MLA checkpoints**, then block-sparse selection (the LSA/DSA/NSA primitive).
7. **RoPE scaling**, so long-context models can be served past their training length.
8. **Batching and a server.** Last because it changes the performance model: once
   several sequences share a step, weight traffic amortises and the metrics from
   `harness/ledger.jsonl` stop describing the engine.

Items 1-3 are throughput-neutral and purely about which checkpoints run and how
output is sampled. Item 4 is speed at long context. Items 5-6 are architecture
coverage. Items 7-8 are deployment.

Status of each lives in `harness/features.json`. This file is the order; the ledger
is the evidence.
