# Attention: what the "latent / sparse" family actually is, and what we can ship

Written 2026-11-27 after researching the primary sources. Every number below is
quoted from the cited paper, not measured here.

## First, disambiguation

"Latent sparse attention" names at least three different ideas that get conflated.
They are not interchangeable, and only one of them is a paper by that title.

| Name | Paper | What is compressed | Cached per token |
|---|---|---|---|
| **LSA** (Latent Sparse Attention) | arXiv:2610.42147, Kubicek et al., Prefrontal AI + ETH, Oct 2026 | the *search* for relevant keys: keys are clustered in a learned latent space and queries score cluster centroids, flat or over a tree | full KV, but only `k` keys are read |
| **MLA** (Multi-head Latent Attention) | arXiv:2405.04434, DeepSeek-V2, May 2024 | the KV *representation*: low-rank joint compression to `c_t^KV`, plus a decoupled RoPE carrier | one latent (`d_c=512`) + 64 RoPE dims instead of `2·n_h·d_h` |
| **DSA** (DeepSeek Sparse Attention) | arXiv:2512.02556 (V3.2) and V3.2-Exp, Sep 2025 | the *number* of KV entries attended: a lightning indexer scores all history, top-`k` (2048) are read | MLA latent, plus a separate indexer K cache |
| **NSA** (Native Sparse Attention) | arXiv:2502.11089, DeepSeek-AI, ACL 2025 best paper | three paths over blocks: compressed tokens + selected blocks + sliding window | block-compressed KV |

MLA and LSA are orthogonal: MLA shrinks each KV entry, LSA and DSA reduce how many
entries you read. DSA is literally instantiated *on top of* MLA, which is why the
top-`k` set can be shared across all query heads.

## LSA (the one you named) in detail

- Keys are projected into a smaller latent space, L2-normalised, and pooled in
  blocks of `n_pool`; blocks are clustered bottom-up with Ward's method into
  buckets whose centroids live in a cache. A query scores bucket centroids and
  expands the top-`p` buckets into key indices, always keeping sink and window
  tokens. Search is either flat or hierarchical (tree), which is what makes prefill
  subquadratic.
- Two variants. **D-LSA** distils into a pre-trained model, training only small
  per-layer projections against a frozen model's attention distribution (KL).
  **N-LSA** trains natively on next-token loss and adds no parameters.
- Reported results, one attention layer, single H200: D-LSA is 22.7x faster in
  prefill and 4.6x in decode on GLM-5.3-FLASH at 10M tokens (against that model's
  own sparse attention), and 72.4x / 42.5x on Qwen3.8-27B (against full attention).
  End-to-end serving is 2.2x for GLM and 9.2x for Qwen. N-LSA matches full
  attention and is 1.6x faster per layer at 128k. Code is Apache-2.0 at
  `Prefrontal-AI/latent-sparse-attention`, as an SGLang plugin with H200/B300
  kernels.
- The motivating measurement is worth internalising: **at 10M tokens, 98% of
  sparse attention's time is spent deciding which keys to attend to**, not
  attending. That is why LSA optimises the indexer and not the dot products.

## Why none of these are drop-in for us

Every one of them is a property of the checkpoint, not a kernel flag. You cannot
retrofit LSA, DSA or NSA onto `Qwen3-4B-Q4_K_M`: DSA's indexer is trained against
the dense attention distribution over ~943.7B tokens of sparse continued training,
and D-LSA requires a distillation pass that trains new per-layer projections. MLA
works only because DeepSeek-V2/V3 weights ship with `kv_a_proj`, `kv_b_proj`,
`q_a_proj` and the RoPE split.

So the order has to be: build the machinery that any of them needs, then support
checkpoints that ship the architecture, then (only if we ever train) the
distillation path.

## What is portable, and what it is worth

Attention is about 4% of our decode step at the context lengths in the current
benchmarks (`0.476 ms` of a `12.50 ms` profiled step on Qwen3-8B at kv_len≈18), so
none of this will move pp128 or short-context decode. It matters at long context,
where KV bandwidth, not weights, is the budget.

Concretely, at 32k context on Qwen3-8B with FP32 KV:

```
per step = 36 layers x 2 (K,V) x 32768 x (8 kv_heads x 128) x 4 B  = 9.4 GB
at 561.7 GB/s                                                     = 16.8 ms  ->  60 tok/s
same with FP16 KV                                                 =  4.7 GB  -> 119 tok/s
same with top-k = 2048 (DSA-style)                                = 604 MB   -> bandwidth-free
```

That is the whole argument for this workstream: FP16 KV doubles long-context
decode on its own, a grouped split-K kernel keeps all 64 CUs busy instead of
launching `n_heads` blocks, and sparse selection is the only thing that removes the
linear term entirely.

## Plan

| # | Item | Needs | Payoff | Gate |
|---|---|---|---|---|
| 1 | FP16 KV cache | half-precision store in `KvCache`, convert in `qk_norm_rope_kv_f32`, `half2` loads in attention | 2x KV bandwidth, half the VRAM | parity holds, decode at 32k measured |
| 2 | GQA-grouped split-K flash decoding | one workgroup per KV-head group (4 query heads on Qwen3, 7 on R1-7B), `kv_len` split across workgroups with an LSE reduce, reads `*pos` so `hipGraph` capture survives | 4-7x less KV re-read, 64 CUs busy at long context | decode tok/s at 2k/8k/32k vs current |
| 3 | WMMA FlashAttention-2 prefill | `v_wmma_f32_16x16x16_f16` for `Q·K^T` and `P·V`, double-buffered LDS K/V tiles, online softmax, causal tile skip | prefill at pp512-pp8192; the scalar path is quadratic on ALU | pp512/pp2048 vs llama.cpp Vulkan |
| 4 | Block-sparse gather + top-k | selection kernel over a score row (`DeepSelect` is MIT and top-k oriented), paged gather into the attention kernel | the primitive LSA/DSA/NSA all need | unit test: gather equals dense on a mask |
| 5 | MLA checkpoints | `kv_a_proj`/`kv_b_proj`, decoupled RoPE, W_UK absorbed into W_Q and W_UV into W_O | KV cache -93.3% per the V2 paper, and the only way DSA matters | a DeepSeek-V2-Lite or V3 GGUF loads and matches |
| 6 | LSA | a checkpoint that ships it, or a distillation pass (out of scope for an inference engine) | indexer 22.7x at 10M tokens | not startable yet |

Items 1-3 are unconditional wins on models we already run. Item 4 unlocks the
family. Item 5 depends on a checkpoint we do not have on disk. Item 6 is blocked on
training, so it stays a watch item.

Measured status of each item lives in `harness/features.json`; this document is the
plan, not the ledger.
