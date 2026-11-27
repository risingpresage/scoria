# scoria Architecture

## Crate Dependency DAG

```text
scoria-cli
  ├── scoria-engine
  │     ├── scoria-hip-rt   (raw libamdhip64.so FFI, no scoria-* deps)
  │     ├── scoria-kernels  (.hip -> .hsaco via build.rs, ISA objdump tests)
  │     └── scoria-gguf     (pure Rust mmap GGUF parser + WeightLayout seam)
  └── (CLI, gate benchmarks, io_uring NVMe bench, parity harness)
```

## Three Day-1 Seams

1. **`WeightLayout` / `LinearKernel`** (`scoria-gguf` + `scoria-engine`)
   - Separates on-disk / repacked tensor codecs (`GGUF Q4_K/Q6_K/Q8_0/F16`, `H4F8`, `Q4W32`, `BITCOS`, `QTIPW32`) from execution.
2. **`ExecutionGraph` IR** (`scoria-engine`)
   - Declarative stage schedule supporting:
     - `Stage::Dense` (standard Llama/Qwen decoder layers)
     - `Stage::Looped { layers, passes, cross_pass_gate }` (Inkling 2 Mini prelude-recur-coda loops)
     - `Stage::MoE { top_k, num_experts }` (Inkling 2 Mini / Small routed experts)
   - Sequence mixers (`MixerKind::FullAttn`, `SlidingWindowAttn(window)`, `LinearRecurrent` for Bonsai 2/2.5) and KV keys indexed by `(layer_id, loop_pass)`.
3. **Tiered `TensorStore`** (`scoria-engine`)
   - Unified `TensorTier::{Vram, PinnedRam, NvmeExtent}` abstraction enabling Strata-style `io_uring` `O_DIRECT` prefetching for cold MoE experts and cold KV blocks.
