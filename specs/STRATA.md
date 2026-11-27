# Strata: paging weights that do not fit

Strata is the tier between "everything resident in VRAM" and "buy a bigger card":
keep hot tensors in VRAM, page the rest from NVMe. This document records the
measured numbers on this machine and what they imply for Inkling 2 Mini.

Reproduce with `scoria plan <gguf>` and `scoria strata <gguf>`.

## What the checkpoint actually costs

`scoria plan /home/noorm/models/inkling/Inkling-2-Mini-UD-Q3_K_XL.gguf --ctx 200000`:

```
arch:        inkling2_loop, 24 layers, 48 KV slots (3x loop on 6..18)
context:     1048576 trained
experts:     128, top-8
sliding:     window 1024 on 20 of 24 layers
kv slots:    8 full-history + 40 windowed
per token:   65536 B FP32 / 32768 B FP16 / 16384 B FP8

     ctx          FP32          FP16           FP8
  200000      12.52 GiB       6.26 GiB       3.13 GiB

weights:     12.70 GiB
vram:        15.69 GiB free of 15.92 GiB
max ctx FP32:     43963 tokens
max ctx FP16:     93047 tokens
max ctx FP8:    191214 tokens
```

Three things fall out of that:

1. **The sliding-window split is what makes long context possible at all.** Only
   the four full-history layers (5, 11, 17, 23) grow with context, and two of them
   sit inside the 3x loop, so they cost three caches each: 8 full-history slots
   against 40 windowed ones. Uniform attention at 200k would need 36 slots of full
   history: 56 GiB at FP32. The window turns that into 12.5 GiB.
2. **200k fits at FP8, nearly at FP16.** 191k tokens with 3.0 GiB of KV room today.
   The last 0.3 GiB is either a smaller context cap or a slightly smaller model.
3. **The weights fit, but with nothing to spare.** 12.70 GiB of a 15.92 GiB card
   leaves 3 GiB. That is the argument for Strata only when something else holds
   VRAM, or when the KV cache needs the room.

## What paging the experts would cost

`scoria strata` replays the routing pattern a decode step makes: for each layer,
read the `top_k` routed expert slabs, nothing else.

```
expert set:   11.36 GiB in 48 tensors, slab 1.42-2.85 MiB
per token:    384 slabs = 727.1 MiB routed (6.2% of the expert set)
mode:         buffered pread + FADV_DONTNEED, qd 16
throughput:   1.57 GB/s
paged decode: 486.3 ms/token = 2 tok/s ceiling with zero compute
```

So a fully paged Inkling 2 Mini runs at 2 tokens/s on this disk. Paging is not a
substitute for residency; it is the fallback when residency is impossible.

Two measured details worth keeping:

- **The expert set is 11.36 GiB of the 12.70 GiB checkpoint.** Top-8 of 128 routes
  6.2% of it per token, so a hot-expert cache is the only thing that makes paging
  viable: keeping 6 GiB resident cuts the traffic about 2x and lands near 4 tok/s.
  Whether that pays depends on the activation skew, which needs a working forward
  pass to measure.
- **`O_DIRECT` fails on this model store.** The filesystem is btrfs with
  `compress=zstd:1`, and direct I/O is rejected on compressed files (`EIO`). The
  benchmark detects the rejection and falls back to buffered reads with
  `posix_fadvise(FADV_DONTNEED)`, which is also why 1.57 GB/s is below the
  3.5 GB/s measured for `O_DIRECT` on an uncompressed test file. A Strata store
  needs its own subvolume with `chattr +C` set *before* the weights are written;
  `chattr` on an existing file does not decompress its extents.

## Design

1. **Tiered store.** `TensorTier` already distinguishes `Vram`, `PinnedRam` and
   `NvmeExtent`. What is missing is a residency policy and a transfer queue.
2. **Expert-granular, not tensor-granular.** A 3-D expert tensor is a contiguous
   array of slabs, so one expert is one `pread`. Slab addresses are already
   computable from the tensor table, which is what `scoria strata` does.
3. **Resident-first.** Load as much of the expert set as fits after the KV
   entitlement, LRU the rest. The hit rate is the only number that matters, and it
   is measurable once a forward pass exists.
4. **Prefetch one layer ahead.** With 8 experts per layer and 1.4-2.9 MiB slabs, a
   queue depth of 16 covers a layer's fetch while the previous layer computes.
5. **Uncompressed store.** `noatime`, `compress=no`, `chattr +C` on the directory,
   and `O_DIRECT` with 4096-aligned buffers, which is the path the gate already
   measures at 3.5 GB/s.

## What is blocked

We cannot yet measure a hot-expert hit rate, or run a single Inkling token, because
`inkling2_loop` needs kernels this engine does not have: relative attention with 16
buckets, per-layer head counts and window masks, 4-tap short convolutions on q/k/v,
attention gates and sinks, block attention residuals, `SiTU`, and a 128-expert MoE
with top-8 routing. `llama.cpp` b11050 cannot load the architecture either, so
there is no reference logits from it; the architecture source is the transformers
`Inkling2Loop` implementation the ingine project references.

The metadata groundwork landed first and is done: `scoria inspect` and `scoria plan`
read the checkpoint, and the KV budget honours per-layer windows and looped blocks.
