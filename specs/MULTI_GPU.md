# Multi-GPU & Multi-Arch Testing

Two separate goals get called "test on multiple GPUs". They need different work:

1. **Multi-arch portability** — the same engine running correctly on a *different AMD
   architecture* (`gfx1201` vs `gfx942` vs `gfx950`). This is what the kernels crate is
   already being shaped for.
2. **Multi-device** — one node, N accelerators, split work across them. Not implemented.
   `ExecutionGraph` is a single-device stage schedule, and there is no collective /
   peer-to-peer transport layer. Nothing in this document assumes it exists. If you go
   there, note the in-repo landmine: ROCm 7.16.0 segfaults in `hipMemcpy` when a thread
   calls `hipSetDevice` and exits before another thread runs HIP — keep HIP calls on a
   persistent thread (see the Mistakes Log in `AGENTS.md`).

Live rental prices: run `./scripts/gpu-prices.sh` (see [Live pricing](#live-pricing)).

## Current state in-tree

| Piece | Status |
|---|---|
| `crates/kernels/build.rs` | Builds **`gfx1201` + `gfx942`** code objects |
| `crates/kernels/hip/arch_compat.h` | RDNA4 branch (`gfx1200`/`gfx1201`, wave32) and CDNA branch (`gfx90a`/`gfx942`/`gfx950`, wave64); `#error` on anything else |
| `crates/kernels/src/lib.rs` | `gate_hsaco_for_arch()` picks a code object from `gcnArchName`; separate ISA assertions for RDNA4 and CDNA3 |
| `crates/hip-rt` | `TargetArch::{Gfx1201, Gfx942, Gfx950, Other}` + `detect_arch()`; `is_rdna4()` |
| `crates/cli/src/gate.rs` | Branches compute benchmark on `ctx.arch.is_rdna4()` (WMMA/SWMMAC vs MFMA) |

So the device-code half of portability is real: memory-path intrinsics, wave size, and the
matrix benchmark already switch per architecture.

## What still blocks it

1. **`gate.rs` hardcodes the arch for output.** `gate.rs:410` writes `"arch": "gfx1201"`,
   `:456` always writes `harness/baselines/gfx1201_gate.json`, `:462` prints a `gfx1201`
   banner. A `gfx942` run therefore overwrites the RDNA4 baseline and mislabels its own
   numbers. Fix: derive all three from `ctx.arch.as_str()` and write
   `harness/baselines/{arch}_gate.json`.
2. **`gfx950` is declared but not built.** `TargetArch::Gfx950` exists and `arch_compat.h`
   handles `__gfx950__`, but `build.rs` has no `gfx950` target and
   `gate_hsaco_for_arch()` returns `None` for it. MI355X/MI350X cannot run the gate today.
3. **`gfx1200` maps to the `gfx1201` code object** (`lib.rs:10`). A `gfx1201` ELF will not
   load on Navi 44 silicon. Either add a `gfx1200` build or build `gfx12-generic`.
4. **`is_rdna4()` matches only `Gfx1201`**, so `gfx1200` would take the CDNA benchmark path
   even though `arch_compat.h` treats it as RDNA4. Should be `Gfx1201 | Gfx1200`.
5. **RDNA3 (`gfx1100`) is not supported at all.** `arch_compat.h` `#error`s on it and the
   kernels use `_gfx12`-suffixed WMMA builtins. A cheap RX 7900 XTX cannot build the crate.
6. **Phase 0 item 5 is host-specific.** The PCIe DMA + NVMe `O_DIRECT` test assumes the
   local P31 + 980 pair. On a rented box expect `null` there and do not read it as a
   regression.

## Target matrix

Prices are from `./scripts/gpu-prices.sh` (fetched 2026-11-27) where a tracked source carries
the card. Three workstations/resold parts only appear at scattered hosts, so they are marked
`≈` and are not produced by the script. Everything moves — re-run it, do not trust this table.

| ROCm target | Card | VRAM | Wave | Matrix path | Rentable at | $/GPU-hr |
|---|---|---|---|---|---|---|
| `gfx1201` | RX 9070 XT | 16 GB | 32 | `v_wmma` + `v_swmmac` | Salad | 0.20 on-demand / 0.07 preemptible |
| `gfx1200` | RX 9060 XT | 16 GB | 32 | `v_wmma` + `v_swmmac` | Salad | 0.13 / 0.05 |
| `gfx1100` | RX 7900 XTX | 24 GB | 32 | `v_wmma` (gfx11 form) | Salad | 0.17 / 0.07 |
| `gfx1100` | Radeon PRO W7900 | 48 GB | 32 | `v_wmma` (gfx11 form) | individual hosts | ≈0.38–0.70 |
| `gfx942` | MI300X | 192 GB | 64 | `v_mfma` | OpenRelay, DigitalOcean, RunPod, Azure | 1.99–2.59 on-demand, 1.11 Azure spot |
| `gfx942` | MI300A | 128 GB | 64 | `v_mfma` | Azure | list only |
| `gfx950` | MI350X | 288 GB | 64 | `v_mfma` | OpenRelay, DigitalOcean | 3.99 on-demand, 2.46 spot |
| `gfx950` | MI355X | 288 GB | 64 | `v_mfma` + `v_wmma_scale_*` | OpenRelay, Oracle, Daytona | 6.30 on-demand, ~2.6–4.5 spot |
| `gfx90a` | MI250X | 128 GB | 64 | `v_mfma` | resellers | ≈0.79 |

Practical reading:

- **You already have the only card that matters for the primary target** — a local
  RX 9070 XT (`gfx1201`). Renting another `gfx1201` buys you nothing but a second data point.
- **`gfx1200` is the cheapest genuinely-different RDNA4 test** (~$0.13/hr). Half the CUs and
  a 128-bit bus, same ISA family — it catches assumptions about CU count, L2/L3 size and
  bandwidth that a second 9070 XT cannot.
- **`gfx942` is the highest-value portability test** (~$2/hr, ~8 hours = ~$16). Wave64, MFMA,
  192 GB. It exercises the entire CDNA branch of `arch_compat.h` and will expose any wave32
  or `_gfx12` assumption that leaked out of the RDNA4 path.
- **`gfx950` only matters once `build.rs` emits it.** Its distinguishing feature is
  `v_wmma_scale_*`, which `gfx1201` bans — the useful assertion is that a `gfx950` build
  contains it and an `gfx1201` build does not.
- **`gfx1100`/RDNA3 is cheap but currently unbuildable.** Worth adding only if you want
  RDNA3 coverage; the work is a `#elif defined(__gfx11__)` branch in `arch_compat.h` plus
  gfx11 WMMA builtins.

## Portability contract (recommended)

Keep architecture knowledge in exactly three places so they cannot drift:

1. `arch_compat.h` — intrinsics and wave size.
2. `gate_hsaco_for_arch()` — target → code object.
3. `TargetArch` — what `detect_arch()` can report.

Then add:

- **One build-target list** in `build.rs` (`const TARGETS: &[&str]`) instead of repeating
  each arch in the compile calls, `lib.rs` `include_bytes!`, and `gate_hsaco_for_arch()`.
- **A negative ISA assertion per arch.** The existing tests assert required mnemonics are
  present; also assert foreign ones are *absent*. For `gfx942`:
  `!asm.contains("v_wmma_") && !asm.contains("v_swmmac_") && !asm.contains("ds_load_tr")`.
  For `gfx1201`: `!asm.contains("v_mfma_") && !asm.contains("v_wmma_scale_")`.
  This is what stops a per-arch `#ifdef` from silently emitting the wrong instruction.
- **Arch-parameterized baselines**: `harness/baselines/{arch}_gate.json`, with the gate
  reading and comparing against the file for the arch it detected.

## Running the gate on a rented box

1. Confirm ROCm matches the pinned version (`rocminfo`, `hipcc --version`). The gate's soak
   numbers are asserted against ROCm 7.16.0; a different runtime invalidates the comparison.
2. `rocminfo | grep -o 'gfx[0-9]*'` — must be a target in the matrix above, and must match
   the code object `gate_hsaco_for_arch()` selects.
3. `./scripts/init.sh`
4. `cargo run --release -p scoria-cli -- gate`
5. Pull back `harness/baselines/<arch>_gate.json`.
6. Expect the NVMe half of item 5 to be `null`; do not compare it across hosts.
7. Budget from `./scripts/gpu-prices.sh`, not from this file.

Cost reference: an 8-hour `gfx1201` validation session is ~$1.60 on-demand; the same on
`gfx942` is ~$16; on `gfx950`, ~$50–70.

## Live pricing

`scripts/gpu-prices.sh` reads public price APIs on every run and caches the result to
`harness/gpu-prices.json` with a `fetched_at` stamp.

```
./scripts/gpu-prices.sh                # board, cheapest first
./scripts/gpu-prices.sh --gpu 9070     # just the RDNA4 cards
./scripts/gpu-prices.sh --json         # normalized records
./scripts/gpu-prices.sh --refresh      # force a fetch, ignore cache
```

- `--all` widens past the AMD filter.
- `PRICE-AS-OF` is each upstream quote's own date, not the fetch date. A `*` marks a quote
  older than 14 days — treat those as indicative only.
- Sources: an hourly-refreshed aggregate, Azure's own retail price API, and a static table
  for the Salad-only consumer AMD cards.
- The cache is considered fresh for 6h (`SCORIA_PRICE_MAX_AGE_H`). If every source is
  unreachable the script reuses the cache and prints its age as a warning rather than
  inventing numbers.

### Known data caveats

- **Advertised ≠ available.** The cheapest listing is routinely sold out. Verify stock at
  the provider before committing to a plan.
- **Vultr's MI355X `$2.59` is the preemptible SKU** and is out of stock; the aggregate
  source labels it on-demand. Treat it as spot.
- **Azure list prices are per 8-GPU instance**, divided by 8 in the board. Hyperscaler
  capacity is normally sold on contract, so the list price is an upper bound, not a quote.
- Consumer AMD cards are only carried by Salad (a distributed consumer-GPU cloud), on
  individually-owned nodes that can disconnect. Assume preemptible unless you pay for the
  High tier.
