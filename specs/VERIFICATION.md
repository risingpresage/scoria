# Verification & Benchmark Protocol

## Phase 0 Hardware Gate (`scoria gate`)
All measurements written to `harness/baselines/gfx1201_gate.json`:
1. **ROCm 7.16.0 stability soak:** 100 cycles (20 `--quick`) of 1 -> 4 -> 8 GiB `DeviceBuf` alloc, H2D write, kernel launch, sync, free. Must achieve 100/100 passes with zero faults/hangs.
2. **VRAM streaming & L3 preservation:** Compare `stream_bw_default` vs `stream_bw_nt` (`TH_LOAD_NT`) and measure 32 MiB `l3_probe_ht` (`TH_LOAD_HT`) hit rate immediately after streaming 1 GiB.
3. **Matrix throughput:** Measure `v_wmma` (FP16, FP8, IU4) and `v_swmmac` (FP16, FP8) with 8 independent accumulators across 4096 Wave32s (`512 x 256`); record dense vs sparse ratio.
4. **Register LUT & BITCOS unpack:** Compare `v_permlane16_var_b32` register LUT vs LDS LUT, and measure `BITCOS` popcount unpack rate.
5. **PCIe DMA & NVMe `O_DIRECT`:** Measure 512 MiB pinned H2D/D2H bandwidth and `io_uring` `O_DIRECT` sequential 256 KiB read bandwidth (cleaning up all temporary files).

## Phase 1–3 Parity & Throughput Rules
- Greedy decode parity (`temperature = 0`, fixed seed) and top-k logit max-abs diff against `llama.cpp` on local GGUF models in `~/models`.
- Open-loop serving benchmarks (`np=1, 4, 8, 16`) compared against pinned baselines in `harness/baselines/`.
