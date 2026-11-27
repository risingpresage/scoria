# Parity & Benchmark Skill

1. Run greedy decode (`temperature = 0`, fixed seed) on local GGUF models in `~/models`.
2. Compare generated token IDs and top-k logits against `llama.cpp` reference outputs.
3. Run open-loop throughput sweeps (`np=1, 4, 8, 16`) and compare against pinned JSON baselines in `harness/baselines/`.
4. Record raw numbers and hardware telemetry (`rocm-smi`) before claiming improvement.
