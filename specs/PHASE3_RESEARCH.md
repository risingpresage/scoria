# Phase 3 Technical Research — Bonsai 2.5 (`qwen35` + Hadamard + `PTQ1_0`/`Q2_0`) & Inkling 2 Mini (`inkling2_loop` + `NVFP4` + Strata + `dflash`)

All GGUF metadata, tensor shapes, and bytes-per-element measurements below were inspected directly from the local downloaded models on 2026-11-26. Block structs and PR references match the upstream `llama.cpp` and `PrismML-Eng/llama.cpp` specifications.

## 1. Local Target Inventory (Already On Disk)

| Role | File | Size | Arch | Tensor Types (`type_id: count`) |
| :--- | :--- | ---: | :--- | :--- |
| **Phase 3A.1 stepping stone** | `/home/noorm/models/Qwen3.8-9B-heretic-uncensored.i1-Q6_K.gguf` | 7.36 GB | `qwen35` | `F32: 177`, `Q6_K(14): 250` (`llama-bench` Vulkan `pp128=3112.13`, `tg64=77.99`) |
| **Bonsai 2 (2B) Q2** | `/home/noorm/models/bonsai/Ternary-Bonsai-2-2B-Q2_0.gguf` | 522 MB | `qwen35` | `F32`, `Q2_0(42)` |
| **Bonsai 2 (2B) PTQ1** | `/home/noorm/models/bonsai/Ternary-Bonsai-2-2B-PTQ1_0.gguf` | 410 MB | `qwen35` | `F32`, `PTQ1_0(143)` |
| **Bonsai 2.5 (27B) Q2+MTP** | `/home/noorm/models/bonsai/Ternary-Bonsai-2.5-27B-Q2_0-mtp.gguf` | 7.91 GB | `qwen35` | `F32: 456`, `Q4_0(2): 8`, `Q2_0(42): 402` |
| **Bonsai 2.5 (27B) PTQ1+MTP** | `/home/noorm/models/bonsai/Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf` | 6.23 GB | `qwen35` | `F32: 456`, `Q4_0(2): 8`, `PTQ1_0(143): 402` |
| **Inkling 2 Mini (NVFP4)** | `/home/noorm/models/inkling/Inkling-2-Mini-NVFP4.gguf` | 17.63 GB | `inkling2_loop` | `F32: 518` (0.148 GB), `Q8_0(8): 186` (6.002 GB), `NVFP4(40): 40` (11.466 GB) |
| **Inkling 2 Mini (UD-Q3_K_XL)** | `/home/noorm/models/inkling/Inkling-2-Mini-UD-Q3_K_XL.gguf` | 13.63 GB | `inkling2_loop` | 13 mixed types (`0, 8, 11..14, 17, 18, 20..22, 30, 40`) |
| **Inkling DSpark drafter** | `/home/noorm/models/inkling/dspark-Inkling-2-Mini-Q8_0.gguf` | 0.50 GB | `dflash` | `F32: 28`, `Q8_0(8): 27`, `BF16(30): 3` |

## 2. Track 3A — `qwen35` + PrismML Hadamard + `Q2_0` / `PTQ1_0` + MTP

### 2.1 Hyperparameters (`Ternary-Bonsai-2.5-27B-*-mtp.gguf`)
- `qwen35.block_count = 65` (64 backbone layers `blk.0..63` + 1 MTP layer `blk.64`, `nextn_predict_layers = 1`)
- `embedding_length = 5120`, `feed_forward_length = 17408`, `context_length = 262144`
- Attention: `head_count = 24`, `head_count_kv = 4`, `key_length = value_length = 256`
- Partial MRoPE: `rope.dimension_count = 64` (out of `256` head dim), `rope.dimension_sections = [11, 11, 10, 0]`, `rope.freq_base = 10000000.0`
- Linear recurrent SSM (GatedDeltaNet): `ssm.conv_kernel = 4`, `ssm.state_size = 128`, `ssm.group_count = 16`, `ssm.time_step_rank = 48`, `ssm.inner_size = 6144`, `full_attention_interval = 4`

### 2.2 Hybrid Layer Cadence (3:1 Linear Recurrent : Full Attention)
- **Layers `(l + 1) % 4 != 0` (48 GatedDeltaNet layers: `blk.0, 1, 2, 4, 5, 6, ...`)**:
  - `attn_qkv.weight` `[5120, 10240]` -> splits into `Q` (`2048` = `16 * 128`), `K` (`2048` = `16 * 128`), `V` (`6144` = `48 * 128`)
  - Causal depthwise conv: `ssm_conv1d.weight` `[4, 10240]` (`F32`, sliding window `3` + current token over concatenated QKV channels) + SiLU
  - Recurrent decay/update scalars: `ssm_alpha.weight [5120, 48]`, `ssm_beta.weight [5120, 48]`, `ssm_a [48]`, `ssm_dt.bias [48]` (all `F32`)
  - Delta rule recurrence per head (`48` heads of `128 x 128` matrix `S` -> `48 * 128 * 128 * 4 B = 3.0 MiB/layer`, `144 MiB` total constant-memory O(1) state across all 48 layers!)
  - Output: `ssm_norm [128]` * `silu(attn_gate [5120, 6144])` -> `ssm_out [6144, 5120]`
- **Layers `(l + 1) % 4 == 0` (16 Full-Attention layers: `blk.3, 7, 11, ...`)**:
  - `attn_q [5120, 12288]` (outputs packed `Q` + attention output gate, `24 * 256 * 2`), `attn_k/v [5120, 1024]`, per-head `attn_q_norm/k_norm [256]`, `attn_output [6144, 5120]`
  - Only 16 layers need KV cache (`4 * 256 = 1024` dims/token -> tiny KV footprint).
- **`blk.64` (MTP layer, quantized `Q4_0` + `F32`)**:
  - Additional tensors: `nextn.enorm [5120]`, `nextn.hnorm [5120]`, `nextn.eh_proj [10240, 5120]`, `nextn.shared_head_norm [5120]`.
  - Runs 1 extra transformer step conditioned on previous hidden state + draft token embedding, yielding 1 free speculative draft token per verify step (`1.6x–1.9x` effective decode boost).

### 2.3 PrismML Hadamard Activation Transform (`prism.hadamard.*`)
Bonsai weights sit in a blockwise normalized Sylvester Walsh-Hadamard rotated basis; activations **must** be rotated on-the-fly before listed matmuls:
- `prism.hadamard.block_size = 1024`
- `prism.hadamard.transform = normalized-sylvester-walsh-hadamard` (`1/sqrt(1024) = 1/32`)
- `prism.hadamard.axis = input-last-dimension`
- `prism.hadamard.sign_mode = explicit` (`sign_widths = [5120, 6144, 17408]`, `sign_values` length `28672` = `5120 + 6144 + 17408` of `±1` int32 signs keyed by projection input width `K`)
- `prism.hadamard.weight_names` (`401` tensors rotated on input)
- `prism.hadamard.inverse_weight_names = ["token_embd.weight"]` (`token_embd` needs inverse/post-unrotation handling; also note `gdn_v_grouped = True`)
- **RDNA 4 Kernel Design (`fwht1024_f32`)**:
  - One 32-thread Wave32 per 1024-element block (or 4 waves across `5120` width): multiply `32` elements by `sign_values[idx] * (1.0f / 32.0f)`, run 5 stages (`stride 1..16`) purely in registers via `__shfl_xor`, store through 4 KiB LDS (`32 * 32 * 4 B`), run 5 stages (`stride 32..512`). Can be fused directly into `rmsnorm` / `silu_mul` / GEMV prologue!

### 2.4 Exact Quantization Block Layouts (Byte-Verified from Disk)
1. **`GGML_TYPE_Q2_0` (type ID `42`, PR `#29600`, observed `36.00 B / 128` -> `18.00 B / 64 weights = 2.25 bpw`)**:
   ```c
   // QK_Q2_0 = 64 elements, 18 bytes
   struct block_q2_0 {
       ggml_half d;      // 2 bytes
       uint8_t   qs[16]; // 16 bytes: 64 x 2-bit weights (4 per byte)
   };
   ```
   Supported upstream in `llama.cpp >= b11556`.
2. **`GGML_TYPE_PTQ1_0` (type ID `143`, `PrismML-Eng` PR `#29672`, observed `28.00 B / 128 weights = 1.75 bpw`)**:
   ```c
   // QK_PTQ1_0 = 128 elements, 28 bytes (7.91 GB Q2_0 shrinks to 6.23 GB)
   struct block_ptq1_0 {
       uint8_t   qs[24]; // 24 bytes: packed 5 trits per byte (base-3 encoding, 120 + grid)
       uint8_t   qh[2];  // 2 bytes: high bits / grid shifts
       ggml_half d;      // 2 bytes: FP16 group scale
   };
   ```
   **RDNA 4 Repack Target (`BITCOS`)**: Unpack base-3 `PTQ1_0` once at model load into 2-bitplane (`u32 pos_mask`, `u32 neg_mask`) + FP16 group-128 scale -> GEMV inner loop runs `v_bcnt_u32_b32` hardware popcount (`4321.57 G-weights/s` verified in Phase 0) against quantized sign-bit activation bitplanes! `6.23 GB` fits completely inside the RX 9070 XT's `16 GiB` VRAM with `> 9.5 GB` left over!

## 3. Track 3B — `inkling2_loop` + `NVFP4` + Strata NVMe/RAM Paging + `dflash`

Target **`Inkling-2-Mini-NVFP4.gguf` (`17.628 GB`)** first rather than `UD-Q3_K_XL`: it uses only **3 tensor types** (`F32: 518` = `0.148 GB`, `Q8_0: 186` = `6.002 GB`, `NVFP4: 40` = `11.466 GB`), vs 13 mixed types in `UD-Q3_K_XL`. Note layers `20..23` use `Q8_0` experts and layers `0..19` use `NVFP4` experts.

### 3.1 Exact `GGML_TYPE_NVFP4` Layout (Type ID `40`, Byte-Verified `9.00 B / 16 weights`)
Measured directly from `blk.0.ffn_down_exps.weight` (`dims=[1152, 2304, 128]`, `191,102,976` bytes / `339,738,624` elems = `0.5625 B/elem` = `9.00 B / 16 weights`, `4.50 bpw`):
- **Micro-block (`16` weights = `9` bytes)** or paired `64`-weight superblock (`36` bytes = `4 x ue4m3 scales + 32B e2m1 nibbles`, verify exact byte ordering against `llama.cpp` `ggml_nvfp4` PR before implementation):
  - `E2M1` 4-bit weights (lookup table `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` with sign bit) + `UE4M3` micro-scales + per-expert FP32 tensor scale (`blk.{l}.ffn_{down,gate_up}_exps.scale [128]`).
- **RDNA 4 Kernel Design**: `v_permlane16_var_b32` 16-entry register LUT (`1824.35 G-lookups/s` verified in Phase 0 gate) maps packed `E2M1` nibbles directly to FP16/FP32 values in registers without LDS bank conflicts.

### 3.2 Looped MoE Graph & Novel Operators (`inkling2_loop`, PR `#32694`)
- **Geometry**: `24` physical blocks (`0..23`), `n_embd = 2304`, `n_ff = n_ff_exp = 1152`, `vocab = 201024` (unpadded `200058`), `128` experts (`top_k = 8`, `expert_weights_scale = 8.0`).
- **Loop Schedule (48 logical steps)**: `loop.start = 6`, `loop.end = 18`, `loop.count = 3`, `loop.varying = False` -> blocks `0..5` (1x), `6..17` (**3x**), `18..23` (1x) = `6 + 12*3 + 6 = 48` logical steps (already modeled by `ExecutionGraph` in `crates/engine/src/lib.rs`, test `expand_inkling2_mini_looped_schedule`). Note KV/conv state keys MUST be indexed by `(layer_id, loop_pass)`!
- **Operators discovered in tensor dump (must implement for parity)**:
  1. **No RoPE (`rope.dimension_count = 0`)**: Uses learned/relative position attention (`attn_rel [2304, 384]`, `attn_rel_bias [16, 1024]`, `attn_rel_clip [16, 24]`), plus `attn_q_clip [128, 24]`, `attn_sink_proj [2304, 24]`, sliding window `1024` pattern, and log-scale attention (`log_scale_floor = 8192`, `log_scale_alpha = 0.1`).
  2. **4 ShortConvs per layer (`shortconv.l_cache = 4`)**: `attn_k_conv`, `attn_v_conv`, `attn_post_conv`, `ffn_post_conv` (`[4, D]` depthwise causal conv with length-3 rolling cache keyed by `(layer_id, loop_pass)`). Also `attn_v_inp_norm` / `attn_v_conv_proj` at block 5 (`blk.5.attn_res_query [2304, 3]`).
  3. **Block Attention Residuals (`attn_res.block_size = 6`)**: `attn_res_query`, `ffn_res_query`, `output_res_query` mix across residual block summaries every 6 layers.
  4. **Cross-Pass Low-Rank Loop Controllers (`blk.6..17`, `gate_rank = 64`)**: `loop_{q,k,v,ffn}_ctrl_{down [4608, 64], forget [64, 2304], gain [64, 2304]}` + `loop_msg_query [2304, 4]`. Note `loop_varying = False`.
  5. **SiTU Activation + Final Logit Softcap**: `activation.situ_beta = 4.0`, `activation.situ_linear_beta = 25.0`; `final_logit_softcapping = 30.0` (`30 * tanh(logits / 30)`).
  6. **MoE Routing**: `ffn_gate_inp [2304, 128]`, router bias `exp_probs_b.bias [128]`, shared/activated `ffn_gate_up_exps` + `ffn_down_exps`.

### 3.3 Strata 3-Tier Expert Paging Budget on RX 9070 XT (`16 GiB` VRAM)
- Non-expert weights (`token_embd`, `output`, attention, norms, shortconv, routers, loop controllers): **`1.818 GB`** -> 100% resident in VRAM.
- Experts total: **`15.798 GB`** across `24` layers (`NVFP4` layers `0..19` = `~4.49 MiB/expert`; `Q8_0` layers `20..23` = `~7.96 MiB/expert`; blended `5.14 MiB/expert`).
- Residency split: keep `~11–11.5 GB` of hottest experts in VRAM (`~70–75%` of total experts, biased toward looped blocks `6..17` which execute 3x per token!), and store the remaining `~4.3–4.8 GB` cold experts in pinned host RAM (PCIe DMA measured `6.96 GB/s`) or `io_uring` `O_DIRECT` NVMe (P31 `3.56 GB/s`, 980 `2.54 GB/s`), overlapping expert DMA prefetch on a second `HipStream` during the previous attention/conv stage.

### 3.4 `dspark-Inkling-2-Mini-Q8_0.gguf` (`dflash` Block-Diffusion Speculative Drafter)
- `3` layers, `n_embd = 2304`, `n_ff = 9216`, `head_count = 24`, `head_count_kv = 8`, `head_dim = 128`, `rope.freq_base = 1e6`, `block_size = 8`, `target_layers = [5, 23, 41, 48]`, `has_markov_context = True`, `final_logit_softcapping = 30.0`.
- Predicts `block_size = 8` tokens in parallel conditioned on hidden states tapped from logical steps `[5, 23, 41, 48]` + `joint_proj` / `markov_w{1,2}` / `mask_embd`.

## 4. Oracle Prerequisites & Recommended Execution Sequence

Current `/home/noorm/llama.cpp/llama-b11050` is **too old** for Bonsai (`Q2_0` requires `>= b11556`; `PTQ1_0` requires `PrismML-Eng/llama.cpp`) and Inkling 2 Mini (`>= b12600` for PR `#32694` + `#32218`). Before starting Phase 3 parity tests:
1. Build upstream `llama.cpp` (`>= b12600`) in `/home/noorm/llama.cpp/upstream/` for `Qwen3.8-9B` (`qwen35`), Bonsai `Q2_0`, `Inkling-2-Mini-NVFP4`, and `dspark`.
2. Build `PrismML-Eng/llama.cpp` in `/home/noorm/llama.cpp/prismml/` for Bonsai `PTQ1_0`.

Recommended milestone order:
1. **3A.1 — `qwen35` backbone** on `Qwen3.8-9B-heretic-uncensored.i1-Q6_K.gguf` (reuses Phase 2 `Q6_K` kernels; only adds GatedDeltaNet + partial MRoPE + attention output gate). Target: beat `llama-bench` Vulkan `3112.13 pp128 / 77.99 tg64`.
2. **3A.2 — `Q2_0` + Wave32 `fwht1024_f32`** on `Ternary-Bonsai-2-2B-Q2_0.gguf`, then scale to `Ternary-Bonsai-2.5-27B-Q2_0-mtp.gguf` (`7.91 GB`, fits in 16 GB VRAM).
3. **3A.3 — `PTQ1_0` -> BITCOS `v_bcnt_u32_b32` + MTP `blk.64`** on `Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf` (`6.23 GB`).
4. **3B.1 — `NVFP4` `v_permlane16_var_b32` LUT + `inkling2_loop`** on `Inkling-2-Mini-NVFP4.gguf` (verify layer-by-layer tap parity against `llama.cpp >= b12600` using small-batch / CPU-backed fallback first).
5. **3B.2 — Strata async VRAM + Pinned-RAM / `io_uring` `O_DIRECT` expert paging** (profile hit rate across prompts; verify zero GPU fault and asynchronous DMA overlap).
6. **3B.3 — `dspark-Inkling-2-Mini` `dflash` 8-token block-diffusion drafting** integrated with batch-8 target verification.
