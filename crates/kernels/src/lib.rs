pub const GATE_KERNELS_GFX1201_HSACO: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gate_kernels_gfx1201.hsaco"));

pub const GATE_KERNELS_GFX942_HSACO: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gate_kernels_gfx942.hsaco"));

pub const INFER_KERNELS_GFX1201_HSACO: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/infer_kernels_gfx1201.hsaco"));

pub const INFER_KERNELS_GFX942_HSACO: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/infer_kernels_gfx942.hsaco"));

pub fn gate_hsaco_for_arch(gcn_arch: &str) -> Option<&'static [u8]> {
    let base = gcn_arch.split(':').next().unwrap_or(gcn_arch);
    match base {
        "gfx1200" | "gfx1201" => Some(GATE_KERNELS_GFX1201_HSACO),
        "gfx942" => Some(GATE_KERNELS_GFX942_HSACO),
        _ => None,
    }
}

pub fn infer_hsaco_for_arch(gcn_arch: &str) -> Option<&'static [u8]> {
    let base = gcn_arch.split(':').next().unwrap_or(gcn_arch);
    match base {
        "gfx1200" | "gfx1201" => Some(INFER_KERNELS_GFX1201_HSACO),
        "gfx942" => Some(INFER_KERNELS_GFX942_HSACO),
        _ => None,
    }
}

pub mod names {
    pub const SOAK_TOUCH: &str = "soak_touch";
    pub const STREAM_BW_DEFAULT: &str = "stream_bw_default";
    pub const STREAM_BW_NT: &str = "stream_bw_nt";
    pub const L3_PROBE_HT: &str = "l3_probe_ht";
    pub const WMMA_BENCH_F16: &str = "wmma_bench_f16";
    pub const WMMA_BENCH_FP8: &str = "wmma_bench_fp8";
    pub const WMMA_BENCH_IU4: &str = "wmma_bench_iu4";
    pub const SWMMAC_BENCH_F16: &str = "swmmac_bench_f16";
    pub const SWMMAC_BENCH_FP8: &str = "swmmac_bench_fp8";
    pub const MFMA_BENCH_F16: &str = "mfma_bench_f16";
    pub const MFMA_BENCH_FP8: &str = "mfma_bench_fp8";
    pub const MFMA_BENCH_I8: &str = "mfma_bench_i8";
    pub const LUT_BENCH_PERMLANE: &str = "lut_bench_permlane";
    pub const LUT_BENCH_LDS: &str = "lut_bench_lds";
    pub const BITCOS_BENCH: &str = "bitcos_bench";
}

pub mod infer_names {
    pub const EMBED_LOOKUP_F32: &str = "embed_lookup_f32";
    pub const EMBED_LOOKUP_F16: &str = "embed_lookup_f16";
    pub const EMBED_LOOKUP_Q8_0: &str = "embed_lookup_q8_0";
    pub const EMBED_LOOKUP_Q4_K: &str = "embed_lookup_q4_k";
    pub const EMBED_LOOKUP_Q6_K: &str = "embed_lookup_q6_k";
    pub const RMSNORM_F32: &str = "rmsnorm_f32";
    pub const QK_RMSNORM_F32: &str = "qk_rmsnorm_f32";
    pub const LINEAR_F32: &str = "linear_f32";
    pub const LINEAR_F16: &str = "linear_f16";
    pub const LINEAR_Q5_0: &str = "linear_q5_0";
    pub const LINEAR_Q8_0: &str = "linear_q8_0";
    pub const LINEAR_Q4_K: &str = "linear_q4_k";
    pub const LINEAR_Q6_K: &str = "linear_q6_k";
    pub const ROPE_NEOX_F32: &str = "rope_neox_f32";
    pub const KV_CACHE_APPEND_F32: &str = "kv_cache_append_f32";
    pub const QK_NORM_ROPE_KV_F32: &str = "qk_norm_rope_kv_f32";
    pub const GQA_ATTN_F32: &str = "gqa_attn_f32";
    pub const SILU_MUL_F32: &str = "silu_mul_f32";
    pub const ADD_INPLACE_F32: &str = "add_inplace_f32";
    pub const ADD_RMSNORM_F32: &str = "add_rmsnorm_f32";
    pub const DECODE_ADVANCE_U32: &str = "decode_advance_u32";
    pub const ARGMAX_STAGE1_F32: &str = "argmax_stage1_f32";
    pub const ARGMAX_STAGE2_F32: &str = "argmax_stage2_f32";
    pub const GEMV_F16_F32: &str = "gemv_f16_f32";
    pub const GEMV_Q5_0_F32: &str = "gemv_q5_0_f32";
    pub const GEMV_Q8_0_F32: &str = "gemv_q8_0_f32";
    pub const GEMV_Q4_K_F32: &str = "gemv_q4_k_f32";
    pub const GEMV_Q4_K_R4_F32: &str = "gemv_q4_k_r4_f32";
    pub const GEMV_Q4_K_R8_F32: &str = "gemv_q4_k_r8_f32";
    pub const GEMV_Q4_K_GATE_UP_SILU_F32: &str = "gemv_q4_k_gate_up_silu_f32";
    pub const QUANTIZE_Q8_1_F32: &str = "quantize_q8_1_f32";
    pub const GEMV_Q4_K_Q8_1_F32: &str = "gemv_q4_k_q8_1_f32";
    pub const GEMV_Q4_K_GATE_UP_SILU_Q8_1_F32: &str = "gemv_q4_k_gate_up_silu_q8_1_f32";
    pub const GEMV_Q6_K_F32: &str = "gemv_q6_k_f32";
    pub const WMMA_PROBE_F16: &str = "wmma_probe_f16";
    pub const GEMM_WMMA_F16: &str = "gemm_wmma_f16";
    pub const GEMM_WMMA_Q5_0: &str = "gemm_wmma_q5_0";
    pub const GEMM_WMMA_Q8_0: &str = "gemm_wmma_q8_0";
    pub const GEMM_WMMA_Q4_K: &str = "gemm_wmma_q4_k";
    pub const GEMM_WMMA_Q6_K: &str = "gemm_wmma_q6_k";
    pub const GEMM_WMMA_NARROW_F16: &str = "gemm_wmma_narrow_f16";
    pub const GEMM_WMMA_NARROW_Q5_0: &str = "gemm_wmma_narrow_q5_0";
    pub const GEMM_WMMA_NARROW_Q8_0: &str = "gemm_wmma_narrow_q8_0";
    pub const GEMM_WMMA_NARROW_Q4_K: &str = "gemm_wmma_narrow_q4_k";
    pub const GEMM_WMMA_NARROW_Q6_K: &str = "gemm_wmma_narrow_q6_k";
}

#[cfg(test)]
mod tests {
    use super::{
        GATE_KERNELS_GFX1201_HSACO, GATE_KERNELS_GFX942_HSACO, INFER_KERNELS_GFX1201_HSACO,
        INFER_KERNELS_GFX942_HSACO,
    };
    use std::fs;
    use std::process::Command;

    fn disasm(hsaco: &[u8], mcpu: &str, tag: &str) -> String {
        let tmp = std::env::temp_dir().join(format!(
            "scoria_isa_{}_{}_{}.hsaco",
            tag,
            mcpu,
            std::process::id()
        ));
        fs::write(&tmp, hsaco).expect("write hsaco temp");
        let output = Command::new("/opt/rocm/lib/llvm/bin/llvm-objdump")
            .args(["-d", &format!("--mcpu={mcpu}")])
            .arg(&tmp)
            .output()
            .expect("run llvm-objdump");
        let _ = fs::remove_file(&tmp);
        assert!(
            output.status.success(),
            "llvm-objdump ({mcpu}) failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).into_owned()
    }

    #[test]
    fn verify_gfx1201_isa_emission() {
        let asm = disasm(GATE_KERNELS_GFX1201_HSACO, "gfx1201", "rdna4");
        let required = [
            "v_wmma_f32_16x16x16_f16",
            "v_wmma_f32_16x16x16_fp8_fp8",
            "v_wmma_i32_16x16x32_iu4",
            "v_swmmac_f32_16x16x32_f16",
            "v_swmmac_f32_16x16x32_fp8_fp8",
            "v_permlane16_var_b32",
            "th:TH_LOAD_NT",
            "th:TH_LOAD_HT",
            "s_wait_loadcnt",
            "v_bcnt_u32_b32",
        ];
        for mnem in required {
            assert!(
                asm.contains(mnem),
                "expected gfx1201 mnemonic `{mnem}` not emitted"
            );
        }
    }

    #[test]
    fn verify_gfx942_cdna_isa_emission() {
        let asm = disasm(GATE_KERNELS_GFX942_HSACO, "gfx942", "cdna3");
        let required = [
            "v_mfma_f32_16x16x16_f16",
            "v_mfma_f32_16x16x32_fp8_fp8",
            "v_mfma_i32_16x16x32_i8",
            "ds_bpermute_b32",
            "global_load_dwordx4",
            "v_bcnt_u32_b32",
        ];
        for mnem in required {
            assert!(
                asm.contains(mnem),
                "expected gfx942 mnemonic `{mnem}` not emitted"
            );
        }
    }

    #[test]
    fn verify_infer_gfx1201_isa() {
        let asm = disasm(INFER_KERNELS_GFX1201_HSACO, "gfx1201", "infer_rdna4");
        for sym in [
            "rmsnorm_f32",
            "qk_rmsnorm_f32",
            "embed_lookup_q6_k",
            "decode_advance_u32",
            "linear_q5_0",
            "linear_q4_k",
            "linear_q6_k",
            "gemv_q4_k_r4_f32",
            "gemv_q4_k_r8_f32",
            "gemv_q4_k_q8_1_f32",
            "v_dot4_i32_iu8",
            "gqa_attn_f32",
            "gemm_wmma_q4_k",
            "gemm_wmma_narrow_q4_k",
            "v_wmma_f32_16x16x16_f16",
        ] {
            assert!(asm.contains(sym), "missing symbol `{sym}` in gfx1201 asm");
        }
        for banned in [
            "ds_load_tr",
            "v_bitop3_b32",
            "v_cvt_pk_f16_fp8",
            "v_wmma_scale_",
        ] {
            assert!(
                !asm.contains(banned),
                "banned gfx1201 mnemonic `{banned}` emitted"
            );
        }
    }

    #[test]
    fn verify_infer_gfx942_isa() {
        let asm = disasm(INFER_KERNELS_GFX942_HSACO, "gfx942", "infer_cdna3");
        for sym in [
            "rmsnorm_f32",
            "qk_rmsnorm_f32",
            "embed_lookup_q6_k",
            "decode_advance_u32",
            "linear_q5_0",
            "linear_q4_k",
            "linear_q6_k",
            "gemv_q4_k_r4_f32",
            "gqa_attn_f32",
        ] {
            assert!(asm.contains(sym), "missing symbol `{sym}` in gfx942 asm");
        }
    }
}
