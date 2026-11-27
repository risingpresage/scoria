pub const GATE_KERNELS_GFX1201_HSACO: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gate_kernels_gfx1201.hsaco"));

pub const GATE_KERNELS_GFX942_HSACO: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gate_kernels_gfx942.hsaco"));

pub fn gate_hsaco_for_arch(gcn_arch: &str) -> Option<&'static [u8]> {
    let base = gcn_arch.split(':').next().unwrap_or(gcn_arch);
    match base {
        "gfx1200" | "gfx1201" => Some(GATE_KERNELS_GFX1201_HSACO),
        "gfx942" => Some(GATE_KERNELS_GFX942_HSACO),
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

#[cfg(test)]
mod tests {
    use super::{GATE_KERNELS_GFX1201_HSACO, GATE_KERNELS_GFX942_HSACO};
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
}
