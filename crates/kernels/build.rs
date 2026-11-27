use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn compile_arch(hip_src: &Path, arch: &str, out_path: &Path) {
    let output = Command::new("/opt/rocm/lib/llvm/bin/clang++")
        .args([
            "-x",
            "hip",
            "--offload-device-only",
            "--no-gpu-bundle-output",
            &format!("--offload-arch={arch}"),
            "-O3",
            "--rocm-path=/opt/rocm",
        ])
        .arg(hip_src)
        .arg("-o")
        .arg(out_path)
        .output()
        .expect("failed to invoke ROCm clang++");

    if !output.status.success() {
        panic!(
            "{arch} kernel compilation failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let hip_src = manifest_dir.join("hip/gate_kernels.hip");
    let compat_hdr = manifest_dir.join("hip/arch_compat.h");
    println!("cargo:rerun-if-changed={}", hip_src.display());
    println!("cargo:rerun-if-changed={}", compat_hdr.display());

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    compile_arch(
        &hip_src,
        "gfx1201",
        &out_dir.join("gate_kernels_gfx1201.hsaco"),
    );
    compile_arch(
        &hip_src,
        "gfx942",
        &out_dir.join("gate_kernels_gfx942.hsaco"),
    );
}
