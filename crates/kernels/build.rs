use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// ROCm root. `SCORIA_ROCM_PATH` wins (Nix and non-standard installs), then `ROCM_PATH`,
/// then the conventional `/opt/rocm`.
fn rocm_path() -> PathBuf {
    env::var_os("SCORIA_ROCM_PATH")
        .or_else(|| env::var_os("ROCM_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/opt/rocm"))
}

/// The clang that can cross-compile HIP device code. Stock clang cannot: it needs the
/// ROCm LLVM so `--rocm-path` can find OCML/OCKL. `SCORIA_ROCM_CLANG` overrides.
fn rocm_clang(rocm: &Path) -> PathBuf {
    env::var_os("SCORIA_ROCM_CLANG")
        .map(PathBuf::from)
        .unwrap_or_else(|| rocm.join("lib/llvm/bin/clang++"))
}

fn compile_arch(hip_src: &Path, arch: &str, out_path: &Path) {
    let rocm = rocm_path();
    let clang = rocm_clang(&rocm);
    let mut cmd = Command::new(&clang);
    cmd.args([
        "-x",
        "hip",
        "--offload-device-only",
        "--no-gpu-bundle-output",
        &format!("--offload-arch={arch}"),
        "-O3",
    ]);
    // An explicit `--rocm-path` is only useful when it exists; otherwise let clang use
    // its own defaults rather than pointing it at a directory that is not there.
    if rocm.is_dir() {
        cmd.arg(format!("--rocm-path={}", rocm.display()));
    }
    // Nixpkgs ships clang and the OCML/OCKL device bitcode as separate outputs, so the
    // device library path has to be passed explicitly or `cosf`/`expf` fail to link.
    if let Some(libs) = env::var_os("SCORIA_ROCM_DEVICE_LIB").map(PathBuf::from)
        && libs.is_dir()
    {
        cmd.arg(format!("--rocm-device-lib-path={}", libs.display()));
    }
    let output = cmd
        .arg(hip_src)
        .arg("-o")
        .arg(out_path)
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "failed to run {} (set SCORIA_ROCM_CLANG to a ROCm clang++): {e}",
                clang.display()
            )
        });
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
    let gate_src = manifest_dir.join("hip/gate_kernels.hip");
    let infer_src = manifest_dir.join("hip/infer_kernels.hip");
    let compat_hdr = manifest_dir.join("hip/arch_compat.h");
    println!("cargo:rerun-if-changed={}", gate_src.display());
    println!("cargo:rerun-if-changed={}", infer_src.display());
    println!("cargo:rerun-if-changed={}", compat_hdr.display());
    println!("cargo:rerun-if-env-changed=SCORIA_ROCM_CLANG");
    println!("cargo:rerun-if-env-changed=SCORIA_ROCM_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=SCORIA_ROCM_DEVICE_LIB");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    compile_arch(
        &gate_src,
        "gfx1201",
        &out_dir.join("gate_kernels_gfx1201.hsaco"),
    );
    compile_arch(
        &gate_src,
        "gfx942",
        &out_dir.join("gate_kernels_gfx942.hsaco"),
    );
    compile_arch(
        &infer_src,
        "gfx1201",
        &out_dir.join("infer_kernels_gfx1201.hsaco"),
    );
    compile_arch(
        &infer_src,
        "gfx942",
        &out_dir.join("infer_kernels_gfx942.hsaco"),
    );
}
