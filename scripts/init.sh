#!/usr/bin/env bash
set -euo pipefail

echo "[init] Checking Rust toolchain..."
rustc --version
cargo --version

echo "[init] Checking ROCm LLVM compiler and disassembler..."
test -x /opt/rocm/lib/llvm/bin/clang++
test -x /opt/rocm/lib/llvm/bin/llvm-objdump
/opt/rocm/lib/llvm/bin/clang++ --version | head -n 1

echo "[init] Checking ROCm gfx1201 device and libraries..."
rocminfo_out="$(rocminfo)"
grep -q "gfx1201" <<<"$rocminfo_out"
test -e /opt/rocm/lib/libamdhip64.so
test -e /usr/lib/liburing.so

echo "[init] NVMe devices:"
lsblk -d -o NAME,MODEL,SIZE,TRAN | grep -E "NAME|nvme" || true

echo "[init] Environment OK."
