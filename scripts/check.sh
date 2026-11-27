#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[check] cargo fmt --check"
cargo fmt --all -- --check

echo "[check] cargo clippy"
cargo clippy --workspace --all-targets -- -D warnings

echo "[check] lint-arch"
./scripts/lint-arch.sh

echo "[check] cargo test"
cargo test --workspace

# Tests that push the whole 48-visit stack through the real checkpoint are unaffordable in
# a debug build; this is where they actually run.
echo "[check] heavy release tests"
cargo test --release -p scoria-engine --test inkling2_kv_ring

echo "[check] ALL PASSED"
