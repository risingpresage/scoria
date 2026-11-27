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

echo "[check] ALL PASSED"
