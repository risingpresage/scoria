#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

fail() {
    echo "[lint-arch] FAIL: $1" >&2
    exit 1
}

# 1. AGENTS.md line cap & CLAUDE.md bridge
agents_lines=$(wc -l < AGENTS.md)
if [ "$agents_lines" -gt 120 ]; then
    fail "AGENTS.md has $agents_lines lines (limit 120)"
fi
grep -qx "@AGENTS.md" CLAUDE.md || fail "CLAUDE.md must contain @AGENTS.md"

if [ -f crates/kernels/AGENTS.md ]; then
    k_lines=$(wc -l < crates/kernels/AGENTS.md)
    [ "$k_lines" -le 50 ] || fail "crates/kernels/AGENTS.md has $k_lines lines (limit 50)"
fi

# 2. No C++ host files under crates/
if find crates -type f \( -name "*.cpp" -o -name "*.cc" -o -name "*.cxx" \) | grep -q .; then
    fail "Found C++ host source files under crates/ (only standalone .hip kernels allowed)"
fi

# 3. No banned gfx1250/unsupported mnemonics in .hip kernels
if ls crates/kernels/hip/*.hip >/dev/null 2>&1; then
    if grep -rnE "ds_load_tr|v_bitop3_b32|v_cvt_pk_f16_fp8|v_wmma_scale_" crates/kernels/hip/*.hip; then
        fail "Found banned unsupported mnemonic in crates/kernels/hip/*.hip"
    fi
fi

# 4. Never invoke hipcc --genco in crates/ or scripts/
if grep -rn "hipcc --genco" crates/ scripts/init.sh scripts/check.sh 2>/dev/null; then
    fail "Found forbidden 'hipcc --genco' invocation"
fi

# 5. Enforce crate dependency DAG in Cargo.toml
python3 - << 'PY'
import pathlib, re, sys

rules = {
    "crates/hip-rt/Cargo.toml": ["scoria-kernels", "scoria-gguf", "scoria-engine", "scoria-cli"],
    "crates/gguf/Cargo.toml": ["scoria-hip-rt", "scoria-kernels", "scoria-engine", "scoria-cli"],
    "crates/kernels/Cargo.toml": ["scoria-hip-rt", "scoria-gguf", "scoria-engine", "scoria-cli"],
    "crates/engine/Cargo.toml": ["scoria-cli"],
}

for path, forbidden in rules.items():
    p = pathlib.Path(path)
    if not p.exists():
        print(f"Missing {path}", file=sys.stderr)
        sys.exit(1)
    text = p.read_text()
    for dep in forbidden:
        if re.search(rf"(?m)^\s*{re.escape(dep)}\s*=", text):
            print(f"DAG violation: {path} must not depend on {dep}", file=sys.stderr)
            sys.exit(1)

# 6. Check relative markdown links in AGENTS.md and specs/*.md
md_files = [pathlib.Path("AGENTS.md")] + sorted(pathlib.Path("specs").glob("*.md"))
link_re = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
for md in md_files:
    content = md.read_text()
    for m in link_re.finditer(content):
        target = m.group(1).split("#")[0]
        if not target or "://" in target:
            continue
        resolved = (md.parent / target).resolve()
        if not resolved.exists():
            print(f"Broken link in {md}: {target}", file=sys.stderr)
            sys.exit(1)
PY

echo "[lint-arch] OK"
