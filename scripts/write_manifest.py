#!/usr/bin/env python3
"""Write PUBLIC_TREE_MANIFEST.json: path, bytes and sha256 of every file of this tree.

Usage: python scripts/write_manifest.py [--check]
With --check the manifest is compared with the tree and the script exits 1 on any difference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "PUBLIC_TREE_MANIFEST.json"
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "outputs"}


def entries() -> list[dict]:
    rows = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p == OUT or any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts):
            continue
        if p.suffix in {".pyc", ".DS_Store"} or p.name == ".DS_Store":
            continue
        rows.append({"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size,
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    rows = entries()
    if a.check:
        old = json.loads(OUT.read_text())["files"]
        if old != rows:
            o = {r["path"]: r for r in old}; n = {r["path"]: r for r in rows}
            print("differences:", sorted(set(o) ^ set(n)) or [k for k in n if o.get(k) != n[k]][:20])
            raise SystemExit(1)
        print("manifest matches:", len(rows), "files")
        return
    OUT.write_text(json.dumps({"files": rows, "n_files": len(rows)}, indent=1) + "\n")
    print("wrote", OUT.name, "with", len(rows), "files")


if __name__ == "__main__":
    main()
