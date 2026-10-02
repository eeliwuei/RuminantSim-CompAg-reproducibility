#!/usr/bin/env python3
"""Build a development case from its configs and the restricted tables (review round 3 F6; R3F, 2026-09-25).

Public entry point of ``ration_reliability.build.dev_case`` (the logic that used to live, unauditable, inside the
restricted directory).  The restricted numbers the logic needs are read from the holder-side file
``data/restricted_local/dev_case_v1/build_dev_case_v1.py`` (``RESTRICTED_CONSTANTS``, parsed with ``ast.literal_eval``,
never executed); the restricted tables are read from ``data/restricted_local/``; outputs are written only there.

Usage (repository root)::

    PYTHONDONTWRITEBYTECODE=1 python scripts/build_dev_case.py --check      # build in memory, compare with the
                                                                             # outputs in the case directory
    PYTHONDONTWRITEBYTECODE=1 python scripts/build_dev_case.py              # (re)write the outputs + build_identity.json
    PYTHONDONTWRITEBYTECODE=1 python scripts/build_dev_case.py --out-dir data/restricted_local/<scratch>

Exit status: 0 ok (``--check``: every output byte-identical); 1 a build check failed (``--check``: outputs differ);
2 inputs missing (listed; nothing built, nothing written); 3 bad arguments.  The build includes the K4/K4c
pre-checks (one nominal LP, 20,000 opt-stream draws; about one second): a pre-check of the defined problem, not a
run and not a result.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from ration_reliability.build.dev_case import DEV_CASE_V1, run_build  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=REPO, help="repository root (default: this checkout)")
    ap.add_argument("--case", default=DEV_CASE_V1.case_id, help="case id (only dev_case_v1 is defined)")
    ap.add_argument("--constants", default=None,
                    help=f"restricted constants file (default: {DEV_CASE_V1.constants_file})")
    ap.add_argument("--out-dir", default=None, help="output directory inside data/restricted_local/ "
                                                    f"(default: {DEV_CASE_V1.case_dir})")
    ap.add_argument("--check", action="store_true", help="build in memory and compare with --compare-dir; write nothing")
    ap.add_argument("--compare-dir", default=None, help=f"directory compared by --check (default: {DEV_CASE_V1.case_dir})")
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return 3 if exc.code else 0
    if args.case != DEV_CASE_V1.case_id:
        print(f"unknown case {args.case!r}; defined: {DEV_CASE_V1.case_id}", file=sys.stderr)
        return 3
    if args.check and args.out_dir:
        print("--check writes nothing; --out-dir is not allowed with it", file=sys.stderr)
        return 3
    root = args.root.resolve()
    constants = Path(args.constants) if args.constants else root / DEV_CASE_V1.constants_file
    try:
        return run_build(root, constants_path=constants, entry_point=Path(__file__).resolve(), out_dir=args.out_dir,
                         check_only=args.check, compare_dir=args.compare_dir)
    except ValueError as exc:            # e.g. an output directory outside data/restricted_local/
        print(f"build refused: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
