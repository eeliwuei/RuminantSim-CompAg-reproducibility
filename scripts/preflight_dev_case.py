#!/usr/bin/env python3
"""Preflight of the dev_case inputs: which files a driver needs, which are missing or stale, how to rebuild them.

Review round 3 (instruction F-4; R3F, 2026-09-25).  Reads nothing but file bytes (sha256), writes nothing, never
prints a restricted value and never produces a placeholder result.  Standard library only.

Usage (repository root)::

    python scripts/preflight_dev_case.py                         # build inputs + outputs of dev_case_v1
    python scripts/preflight_dev_case.py --driver run_dev_case_v1   # plus the driver's extra files
    python scripts/preflight_dev_case.py --json                  # machine-readable report
    python scripts/preflight_dev_case.py --driver run_dev_case_v1 --run-type pilot
                                                                 # strict build identity: no sidecar or builder
                                                                 # code changed since the build -> exit 1 (FIX3_DEF)

Exit status: 0 READY; 1 INCONSISTENT (present but stale: rebuild with scripts/build_dev_case.py); 2 MISSING (BLOCKED:
the list says which files are missing and whether they can be rebuilt from the package); 3 bad arguments.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from ration_reliability.build.preflight import (  # noqa: E402
    DRIVER_PROFILES,
    EXIT_INCONSISTENT,
    EXIT_USAGE,
    format_preflight,
    preflight_dev_case,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=REPO)
    ap.add_argument("--driver", default=None, choices=sorted(DRIVER_PROFILES))
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    ap.add_argument("--run-type", default=None, choices=["smoke", "debug", "pilot", "official"],
                    help="pilot/official: a missing build sidecar or builder code changed since the build is "
                         "INCONSISTENT (exit 1); otherwise a warning")
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return EXIT_USAGE if exc.code else 0
    try:
        rep = preflight_dev_case(args.root.resolve(), driver=args.driver, run_type=args.run_type)
    except Exception as exc:  # noqa: BLE001 -- a short line, never a traceback
        print(f"[preflight] internal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_INCONSISTENT
    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=1))
    else:
        print(format_preflight(rep))
    return int(rep["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
