#!/usr/bin/env python3
"""Recompute every reported confidence bound and target decision from the aggregate tables.

The article reports, for each frozen rule and test cell, a one-sided Clopper--Pearson upper bound on the
failure rate (per-claim error 0.05/10,000), lower bounds on the all-ration failure floor of a frozen pool,
and conservative paired intervals built from four one-sided binomial tails (per-tail error 0.05/40,000).
This script rebuilds all of them from the counts stored in ``data/aggregate_results`` and compares them
with the stored values. It needs only numpy and scipy.

Usage::

    python extensions/analysis/recompute_bounds_from_aggregates.py --data data/aggregate_results
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scipy.stats import beta

RATE_ALPHA = 0.05 / 10_000          # one-sided, per rate claim
PAIR_GAMMA = 0.05 / (4 * 10_000)    # one-sided, per tail of a paired claim


def cp_upper(k: int, n: int, a: float) -> float:
    return 1.0 if k >= n else float(beta.ppf(1.0 - a, k + 1, n - k))


def cp_lower(k: int, n: int, a: float) -> float:
    return 0.0 if k <= 0 else float(beta.ppf(a, k, n - k + 1))


def rows(p: Path) -> list[dict]:
    with p.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def check_rates(p: Path, k_col: str, n_col: str, up_col: str, out: list) -> None:
    worst = 0.0
    rs = rows(p)
    for r in rs:
        k, n = int(float(r[k_col])), int(float(r[n_col]))
        worst = max(worst, abs(cp_upper(k, n, RATE_ALPHA) - float(r[up_col])))
    out.append({"table": p.name, "kind": "rate upper bound", "records": len(rs), "max_abs_error": worst})


def check_floors(p: Path, k_col: str, n_col: str, lo_col: str, out: list) -> None:
    worst = 0.0
    rs = rows(p)
    for r in rs:
        k, n = int(float(r[k_col])), int(float(r[n_col]))
        worst = max(worst, abs(cp_lower(k, n, RATE_ALPHA) - float(r[lo_col])))
    out.append({"table": p.name, "kind": "pool floor lower bound", "records": len(rs), "max_abs_error": worst})


def check_pairs(p: Path, n10_col: str, n01_col: str, n_col: str, lo_col: str, hi_col: str, out: list) -> None:
    worst = 0.0
    rs = rows(p)
    for r in rs:
        n10, n01, n = int(float(r[n10_col])), int(float(r[n01_col])), int(float(r[n_col]))
        lo = cp_lower(n10, n, PAIR_GAMMA) - cp_upper(n01, n, PAIR_GAMMA)
        hi = cp_upper(n10, n, PAIR_GAMMA) - cp_lower(n01, n, PAIR_GAMMA)
        worst = max(worst, abs(lo - float(r[lo_col])), abs(hi - float(r[hi_col])))
    out.append({"table": p.name, "kind": "paired interval", "records": len(rs), "max_abs_error": worst})


def headline_ranges(data: Path) -> dict:
    """Ranges quoted in the article, rebuilt from the tables (percent)."""
    cov = rows(data / "supplementary_tables/R6_coverage_risks.csv")
    mea = rows(data / "supplementary_tables/R6_measurement_risks.csv")
    att = rows(data / "supplementary_tables/R6_attribution_risks.csv")
    cont = rows(data / "supplementary_tables/R6_continuous_24_registered_risks_safe.csv")

    def rng(vals):
        vals = [100 * v for v in vals]
        return [round(min(vals), 3), round(max(vals), 3)]

    out = {}
    for case in ("A", "C"):
        full = [float(r["risk"]) for r in cov if r["case"] == case and r["policy"] == "Q2_safe_delta"]
        out[f"full-pool risk, case {case} (18 fresh tests)"] = rng(full)
        out[f"full-pool max adjusted upper, case {case}"] = round(100 * max(float(r["adjusted_upper"]) for r in cov if r["case"] == case and r["policy"] == "Q2_safe_delta"), 4)
        cid = "dev_case_v3a" if case == "A" else "dev_case_v3c"
        ua = [float(r["failure_rate"]) for r in mea if r["case"] == cid and r["policy"] == "Q2_postUA_0p5"]
        out[f"full-pool uncertainty-aware risk at 0.5 SD noise, case {case}"] = rng(ua)
        for q in ("Q0", "Q1", "Q2", "Q3", "Q4"):
            d = [float(r["failure_rate"]) for r in att if r["case"] == cid and r["world"] == "SD-H0" and r["policy"] == f"{q}_delta"]
            out[f"nutrient-informed risk on {q}, case {case}, TAB/C0"] = rng(d)
    for case in ("R0", "B"):
        for pol in ("oldQ3_delta", "new_trainingbest_static"):
            v = [float(r["failure_rate"]) for r in cont if r["case"] == case and r["policy"] == pol]
            out[f"{case}: {pol}"] = rng(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/aggregate_results"))
    ap.add_argument("--tolerance", type=float, default=1e-9)
    a = ap.parse_args()
    st = a.data / "supplementary_tables"
    checks: list = []
    check_rates(st / "R6_attribution_risks.csv", "failure_count", "n_states", "adjusted_upper", checks)
    check_rates(st / "R6_coverage_risks.csv", "k", "n", "adjusted_upper", checks)
    check_rates(st / "R6_measurement_risks.csv", "failure_count", "n_states", "adjusted_upper_recomputed", checks)
    check_rates(st / "R6_information_rates.csv", "failure_count", "n_states", "adjusted_upper", checks)
    check_rates(st / "R6_reverse_rates.csv", "failure_count", "n_states", "adjusted_upper", checks)
    check_rates(st / "R6_continuous_24_registered_risks_safe.csv", "failure_count", "n", "adjusted_one_sided_upper", checks)
    check_floors(st / "R6_attribution_fixed_menu_lower.csv", "all_fail_count", "n_states", "adjusted_lower", checks)
    check_floors(st / "R6_continuous_6_fixed_menu_floors_safe.csv", "all_fail_count", "n", "adjusted_lower", checks)
    check_pairs(st / "R6_attribution_pairs.csv", "n10", "n01", "n_states", "ci_low", "ci_high", checks)
    check_pairs(st / "R6_coverage_pairs.csv", "policy_only_failure", "comparator_only_failure", "n", "lower", "upper", checks)
    check_pairs(st / "R6_measurement_pairs.csv", "policy_only_failure", "comparator_only_failure", "n_states", "adjusted_lower", "adjusted_upper", checks)
    check_pairs(st / "R6_information_paired.csv", "recourse_only_failure", "constant_only_failure", "n", "adjusted_difference_lower", "adjusted_difference_upper", checks)
    check_pairs(st / "R6_reverse_paired.csv", "recourse_only_failure", "constant_only_failure", "n", "adjusted_difference_lower", "adjusted_difference_upper", checks)
    check_pairs(st / "R6_continuous_30_registered_pairs_safe.csv", "policy_only_failure", "comparator_only_failure", "n", "adjusted_lower", "adjusted_upper", checks)
    ok = all(c["max_abs_error"] <= a.tolerance for c in checks)
    report = {"status": "PASS" if ok else "FAIL", "tolerance": a.tolerance, "checks": checks, "headline_ranges_percent": headline_ranges(a.data)}
    print(json.dumps(report, indent=1))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
