#!/usr/bin/env python3
"""Draw the four main-text figures from the aggregate result tables.

No simulation is run: every plotted value is read from the CSV files under
``data/aggregate_results`` and written back as a JSON record next to the figures so that
the plotted numbers can be checked against the tables.

Usage::

    python extensions/figures/make_main_figures.py --data data/aggregate_results --out paper/figures
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

COLORS = ["#0072B2", "#D55E00", "#009E73"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p: Path) -> list[dict]:
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "lines.markeredgewidth": 1.3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 1.2,
        "xtick.major.width": 1.2, "ytick.major.width": 1.2, "savefig.dpi": 300,
    })


def save(fig, out: Path, name: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    return {"pdf_sha256": sha(out / f"{name}.pdf"), "png_sha256": sha(out / f"{name}.png")}


# ----------------------------------------------------------------------------- Fig. 1
def fig_decision_structure(out: Path) -> dict:
    fig, ax = plt.subplots(figsize=(5.2, 5.55))
    fig.subplots_adjust(left=0.012, right=0.988, top=0.995, bottom=0.005)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    blue, green, ink, grey = "#21576B", "#376849", "#1F3039", "#707C82"

    def box(x, y, w, h, title, body, edge=blue, fill="#F1F5F6"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.007,rounding_size=0.012",
                                    linewidth=1.4, edgecolor=edge, facecolor=fill))
        ax.text(x + w / 2, y + h - 0.018, title, ha="center", va="top", fontsize=9.2, fontweight="bold", color=edge)
        ax.text(x + w / 2, y + h - 0.052, body, ha="center", va="top", fontsize=8.5, color=ink, linespacing=1.23)

    def arrow(x1, y1, x2, y2, color=blue, style="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
                                     linewidth=1.4, color=color, linestyle=style))

    box(0.045, 0.875, 0.91, 0.103, "Shared case and evaluation definition",
        "Reference cow and inventory; fixed nutrient targets\nPlanned supply; declared composition distribution",
        edge=green, fill="#EDF4EE")
    box(0.045, 0.72, 0.91, 0.113, "Training: freeze one candidate pool Q",
        "Build Q from admissible rations and physical mixtures\nFreeze the fixed ration, the selection rules and DM moments",
        edge=green, fill="#EDF4EE")
    arrow(0.5, 0.867, 0.5, 0.840, green)
    ax.plot([0.025, 0.975], [0.685, 0.685], color=grey, linewidth=1.4, linestyle=(0, (4, 3)))
    ax.text(0.5, 0.696, "Rules fixed before independent test sampling", ha="center", va="center", fontsize=8.5, color=grey)
    box(0.045, 0.56, 0.91, 0.088, "Test state from the declared model",
        "Nutrient concentrations θ and actual ingredient DM d", edge=grey)
    arrow(0.5, 0.674, 0.5, 0.655, green)
    box(0.045, 0.39, 0.41, 0.12, "Fixed choice from Q", "No nutrient observation\nFeed the frozen fixed ration")
    box(0.545, 0.39, 0.41, 0.12, "Nutrient-informed choice from Q", "Ideal non-DM nutrients observed\nFrozen score; DM hidden")
    arrow(0.25, 0.552, 0.25, 0.518); arrow(0.75, 0.552, 0.75, 0.518)
    box(0.045, 0.257, 0.41, 0.082, "Fixed as-fed ration", "Same q in every test state")
    box(0.545, 0.257, 0.41, 0.082, "Selected as-fed ration", "One ration q(θ) from Q")
    arrow(0.25, 0.382, 0.25, 0.347); arrow(0.75, 0.382, 0.75, 0.347)
    box(0.045, 0.115, 0.91, 0.10, "Common evaluation through the selected NASEM chain",
        "Chosen q with realised nutrients and DM; original domain\nFailure = known violation or unresolved verdict", edge=grey)
    arrow(0.25, 0.249, 0.25, 0.223); arrow(0.75, 0.249, 0.75, 0.223)
    arrow(0.5, 0.552, 0.5, 0.223, grey, (0, (3, 3)))
    ax.text(0.5, 0.388, "Actual DM: evaluator only", ha="center", va="center", rotation=90, fontsize=8.5, color=grey,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7})
    box(0.045, 0.012, 0.91, 0.079, "Report per case, dependence structure and training replicate",
        "Failure rates; paired comparisons; descriptive costs", edge=grey, fill="#FAFBFB")
    arrow(0.5, 0.107, 0.5, 0.098, grey)
    return {"name": "Fig1_decision_structure", "records": [], **save(fig, out, "Fig1_decision_structure")}


# ----------------------------------------------------------------------------- Fig. 2
def fig_candidate_pools(data: Path, out: Path) -> dict:
    rs = data / "supplementary_tables/R6_attribution_risks.csv"
    ps = data / "supplementary_tables/R6_attribution_pairs.csv"
    rates, pairs = rows(rs), rows(ps)
    fig = plt.figure(figsize=(5.2, 5.55))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 1], hspace=0.65, wspace=0.30)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    records = []
    for col, (ax, case, name, target) in enumerate(zip(axs, ["dev_case_v3a", "dev_case_v3c"], ["A", "C"], [5, 2])):
        for mode, marker, offset in [("static", "s", -0.13), ("delta", "o", 0.13)]:
            for root in range(3):
                for j, q in enumerate(["Q0", "Q1", "Q2", "Q3", "Q4"]):
                    r = next(r for r in rates if r["case"] == case and r["world"] == "SD-H0"
                             and r["root"] == f"r{root}" and r["policy"] == q + "_" + mode)
                    x = j + offset + (root - 1) * 0.07
                    v, u = 100 * float(r["failure_rate"]), 100 * float(r["adjusted_upper"])
                    ax.plot([x, x], [v, u], color=COLORS[root], linewidth=1.4)
                    ax.plot(x, u, "_", color=COLORS[root], markersize=4, markeredgewidth=1.3)
                    ax.plot(x, v, marker=marker, color=COLORS[root],
                            markerfacecolor="white" if mode == "static" else COLORS[root], markersize=3.8, markeredgewidth=1.2)
                    records.append({"case": name, "root": root, "policy": q + "_" + mode, "risk_percent": v, "upper_percent": u})
        ax.axhline(target, color="#666666", linestyle="--", linewidth=1.3)
        ax.set_ylim(0, 16); ax.set_xlim(-0.45, 4.45)
        ax.set_xticks(range(5), ["Q0", "Q1", "Q2", "Q3", "Q4"]); ax.set_xlabel("Candidate pool")
        ax.set_yticks([0, 5, 10, 15]); ax.set_ylabel("Failure probability (%)" if col == 0 else "")
        ax.set_title(f"({chr(97 + col)}) Case {name}", loc="left", fontweight="bold", fontsize=9.5)
        ax.grid(axis="y", color="#E5E5E5", linewidth=1.1); ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(gs[1, :]); labels = []
    cells = [(c, n, r) for c, n in [("dev_case_v3a", "A"), ("dev_case_v3c", "C")] for r in range(3)]
    for j, (case, name, root) in enumerate(cells):
        p = next(r for r in pairs if r["case"] == case and r["world"] == "SD-H0" and r["root"] == f"r{root}"
                 and r["policy"] == "Q3_delta" and r["comparator"] == "Q2_delta")
        v, lo, hi = [100 * float(p[x]) for x in ["difference", "ci_low", "ci_high"]]
        ax.plot([lo, hi], [j, j], color=COLORS[root], linewidth=1.4)
        ax.plot([lo, hi], [j, j], "|", color=COLORS[root], markersize=7, markeredgewidth=1.4)
        ax.plot(v, j, "o", color=COLORS[root], markersize=4)
        labels.append(f"{name}, replicate {root}")
        records.append({"case": name, "root": root, "paired_policy": "Q3_delta", "paired_comparator": "Q2_delta",
                        "difference_points": v, "lower_points": lo, "upper_points": hi})
    ax.axvline(0, color="#666666", linewidth=1.3); ax.set_ylim(5.5, -0.5); ax.set_yticks(range(6), labels)
    ax.set_xlabel("Q3 minus Q2 failure (percentage points)")
    ax.set_title("(c) Compression: paired differences", loc="left", fontweight="bold", fontsize=9.5)
    ax.grid(axis="x", color="#E5E5E5", linewidth=1.1); ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([], [], color=c, marker="o", linestyle="", label=f"Replicate {i}", markersize=4) for i, c in enumerate(COLORS)]
    handles += [Line2D([], [], color="#555555", marker="s", markerfacecolor="white", linestyle="", label="Fixed rule", markersize=4),
                Line2D([], [], color="#555555", marker="o", linestyle="", label="Nutrient-informed rule", markersize=4)]
    fig.legend(handles=handles, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.55, 0.003),
               fontsize=8.5, columnspacing=0.7, handletextpad=0.2)
    fig.subplots_adjust(left=0.18, right=0.97, top=0.935, bottom=0.17)
    return {"name": "Fig2_candidate_pools", "records": records, "sources": {rs.name: sha(rs), ps.name: sha(ps)},
            **save(fig, out, "Fig2_candidate_pools")}


# ----------------------------------------------------------------------------- Fig. 3
def fig_coverage_measurement(data: Path, out: Path) -> dict:
    cov_p = data / "supplementary_tables/R6_coverage_risks.csv"
    mea_p = data / "supplementary_tables/R6_measurement_risks.csv"
    cov, mea = rows(cov_p), rows(mea_p)
    plt.rcParams.update({"axes.linewidth": 1.25, "xtick.major.width": 1.25, "ytick.major.width": 1.25})
    fig, axs = plt.subplots(2, 2, figsize=(5.2, 5.55))
    records = []
    for col, case in enumerate(["A", "C"]):
        ax = axs[0, col]; ys, labels = [], []
        for source_i, source in enumerate(["C0", "C1_Table6_matched", "C2_Corn4D_author"]):
            for pool_i, policy in enumerate(["Q3_delta", "Q2_safe_delta"]):
                y = 2 * source_i + pool_i; ys.append(y)
                labels.append(f"{source[:2]}: " + ("compressed Q3" if pool_i == 0 else "full Q2"))
                sel = [r for r in cov if r["case"] == case and r["target"] == source and r["policy"] == policy]
                assert len(sel) == 3, (case, source, policy)
                for r in sel:
                    root = int(r["root"]); rate = 100 * float(r["risk"]); upper = 100 * float(r["adjusted_upper"])
                    yy = y + (root - 1) * 0.16
                    ax.plot([rate, upper], [yy, yy], color=COLORS[root], linewidth=1.35)
                    ax.plot(upper, yy, marker="|", color=COLORS[root], markersize=7, markeredgewidth=1.35)
                    ax.plot(rate, yy, marker="o", color=COLORS[root], markersize=4.5)
                    records.append({"experiment": "coverage", "case": case, "target": source, "root": root,
                                    "policy": policy, "n": int(r["n"]), "risk_percent": rate, "upper_percent": upper})
        ax.set_yticks(ys, labels=labels if col == 0 else [""] * len(labels)); ax.set_ylim(5.5, -0.5)
        ax.set_xlim(0, 4 if case == "A" else 2.2)
        ax.axvline(1 if case == "C" else 5, color="#666666", linestyle="--", linewidth=1.25)
        ax.set_title(f"({chr(97 + col)}) Full versus compressed pool, {case}", loc="left", fontweight="bold", fontsize=9.5, pad=8)
        ax.set_xlabel("Failure probability (%)")
        for y in [1.5, 3.5]:
            ax.axhline(y, color="#DCDCDC", linewidth=1.1, zorder=0)
        ax.grid(axis="x", color="#E5E5E5", linewidth=1.1, zorder=0); ax.spines[["top", "right"]].set_visible(False)

        policies = ["Q3_ideal_delta", "Q3_raw_noise_0p5", "Q3_postmean_0p5", "Q3_postUA_0p5", "Q2_postUA_0p5"]
        labels = ["Ideal nutrients: Q3", "Raw observation: Q3", "Posterior mean: Q3", "Uncertainty-aware: Q3", "Uncertainty-aware: Q2"]
        ax = axs[1, col]
        case_id = "dev_case_v3a" if case == "A" else "dev_case_v3c"
        for y, policy in enumerate(policies):
            sel = [r for r in mea if r["case"] == case_id and r["policy"] == policy]
            assert len(sel) == 3, (case_id, policy)
            for r in sel:
                root = int(r["training_root_index"]); rate = 100 * float(r["failure_rate"])
                upper = 100 * float(r["adjusted_upper_recomputed"]); yy = y + (root - 1) * 0.16
                ax.plot([rate, upper], [yy, yy], color=COLORS[root], linewidth=1.35)
                ax.plot(upper, yy, marker="|", color=COLORS[root], markersize=7, markeredgewidth=1.35)
                ax.plot(rate, yy, marker="o", color=COLORS[root], markersize=4.5)
                records.append({"experiment": "measurement", "case": case, "target": "C0", "root": root,
                                "policy": policy, "n": int(r["n_states"]), "risk_percent": rate, "upper_percent": upper})
        ax.set_yticks(range(len(labels)), labels=labels if col == 0 else [""] * len(labels)); ax.set_ylim(4.5, -0.5)
        ax.set_xlim(0, 12 if case == "A" else 10)
        ax.axvline(5 if case == "A" else 2, color="#666666", linestyle="--", linewidth=1.25)
        ax.set_xlabel("Failure probability (%)")
        ax.set_title(f"({chr(99 + col)}) Nutrient noise 0.5 SD, {case}", loc="left", fontweight="bold", fontsize=9.5, pad=8)
        ax.grid(axis="x", color="#E5E5E5", linewidth=1.1, zorder=0); ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([], [], color=c, marker="o", linewidth=1.35, label=f"Replicate {i}", markersize=4.5) for i, c in enumerate(COLORS)]
    fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.55, 0.014), frameon=False,
               handlelength=1.0, handletextpad=0.25, columnspacing=0.7, fontsize=8.5)
    fig.subplots_adjust(left=0.34, right=0.97, top=0.94, bottom=0.15, wspace=0.35, hspace=0.42)
    return {"name": "Fig3_coverage_measurement", "records": records, "sources": {cov_p.name: sha(cov_p), mea_p.name: sha(mea_p)},
            **save(fig, out, "Fig3_coverage_measurement")}


# ----------------------------------------------------------------------------- Fig. 4
def fig_failure_vs_shortage(data: Path, out: Path) -> dict:
    a_p = data / "derived_tables/R7_descriptive_same_protocol_differences.csv"
    r_p = data / "derived_tables/R7_NEL_all96_rule_records_safe.csv"
    a, r = rows(a_p), rows(r_p)
    key = lambda z: (z["family"], z["case"], z["scenario"], z["training_rep_id"], z["policy"])  # noqa: E731
    risk = {key(z): float(z["joint_failure_rate_total"]) for z in r}
    a = [z for z in a if z["constraint_id"] == "PN-NEL-FIXEDDMI"]
    configs = [
        ("a  Full pool: coverage", "coverage_formal", [("Q2_safe_delta", "Q3_delta", "o", "Q2 − Q3")]),
        ("b  Full pool: measurement", "measurement_formal", [("Q2_ideal_delta", "Q3_ideal_delta", "o", "Ideal"),
                                                            ("Q2_postUA_0p1", "Q3_postUA_0p1", "s", "Noise 0.1"),
                                                            ("Q2_postUA_0p5", "Q3_postUA_0p5", "^", "Noise 0.5")]),
        ("c  Uncertainty-aware score: noise 0.1", "measurement_formal", [("Q3_postUA_0p1", "Q3_raw_noise_0p1", "o", "UA − raw"),
                                                                       ("Q3_postUA_0p1", "Q3_postmean_0p1", "s", "UA − mean")]),
        ("d  Uncertainty-aware score: noise 0.5", "measurement_formal", [("Q3_postUA_0p5", "Q3_raw_noise_0p5", "o", "UA − raw"),
                                                                       ("Q3_postUA_0p5", "Q3_postmean_0p5", "s", "UA − mean")]),
    ]
    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.linewidth": 1, "xtick.major.width": 1, "ytick.major.width": 1, "lines.linewidth": 1.1, "pdf.fonttype": 42})
    fig, axs = plt.subplots(2, 2, figsize=(5.6, 5.65)); colors = {"A": "#2166AC", "C": "#B2182B"}
    points = []
    for ax, (title, fam, groups) in zip(axs.flat, configs):
        for pol, comp, marker, lab in groups:
            for z in [z for z in a if z["family"] == fam and z["policy"] == pol and z["comparator"] == comp]:
                kp = key(z); kc = kp[:-1] + (comp,)
                x = 100 * (risk[kp] - risk[kc]); y = 100 * float(z["normalized_top_ceil5pct_mean_delta"])
                ax.scatter(x, y, s=27, c=colors[z["case"]], marker=marker, linewidths=1.0, edgecolors="white", alpha=0.9, zorder=3)
                points.append({**{k: z[k] for k in ["family", "case", "scenario", "training_rep_id", "policy", "comparator"]},
                               "risk_difference_percentage_points": x, "NEL_tail_difference_percent_of_target": y, "panel": title[0]})
        ax.axhline(0, color="#555555", lw=1); ax.axvline(0, color="#777777", lw=1)
        ax.set_title(title, loc="left", pad=8)
        ax.set_xlabel("Joint failure difference (pp)", labelpad=5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_ylabel("NEL top-5% mean shortage difference\n(% of target)", labelpad=2)
        ax.grid(color="#DDDDDD", linewidth=1, alpha=0.6); ax.set_axisbelow(True)
        handles = [Line2D([], [], color="#555555", marker=m, ls="", markersize=4.5, label=lab) for _, _, m, lab in groups]
        if title[0] != "a":
            ax.legend(handles=handles, fontsize=8.0, frameon=False, loc="best", handletextpad=0.3, borderpad=0.2)
    fig.legend(handles=[Line2D([], [], color=c, marker="o", ls="", markersize=5, label="Case " + k) for k, c in colors.items()],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.007), fontsize=8.5)
    fig.subplots_adjust(left=0.18, right=0.99, top=0.95, bottom=0.155, hspace=0.48, wspace=0.68)
    return {"name": "Fig4_failure_vs_shortage", "records": points, "sources": {a_p.name: sha(a_p), r_p.name: sha(r_p)},
            **save(fig, out, "Fig4_failure_vs_shortage")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/aggregate_results"))
    ap.add_argument("--out", type=Path, default=Path("paper/figures"))
    args = ap.parse_args()
    style()
    bindings = [fig_decision_structure(args.out), fig_candidate_pools(args.data, args.out),
                fig_coverage_measurement(args.data, args.out), fig_failure_vs_shortage(args.data, args.out)]
    (args.out / "figure_bindings.json").write_text(json.dumps({"new_draws": 0, "figures": bindings}, indent=1) + "\n")
    for b in bindings:
        print(b["name"], "records:", len(b["records"]))


if __name__ == "__main__":
    main()
