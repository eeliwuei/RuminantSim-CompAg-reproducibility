# Candidate availability and uncertainty-aware selection shape the reliability of dairy rations under variable feed composition

Code, specifications, aggregate result tables and figure scripts for the article of the same title by
Wei Liu and Seojin Jang (DeepFarm Inc.).

The study asks which of two levers restores the model-defined reliability of a dairy ration when feed
composition varies: the set of candidate rations available at the moment of choice, or the information
observed before choosing. One reference cow, selected NASEM (2021) equations, matched candidate pools and
20,000 independent test states per training replicate are used throughout; the actual dry matter of each
ingredient is hidden from every selection rule and enters only the evaluation.

## What this repository reproduces

| Part | Where | Needs restricted inputs? |
|---|---|---|
| Every reported confidence bound and target decision, rebuilt from the stored counts | `extensions/analysis/recompute_bounds_from_aggregates.py` | no |
| The four main-text figures, drawn from the aggregate tables | `extensions/figures/make_main_figures.py` | no |
| Exact-tail and algebra checks on the research-extension tables | `extensions/analysis/check_exact_and_algebra.py` | no |
| Formulation and evaluation engine (data model, uncertainty factory, optimisation, evaluator, run records) with unit, numerical and integration tests and a synthetic demo | `src/ration_reliability/`, `tests/`, `scripts/run_synthetic_demo.py` | no |
| Specification of the reference problem, constraint table and uncertainty layers | `configs/`, `docs/` | values withheld (see *Data policy*) |
| Readback scripts that produced the aggregate tables from the saved test cells of the later protocols | `extensions/readback_from_saved_cells/` | yes (the controlled study archive) |
| Saved-state readers for the shortage reanalysis and the fixed-action coefficient sensitivity | `extensions/saved_state_readers/` | yes (the controlled study archive) |
| SHA-256 digests of the six frozen protocols of the later experiments | `extensions/protocols/PROTOCOL_SHA256.txt` | — |
| Observed-composition LP re-solve control (exploratory, development streams) | `experiments/E4_resolve_lp_control/`, results in `data/aggregate_results/resolve_lp_control/` | yes to rerun (the reference problem); results provided |
| Released as-fed rations of every fixed rule (one decimal) and equation-derived requirement amounts, with the identifiability check | `data/aggregate_results/released_rations_and_requirements/` | no |

## Layout

```
src/ration_reliability/      engine (not installed; tests put src/ on sys.path)
experiments/                 development and official-run drivers, freeze pins (original study)
scripts/                     environment check, dev-case build / preflight, synthetic demo, output verifiers
tests/                       unit/, numerical/, integration/
configs/                     study configuration (restricted numbers are null + value_ref)
data/aggregate_results/      the aggregate tables behind every number in the article
   supplementary_tables/     Supplementary Data tables (S6–S15, R6_* families, v3/ per-case per-root tables)
   derived_tables/           shortage, denominator and coefficient-sensitivity tables (R7_*)
   released_rations_and_requirements/  fixed-rule rations, derived requirement amounts, identifiability check
   resolve_lp_control/       results of the observed-composition LP re-solve control
data/synthetic_test_only/    invented data for the software demo
docs/                        reference problem, uncertainty layers, decision timing and units, engine API
extensions/                  analysis, figures, readback and saved-state scripts of the later protocols
paper/figures/               the main-text figures as submitted
PUBLIC_TREE_MANIFEST.json    sha256 of every file of this tree
```

Most specification documents and configuration comments are written in Chinese; code, identifiers, data
tables and this README are in English.

## Quick start

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements-lock.txt            # engine and tests
pip install -r requirements-extensions.txt      # numpy, scipy, pandas, matplotlib for the extension scripts

python extensions/analysis/recompute_bounds_from_aggregates.py --data data/aggregate_results
python extensions/figures/make_main_figures.py --data data/aggregate_results --out paper/figures
python -m pytest -q tests
```

The first command rebuilds all one-sided Clopper–Pearson upper bounds (per-claim error 0.05/10,000), the
all-ration failure floors of the frozen pools, and the four-tail paired intervals (per-tail error
0.05/40,000) from the counts in the tables, compares them with the stored values and prints the headline
ranges quoted in the article. The second redraws Figs. 1–4 and writes `figure_bindings.json` with every
plotted value.

## Data policy

The numerical inputs taken from NASEM (2021) *Nutrient Requirements of Dairy Cattle* (8th rev. ed.,
doi:10.17226/25806) are not redistributed, nor are journal table values whose redistribution licence is
undecided, the frozen ration vectors (from which those values could be recovered), fitted composition
models, saved simulation states or random-stream seeds. Configuration files therefore carry `null` plus a
`value_ref` pointer where a restricted number belongs. The engine, the tests and the synthetic demo run
without them; the reference problem itself runs only after the inputs have been rebuilt from a licensed copy
(`docs/` and `data/README.md`). The aggregate tables contain counts, rates, bounds, paired outcomes and
normalised shortage summaries only. Readback and saved-state scripts are provided for provenance and run
only against the controlled study archive, which is available to editors and reviewers from the
corresponding author.

## Citation

See `CITATION.cff`. Until the article appears, cite the repository release.

## Licence

MIT (see `LICENSE`).
