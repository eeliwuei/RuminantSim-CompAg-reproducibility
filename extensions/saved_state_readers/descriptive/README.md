# R7.2 saved-margin descriptive reanalysis

`reanalyse_saved_descriptive.py` reads the **unchanged extracted R7 core reviewer bundle**. It uses only Python's standard library and NumPy; it does not import the nutritional model, core evaluator, uncertainty model or previous tail-reanalysis implementation. All 240 core inputs it uses are bound to the original core manifest and rehashed after reading.

Run with a new output directory:

```sh
python reanalyse_saved_descriptive.py \
  --core /path/to/extracted_R7_core \
  --reference-tail /path/to/R7_1/normalized_severity.csv \
  --received-top5 /path/to/top5_is_rescaled_mean_864.csv \
  --output /path/to/new_descriptive_results
```

The reference-tail table can be produced by the already qualified R7.1 `reproduce_saved_tails.py`. It is a comparison input, never the source of the new values. The separately received top5 CSV is also only a comparison input. Required third-party access conditions of the core are unchanged. A local run does not establish lawful external reviewer access.

The latest actual run is `results_v1/`. Its receipt records input/output/script hashes, versions, maximum comparison errors and scope. Python 3.14.7 and NumPy 2.4.6 were used. The code requires a new output directory and never overwrites original data. No random generator, model evaluator, selection algorithm, new confidence interval or new significance test is used.

## Derived tables

- `top5_identity_and_occurrence_864.csv`: all nine constraints for 96 policy-cell records; positive deficits, original-tolerance violations, defined/unknown denominators, quantiles, exact `n/ceil(.05n)` identity and conditional severity. Values are dimensionless ratios. The all-defined `conditional_mean` name is retained only for compatibility with the earlier table; its denominator includes zero deficits.
- `NEL_frequency_conditional_severity_96.csv`: NEL subset with original joint failure counts. Positive frequency uses defined states; violation and joint failure rates labelled `_total` use all 20,000 states.
- `high_noise_energy_frequency_and_conditional_severity_6.csv`: six same-Q3 posterior-mean-plus-variance versus raw-observation pairs at noise multiplier 0.5. Deltas are policy minus comparator. These are descriptive comparisons, not newly tested hypotheses.
- `fullQ2_vs_Q3_energy_frequency_and_severity_36.csv`: 18 coverage comparisons plus 18 measurement comparisons (six at each ideal, 0.1 and 0.5 condition). Keeping protocol/case/replicate identities prevents interpreting these as 36 independent methods or pooling their observations into a fixed-rule certificate.
- `common_defined_comparisons_42.csv`: the above 42 comparisons restricted to common-defined NEL positions, with overlap and differing violation-set counts.
- `compression_full_information_decomposition_6.csv`: six original A/C TAB attribution cells. Saved bitpacked candidate loss flags are decoded with the original `big` bit order, their Q2/Q3 all-fail indicators recomputed, then compared with stored oracle witnesses and actual chosen-action losses. The legacy `selection_mistake_count` is checked numerically and labelled `oracle_excess_failures` in the new table.

Normalisation is independently reconstructed from the bound frozen parent `main_event_row_specs` and each state's saved DM. The NEL threshold is the actual selected-chain requirement, not the shifted linear bound. Ca and P denominators add respectively 0.9 and 1.0 times saved total DM to the frozen constant bound. Other rows use the absolute frozen threshold. These are the existing definitions, not new biological requirements.

`mathematical_interpretation.md` gives the exact three-term risk decomposition, hidden-state counterexample, marginal-score/union-risk ranking reversal, variance derivative and top5 identity. `check_mathematical_examples.py` runs deterministic illustrative algebra checks; `mathematical_identity_checks_v2.json` records PASS (the earlier receipt is preserved; v2 explicitly defines the illustrative margin as raw signed margin plus original tolerance). They are not scientific simulations.

## Scope

The run recomputes summaries from 1,920,000 saved policy-state positions and reads 94,620,000 saved candidate-flag positions for the six compression cells. This counts data readback, **not new canonical nutritional evaluations**. It does not regenerate inputs, prove nutritional equations independently, reselect actions, establish causality for the observed tradeoff or quantify animal harm. Unknowns and all adverse results remain present. The external R7.2 ZIP and its proposed other tables/scripts were not received; these are newly written local reproductions from the actual archived arrays.

## Separate sensitivity cross-audit

`audit_final_fmcp_sensitivity.py` independently checks the separately executed final-rule fMCP sensitivity. It reconstructs all96×5 saved-action margin/label sets through separate fecal-DE and urinary-UE terms, compares all480 rows and360 statewise paired counts, and inverts the binomial CDF for all480 diagnostic bounds. It imports no nutritional model or evaluator. Actual receipt and limitations are in `sensitivity_crossaudit/`.

```sh
python audit_final_fmcp_sensitivity.py \
  --core /path/to/extracted_R7_core \
  --sensitivity /path/to/sensitivity_directory \
  --baseline-nel /path/to/descriptive/results_v1/NEL_frequency_conditional_severity_96.csv \
  --output /path/to/new_sensitivity_crossaudit
```

The sensitivity directory must include its frozen protocol/source/input list plus completed `results_v1`. Its canonical spot verification is producer evidence, not a second canonical run by this independent aggregate/statewise audit. New scenarios include two rows where positive-deficit frequency exceeds5%; the baseline864-row top5 identity must not be generalised to all480 sensitivity records.
