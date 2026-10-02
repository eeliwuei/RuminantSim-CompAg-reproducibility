# R7.2 scientific reanalysis add-on

This add-on depends on the unchanged extracted R7 core reviewer bundle identified in CORE_DEPENDENCY.json. It contains no newly sampled states or newly selected policies. Run each component into a new output directory using Python with requirements.txt. Python 3.14.7 was used for qualification. The core has its own original runtime inputs.

```sh
python descriptive/reanalyse_saved_descriptive.py --core /path/to/extracted_R7_core --reference-tail descriptive/comparison_inputs/normalized_severity.csv --received-top5 descriptive/comparison_inputs/top5_is_rescaled_mean_864.csv --output /path/to/new_descriptive_output
python sensitivity/reproduce_final_fmcp_sensitivity.py --core /path/to/extracted_R7_core --output /path/to/new_sensitivity_output
```

The descriptive reader reproduces 864 rows and six compression contrasts from saved margins/flags; the comparison CSVs do not supply its new values. The sensitivity reader applies the five previously used hypothetical fMCP levels to final frozen actions/states, checks a canonical subset and produces 480 scenario records and 360 descriptive pairs. All methods, denominator distinctions and limitations are in the component READMEs.

No new paired or tail significance tests are run. Diagnostic CP bounds are post-hoc model sensitivity, not new prospective certification. External full-model differences and nutritional/field validity remain outside scope. The original core remains subject to actual source-specific access arrangements; this locally executable add-on does not establish external reviewer access or redistribution permission.

Expected safe result CSVs are provided for byte-level comparison. Qualification receipts are separate from scientific values. All original core files must remain unchanged.
