# Data

* `synthetic_test_only/` — invented inputs for the tests and the synthetic demo. They are marked
  `is_synthetic: true`, are rejected by the validator in `pilot` / `official` mode and say nothing about real feeds.
* `data_dictionary.md` — fields, units and keys of every holder-side table the code reads.
* `restricted_local/` — **not distributed.** The code expects the licensed inputs here (NASEM 2021 Table 19-1/19-3
  transcriptions, the NASEM feed-library extract, Table 5-1 / Table 21-3 values, build constants and the build
  outputs derived from them). NASEM (2021) is © National Academy of Sciences, all rights reserved; values from it,
  and quantities that return them in one step, are not published in this repository. A holder of a licensed copy
  rebuilds the directory as described in the top-level `README.md`; `scripts/preflight_dev_case.py` lists what is
  missing and never prints a value.

Nothing under `restricted_local/` may be committed (see `.gitignore`).
