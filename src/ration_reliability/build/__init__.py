"""Case builders (review round 3, F6; R3F): auditable build logic and input preflight for development cases.

* ``ration_reliability.build.dev_case`` -- the generic logic that turns ``configs/dev_case_v1/*.yaml`` and the
  restricted tables into the engine problem ``dev_case_v1_problem.yaml`` and its companion files.  It was moved out of
  the restricted directory so that the transformation from the sources to the problem can be audited without the
  restricted data.  The module holds **no restricted number**: the few NASEM constants the logic needs are read at run
  time from the restricted directory (``RESTRICTED_CONSTANTS`` of the holder-side wrapper
  ``data/restricted_local/dev_case_v1/build_dev_case_v1.py``).
* ``ration_reliability.build.preflight`` -- lists the files a dev_case driver needs, which of them are missing or stale,
  and how each can be rebuilt; drivers call ``require_dev_case_inputs`` before they read anything.

Submodules are imported explicitly (``from ration_reliability.build import preflight``); importing this package does
not import numpy / scipy / pandas, so the preflight also runs in an environment where the numerical stack is broken.
"""

__all__ = ["dev_case", "preflight"]
