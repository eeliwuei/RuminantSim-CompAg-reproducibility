"""Method registry.  Methods are loaded lazily by name.

To add a method (other agents): create a new module in this package exposing ``solve`` with the
:class:`SolveFn` signature, then add **one line** to :data:`REGISTRY` below.  Do not edit the
other engine modules; request changes through the log instead.

Every method must

* receive the same :class:`~ration_reliability.datamodel.RationProblem`;
* return a :class:`~ration_reliability.datamodel.SolveResult` with a physical ``q`` (kg
  as-fed/head/d) or no ration at all;
* use only the draws it is entitled to (``opt`` for fitting; ``validation`` only inside explicit
  selection/calibration routines; never ``test``), checked with
  :func:`~ration_reliability.uncertainty.require_stream`;
* leave scoring to :func:`ration_reliability.evaluation.evaluate`.
"""

from __future__ import annotations

import importlib
from typing import Any, Mapping, Optional, Protocol

import numpy as np

__all__ = ["REGISTRY", "METHOD_EQUIVALENCES", "SolveFn", "get_method", "available_methods",
           "equivalence_annotations"]

#: method id -> ``"module.path:function"`` (lazy).
REGISTRY: dict[str, str] = {
    "M0_nominal": "ration_reliability.optimization.m0_nominal:solve",
    "M1_safety_margin": "ration_reliability.optimization.safety_margin:solve",
    "M2_joint_chance_saa": "ration_reliability.optimization.chance_saa:solve",
    "M2b_marginal_bonferroni_saa": "ration_reliability.optimization.chance_saa:solve_marginal_bonferroni",
    "M3a_box_robust": "ration_reliability.optimization.robust:solve_box_robust",
    "M3b_budget_robust": "ration_reliability.optimization.robust:solve_budget_robust",
    "M3c_scenario_set_robust": "ration_reliability.optimization.robust:solve_scenario_set_robust",
}


#: Mathematically equivalent method pairs (contract §9 "均值方法的等价性", acceptance D06).  A
#: report must merge each pair into one method or print the equivalence next to both rows; the
#: pair is never counted as two contributions.  Conditions are the ones under which the two LPs
#: are identical (numerically checked in the unit/integration tests named in ``evidence``).
METHOD_EQUIVALENCES: tuple[dict[str, Any], ...] = (
    {"pair_id": "EQ-M1-M3a",
     "methods": ("M1_safety_margin", "M3a_box_robust"),
     "condition": "M1 with margin_scale='sd' and apply_to_dm=True, and M3a on the box mu +/- k*sigma with the same "
                  "sigma, centred on the nominal composition table and d_hat (BoxUncertaintySet.from_mean_sd with "
                  "the nominal centre): both are the row-wise worst case over the same box (identical LP). M3a with "
                  "params k builds its box around the opt-draw sample mean instead, so the centres differ",
     "difference_that_remains": "M1 selects k on the validation stream; the robust set of M3a is a declared set",
     "evidence": "src/ration_reliability/optimization/safety_margin.py module doc; "
                 "tests/unit/test_safety_margin.py; tests/unit/test_redteam_engine_fixes.py::"
                 "test_m1_and_m3a_on_the_same_box_are_one_method",
     "reporting_rule": "report once, or label both rows 'equivalent to EQ-M1-M3a'"},
    {"pair_id": "EQ-M0-MODES",
     "methods": ("M0_nominal[coefficient_mode=draw_mean]", "M0_nominal[coefficient_mode=nominal_point]"),
     "condition": "zero uncertainty centred on the nominal state (or draw means of the products d_i a_ij equal to "
                  "the nominal products): the mean-coefficient LP equals the nominal LP",
     "difference_that_remains": "none under the condition",
     "evidence": "src/ration_reliability/optimization/m0_nominal.py module doc; reports/mean_nominal_equivalence.md",
     "reporting_rule": "one method (M0)"},
    {"pair_id": "EQ-M3c-M2-ALPHA0",
     "methods": ("M3c_scenario_set_robust", "M2_joint_chance_saa[alpha_train=0]"),
     "condition": "same opt scenarios: floor(alpha N) = 0 forces every z_s = 0, i.e. the scenario-set robust LP",
     "difference_that_remains": "none under the condition",
     "evidence": "src/ration_reliability/optimization/robust.py module doc (scenario set)",
     "reporting_rule": "report as the alpha -> 0 end of the M2 curve, not as an extra method"},
)


def equivalence_annotations(method_id: str) -> tuple[dict[str, Any], ...]:
    """Equivalence records (:data:`METHOD_EQUIVALENCES`) that involve ``method_id``."""
    out = []
    for rec in METHOD_EQUIVALENCES:
        if any(m == method_id or m.startswith(method_id + "[") for m in rec["methods"]):
            out.append(dict(rec))
    return tuple(out)


class SolveFn(Protocol):
    """Signature every registered method must implement."""

    def __call__(self, problem: Any, *, d_hat: Optional[np.ndarray] = None, opt_draws: Any = None,
                 params: Optional[Mapping[str, Any]] = None, solver_options: Any = None) -> Any: ...


def available_methods() -> tuple[str, ...]:
    """Registered method ids (sorted)."""
    return tuple(sorted(REGISTRY))


def get_method(name: str) -> SolveFn:
    """Import and return the ``solve`` callable registered under ``name``."""
    try:
        target = REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown method {name!r}; registered: {available_methods()}") from None
    mod_name, _, fn_name = target.partition(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name or "solve")
    return fn
