"""Reference energy check: the linear fixed-DMI NEL row versus the project's nonlinear Chapter 3 chain.

Review round 3 (F3; next-round instruction D).  Status (contract section 2.3): ``implemented``,
``unit_passed`` (``tests/numerical/test_energy_reference.py``; synthetic hand-computed cases only); evidence
level ``code_tested``.  Passing the tests shows that the per-state verdicts are computed as documented --
not that any diet meets any cow's energy needs.

Why
---
The fixed-DMI linearisation (:mod:`.energy`) freezes part of the digestibility at the feed means
(Eq 3-3a dNDF_base at the mean NDF and lignin; Eq 3-9 concentrations and DMI/BW at the scenario DMI), whereas
the whole-diet chain :func:`.energy.nonlinear_diet_nel` recomputes them for each drawn state.  The linear row
is therefore **not** a draw-wise lower bound: with the drawn forage NDF below its mean it can lie above the
chain (the round-3 probe: +0.363398 Mcal/d on a synthetic diet).  The linear row may still be used to
*generate* candidates; every executed ration is then scored state by state by both models here, and the
disagreements are reported as false passes / false fails with the error distribution.  A difference is a
**model difference** (``model_difference_not_animal_outcome``), never an animal result.

What the reference chain is -- and is not
-----------------------------------------
:func:`.energy.nonlinear_diet_nel` implements NASEM (2021) Eq 3-1 ... 3-12 for the whole diet with the
state's own DM supply as DMI, its own diet starch (Eq 3-5a) and DMI/BW (Eq 3-5a/3-5b), the drawn NDF in
Eq 3-3a, and Eq 3-9 at the diet concentrations.  It keeps fixed: undigested microbial CP ``fMCP`` (16.5 g/kg
DMI, the Table 19-1 convention of p.27; the model proper predicts microbial CP, Eq 3-6b), endogenous fecal ROM
34.3 g/kg DMI, FA and lignin at the feed values, the RUP / dRUP fractions and the base starch and FA
digestibilities.  It assumes adequate RDP (p.24, p.26: energy is overestimated if RDP is deficient) and has no
supplemental NPN, fat supplements, monensin or infusions.  It is **not** the full NASEM model or the
``nasem_dairy`` software, and it is not an animal validation (``reports/energy_domain_audit.md`` §3).
A verdict of this chain is therefore labelled ``verdict_model_status = research_assumption`` in every summary
(``verdict_status`` block; red team D/C): a sourced requirement threshold does not make the verdict model
sourced -- a row judged by it is ``requirement_sourced/model_assumption``.

Supply is not intake: the reference uses DMI = the DM actually supplied in the state (``sum_i q_i d_i``)
and assumes it is eaten; eating behaviour is not modelled or validated.

Decision information (contract T2.1): the executed ration ``q`` (kg as fed) is fixed.  In every state the
realised DM is ``x_s = q * d_s``; it is never renormalised to the planned DM and ``q`` is never re-derived
from the hidden ``d_s`` (no re-feeding with the hidden DM).

API (also ``docs/ENGINE_API.md`` §14)
-------------------------------------
::

    spec = EnergyReferenceSpec.from_constraint(lin, constraint_spec)       # the PN-NEL-FIXEDDMI row
    res  = reference_energy_check(q, draws, spec)                          # EnergyReferenceResult, per state
    res.summary()                                                          # counts, false pass/fail, error distribution
    res.class_counts(mask)                                                 # class counts inside any state mask
    jt   = joint_with_reference_energy(eval_result, res)                   # joint event with the chain's energy verdict

Per state ``s`` (``R`` = requirement = ``bound + C0``; ``tol`` = the row's numerical tolerance, Mcal/d):

* linear supply ``L_s = sum_i x_si e_si + C0`` with ``e`` the ``NEL_fixedDMI`` column of the draws (checked
  against ``lin.density`` of the same composition; a mismatch means another world and is refused);
  linear margin ``L_s - R`` = the public evaluator's margin of the row; linear fail iff ``-(L_s - R) > tol``;
* reference supply ``N_s`` = :func:`.energy.nonlinear_diet_nel` over the used feeds (``q_i != 0``) with the
  drawn composition columns of ``lin.nutrient_map``; reference fail iff ``-(N_s - R) > tol``;
* ``difference_mcal_d = L_s - N_s`` (> 0: the linear row lies above the reference chain);
* ``state_class``: ``agree_pass``, ``agree_fail``, ``false_pass`` (linear pass, reference fail),
  ``false_fail`` (linear fail, reference pass), ``reference_undefined`` (missing cell of a used feed,
  ``support_violation`` of a used feed -- lignin > NDF or a fraction outside [0, 1] -- or no DM supplied) or
  ``linear_undefined`` (missing cell: the evaluator's ``undefined``);
* ``analysis_anomaly``: a used feed's drawn CP + NDF + starch + EE + ash > 100 % DM or Eq 3-1 ROM < 0
  (``analysis_overlap_or_measurement_anomaly``); the reference is still computed and the state stays in
  every denominator -- the flag only lets the report split the verdicts.

Nothing is clipped, normalised or dropped; undefined states stay in the denominator (``summary`` gives the
reference violation rate as the bounds ``[n_fail / S, (n_fail + n_undefined) / S]``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..datamodel import ConstraintKind, ConstraintSpec, Sense
from ..hashing import stable_hash
from .energy import (
    ANALYSIS_ANOMALY_CLASS,
    ENERGY_COLUMN_ID,
    LINEARISATION_ID,
    SUPPORT_VIOLATION_CLASS,
    NELLinearisation,
    _FMCP_TABLE,
    classify_composition_states,
    nonlinear_diet_nel,
)

__all__ = [
    "REFERENCE_CHAIN_ID",
    "STATE_CLASSES",
    "UNDEFINED_REASONS",
    "MODEL_DIFFERENCE_LABEL",
    "VERDICT_MODEL_STATUS",
    "VERDICT_STATUS_LABEL",
    "REFERENCE_CHAIN_ASSUMPTIONS",
    "EnergyReferenceSpec",
    "EnergyReferenceResult",
    "reference_energy_check",
    "joint_with_reference_energy",
]

#: Version tag of the reference chain configuration (bump on any change of what is held fixed).
REFERENCE_CHAIN_ID = "NASEM2021_ch3_chain_fixedFMCP_suppliedDMI_v1"
#: Per-state classes, in reporting order.
STATE_CLASSES = ("agree_pass", "agree_fail", "false_pass", "false_fail", "reference_undefined", "linear_undefined")
#: Why a reference verdict is undefined (the state stays in every denominator).
UNDEFINED_REASONS = ("missing_data", SUPPORT_VIOLATION_CLASS, "no_dm_supplied", "chain_error")
#: Label carried by every result: a disagreement is a difference between two models, not an animal outcome.
MODEL_DIFFERENCE_LABEL = "model_difference_not_animal_outcome"
#: Status of the reference energy *verdict model* (review round 3 red team, D/C): the requirement it is compared
#: with comes from the source equations, but the verdict rests on the project's reference chain and its assumptions
#: below, so a verdict of this model is ``research_assumption`` -- not "sourced".  Composite label for a row whose
#: threshold is sourced and whose verdict model is this chain: :data:`VERDICT_STATUS_LABEL`.
VERDICT_MODEL_STATUS = "research_assumption"
VERDICT_STATUS_LABEL = "requirement_sourced/model_assumption"
#: What the reference chain holds fixed or assumes (every energy verdict of the main reference event depends on it).
REFERENCE_CHAIN_ASSUMPTIONS = (
    "undigested microbial CP fixed at 16.5 g/kg DMI (p.27, Table 19-1 convention), not predicted (Eq 3-6b)",
    "DMI = the DM supplied in the state, all of it eaten (intake not modelled or validated)",
    "RDP adequate (p.24, p.26: energy is overestimated if RDP is deficient)",
    "forage NDF adequate for rumen conditions (p.24)",
    "FA and lignin at the feed values; base starch and FA digestibilities fixed",
    "no supplemental NPN, fat supplements, monensin or infusions; not the full NASEM model or nasem_dairy",
)
#: g undigested bacterial CP / kg DMI held fixed in the chain (p.27, Table 19-1 convention); one source.
_FMCP_DEFAULT = _FMCP_TABLE


def _ro(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    a.setflags(write=False)
    return a


def _f(v: Any) -> Optional[float]:
    return None if v is None or not math.isfinite(float(v)) else float(v)


# ---------------------------------------------------------------------------------------------
# specification
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class EnergyReferenceSpec:
    """What the reference check compares (one energy row).

    ``lin``: the linearisation of the row (its feeds carry the fixed per-feed parameters and its settings
    the body weight and the Eq 3-10a protein outputs used by the chain).  ``bound_mcal_d``: the engine
    row's bound ``NEL_req - C0`` (so the linear margin is exactly the evaluator's).  ``tolerance_mcal_d``:
    the row's numerical tolerance, used for both verdicts.  ``fmcp_g_per_kg_dmi``: undigested microbial CP
    held fixed in the chain (default 16.5, p.27).  ``column_check_atol``: largest accepted difference
    between the draws' energy column and ``lin.density`` of the same composition.
    """

    lin: NELLinearisation
    bound_mcal_d: float
    tolerance_mcal_d: float
    constraint_id: str = "PN-NEL-FIXEDDMI"
    fmcp_g_per_kg_dmi: float = _FMCP_DEFAULT
    column_check_atol: float = 1e-9

    def __post_init__(self) -> None:
        if not isinstance(self.lin, NELLinearisation):
            raise TypeError("EnergyReferenceSpec.lin must be an energy.NELLinearisation")
        for f in ("bound_mcal_d", "tolerance_mcal_d", "fmcp_g_per_kg_dmi", "column_check_atol"):
            v = float(getattr(self, f))
            if not math.isfinite(v):
                raise ValueError(f"EnergyReferenceSpec.{f} must be finite")
            object.__setattr__(self, f, v)
        if self.tolerance_mcal_d < 0 or self.fmcp_g_per_kg_dmi < 0 or self.column_check_atol < 0:
            raise ValueError("EnergyReferenceSpec: tolerance, fMCP and column_check_atol must be >= 0")

    @classmethod
    def from_constraint(cls, lin: NELLinearisation, constraint: ConstraintSpec, **kw: Any) -> "EnergyReferenceSpec":
        """Spec of an engine NEL row built by :func:`.energy.nel_constraint_spec` (bound, tolerance, id)."""
        if (constraint.kind is not ConstraintKind.SUPPLY or constraint.sense is not Sense.GE
                or dict(constraint.terms) != {ENERGY_COLUMN_ID: 1.0} or constraint.unit != "Mcal/d"):
            raise ValueError(f"{constraint.constraint_id}: not a '{ENERGY_COLUMN_ID} supply >= bound' row in Mcal/d")
        return cls(lin, float(constraint.bound), float(constraint.numerical_tolerance),
                   constraint_id=constraint.constraint_id, **kw)

    @property
    def requirement_mcal_d(self) -> float:
        """NEL requirement ``R = bound + C0`` (Mcal/d)."""
        return float(self.bound_mcal_d + self.lin.constant_mcal_d)

    def fingerprint(self) -> str:
        return stable_hash("EnergyReferenceSpec/v1", REFERENCE_CHAIN_ID, LINEARISATION_ID, self.lin.fingerprint(),
                           self.constraint_id, self.bound_mcal_d, self.tolerance_mcal_d, self.fmcp_g_per_kg_dmi,
                           self.column_check_atol)

    def to_record(self) -> dict[str, Any]:
        """JSON-safe description without feed values (those live in the linearisation record)."""
        return {"reference_chain_id": REFERENCE_CHAIN_ID, "linearisation_id": LINEARISATION_ID,
                "linearisation_fingerprint": self.lin.fingerprint(), "constraint_id": self.constraint_id,
                "bound_mcal_d": self.bound_mcal_d, "requirement_mcal_d": self.requirement_mcal_d,
                "tolerance_mcal_d": self.tolerance_mcal_d, "fmcp_g_per_kg_dmi": self.fmcp_g_per_kg_dmi,
                "column_check_atol": self.column_check_atol, "fingerprint": self.fingerprint(),
                "held_fixed": ["fMCP", "efROM 34.3 g/kg DMI", "FA", "lignin", "RUP/dRUP", "base starch/FA digestibility"],
                "dmi_basis": "supplied DM of the state (assumed eaten; intake not modelled)",
                "not": "not the full NASEM model or software; not an animal validation"}


# ---------------------------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class EnergyReferenceResult:
    """Per-state verdicts of one executed ration (arrays ``[S]``, read-only).  See the module docstring."""

    constraint_id: str
    spec_fingerprint: str
    q_hash: str
    ingredient_ids: tuple[str, ...]
    stream_id: Optional[str]
    model_fingerprint: Optional[str]
    requirement_mcal_d: float
    tolerance_mcal_d: float
    energy_column_source: str
    max_abs_column_deviation: Optional[float]
    linear_supply_mcal_d: np.ndarray
    linear_margin_mcal_d: np.ndarray
    linear_defined: np.ndarray
    linear_violated: np.ndarray
    reference_supply_mcal_d: np.ndarray
    reference_margin_mcal_d: np.ndarray
    reference_defined: np.ndarray
    reference_violated: np.ndarray
    difference_mcal_d: np.ndarray
    state_class: np.ndarray
    undefined_reason: np.ndarray
    analysis_anomaly: np.ndarray
    support_violation: np.ndarray
    diet_dmi_kg_d: np.ndarray
    diet_starch_pct: np.ndarray
    label: str = MODEL_DIFFERENCE_LABEL

    @property
    def n_states(self) -> int:
        return int(self.state_class.shape[0])

    def class_counts(self, mask: Optional[np.ndarray] = None) -> dict[str, int]:
        """Counts of :data:`STATE_CLASSES` among the states selected by ``mask`` (default: all)."""
        m = np.ones(self.n_states, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        if m.shape != (self.n_states,):
            raise ValueError("class_counts: mask must have shape [S]")
        return {c: int(np.sum(m & (self.state_class == c))) for c in STATE_CLASSES}

    def summary(self) -> dict[str, Any]:
        """Counts, rates and the error distribution; denominators keep every state."""
        S = self.n_states
        c = self.class_counts()
        lin_pass = self.linear_defined & ~self.linear_violated
        lin_fail = self.linear_defined & self.linear_violated
        ref_pass = self.reference_defined & ~self.reference_violated
        ref_fail = self.reference_defined & self.reference_violated
        n_ref_und = int(np.sum(~self.reference_defined))
        both = self.linear_defined & self.reference_defined
        n_lp_def = int(np.sum(lin_pass & self.reference_defined))
        n_lf_def = int(np.sum(lin_fail & self.reference_defined))
        dd = self.difference_mcal_d[np.isfinite(self.difference_mcal_d)]
        qs = np.quantile(dd, [0.01, 0.05, 0.5, 0.95, 0.99]) if dd.size else [None] * 5
        reasons = {r: int(np.sum(self.undefined_reason == r)) for r in UNDEFINED_REASONS}
        an = self.analysis_anomaly
        return {
            "label": self.label, "constraint_id": self.constraint_id, "spec_fingerprint": self.spec_fingerprint,
            "q_hash": self.q_hash, "stream_id": self.stream_id, "model_fingerprint": self.model_fingerprint,
            "requirement_mcal_d": self.requirement_mcal_d, "tolerance_mcal_d": self.tolerance_mcal_d,
            "energy_column_source": self.energy_column_source,
            "max_abs_column_deviation": self.max_abs_column_deviation,
            "n_states": S, "class_counts": c,
            "n_linear_defined": int(np.sum(self.linear_defined)), "n_linear_pass": int(np.sum(lin_pass)),
            "n_linear_fail": int(np.sum(lin_fail)),
            "n_reference_defined": int(np.sum(self.reference_defined)), "n_reference_pass": int(np.sum(ref_pass)),
            "n_reference_fail": int(np.sum(ref_fail)), "n_reference_undefined": n_ref_und,
            "reference_undefined_reasons": reasons,
            "n_both_defined": int(np.sum(both)),
            "linear_violation_rate": float(np.sum(lin_fail) / S) if S else None,
            "reference_violation_rate_lower": float(np.sum(ref_fail) / S) if S else None,
            "reference_violation_rate_upper": float((np.sum(ref_fail) + n_ref_und) / S) if S else None,
            "false_pass_share_of_all": float(c["false_pass"] / S) if S else None,
            "false_fail_share_of_all": float(c["false_fail"] / S) if S else None,
            "false_pass_share_of_linear_pass": float(c["false_pass"] / n_lp_def) if n_lp_def else None,
            "false_fail_share_of_linear_fail": float(c["false_fail"] / n_lf_def) if n_lf_def else None,
            "difference_mcal_d": {"n": int(dd.size), "mean": _f(dd.mean()) if dd.size else None,
                                  "sd": _f(dd.std(ddof=1)) if dd.size > 1 else None,
                                  "min": _f(dd.min()) if dd.size else None, "q01": _f(qs[0]), "q05": _f(qs[1]),
                                  "q50": _f(qs[2]), "q95": _f(qs[3]), "q99": _f(qs[4]),
                                  "max": _f(dd.max()) if dd.size else None,
                                  "share_linear_above_reference": float(np.mean(dd > 0)) if dd.size else None,
                                  "definition": "linear supply - reference chain supply (Mcal/d), states where both "
                                                "are defined"},
            "n_analysis_anomaly_states": int(np.sum(an)),
            "analysis_anomaly_class": ANALYSIS_ANOMALY_CLASS,
            "class_counts_in_analysis_anomaly_states": self.class_counts(an),
            "n_support_violation_states": int(np.sum(self.support_violation)),
            "verdict_status": {"verdict_model": REFERENCE_CHAIN_ID, "verdict_model_status": VERDICT_MODEL_STATUS,
                               "label_for_a_sourced_requirement": VERDICT_STATUS_LABEL,
                               "assumptions": list(REFERENCE_CHAIN_ASSUMPTIONS),
                               "note": "a reference verdict depends on the project chain's assumptions; a requirement "
                                       "with a sourced threshold does not make the verdict model sourced"},
            "notes": ("q fixed; x = q * d_state, never renormalised, q never re-derived from the hidden DM",
                      "reference = project Chapter 3 chain (fixed fMCP, FA, lignin; DMI = supplied DM); not full NASEM",
                      "undefined states stay in every denominator; nothing clipped or dropped"),
        }


# ---------------------------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------------------------

def _q_vector(q: Any, n: int) -> np.ndarray:
    qv = np.array(getattr(q, "q_as_fed", q), dtype=float)
    if qv.shape != (n,) or not np.all(np.isfinite(qv)):
        raise ValueError(f"reference_energy_check: q must be finite with shape ({n},)")
    if np.any(qv < 0):
        raise ValueError("reference_energy_check: q must be >= 0 (the structural rule q >= 0 is the evaluator's)")
    return qv


def reference_energy_check(q: Any, draws: Any, spec: EnergyReferenceSpec) -> EnergyReferenceResult:
    """Score the executed ration ``q`` in every drawn state with the linear row and the reference chain.

    Parameters
    ----------
    q : ``[I]`` kg as fed / head / d (or an object with ``q_as_fed``), in the order of ``draws.ingredient_ids``
        (= ``spec.lin.ingredient_ids``; no reordering).  Fixed: never re-derived from the drawn DM.
    draws : a :class:`~ration_reliability.uncertainty.base.DrawSet` (``theta [S, I, J]`` canonical fractions,
        ``d [S, I]`` true DM fractions).  It must carry the columns of ``spec.lin.nutrient_map``; if it carries
        ``NEL_fixedDMI`` that column is used for the linear row (it is what the public evaluator uses) after
        checking it against ``spec.lin.density``.
    spec : :class:`EnergyReferenceSpec`.

    Returns
    -------
    :class:`EnergyReferenceResult` (per-state arrays; ``summary()`` for counts and the error distribution).
    """
    lin = spec.lin
    ids = tuple(draws.ingredient_ids)
    if ids != tuple(lin.ingredient_ids):
        raise ValueError("reference_energy_check: draws and linearisation have different ingredient order")
    nut = list(draws.nutrient_ids)
    theta = np.asarray(draws.theta, dtype=float)
    d = np.asarray(draws.d, dtype=float)
    S, I = d.shape
    qv = _q_vector(q, I)
    used = qv != 0.0
    iu = np.flatnonzero(used)
    x = d * qv[None, :]                                          # realised DM, NOT renormalised

    # ---- linear row: the draws' energy column, checked against the linearisation of the same composition
    e_lin = lin.density(theta, nut)                                # [S, I]; NaN propagates
    max_dev: Optional[float] = None
    if ENERGY_COLUMN_ID in nut:
        e = theta[..., nut.index(ENERGY_COLUMN_ID)]
        fin_e, fin_l = np.isfinite(e), np.isfinite(e_lin)
        if np.any(fin_e[:, iu] != fin_l[:, iu]):
            raise ValueError("reference_energy_check: energy column and linearisation disagree on missing cells")
        both = fin_e & fin_l
        max_dev = float(np.max(np.abs(e[both] - e_lin[both]))) if both.any() else 0.0
        if max_dev > spec.column_check_atol:
            raise ValueError(f"reference_energy_check: the draws' {ENERGY_COLUMN_ID} column differs from the "
                             f"linearisation by {max_dev:.3g} (> {spec.column_check_atol:g}): another world")
        source = "draw_column"
    else:
        e, source = e_lin, "linearisation_density"
    xe_u = x[:, iu] * e[:, iu]
    lin_def = np.all(np.isfinite(xe_u), axis=1)
    part = np.where(lin_def, np.sum(np.where(np.isfinite(xe_u), xe_u, 0.0), axis=1), np.nan)
    C0 = float(lin.constant_mcal_d)
    R = spec.requirement_mcal_d
    tol = spec.tolerance_mcal_d
    lin_supply = part + C0
    lin_margin = part - spec.bound_mcal_d                          # = evaluator margin of the row
    with np.errstate(invalid="ignore"):
        lin_viol = lin_def & (-lin_margin > tol)

    # ---- composition classes (labels only) and the reference chain
    cls = classify_composition_states(theta, nut, ids, lin=lin)
    anomaly = cls.analysis_anomaly[:, iu].any(axis=1) if iu.size else np.zeros(S, dtype=bool)
    support = cls.support_violation[:, iu].any(axis=1) if iu.size else np.zeros(S, dtype=bool)
    fmap = dict(lin.nutrient_map)                                  # engine id -> composition field
    missing_cols = [n for n in fmap if n not in nut]
    if missing_cols:
        raise KeyError(f"reference_energy_check: draws lack the composition columns {missing_cols}")
    jn = {fld: nut.index(eid) for eid, fld in fmap.items()}
    feeds_u = [lin.feeds[i] for i in iu]
    st = lin.settings
    ref_supply = np.full(S, np.nan)
    dmi = np.full(S, np.nan)
    starch = np.full(S, np.nan)
    reason = np.full(S, "", dtype=object)
    comp_u = theta[:, iu, :][:, :, list(jn.values())] if iu.size else np.zeros((S, 0, len(jn)))
    miss = ~(np.all(np.isfinite(x[:, iu]), axis=1) & np.all(np.isfinite(comp_u), axis=(1, 2)))
    for s in range(S):
        if miss[s]:
            reason[s] = "missing_data"
            continue
        if support[s]:
            reason[s] = SUPPORT_VIOLATION_CLASS
            continue
        xs = x[s, iu]
        if not iu.size or not float(xs.sum()) > 0.0:
            reason[s] = "no_dm_supplied"
            continue
        comps = [{fld: float(theta[s, i, j] * 100.0) for fld, j in jn.items()} for i in iu]
        try:
            nl = nonlinear_diet_nel(xs, feeds_u, body_weight_kg=st.body_weight_kg, milk_cp_kg_d=st.milk_cp_kg_d,
                                    body_gain_cp_kg_d=st.body_gain_cp_kg_d, compositions=comps,
                                    fmcp_g_per_kg_dmi=spec.fmcp_g_per_kg_dmi)
        except ValueError:
            reason[s] = "chain_error"
            continue
        ref_supply[s], dmi[s], starch[s] = nl["NEL"], nl["DMI"], nl["starch_pct"]
    ref_def = np.isfinite(ref_supply)
    ref_margin = ref_supply - R
    with np.errstate(invalid="ignore"):
        ref_viol = ref_def & (-ref_margin > tol)
    diff = np.where(lin_def & ref_def, lin_supply - ref_supply, np.nan)

    klass = np.empty(S, dtype="<U20")
    lp, rp = lin_def & ~lin_viol, ref_def & ~ref_viol
    klass[:] = "reference_undefined"
    klass[lin_def & ref_def & lp & rp] = "agree_pass"
    klass[lin_def & ref_def & ~lp & ~rp] = "agree_fail"
    klass[lin_def & ref_def & lp & ~rp] = "false_pass"
    klass[lin_def & ref_def & ~lp & rp] = "false_fail"
    klass[~lin_def] = "linear_undefined"
    return EnergyReferenceResult(
        constraint_id=spec.constraint_id, spec_fingerprint=spec.fingerprint(),
        q_hash=stable_hash("q/v1", ids, qv), ingredient_ids=ids,
        stream_id=getattr(draws, "stream_id", None), model_fingerprint=getattr(draws, "model_fingerprint", None),
        requirement_mcal_d=R, tolerance_mcal_d=tol, energy_column_source=source, max_abs_column_deviation=max_dev,
        linear_supply_mcal_d=_ro(lin_supply), linear_margin_mcal_d=_ro(lin_margin), linear_defined=_ro(lin_def),
        linear_violated=_ro(lin_viol), reference_supply_mcal_d=_ro(ref_supply), reference_margin_mcal_d=_ro(ref_margin),
        reference_defined=_ro(ref_def), reference_violated=_ro(ref_viol), difference_mcal_d=_ro(diff),
        state_class=_ro(klass), undefined_reason=_ro(reason.astype("<U24")), analysis_anomaly=_ro(anomaly),
        support_violation=_ro(support), diet_dmi_kg_d=_ro(dmi), diet_starch_pct=_ro(starch))


# ---------------------------------------------------------------------------------------------
# joint event with the reference energy verdict (for the unified reference evaluation)
# ---------------------------------------------------------------------------------------------

def joint_with_reference_energy(eval_result: Any, energy: EnergyReferenceResult, *,
                                member_ids: Optional[Sequence[str]] = None,
                                margin_atol: float = 1e-9) -> dict[str, Any]:
    """Joint violation event of ``eval_result`` with the energy row judged by the reference chain.

    ``eval_result``: the public evaluator's :class:`~ration_reliability.datamodel.EvaluationResult` of the same
    ``q`` on the same draws (checked: same ``q_hash``, same number of states, same stream id when both are
    known, and -- when the energy row is in ``eval_result`` -- its margins equal ``energy.linear_margin_mcal_d``
    within ``margin_atol``).  ``member_ids``: the rows of the event (default: every
    ``probabilistic_nutrition`` row of ``eval_result``).  The energy row's linear verdict is replaced by the
    reference verdict if the row is a member; every other row keeps the evaluator's verdict.

    Returns arrays ``joint_violation``, ``joint_unknown`` (no violation but an undefined member, including a
    reference-undefined energy row) and the counts ``n_violated``, ``n_unknown``, ``rate_lower``, ``rate_upper``
    (denominator = all states).
    """
    ids = list(eval_result.constraint_ids)
    S = int(eval_result.n_draws)
    if S != energy.n_states:
        raise ValueError("joint_with_reference_energy: different numbers of states")
    if getattr(eval_result, "q_hash", None) != energy.q_hash:
        raise ValueError("joint_with_reference_energy: evaluator result and energy check are for different q")
    sid = getattr(eval_result, "draw_stream_id", None)
    if sid is not None and energy.stream_id is not None and sid != energy.stream_id:
        raise ValueError("joint_with_reference_energy: different draw streams")
    if energy.constraint_id in ids:
        k = ids.index(energy.constraint_id)
        m_ev = np.asarray(eval_result.margin[:, k], dtype=float)
        both = np.isfinite(m_ev) & np.isfinite(energy.linear_margin_mcal_d)
        if np.any(np.isfinite(m_ev) != energy.linear_defined) or \
                (both.any() and float(np.max(np.abs(m_ev[both] - energy.linear_margin_mcal_d[both]))) > margin_atol):
            raise ValueError("joint_with_reference_energy: evaluator energy margins differ from the linear margins "
                             "of the reference check (another world, q or spec)")
    if member_ids is None:
        members = [c for c, kls in zip(ids, eval_result.constraint_classes) if kls == "probabilistic_nutrition"]
    else:
        members = list(member_ids)
        unknown = [m for m in members if m not in ids and m != energy.constraint_id]
        if unknown:
            raise KeyError(f"joint_with_reference_energy: rows {unknown} are not in the evaluator result")
    other = [ids.index(m) for m in members if m != energy.constraint_id]
    viol = np.asarray(eval_result.violated, dtype=bool)
    und = np.asarray(eval_result.undefined, dtype=bool)
    v_o = viol[:, other].any(axis=1) if other else np.zeros(S, dtype=bool)
    u_o = und[:, other].any(axis=1) if other else np.zeros(S, dtype=bool)
    energy_member = energy.constraint_id in members
    if energy_member:
        v_e = np.asarray(energy.reference_violated, dtype=bool)
        u_e = ~np.asarray(energy.reference_defined, dtype=bool)
    else:
        v_e = u_e = np.zeros(S, dtype=bool)
    jv = v_o | v_e
    ju = ~jv & (u_o | u_e)
    return {"member_ids": members, "energy_row_member": energy_member, "energy_verdict": "reference_chain",
            "joint_violation": _ro(jv), "joint_unknown": _ro(ju), "n_states": S,
            "n_violated": int(jv.sum()), "n_unknown": int(ju.sum()),
            "rate_lower": float(jv.sum() / S) if S else None,
            "rate_upper": float((jv.sum() + ju.sum()) / S) if S else None,
            "label": MODEL_DIFFERENCE_LABEL}
