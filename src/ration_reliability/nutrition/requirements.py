"""NASEM 2021 dry-matter intake and factorial absorbed Ca / P requirements (A layer, B-133).

Status (contract section 2.3): ``implemented``, ``unit_passed``
(``tests/numerical/test_nutrition_requirements.py``: every row of ``reports/model_audit.md`` section 8).
The section 8.1 rows are one operating point (multiparous, DIM 200, MatBW = BW = 700 kg), at which
several coefficient slips cancel; the non-degenerate working points of section 8.5 (primiparous
DIM 15, MatBW != BW, BW != 700 kg, gestation beyond day 190, mixed parity) are hand-computed and
tested separately (FIX5).
No ``nasem_dairy`` code check has been possible (B-109); where the book is internally inconsistent the
alternatives are implemented side by side and the caller must choose explicitly -- nothing here picks
a version on the user's behalf (DECISIONS D-135; ``configs/animal_profile.yaml``
``version_bundle_until_code_check``).

Source: NASEM (2021) *Nutrient Requirements of Dairy Cattle*, 8th rev. ed., local PDF
``data/restricted_local/nasem_dairy_2021_ncbi_bookshelf.pdf`` (source id ``SRC-C-NASEM21-T19``).
Page numbers are **printed** pages (PDF page = printed + 20).  Equation constants below are the
printed equation coefficients (as transcribed in ``reports/model_audit.md`` sections 3-6 and re-read from
the PDF text); per-feed absorption coefficients are **not** in this module -- they are read from a
restricted table (:func:`load_absorption_coefficient_table`).

Inputs carry units
------------------
Every public function takes :class:`Qty` inputs (``Qty(value, unit)``); bare numbers are refused.
Units come from :mod:`ration_reliability.normalization.units` (``kg``, ``kg/d``, ``g/d``, ``%``,
``fraction``, ``Mcal/d``, ``Mcal/kg``, ...), plus ``d`` (days) for day counts and ``1`` for the
dimensionless indicators (parity indicator / ``An_Parity``, BCS on the 1-5 scale).  Every function
returns an :class:`EquationResult` (or :class:`FactorialRequirement`) that records the equation
numbers, printed pages, inputs and intermediates.

Book inconsistencies handled explicitly (``reports/model_audit.md`` section 11)
------------------------------------------------------------------------------
* DMI: :func:`dmi_eq2_1` (Eq 2-1, p.12) and :func:`dmi_eq20_21_literal` (Eq 20-21 as printed, p.423;
  the BCS term reads ``-0.689 - 1.87 (An_Parity - 1) BCS`` and the unbalanced opening parenthesis is
  read as enclosing the whole sum, as in model_audit section 3.4).  Both exist; no default.
* Mineral growth term: option ``A_An_BWgain`` (model Eq 20-374 / 20-387 driven by An_BWgain) or
  ``B_zero_mature`` (Chapter 7 growth equations are for growing cattle; 0 for a mature cow) --
  PUD-P2a-03.  No default.
* Milk P constant: :func:`p_lactation_eq20_389` (0.48, p.456) or :func:`p_lactation_eq7_8b`
  (0.49, p.113) -- PUD-P2a-04.  No default.
* Gestation: ``text_rule`` (Ca: Eq 7-3 only beyond day 190, p.106-107; P: 0 below day 190, p.112) or
  ``literal_equation`` (equation evaluated at the given day; model Eq 20-375 / 20-388 print no
  day-190 condition) -- code check pending (``configs/animal_profile.yaml`` mineral_gestation_rule).
  Ca constant ``Eq 7-3`` (0.02456, p.107) or ``Eq 20-375`` (0.0245, p.455).  No defaults.
* Maintenance in the linear absorbed-mineral row: ``actual_D`` (0.9 g Ca / 1.0 g P per kg of the
  scenario's own DM supply) or ``fixed_DMI`` -- PUD-P2a-05.  No default.

Absorbed-mineral constraint rows (model_audit section 6.2)
----------------------------------------------------------
``Abs_CaIn = sum_f Fd_CaIn_f * Fd_acCa_f`` (Eq 20-370 / 20-371, p.454) and the P analogue (Eq 20-381 /
20-382, p.455) are linear in the executed ration ``q`` for fixed ACs.  :func:`absorbed_mineral_row`
returns the engine terms ``{"C:AC_Ca:Ca": 1, "DM": -0.0009}`` (Ca, maintenance on actual D) etc., and
:meth:`AbsorbedMineralRow.to_constraint_spec` refuses to label a row derived under open decisions as
``sourced``, or as ``research_scenario_assumption`` without an explicit
:class:`~ration_reliability.io.config.PendingOverride` (B-133 builder rule).

Nothing here is a nutritional result: passing the tests shows that the printed equations are
computed as transcribed, not that the requirements are adequate for any animal (contract section 8).
"""

from __future__ import annotations

import csv
import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from ..datamodel import (
    ConstraintClass,
    ConstraintKind,
    ConstraintSpec,
    DMSource,
    IngredientRecord,
    Provenance,
    Sense,
    ValueStatus,
)
from ..errors import InvalidProblemError, UnitError
from ..hashing import file_sha256
from ..io.config import PendingOverride, PendingRelabelError, check_pending_relabel
from ..normalization import units as U

__all__ = [
    "NASEM_SOURCE_ID",
    "Qty",
    "EquationRef",
    "EquationResult",
    "RequirementChoices",
    "FactorialRequirement",
    "GROWTH_OPTIONS",
    "GESTATION_RULES",
    "CA_GESTATION_EQUATIONS",
    "MILK_P_EQUATIONS",
    "MAINTENANCE_BASES",
    "milk_net_energy_eq3_14b",
    "milk_energy_output_eq20_220",
    "dmi_eq2_1",
    "dmi_eq20_21_literal",
    "ca_maintenance_eq7_1",
    "milk_ca_concentration_eq7_4",
    "ca_lactation_eq20_376",
    "ca_growth_eq7_2",
    "ca_gestation_eq7_3",
    "p_maintenance_eq7_5b",
    "p_lactation_eq20_389",
    "p_lactation_eq7_8b",
    "p_growth_eq7_6",
    "p_gestation_eq7_7",
    "ca_requirement_factorial",
    "p_requirement_factorial",
    "absorbed_density_pct_dm",
    "display_dietary_concentration_pct_dm",
    "AC_TABLE_COLUMNS",
    "ACEntry",
    "AbsorptionCoefficientTable",
    "load_absorption_coefficient_table",
    "absorption_coefficients_for",
    "attach_absorption_coefficients",
    "AbsorbedMineralRow",
    "absorbed_mineral_row",
]

#: Source-registry id of the local NASEM 2021 PDF (``sources/source_registry.csv``).
NASEM_SOURCE_ID = "SRC-C-NASEM21-T19"

GROWTH_OPTIONS = ("A_An_BWgain", "B_zero_mature")
GESTATION_RULES = ("text_rule", "literal_equation")
CA_GESTATION_EQUATIONS = ("Eq 7-3", "Eq 20-375")
MILK_P_EQUATIONS = ("Eq 20-389", "Eq 7-8b")
MAINTENANCE_BASES = ("actual_D", "fixed_DMI")

#: Open decisions (``configs/animal_profile.yaml`` / ``configs/constraints.yaml``) behind each choice.
_DECISION_OF = {
    "profile": "PUD-P2a-01",
    "bcs": "PUD-P2a-02",
    "growth_option": "PUD-P2a-03",
    "milk_p_equation": "PUD-P2a-04",
    "maintenance_basis": "PUD-P2a-05",
    "gestation_rule": "PUD-P2a-06",
    "dmi_version": "PUD-P2a-08",
}

#: Decisions behind any DMI-dependent quantity (FIX5, red-team finding on B-G1-04): the DMI version
#: (Eq 2-1 vs Eq 20-21 literal, +5.1 %; PUD-P2a-08) and the BCS = 3.0 scenario input of both DMI
#: equations (PUD-P2a-02).  ``dmi_label`` is free text and is *not* used to waive them: every DMI in
#: this project comes from one of the two equations with the BCS assumption.
_DMI_DECISIONS = (_DECISION_OF["bcs"], _DECISION_OF["dmi_version"])


def _with(ids: Sequence[str], extra: Sequence[str]) -> tuple[str, ...]:
    """``ids`` followed by the members of ``extra`` not yet present (order kept, no duplicates)."""
    out = list(ids)
    for d in extra:
        if d not in out:
            out.append(d)
    return tuple(out)


# --------------------------------------------------------------------------------------------
# inputs and results
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Qty:
    """A numeric input with its unit, e.g. ``Qty(700, "kg")``, ``Qty(3.26, "%")``, ``Qty(200, "d")``."""

    value: float
    unit: str

    def as_tuple(self) -> tuple[float, str]:
        return (float(self.value), self.unit)


_DAY_UNITS = {"d": 1.0}


def _val(x: Any, target: str, name: str) -> float:
    """``x`` (a :class:`Qty`) expressed in ``target``; bare numbers and unknown units are refused."""
    if not isinstance(x, Qty):
        raise TypeError(f"{name}: pass Qty(value, unit); bare numbers are refused (inputs carry units)")
    try:
        v = float(x.value)
    except (TypeError, ValueError):
        raise TypeError(f"{name}: value must be numeric") from None
    if not math.isfinite(v):
        raise ValueError(f"{name}: value must be finite")
    if target == "d":
        if x.unit not in _DAY_UNITS:
            raise UnitError(f"{name}: day count needs unit 'd', got {x.unit!r}")
        return v * _DAY_UNITS[x.unit]
    if target == "1":
        if x.unit != "1":
            raise UnitError(f"{name}: dimensionless input needs unit '1', got {x.unit!r}")
        return v
    return float(U.convert(v, x.unit, target))


@dataclass(frozen=True)
class EquationRef:
    """One printed equation (or table / text passage) with its printed page."""

    equation: str
    printed_page: int
    note: str = ""
    source_id: str = NASEM_SOURCE_ID

    @property
    def locator(self) -> str:
        """``NASEM 2021 Eq 2-1, p.12 (pdf p.32)``."""
        return f"NASEM 2021 {self.equation}, p.{self.printed_page} (pdf p.{self.printed_page + 20})"

    def to_dict(self) -> dict[str, Any]:
        return {"equation": self.equation, "printed_page": self.printed_page, "pdf_page": self.printed_page + 20,
                "source_id": self.source_id, "note": self.note}


@dataclass(frozen=True)
class EquationResult:
    """Result of one equation: value, unit, equation/page metadata, inputs and intermediates."""

    quantity: str
    value: float
    unit: str
    equations: tuple[EquationRef, ...]
    inputs: Mapping[str, tuple[float, str]] = field(default_factory=dict)
    intermediates: Mapping[str, float] = field(default_factory=dict)
    variant: Optional[str] = None
    notes: tuple[str, ...] = ()

    def as_qty(self) -> Qty:
        """The result as an input for the next equation."""
        return Qty(self.value, self.unit)

    @property
    def locators(self) -> tuple[str, ...]:
        return tuple(r.locator for r in self.equations)

    def to_dict(self) -> dict[str, Any]:
        return {"quantity": self.quantity, "value": self.value, "unit": self.unit,
                "equations": [r.to_dict() for r in self.equations], "inputs": {k: list(v) for k, v in self.inputs.items()},
                "intermediates": dict(self.intermediates), "variant": self.variant, "notes": list(self.notes)}


def _inputs(**kw: Qty) -> dict[str, tuple[float, str]]:
    return {k: v.as_tuple() for k, v in kw.items()}


@dataclass(frozen=True)
class RequirementChoices:
    """The version choices a factorial requirement depends on (all required, no defaults).

    ``growth_option``: ``A_An_BWgain`` | ``B_zero_mature`` (PUD-P2a-03);
    ``gestation_rule``: ``text_rule`` | ``literal_equation`` (code check pending, PUD-P2a-06);
    ``ca_gestation_equation``: ``Eq 7-3`` (0.02456) | ``Eq 20-375`` (0.0245);
    ``milk_p_equation``: ``Eq 20-389`` (0.48) | ``Eq 7-8b`` (0.49) (PUD-P2a-04).
    """

    growth_option: str
    gestation_rule: str
    ca_gestation_equation: str
    milk_p_equation: str

    def __post_init__(self) -> None:
        for name, allowed in (("growth_option", GROWTH_OPTIONS), ("gestation_rule", GESTATION_RULES),
                              ("ca_gestation_equation", CA_GESTATION_EQUATIONS),
                              ("milk_p_equation", MILK_P_EQUATIONS)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")

    def open_decisions(self, element: str) -> tuple[str, ...]:
        """Decision ids behind these *choices* for ``element`` (``Ca`` or ``P``).

        The DMI decisions (PUD-P2a-02, PUD-P2a-08) are not choices of this object; the factorial
        functions add them because the maintenance term depends on DMI (see ``_DMI_DECISIONS``).
        """
        ids = [_DECISION_OF["profile"], _DECISION_OF["growth_option"], _DECISION_OF["gestation_rule"]]
        if element == "P":
            ids.append(_DECISION_OF["milk_p_equation"])
        return tuple(ids)


@dataclass(frozen=True)
class FactorialRequirement:
    """Absorbed requirement of one element = maintenance + lactation + growth + gestation (g/d)."""

    element: str
    total: EquationResult
    maintenance: EquationResult
    lactation: EquationResult
    growth: EquationResult
    gestation: EquationResult
    choices: RequirementChoices
    dmi_used: tuple[float, str]
    dmi_label: str
    open_decisions: tuple[str, ...]

    @property
    def value(self) -> float:
        """Total absorbed requirement (g/d)."""
        return self.total.value

    @property
    def unit(self) -> str:
        return self.total.unit

    def to_dict(self) -> dict[str, Any]:
        return {"element": self.element, "value_g_per_d": self.value,
                "total": self.total.to_dict(), "maintenance": self.maintenance.to_dict(),
                "lactation": self.lactation.to_dict(), "growth": self.growth.to_dict(),
                "gestation": self.gestation.to_dict(), "choices": dataclasses.asdict(self.choices),
                "dmi_used": list(self.dmi_used), "dmi_label": self.dmi_label,
                "open_decisions": list(self.open_decisions)}


# --------------------------------------------------------------------------------------------
# milk energy and DMI
# --------------------------------------------------------------------------------------------

def milk_net_energy_eq3_14b(fat: Qty, true_protein: Qty, lactose: Qty) -> EquationResult:
    """NEL of milk (Mcal/kg) = 9.29 fat + 5.85 true protein + 3.95 lactose, all kg/kg milk.

    Eq 3-14b (p.30); target form Eq 20-218 (p.441).  Inputs are milk concentrations (``%`` or
    ``fraction`` of milk).  NASEM default lactose when not measured: 4.85 % (p.30).
    """
    f = _val(fat, "fraction", "fat")
    tp = _val(true_protein, "fraction", "true_protein")
    lac = _val(lactose, "fraction", "lactose")
    for n, v in (("fat", f), ("true_protein", tp), ("lactose", lac)):
        if not 0.0 <= v < 1.0:
            raise ValueError(f"{n} must be a fraction of milk in [0, 1)")
    terms = {"fat_term": 9.29 * f, "true_protein_term": 5.85 * tp, "lactose_term": 3.95 * lac}
    return EquationResult("milk_NEL_concentration", sum(terms.values()), "Mcal/kg",
                          (EquationRef("Eq 3-14b", 30), EquationRef("Eq 20-218", 441, "target form (Milk_NEpTrg)")),
                          _inputs(fat=fat, true_protein=true_protein, lactose=lactose), terms)


def milk_energy_output_eq20_220(nel_milk: Qty, milk_yield: Qty) -> EquationResult:
    """Milk NE output (Mcal/d) = milk NEL concentration x milk yield.

    Eq 20-220 (p.441, target milk: ``Milk_NEuseTrg``, the input of Eq 20-21); Eq 20-221 (same
    arithmetic for predicted milk, cited as NV-02 in model_audit section 8).
    """
    nel = _val(nel_milk, "Mcal/kg", "nel_milk")
    my = _val(milk_yield, "kg/d", "milk_yield")
    if nel < 0 or my < 0:
        raise ValueError("nel_milk and milk_yield must be >= 0")
    return EquationResult("MilkE", nel * my, "Mcal/d",
                          (EquationRef("Eq 20-220", 441, "target milk (Milk_NEuseTrg)"),
                           EquationRef("Eq 20-221", 441, "same arithmetic, predicted milk")),
                          _inputs(nel_milk=nel_milk, milk_yield=milk_yield))


def _dmi_common(milk_energy: Qty, body_weight: Qty, bcs: Qty, days_in_milk: Qty) -> tuple[float, float, float, float, list[str]]:
    me = _val(milk_energy, "Mcal/d", "milk_energy")
    bw = _val(body_weight, "kg", "body_weight")
    b = _val(bcs, "1", "bcs")
    dim = _val(days_in_milk, "d", "days_in_milk")
    if not 1.0 <= b <= 5.0:
        raise ValueError("bcs must be on the 1-5 scale")
    if bw <= 0 or me < 0 or dim < 0:
        raise ValueError("body_weight must be > 0, milk_energy and days_in_milk >= 0")
    notes = []
    if not 1.0 <= dim <= 368.0:
        notes.append("days_in_milk outside the 1-368 d range of the Eq 2-1 modelling data (p.12)")
    return me, bw, b, dim, notes


def dmi_eq2_1(parity: Qty, milk_energy: Qty, body_weight: Qty, bcs: Qty, days_in_milk: Qty) -> EquationResult:
    """DMI (kg/d), NASEM 2021 Eq 2-1 (p.12), animal factors only (lactating Holstein cows)::

        [3.7 + 5.7 Parity + 0.305 MilkE + 0.022 BW + (-0.689 - 1.87 Parity) BCS]
          x [1 - (0.212 + 0.136 Parity) exp(-0.053 DIM)]

    ``parity`` is the Eq 2-1 indicator in [0, 1] (0 = all primiparous, 1 = all multiparous; unit
    ``1``).  The printed first bracket "[3.7 + Parity x 5.7)" is a typesetting slip (model_audit
    section 11 no. 1).  Supplied DM is not observed intake (contract C04).
    """
    p = _val(parity, "1", "parity")
    if not 0.0 <= p <= 1.0:
        raise ValueError("Eq 2-1 parity indicator must be in [0, 1]")
    me, bw, b, dim, notes = _dmi_common(milk_energy, body_weight, bcs, days_in_milk)
    bracket = 3.7 + 5.7 * p + 0.305 * me + 0.022 * bw + (-0.689 - 1.87 * p) * b
    dimc = 1.0 - (0.212 + 0.136 * p) * math.exp(-0.053 * dim)
    return EquationResult("DMI", bracket * dimc, "kg/d", (EquationRef("Eq 2-1", 12),),
                          _inputs(parity=parity, milk_energy=milk_energy, body_weight=body_weight, bcs=bcs,
                                  days_in_milk=days_in_milk),
                          {"bracket": bracket, "dim_correction": dimc,
                           "bcs_slope_kg_per_d_per_unit": (-0.689 - 1.87 * p) * dimc},
                          variant="Eq 2-1", notes=tuple(notes))


def dmi_eq20_21_literal(an_parity: Qty, milk_energy: Qty, body_weight: Qty, bcs: Qty,
                        days_in_milk: Qty) -> EquationResult:
    """DMI (kg/d) by Eq 20-21 **as printed** (p.423), ``An_Parity`` in [1, 2] (unit ``1``)::

        (3.7 + 5.7 (An_Parity - 1) + 0.305 MilkE + 0.022 BW + (-0.689 - 1.87 (An_Parity - 1) BCS))
          x (1 - (0.212 + 0.136 (An_Parity - 1)) exp(-0.053 DIM))

    Literal reading used in model_audit section 3.4: ``-0.689`` is not multiplied by BCS; the
    unbalanced opening parenthesis is read as enclosing the whole sum.  (Reading the product with the
    DIM factor as binding only the BCS term is a second literal reading; its value is returned in
    ``intermediates['alt_precedence_reading']`` for information only.)  It contradicts the BCS slopes
    stated on p.13 but comes close to the primiparous Table 21-1 columns (DIM 150 within the printed
    precision; DIM 15 off by 0.07 kg/d; model_audit section 3.5, FIX5 correction).
    Code check pending (B-109, PUD-P2a-06/-08).
    """
    ap = _val(an_parity, "1", "an_parity")
    if not 1.0 <= ap <= 2.0:
        raise ValueError("An_Parity must be in [1, 2]")
    me, bw, b, dim, notes = _dmi_common(milk_energy, body_weight, bcs, days_in_milk)
    k = ap - 1.0
    base = 3.7 + 5.7 * k + 0.305 * me + 0.022 * bw
    bcs_term = -0.689 - 1.87 * k * b
    dimc = 1.0 - (0.212 + 0.136 * k) * math.exp(-0.053 * dim)
    return EquationResult("DMI", (base + bcs_term) * dimc, "kg/d",
                          (EquationRef("Eq 20-21", 423, "literal printed brackets; model_audit section 3.4"),),
                          _inputs(an_parity=an_parity, milk_energy=milk_energy, body_weight=body_weight, bcs=bcs,
                                  days_in_milk=days_in_milk),
                          {"bracket": base + bcs_term, "dim_correction": dimc,
                           "bcs_slope_kg_per_d_per_unit": -1.87 * k * dimc,
                           "alt_precedence_reading": base + bcs_term * dimc},
                          variant="Eq 20-21 literal",
                          notes=tuple(notes) + ("printed brackets differ from Eq 2-1 (model_audit section 11 no. 2)",))


# --------------------------------------------------------------------------------------------
# calcium (absorbed, g/d)
# --------------------------------------------------------------------------------------------

def ca_maintenance_eq7_1(dmi: Qty) -> EquationResult:
    """Metabolic fecal Ca (g/d) = 0.90 x DMI (kg/d); Eq 7-1 (p.106), model Eq 20-373 (p.454)."""
    d = _val(dmi, "kg/d", "dmi")
    if d < 0:
        raise ValueError("dmi must be >= 0")
    return EquationResult("Ca_maintenance", 0.9 * d, "g/d",
                          (EquationRef("Eq 7-1", 106), EquationRef("Eq 20-373", 454)), _inputs(dmi=dmi))


def milk_ca_concentration_eq7_4(milk_true_protein: Qty) -> EquationResult:
    """Milk Ca (g/kg milk) = 0.295 + 0.239 x milk true protein **percent**; Eq 7-4 (p.107).

    The model form Eq 20-376 (p.455) prints ``MlkNP_Milk`` without x100; the percent reading is the
    one that reproduces the book example (NB-02; model_audit section 11 no. 6).
    """
    tp_pct = _val(milk_true_protein, "%", "milk_true_protein")
    if not 0.0 <= tp_pct < 100.0:
        raise ValueError("milk_true_protein must be a percentage of milk")
    return EquationResult("milk_Ca_concentration", 0.295 + 0.239 * tp_pct, "g/kg",
                          (EquationRef("Eq 7-4", 107), EquationRef("Eq 20-376", 455, "x100 missing in print")),
                          _inputs(milk_true_protein=milk_true_protein), {"true_protein_pct": tp_pct})


def ca_lactation_eq20_376(milk_true_protein: Qty, milk_yield: Qty) -> EquationResult:
    """Ca for lactation (g/d) = milk Ca (Eq 7-4) x milk yield (kg/d); Eq 20-376 (p.455)."""
    conc = milk_ca_concentration_eq7_4(milk_true_protein)
    my = _val(milk_yield, "kg/d", "milk_yield")
    if my < 0:
        raise ValueError("milk_yield must be >= 0")
    return EquationResult("Ca_lactation", conc.value * my, "g/d", conc.equations,
                          _inputs(milk_true_protein=milk_true_protein, milk_yield=milk_yield),
                          {"milk_Ca_g_per_kg": conc.value})


def ca_growth_eq7_2(mature_body_weight: Qty, body_weight: Qty, body_weight_gain: Qty) -> EquationResult:
    """Ca for growth (g/d) = 9.83 MatBW^0.22 BW^-0.22 x gain; Eq 7-2 (p.106), model Eq 20-374 (p.455).

    Eq 7-2 is written for growing cattle (ADG); the model drives it with An_BWgain (frame + reserves,
    Eq 20-247).  Whether to apply it to a mature cow is PUD-P2a-03 (see :func:`ca_requirement_factorial`).
    """
    mat = _val(mature_body_weight, "kg", "mature_body_weight")
    bw = _val(body_weight, "kg", "body_weight")
    g = _val(body_weight_gain, "kg/d", "body_weight_gain")
    if mat <= 0 or bw <= 0:
        raise ValueError("body weights must be > 0")
    scale = mat ** 0.22 * bw ** -0.22
    return EquationResult("Ca_growth", 9.83 * scale * g, "g/d",
                          (EquationRef("Eq 7-2", 106), EquationRef("Eq 20-374", 455)),
                          _inputs(mature_body_weight=mature_body_weight, body_weight=body_weight,
                                  body_weight_gain=body_weight_gain), {"maturity_scale": scale})


def _gestation_literal(c: float, a: float, b: float, t: float, bw: float) -> float:
    return c * (math.exp((a - b * t) * t) - math.exp((a - b * (t - 1.0)) * (t - 1.0))) * bw / 715.0


def ca_gestation_eq7_3(days_pregnant: Qty, body_weight: Qty, *, rule: str, equation: str) -> EquationResult:
    """Ca for gestation (g/d), Eq 7-3 (p.107) / model Eq 20-375 (p.455)::

        c [exp((0.05581 - 0.00007 t) t) - exp((0.05581 - 0.00007 (t-1)) (t-1))] x BW / 715

    ``equation``: ``"Eq 7-3"`` (c = 0.02456) or ``"Eq 20-375"`` (c = 0.0245).
    ``rule``: ``"text_rule"`` -- the equation applies only beyond day 190 (p.106-107), so t <= 190 gives
    0; ``"literal_equation"`` -- evaluated at t (the model equation prints no day-190 condition).
    t <= 0 (not pregnant) gives 0 under both rules.
    """
    if rule not in GESTATION_RULES:
        raise ValueError(f"rule must be one of {GESTATION_RULES}")
    if equation not in CA_GESTATION_EQUATIONS:
        raise ValueError(f"equation must be one of {CA_GESTATION_EQUATIONS}")
    t = _val(days_pregnant, "d", "days_pregnant")
    bw = _val(body_weight, "kg", "body_weight")
    if bw <= 0:
        raise ValueError("body_weight must be > 0")
    c = 0.02456 if equation == "Eq 7-3" else 0.0245
    literal = _gestation_literal(c, 0.05581, 0.00007, t, bw) if t > 0 else 0.0
    applies = t > 190.0 if rule == "text_rule" else t > 0
    refs = (EquationRef("Eq 7-3", 107, "c = 0.02456; beyond day 190 (p.106-107)"),
            EquationRef("Eq 20-375", 455, "c = 0.0245; no day-190 condition printed"))
    return EquationResult("Ca_gestation", literal if applies else 0.0, "g/d", refs,
                          _inputs(days_pregnant=days_pregnant, body_weight=body_weight),
                          {"literal_equation_value": literal, "constant": c},
                          variant=f"{equation}; {rule}")


# --------------------------------------------------------------------------------------------
# phosphorus (absorbed, g/d)
# --------------------------------------------------------------------------------------------

def p_maintenance_eq7_5b(dmi: Qty, body_weight: Qty) -> EquationResult:
    """Adult-cow maintenance P (g/d) = 1.0 g/kg DMI (fecal) + 0.0006 g/kg BW (urinary).

    Eq 7-5b (p.112); model Eq 20-384 (urinary), 20-385 (fecal, parity >= 1), 20-386 (sum), p.455.
    """
    d = _val(dmi, "kg/d", "dmi")
    bw = _val(body_weight, "kg", "body_weight")
    if d < 0 or bw <= 0:
        raise ValueError("dmi must be >= 0 and body_weight > 0")
    fecal, urinary = 1.0 * d, 0.0006 * bw
    return EquationResult("P_maintenance", fecal + urinary, "g/d",
                          (EquationRef("Eq 7-5b", 112), EquationRef("Eq 20-384", 455), EquationRef("Eq 20-385", 455),
                           EquationRef("Eq 20-386", 455)),
                          _inputs(dmi=dmi, body_weight=body_weight), {"fecal": fecal, "urinary": urinary})


def _p_lactation(constant: float, ref: EquationRef, milk_true_protein: Qty, milk_yield: Qty) -> EquationResult:
    tp_pct = _val(milk_true_protein, "%", "milk_true_protein")
    my = _val(milk_yield, "kg/d", "milk_yield")
    if not 0.0 <= tp_pct < 100.0 or my < 0:
        raise ValueError("milk_true_protein must be a percentage of milk and milk_yield >= 0")
    conc = constant + 0.13 * tp_pct
    return EquationResult("P_lactation", conc * my, "g/d", (ref,),
                          _inputs(milk_true_protein=milk_true_protein, milk_yield=milk_yield),
                          {"milk_P_g_per_kg": conc, "constant": constant}, variant=ref.equation)


def p_lactation_eq20_389(milk_true_protein: Qty, milk_yield: Qty) -> EquationResult:
    """P for lactation (g/d) = (0.48 + 0.13 x true protein %) x milk yield; Eq 20-389 (p.456).

    The constant differs from Eq 7-8b (0.49, p.113); the book example on p.113 (0.88 g/kg at 3.1 %)
    supports 0.48 (NB-03).  Choice: PUD-P2a-04.
    """
    return _p_lactation(0.48, EquationRef("Eq 20-389", 456), milk_true_protein, milk_yield)


def p_lactation_eq7_8b(milk_true_protein: Qty, milk_yield: Qty) -> EquationResult:
    """P for lactation (g/d) = (0.49 + 0.13 x true protein %) x milk yield; Eq 7-8b (p.113). PUD-P2a-04."""
    return _p_lactation(0.49, EquationRef("Eq 7-8b", 113), milk_true_protein, milk_yield)


def p_growth_eq7_6(mature_body_weight: Qty, body_weight: Qty, body_weight_gain: Qty) -> EquationResult:
    """P for growth (g/d) = (1.2 + 4.635 MatBW^0.22 BW^-0.22) x gain; Eq 7-6 (p.112), Eq 20-387 (p.455).

    Eq 7-6 prints the unit as kg/d; by magnitude it is g/d (model_audit section 11 no. 9).
    """
    mat = _val(mature_body_weight, "kg", "mature_body_weight")
    bw = _val(body_weight, "kg", "body_weight")
    g = _val(body_weight_gain, "kg/d", "body_weight_gain")
    if mat <= 0 or bw <= 0:
        raise ValueError("body weights must be > 0")
    scale = mat ** 0.22 * bw ** -0.22
    return EquationResult("P_growth", (1.2 + 4.635 * scale) * g, "g/d",
                          (EquationRef("Eq 7-6", 112, "unit printed as kg/d; g/d by magnitude"),
                           EquationRef("Eq 20-387", 455)),
                          _inputs(mature_body_weight=mature_body_weight, body_weight=body_weight,
                                  body_weight_gain=body_weight_gain), {"maturity_scale": scale})


def p_gestation_eq7_7(days_pregnant: Qty, body_weight: Qty, *, rule: str) -> EquationResult:
    """P for gestation (g/d), Eq 7-7 (p.112) / model Eq 20-388 (p.455)::

        0.02743 [exp((0.05527 - 0.000075 t) t) - exp((0.05527 - 0.000075 (t-1)) (t-1))] x BW / 715

    ``rule="text_rule"``: the conceptus requirement below day 190 is set to zero in the model (p.112),
    so t < 190 gives 0 (t = 190 is computed); ``"literal_equation"``: evaluated at t (Eq 20-388
    prints no condition).  t <= 0 gives 0.
    """
    if rule not in GESTATION_RULES:
        raise ValueError(f"rule must be one of {GESTATION_RULES}")
    t = _val(days_pregnant, "d", "days_pregnant")
    bw = _val(body_weight, "kg", "body_weight")
    if bw <= 0:
        raise ValueError("body_weight must be > 0")
    literal = _gestation_literal(0.02743, 0.05527, 0.000075, t, bw) if t > 0 else 0.0
    applies = t >= 190.0 if rule == "text_rule" else t > 0
    return EquationResult("P_gestation", literal if applies else 0.0, "g/d",
                          (EquationRef("Eq 7-7", 112, "0 below day 190 (p.112)"),
                           EquationRef("Eq 20-388", 455, "no day-190 condition printed")),
                          _inputs(days_pregnant=days_pregnant, body_weight=body_weight),
                          {"literal_equation_value": literal}, variant=rule)


# --------------------------------------------------------------------------------------------
# factorial totals
# --------------------------------------------------------------------------------------------

def _growth(fn, choices: RequirementChoices, mature_body_weight: Qty, body_weight: Qty,
            body_weight_gain: Qty) -> EquationResult:
    r = fn(mature_body_weight, body_weight, body_weight_gain)
    if choices.growth_option == "A_An_BWgain":
        return dataclasses.replace(r, variant="A_An_BWgain",
                                   notes=r.notes + ("option A: model drives the growth term with An_BWgain "
                                                    "(frame + reserves, Eq 20-247, p.444); PUD-P2a-03",))
    return dataclasses.replace(r, value=0.0, variant="B_zero_mature",
                               intermediates={**r.intermediates, "option_A_value": r.value},
                               notes=r.notes + ("option B: Chapter 7 growth equation is for growing cattle; "
                                                "0 for a mature cow (p.106, p.112); PUD-P2a-03",))


def ca_requirement_factorial(*, dmi: Qty, milk_yield: Qty, milk_true_protein: Qty, body_weight: Qty,
                             mature_body_weight: Qty, body_weight_gain: Qty, days_pregnant: Qty,
                             choices: RequirementChoices, dmi_label: str) -> FactorialRequirement:
    """Absorbed Ca requirement (g/d) = maintenance + lactation + growth + gestation; Eq 20-378 (p.455).

    ``dmi`` is whichever DMI the caller uses (e.g. :func:`dmi_eq2_1` or :func:`dmi_eq20_21_literal`);
    ``dmi_label`` records which one.  Maintenance in a *linear constraint* may instead use the
    scenario's DM supply (see :func:`absorbed_mineral_row`).
    """
    if not dmi_label:
        raise ValueError("dmi_label is required (which DMI equation/version was used)")
    m = ca_maintenance_eq7_1(dmi)
    lac = ca_lactation_eq20_376(milk_true_protein, milk_yield)
    gr = _growth(ca_growth_eq7_2, choices, mature_body_weight, body_weight, body_weight_gain)
    ge = ca_gestation_eq7_3(days_pregnant, body_weight, rule=choices.gestation_rule,
                            equation=choices.ca_gestation_equation)
    parts = {"maintenance": m.value, "lactation": lac.value, "growth": gr.value, "gestation": ge.value}
    total = EquationResult("Ca_absorbed_requirement", sum(parts.values()), "g/d",
                           (EquationRef("Eq 20-378", 455),) + m.equations + lac.equations + gr.equations + ge.equations,
                           _inputs(dmi=dmi, milk_yield=milk_yield, milk_true_protein=milk_true_protein,
                                   body_weight=body_weight, mature_body_weight=mature_body_weight,
                                   body_weight_gain=body_weight_gain, days_pregnant=days_pregnant),
                           parts, variant=f"growth {choices.growth_option}; gestation {choices.gestation_rule}")
    return FactorialRequirement("Ca", total, m, lac, gr, ge, choices, dmi.as_tuple(), dmi_label,
                                _with(choices.open_decisions("Ca"), _DMI_DECISIONS))


def p_requirement_factorial(*, dmi: Qty, milk_yield: Qty, milk_true_protein: Qty, body_weight: Qty,
                            mature_body_weight: Qty, body_weight_gain: Qty, days_pregnant: Qty,
                            choices: RequirementChoices, dmi_label: str) -> FactorialRequirement:
    """Absorbed P requirement (g/d) = maintenance + lactation + growth + gestation; Eq 20-391 (p.456)."""
    if not dmi_label:
        raise ValueError("dmi_label is required (which DMI equation/version was used)")
    m = p_maintenance_eq7_5b(dmi, body_weight)
    lac_fn = p_lactation_eq20_389 if choices.milk_p_equation == "Eq 20-389" else p_lactation_eq7_8b
    lac = lac_fn(milk_true_protein, milk_yield)
    gr = _growth(p_growth_eq7_6, choices, mature_body_weight, body_weight, body_weight_gain)
    ge = p_gestation_eq7_7(days_pregnant, body_weight, rule=choices.gestation_rule)
    parts = {"maintenance": m.value, "lactation": lac.value, "growth": gr.value, "gestation": ge.value}
    total = EquationResult("P_absorbed_requirement", sum(parts.values()), "g/d",
                           (EquationRef("Eq 20-391", 456),) + m.equations + lac.equations + gr.equations + ge.equations,
                           _inputs(dmi=dmi, milk_yield=milk_yield, milk_true_protein=milk_true_protein,
                                   body_weight=body_weight, mature_body_weight=mature_body_weight,
                                   body_weight_gain=body_weight_gain, days_pregnant=days_pregnant),
                           {**parts, "maintenance_fecal": m.intermediates["fecal"],
                            "maintenance_urinary": m.intermediates["urinary"]},
                           variant=f"growth {choices.growth_option}; gestation {choices.gestation_rule}; "
                                   f"milk P {choices.milk_p_equation}")
    return FactorialRequirement("P", total, m, lac, gr, ge, choices, dmi.as_tuple(), dmi_label,
                                _with(choices.open_decisions("P"), _DMI_DECISIONS))


def absorbed_density_pct_dm(requirement: FactorialRequirement, dmi: Qty) -> EquationResult:
    """Absorbed element per kg DM (%DM) = requirement (g/d) / (10 x DMI); NV-18 (display, AC-free)."""
    d = _val(dmi, "kg/d", "dmi")
    if d <= 0:
        raise ValueError("dmi must be > 0")
    return EquationResult(f"{requirement.element}_absorbed_density", requirement.value / (10.0 * d), "%",
                          requirement.total.equations, {"dmi": dmi.as_tuple()},
                          notes=("display only (model_audit section 6.3)",))


def display_dietary_concentration_pct_dm(requirement: FactorialRequirement, dmi: Qty,
                                         mean_absorption_coefficient: Qty) -> EquationResult:
    """%DM display value under an *assumed* diet-average AC (model_audit section 6.3; NV-20).

    Display only, never a constraint: the diet-average AC depends on the formula itself.
    """
    ac = _val(mean_absorption_coefficient, "1", "mean_absorption_coefficient")
    if not 0.0 < ac <= 1.0:
        raise ValueError("mean_absorption_coefficient must be in (0, 1]")
    dens = absorbed_density_pct_dm(requirement, dmi)
    return EquationResult(f"{requirement.element}_dietary_concentration_display", dens.value / ac, "%",
                          dens.equations, {"dmi": dmi.as_tuple(), "mean_absorption_coefficient": (ac, "1")},
                          notes=("display only; assumed diet-average AC (model_audit section 6.3)",))


# --------------------------------------------------------------------------------------------
# absorption-coefficient table (restricted data; values never hard-coded here)
# --------------------------------------------------------------------------------------------

#: Columns of ``data/restricted_local/ac_ca_p_by_ingredient.csv``.
AC_TABLE_COLUMNS = ("ingredient_id", "ingredient_name_zh", "nasem_feed_name", "nasem_feed_id", "survey_role",
                    "category", "element", "coefficient_name", "value", "unit", "value_status", "mapping_type",
                    "nasem_class_label", "locator_table", "printed_page", "pdf_page", "core_id_of_ac", "source_id",
                    "decision_ids", "rationale", "entry_method", "copyright_status")
_AC_ELEMENTS = ("Ca", "P")
_AC_MAPPING_TYPES = ("table19_3_entry", "table7_1_class", "chapter7_text_class", "chapter7_text_default", "no_basis")


@dataclass(frozen=True)
class ACEntry:
    """One ingredient x element absorption coefficient with its provenance."""

    ingredient_id: str
    element: str
    coefficient_name: str
    value: Optional[float]
    value_status: ValueStatus
    mapping_type: str
    nasem_class_label: str
    locator_table: str
    printed_page: Optional[int]
    core_id_of_ac: str
    source_id: str
    decision_ids: str
    rationale: str

    @property
    def usable(self) -> bool:
        """True when a numeric value exists and the status is not pending."""
        return self.value is not None and self.value_status is not ValueStatus.PENDING_USER_DECISION

    @property
    def locator(self) -> str:
        pg = "" if self.printed_page is None else f", p.{self.printed_page} (pdf p.{self.printed_page + 20})"
        cid = f"; core_id {self.core_id_of_ac}" if self.core_id_of_ac else ""
        return f"NASEM 2021 {self.locator_table}{pg}; class '{self.nasem_class_label}'{cid}"

    def provenance(self) -> Provenance:
        """Provenance of the coefficient (``coefficient:<name>`` of the ingredient)."""
        st = self.value_status
        return Provenance(status=st, source_id=self.source_id or None,
                          locator=self.locator if st is ValueStatus.SOURCED else None,
                          rationale=None if st is ValueStatus.SOURCED else f"{self.rationale} [{self.locator}]")


@dataclass(frozen=True)
class AbsorptionCoefficientTable:
    """Validated absorption-coefficient table (read from a restricted CSV)."""

    entries: tuple[ACEntry, ...]
    path: Optional[str] = None
    sha256: Optional[str] = None

    def get(self, ingredient_id: str, element: str) -> ACEntry:
        for e in self.entries:
            if e.ingredient_id == ingredient_id and e.element == element:
                return e
        raise KeyError(f"no AC entry for ({ingredient_id!r}, {element!r})")

    def summary(self) -> dict[str, Any]:
        """Counts by element/status/mapping type -- no coefficient values (safe for tracked reports)."""
        out: dict[str, Any] = {"n_entries": len(self.entries), "sha256": self.sha256,
                               "n_ingredients": len({e.ingredient_id for e in self.entries})}
        for el in _AC_ELEMENTS:
            es = [e for e in self.entries if e.element == el]
            out[el] = {"by_status": _count(e.value_status.value for e in es),
                       "by_mapping_type": _count(e.mapping_type for e in es)}
        return out


def _count(xs: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items()))


def load_absorption_coefficient_table(path: str | Path) -> AbsorptionCoefficientTable:
    """Read and validate an AC table (CSV with :data:`AC_TABLE_COLUMNS`); every issue is listed at once.

    Rules: known elements and mapping types; ``coefficient_name == "AC_<element>"``; unit ``1``;
    ``sourced`` / ``research_scenario_assumption`` rows need a value in (0, 1], a source id and a
    printed page; ``synthetic_test_only`` rows (test fixtures) need a value and a synthetic source id;
    ``sourced`` rows need a non-``no_basis`` mapping; ``pending_user_decision`` rows must have an
    empty value; non-sourced rows need a rationale; ``pdf_page == printed_page + 20``;
    ``table19_3_entry`` rows need ``core_id_of_ac``; no duplicate (ingredient, element).
    """
    path = Path(path)
    errors: list[str] = []
    entries: list[ACEntry] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = tuple(reader.fieldnames or ())
        missing = [c for c in AC_TABLE_COLUMNS if c not in cols]
        if missing:
            raise InvalidProblemError(f"{path.name}: missing columns {missing}")
        seen: set[tuple[str, str]] = set()
        for n, row in enumerate(reader, start=2):
            w = f"{path.name} line {n}"
            iid, el = (row["ingredient_id"] or "").strip(), (row["element"] or "").strip()
            if not iid:
                errors.append(f"{w}: empty ingredient_id")
                continue
            if el not in _AC_ELEMENTS:
                errors.append(f"{w}: element must be one of {_AC_ELEMENTS}")
                continue
            if (iid, el) in seen:
                errors.append(f"{w}: duplicate entry for ({iid}, {el})")
            seen.add((iid, el))
            if row["coefficient_name"] != f"AC_{el}":
                errors.append(f"{w}: coefficient_name must be AC_{el}")
            if row["unit"] != "1":
                errors.append(f"{w}: unit must be '1' (dimensionless fraction absorbed)")
            try:
                st = ValueStatus(row["value_status"])
            except ValueError:
                errors.append(f"{w}: unknown value_status {row['value_status']!r}")
                continue
            if st is ValueStatus.SYNTHETIC_TEST_ONLY and "synthetic" not in (row["source_id"] or "").lower():
                errors.append(f"{w}: synthetic_test_only needs a synthetic source_id")
            mt = row["mapping_type"]
            if mt not in _AC_MAPPING_TYPES:
                errors.append(f"{w}: unknown mapping_type {mt!r}")
            raw = (row["value"] or "").strip()
            val: Optional[float] = None
            if raw:
                try:
                    val = float(raw)
                except ValueError:
                    errors.append(f"{w}: value must be numeric or empty")
            page_raw = (row["printed_page"] or "").strip()
            page = int(page_raw) if page_raw.isdigit() else None
            if page_raw and page is None:
                errors.append(f"{w}: printed_page must be an integer")
            pdf_raw = (row["pdf_page"] or "").strip()
            if page is not None and pdf_raw != str(page + 20):
                errors.append(f"{w}: pdf_page must be printed_page + 20")
            if st is ValueStatus.PENDING_USER_DECISION:
                if val is not None:
                    errors.append(f"{w}: pending_user_decision must have an empty value (no placeholder numbers)")
            else:
                if val is None:
                    errors.append(f"{w}: status {st.value} needs a value")
                elif not (math.isfinite(val) and 0.0 < val <= 1.0):
                    errors.append(f"{w}: absorption coefficient must be in (0, 1]")
                if st is not ValueStatus.SYNTHETIC_TEST_ONLY and page is None:
                    errors.append(f"{w}: status {st.value} needs a printed_page (table/page locator)")
                if not (row["source_id"] or "").strip():
                    errors.append(f"{w}: status {st.value} needs a source_id")
            if st is ValueStatus.SOURCED and mt == "no_basis":
                errors.append(f"{w}: a 'sourced' value cannot have mapping_type 'no_basis'")
            if st is not ValueStatus.SOURCED and not (row["rationale"] or "").strip():
                errors.append(f"{w}: status {st.value} needs a rationale")
            if mt == "table19_3_entry" and not (row["core_id_of_ac"] or "").strip():
                errors.append(f"{w}: table19_3_entry needs core_id_of_ac (link to the double-entered core table)")
            entries.append(ACEntry(iid, el, row["coefficient_name"], val, st, mt, row["nasem_class_label"],
                                   row["locator_table"], page, (row["core_id_of_ac"] or "").strip(),
                                   (row["source_id"] or "").strip(), row["decision_ids"] or "",
                                   row["rationale"] or ""))
    if errors:
        raise InvalidProblemError(f"{path.name}: {len(errors)} issue(s):\n  - " + "\n  - ".join(errors))
    return AbsorptionCoefficientTable(tuple(entries), str(path), file_sha256(path))


def absorption_coefficients_for(table: AbsorptionCoefficientTable, ingredient_ids: Sequence[str],
                                elements: Sequence[str] = _AC_ELEMENTS, *,
                                allowed_statuses: Sequence[str] = ("sourced", "research_scenario_assumption")
                                ) -> dict[str, dict[str, tuple[float, Provenance]]]:
    """``{ingredient_id: {"AC_Ca": (value, provenance), ...}}`` for every requested ingredient.

    Raises :class:`InvalidProblemError` listing every ingredient whose entry is missing, pending (no
    value) or has a status outside ``allowed_statuses`` -- a missing coefficient is never 0.
    """
    allowed = {ValueStatus(s) for s in allowed_statuses}
    out: dict[str, dict[str, tuple[float, Provenance]]] = {}
    problems: list[str] = []
    for iid in ingredient_ids:
        out[iid] = {}
        for el in elements:
            try:
                e = table.get(iid, el)
            except KeyError:
                problems.append(f"{iid}/{el}: no entry in the AC table")
                continue
            if not e.usable:
                problems.append(f"{iid}/{el}: {e.value_status.value} (no value; {e.decision_ids or 'no decision id'})")
                continue
            if e.value_status not in allowed:
                problems.append(f"{iid}/{el}: status {e.value_status.value} not allowed here")
                continue
            out[iid][e.coefficient_name] = (float(e.value), e.provenance())  # type: ignore[arg-type]
    if problems:
        raise InvalidProblemError("absorption coefficients unavailable (never replaced by 0):\n  - "
                                  + "\n  - ".join(problems))
    return out


def attach_absorption_coefficients(ingredients: Sequence[IngredientRecord], table: AbsorptionCoefficientTable,
                                   elements: Sequence[str] = _AC_ELEMENTS, *,
                                   allowed_statuses: Sequence[str] = ("sourced", "research_scenario_assumption")
                                   ) -> tuple[IngredientRecord, ...]:
    """Copies of ``ingredients`` with ``coefficients["AC_<el>"]`` and ``provenance["coefficient:AC_<el>"]``.

    These are the per-ingredient ``c_i`` of the engine terms ``C:AC_Ca:Ca`` / ``C:AC_P:P``.  An
    existing coefficient of the same name is refused (no silent overwrite).
    """
    acs = absorption_coefficients_for(table, [g.ingredient_id for g in ingredients], elements,
                                      allowed_statuses=allowed_statuses)
    out = []
    for g in ingredients:
        clash = sorted(set(acs[g.ingredient_id]) & set(g.coefficients))
        if clash:
            raise InvalidProblemError(f"{g.ingredient_id}: coefficient(s) {clash} already set; refusing to overwrite")
        coefs = dict(g.coefficients)
        prov = dict(g.provenance)
        for name, (v, p) in acs[g.ingredient_id].items():
            coefs[name] = v
            prov[f"coefficient:{name}"] = p
        out.append(dataclasses.replace(g, coefficients=coefs, provenance=prov))
    return tuple(out)


# --------------------------------------------------------------------------------------------
# linear absorbed-mineral rows
# --------------------------------------------------------------------------------------------

#: Maintenance coefficient per kg DM, in kg element per kg DM (canonical units of a supply row).
_MAINT_PER_KG_DM = {"Ca": 0.9e-3, "P": 1.0e-3}
_SUPPLY_REFS = {"Ca": (EquationRef("Eq 20-370", 454), EquationRef("Eq 20-371", 454)),
                "P": (EquationRef("Eq 20-381", 455), EquationRef("Eq 20-382", 455))}


@dataclass(frozen=True)
class AbsorbedMineralRow:
    """Linear absorbed-mineral constraint ``sum_i q_i d_i (c_i a_i - m) >= R`` in engine term form.

    ``terms`` uses the engine grammar (``C:AC_Ca:Ca``, ``DM``); ``bound_g_per_d`` is ``R`` in g/d.
    """

    element: str
    coefficient_name: str
    terms: Mapping[str, float]
    bound_g_per_d: float
    maintenance_basis: str
    requirement: FactorialRequirement
    equations: tuple[EquationRef, ...]
    open_decisions: tuple[str, ...]

    def to_constraint_spec(self, constraint_id: str, *, provenance: Provenance, numerical_tolerance: float,
                           name: Optional[str] = None, pending_override: Optional[PendingOverride] = None,
                           decisions_resolved: Sequence[str] = (), claim_scope: Optional[str] = None,
                           constraint_class: ConstraintClass = ConstraintClass.PROBABILISTIC_NUTRITION
                           ) -> ConstraintSpec:
        """Engine :class:`ConstraintSpec` (kind ``supply``, sense ``ge``, unit ``g/d``, scenario DM).

        Provenance rule (B-133): while any of :attr:`open_decisions` is not in ``decisions_resolved``,
        ``sourced`` is refused and ``research_scenario_assumption`` needs ``pending_override`` (checked by
        :func:`~ration_reliability.io.config.check_pending_relabel`); ``pending_user_decision`` gives a
        spec with ``bound=None`` (cannot be compiled -- the validator refuses it for pilot/official);
        ``synthetic_test_only`` (tests, smoke) is allowed.
        """
        remaining = tuple(d for d in self.open_decisions if d not in set(decisions_resolved))
        st = provenance.status
        notes = ""
        if remaining:
            if st is ValueStatus.SOURCED:
                raise PendingRelabelError([f"{constraint_id}: derived under open decisions {list(remaining)}; "
                                           "it cannot be labelled 'sourced'"])
            if st is ValueStatus.RESEARCH_SCENARIO_ASSUMPTION:
                ovr = check_pending_relabel(ValueStatus.PENDING_USER_DECISION, st, override=pending_override,
                                            where=constraint_id)
                notes = f"pending override {ovr}"
            elif pending_override is not None:
                raise PendingRelabelError([f"{constraint_id}: an override only applies to "
                                           "research_scenario_assumption"])
        elif pending_override is not None:
            raise PendingRelabelError([f"{constraint_id}: no open decision left; the override is meaningless"])
        bound = None if st is ValueStatus.PENDING_USER_DECISION else float(self.bound_g_per_d)
        refs = "; ".join(r.locator for r in self.equations[:4])
        return ConstraintSpec(
            constraint_id=constraint_id,
            name=name or f"absorbed {self.element} supply >= factorial requirement ({self.maintenance_basis})",
            kind=ConstraintKind.SUPPLY, terms=dict(self.terms), sense=Sense.GE, bound=bound, unit="g/d",
            constraint_class=constraint_class, numerical_tolerance=float(numerical_tolerance),
            provenance=provenance, basis="none", dm_source=DMSource.SCENARIO,
            standard_version="NASEM 2021 factorial (absorbed)",
            claim_scope=claim_scope or (f"absorbed {self.element} vs NASEM requirement under fixed ACs; open "
                                        f"decisions {list(remaining)}; not a health or bone outcome"),
            quantity_type="model_quantity", notes=f"{refs}; {notes}".strip("; "))


def absorbed_mineral_row(requirement: FactorialRequirement, *, maintenance_basis: str) -> AbsorbedMineralRow:
    """Terms and bound of the linear absorbed-mineral row (model_audit section 6.2).

    ``maintenance_basis="actual_D"`` (PUD-P2a-05 recommended default, not chosen here): the fecal
    maintenance moves to the left side per kg of the scenario's own DM supply ``D``::

        Ca:  sum_i x_i (10 a_Ca,i AC_Ca,i - 0.9) >= Ca_lact + Ca_growth + Ca_gest        (g/d)
        P:   sum_i x_i (10 a_P,i  AC_P,i  - 1.0) >= P_urinary + P_lact + P_growth + P_gest

    engine terms ``{"C:AC_Ca:Ca": 1, "DM": -0.0009}`` / ``{"C:AC_P:P": 1, "DM": -0.001}``;
    ``"fixed_DMI"``: ``{"C:AC_Ca:Ca": 1}`` / ``{"C:AC_P:P": 1}`` with the full requirement as bound.

    Open decisions of the row (FIX5): the requirement's choices plus PUD-P2a-05; the DMI decisions
    PUD-P2a-02 (BCS) and PUD-P2a-08 (DMI version) are kept only for ``fixed_DMI`` -- there the bound
    contains the DMI-proportional maintenance (Eq 2-1 vs Eq 20-21 literal differ by 5.1 %); under
    ``actual_D`` the bound is DMI-free (the DMI version then acts only through SH-DM-PLAN).
    """
    if maintenance_basis not in MAINTENANCE_BASES:
        raise ValueError(f"maintenance_basis must be one of {MAINTENANCE_BASES}")
    el = requirement.element
    if el not in _MAINT_PER_KG_DM:
        raise ValueError(f"unsupported element {el!r}")
    cname = f"AC_{el}"
    terms: dict[str, float] = {f"C:{cname}:{el}": 1.0}
    parts = requirement.lactation.value + requirement.growth.value + requirement.gestation.value
    if maintenance_basis == "actual_D":
        # the DMI-proportional (fecal) maintenance moves to the left side; P keeps its urinary part
        terms["DM"] = -_MAINT_PER_KG_DM[el]
        bound = parts if el == "Ca" else requirement.maintenance.intermediates["urinary"] + parts
        # the bound no longer contains DMI -> the DMI decisions do not apply to this row
        decisions = tuple(d for d in requirement.open_decisions if d not in _DMI_DECISIONS)
    else:
        bound = requirement.maintenance.value + parts
        # the bound contains 0.9 / 1.0 x DMI -> DMI version and BCS assumption stay open
        decisions = _with(requirement.open_decisions, _DMI_DECISIONS)
    return AbsorbedMineralRow(el, cname, terms, bound, maintenance_basis, requirement,
                              _SUPPLY_REFS[el] + requirement.total.equations,
                              _with(decisions, (_DECISION_OF["maintenance_basis"],)))


# --------------------------------------------------------------------------------------------
# calling layer: the version is fixed by configuration (review round 2, R4 item 2; K4, 2026-09-25)
# --------------------------------------------------------------------------------------------
# The functions above keep every book alternative and pick nothing.  The layer below reads an
# explicit ``requirement_version`` block (e.g. ``configs/dev_case_v1/animal.yaml``) and applies it;
# the adjudication evidence (nasem_dairy code check, commit 9b0b28e, and the book pages) lives in
# that configuration and in ``reports/minimum_case_nutrition_audit.md``.  There are no defaults:
# a missing or unknown field is an error.  The G1 function bodies above are unchanged.

#: DMI functions a configuration may select (both remain available; see module docstring).
DMI_FUNCTIONS = {"dmi_eq2_1": dmi_eq2_1, "dmi_eq20_21_literal": dmi_eq20_21_literal}

_VERSION_KEYS = ("dmi_function", "growth_option", "gestation_rule", "ca_gestation_equation", "milk_p_equation",
                 "maintenance_basis", "decisions_resolved", "row_status", "row_status_rationale")
_VERSION_OPTIONAL_KEYS = ("decisions_resolved_note", "notes")
_COW_KEYS = ("parity_eq2_1", "body_weight", "mature_body_weight", "body_condition_score", "days_in_milk",
             "milk_yield", "milk_fat", "milk_true_protein", "milk_lactose", "days_pregnant",
             "body_weight_gain_for_minerals")


@dataclass(frozen=True)
class RequirementVersion:
    """A fixed, configuration-declared version of the DMI and absorbed-mineral requirement model."""

    dmi_function: str
    choices: RequirementChoices
    maintenance_basis: str
    decisions_resolved: tuple[str, ...]
    row_status: ValueStatus
    row_status_rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {"dmi_function": self.dmi_function, "choices": dataclasses.asdict(self.choices),
                "maintenance_basis": self.maintenance_basis, "decisions_resolved": list(self.decisions_resolved),
                "row_status": self.row_status.value, "row_status_rationale": self.row_status_rationale}


def requirement_version_from_config(block: Mapping[str, Any]) -> RequirementVersion:
    """Parse a ``requirement_version`` block; every key is required and every value is checked.

    ``row_status`` may be ``sourced`` or ``research_scenario_assumption`` (the latter needs a
    non-empty ``row_status_rationale``); ``decisions_resolved`` lists the decision ids the
    configuration settles (their evidence must be recorded next to the block).
    """
    if not isinstance(block, Mapping):
        raise InvalidProblemError("requirement_version: expected a mapping")
    issues = [f"requirement_version: missing key {k!r}" for k in _VERSION_KEYS if k not in block]
    unknown = sorted(set(block) - set(_VERSION_KEYS) - set(_VERSION_OPTIONAL_KEYS))
    issues += [f"requirement_version: unknown key {k!r}" for k in unknown]
    if issues:
        raise InvalidProblemError("; ".join(issues))
    if block["dmi_function"] not in DMI_FUNCTIONS:
        issues.append(f"requirement_version.dmi_function must be one of {sorted(DMI_FUNCTIONS)}")
    if block["maintenance_basis"] not in MAINTENANCE_BASES:
        issues.append(f"requirement_version.maintenance_basis must be one of {MAINTENANCE_BASES}")
    dec = block["decisions_resolved"]
    if isinstance(dec, str) or not isinstance(dec, (list, tuple)) or not all(isinstance(d, str) and d for d in dec):
        issues.append("requirement_version.decisions_resolved must be a list of decision ids")
    try:
        st = ValueStatus(block["row_status"])
    except ValueError:
        st = None
        issues.append("requirement_version.row_status is not a ValueStatus")
    if st is not None and st not in (ValueStatus.SOURCED, ValueStatus.RESEARCH_SCENARIO_ASSUMPTION):
        issues.append("requirement_version.row_status must be 'sourced' or 'research_scenario_assumption'")
    if st is ValueStatus.RESEARCH_SCENARIO_ASSUMPTION and not str(block["row_status_rationale"] or "").strip():
        issues.append("requirement_version.row_status_rationale is required for research_scenario_assumption")
    try:
        choices = RequirementChoices(growth_option=block["growth_option"], gestation_rule=block["gestation_rule"],
                                     ca_gestation_equation=block["ca_gestation_equation"],
                                     milk_p_equation=block["milk_p_equation"])
    except ValueError as exc:
        issues.append(f"requirement_version: {exc}")
        choices = None
    if issues:
        raise InvalidProblemError("; ".join(issues))
    return RequirementVersion(str(block["dmi_function"]), choices, str(block["maintenance_basis"]),  # type: ignore[arg-type]
                              tuple(dec), st, str(block["row_status_rationale"]))  # type: ignore[arg-type]


@dataclass(frozen=True)
class ReferenceCowRequirements:
    """DMI, absorbed Ca/P requirements and the linear mineral rows of one configured reference cow."""

    version: RequirementVersion
    milk_nel: EquationResult
    milk_energy: EquationResult
    dmi: EquationResult
    ca: FactorialRequirement
    p: FactorialRequirement
    ca_row: AbsorbedMineralRow
    p_row: AbsorbedMineralRow

    def summary(self) -> dict[str, Any]:
        return {"version": self.version.to_dict(), "milk_NEL_Mcal_per_kg": self.milk_nel.value,
                "MilkE_Mcal_d": self.milk_energy.value, "DMI_kg_d": self.dmi.value,
                "Ca_absorbed_requirement_g_d": self.ca.value, "P_absorbed_requirement_g_d": self.p.value,
                "Ca_row_bound_g_d": self.ca_row.bound_g_per_d, "P_row_bound_g_d": self.p_row.bound_g_per_d,
                "Ca_row_terms": dict(self.ca_row.terms), "P_row_terms": dict(self.p_row.terms),
                "Ca_row_open_decisions": list(self.ca_row.open_decisions),
                "P_row_open_decisions": list(self.p_row.open_decisions)}


def _cow_input(inputs: Mapping[str, Any], key: str, unit: str) -> Qty:
    blk = inputs.get(key)
    if not isinstance(blk, Mapping) or "value" not in blk:
        raise InvalidProblemError(f"reference_cow.inputs.{key}: missing value block")
    u = blk.get("unit", unit)
    if u != unit:
        raise InvalidProblemError(f"reference_cow.inputs.{key}: unit must be {unit!r}, got {u!r}")
    return Qty(float(blk["value"]), unit)


def requirements_from_config(animal_cfg: Mapping[str, Any]) -> ReferenceCowRequirements:
    """Apply the configured version to the configured reference cow (e.g. ``configs/dev_case_v1/animal.yaml``).

    Reads ``reference_cow.inputs`` (units: kg, d, kg/d, %, and ``1`` for the Eq 2-1 parity indicator and
    BCS) and ``requirement_version``.  When ``derived_expected`` is present, every listed value is
    re-checked within its ``tolerance`` and a mismatch raises (the configuration and the code cannot
    silently disagree).
    """
    version = requirement_version_from_config(animal_cfg.get("requirement_version", {}))
    cow = (animal_cfg.get("reference_cow") or {}).get("inputs") or {}
    missing = [k for k in _COW_KEYS if k not in cow]
    if missing:
        raise InvalidProblemError(f"reference_cow.inputs missing {missing}")
    units = {"parity_eq2_1": "1", "body_weight": "kg", "mature_body_weight": "kg", "body_condition_score": "1",
             "days_in_milk": "d", "milk_yield": "kg/d", "milk_fat": "%", "milk_true_protein": "%",
             "milk_lactose": "%", "days_pregnant": "d", "body_weight_gain_for_minerals": "kg/d"}
    q = {k: _cow_input(cow, k, u) for k, u in units.items()}
    nel = milk_net_energy_eq3_14b(q["milk_fat"], q["milk_true_protein"], q["milk_lactose"])
    me = milk_energy_output_eq20_220(nel.as_qty(), q["milk_yield"])
    if version.dmi_function == "dmi_eq2_1":
        dmi = dmi_eq2_1(q["parity_eq2_1"], me.as_qty(), q["body_weight"], q["body_condition_score"], q["days_in_milk"])
    else:  # Eq 20-21 uses An_Parity in [1, 2] = Eq 2-1 parity indicator + 1
        dmi = dmi_eq20_21_literal(Qty(q["parity_eq2_1"].value + 1.0, "1"), me.as_qty(), q["body_weight"],
                                  q["body_condition_score"], q["days_in_milk"])
    common = dict(dmi=dmi.as_qty(), milk_yield=q["milk_yield"], milk_true_protein=q["milk_true_protein"],
                  body_weight=q["body_weight"], mature_body_weight=q["mature_body_weight"],
                  body_weight_gain=q["body_weight_gain_for_minerals"], days_pregnant=q["days_pregnant"],
                  choices=version.choices, dmi_label=f"{version.dmi_function} (configured)")
    ca = ca_requirement_factorial(**common)
    p = p_requirement_factorial(**common)
    out = ReferenceCowRequirements(version, nel, me, dmi, ca, p,
                                   absorbed_mineral_row(ca, maintenance_basis=version.maintenance_basis),
                                   absorbed_mineral_row(p, maintenance_basis=version.maintenance_basis))
    exp = animal_cfg.get("derived_expected")
    if exp:
        tol = float(exp.get("tolerance", 1e-6))
        got = {"milk_NEL_concentration_Mcal_per_kg": nel.value, "MilkE_Mcal_d": me.value, "DMI_kg_d": dmi.value,
               "Ca_absorbed_requirement_g_d": ca.value, "P_absorbed_requirement_g_d": p.value}
        if version.maintenance_basis == "actual_D":
            got.update({"Ca_row_bound_actual_D_g_d": out.ca_row.bound_g_per_d,
                        "P_row_bound_actual_D_g_d": out.p_row.bound_g_per_d})
        bad = [f"{k}: expected {exp[k]}, computed {v:.9f}" for k, v in got.items()
               if k in exp and abs(float(exp[k]) - v) > tol]
        if bad:
            raise InvalidProblemError("derived_expected mismatch (configuration vs code): " + "; ".join(bad))
    return out


def mineral_constraint_specs(req: ReferenceCowRequirements, *, numerical_tolerance: float,
                             ca_id: str = "PN-CA-ABS", p_id: str = "PN-P-ABS") -> tuple[ConstraintSpec, ConstraintSpec]:
    """Engine constraint specs of the two absorbed-mineral rows under the configured version.

    The configured ``decisions_resolved`` are passed to :meth:`AbsorbedMineralRow.to_constraint_spec`; if
    a row still has an open decision the configuration did not settle, the relabel guard raises (no
    override is created here).
    """
    v = req.version
    prov = Provenance(status=v.row_status, source_id=NASEM_SOURCE_ID if v.row_status is ValueStatus.SOURCED else None,
                      locator=("; ".join(r.locator for r in req.ca_row.equations[:3])
                               if v.row_status is ValueStatus.SOURCED else None),
                      rationale=None if v.row_status is ValueStatus.SOURCED else v.row_status_rationale)
    prov_p = prov if v.row_status is not ValueStatus.SOURCED else Provenance(
        status=v.row_status, source_id=NASEM_SOURCE_ID, locator="; ".join(r.locator for r in req.p_row.equations[:3]))
    ca = req.ca_row.to_constraint_spec(ca_id, provenance=prov, numerical_tolerance=numerical_tolerance,
                                       decisions_resolved=v.decisions_resolved)
    p = req.p_row.to_constraint_spec(p_id, provenance=prov_p, numerical_tolerance=numerical_tolerance,
                                     decisions_resolved=v.decisions_resolved)
    return ca, p


# --------------------------------------------------------------------------------------------
# reference resolution for the dev-case configuration (K4c, 2026-09-25)
# --------------------------------------------------------------------------------------------
# ``configs/dev_case_v1/*.yaml`` do not copy NASEM table values: a value block either carries
# ``value`` itself, or ``value_from`` = "<tracked file relative to the repository root>#<dotted.key>"
# (e.g. the reference-cow inputs already kept in ``configs/animal_profile.yaml``), or ``value_ref`` =
# "restricted_values.yaml#<key>" (restricted values; the caller supplies them -- this function never
# reads a file under ``data/restricted_local``).  ``requirements_from_config`` only accepts resolved
# blocks, so every consumer must go through :func:`resolve_value_refs` first.

_RESTRICTED_DIR_PARTS = ("data", "restricted_local")


def resolve_value_refs(cfg: Any, *, repo_root: Any, restricted_values: Optional[Mapping[str, Any]] = None,
                       tol: float = 1e-9) -> tuple[Any, list[dict[str, Any]]]:
    """Deep copy of ``cfg`` with ``value`` filled in for every ``value_from`` / ``value_ref`` block.

    * ``value_from``: the referenced file must lie inside ``repo_root`` and outside
      ``data/restricted_local`` (a tracked file must not silently pull a restricted value);
      the dotted key path must end at a finite number.  If the block also has ``value``, the two
      must agree within ``tol`` (the file wins nothing: a mismatch is an error).
    * ``value_ref``: looked up in ``restricted_values`` (``{key: value or {"value": ...}}``); without
      that mapping the block is left unresolved and listed as such in the log.

    Returns ``(resolved_cfg, log)``; ``log`` has one entry per reference (config path, reference,
    resolved value or ``None``).  All problems are collected and raised together as
    :class:`InvalidProblemError`.
    """
    import copy

    import yaml

    root = Path(repo_root).resolve()
    out = copy.deepcopy(cfg)
    log: list[dict[str, Any]] = []
    issues: list[str] = []
    cache: dict[Path, Any] = {}

    def _lookup(ref: str) -> Any:
        if "#" not in ref:
            raise KeyError("reference must be '<file>#<dotted.key>'")
        fname, keys = ref.split("#", 1)
        p = (root / fname).resolve()
        if root not in p.parents:
            raise KeyError(f"file {fname!r} is outside the repository root")
        rel = p.relative_to(root).parts
        if rel[:2] == _RESTRICTED_DIR_PARTS:
            raise KeyError(f"value_from may not point into data/restricted_local ({fname!r}); use value_ref")
        if p not in cache:
            if not p.is_file():
                raise KeyError(f"file {fname!r} not found")
            cache[p] = yaml.safe_load(p.read_text(encoding="utf-8"))
        node = cache[p]
        for k in keys.split("."):
            if not isinstance(node, Mapping) or k not in node:
                raise KeyError(f"key {k!r} of {keys!r} not found in {fname!r}")
            node = node[k]
        return node

    def _num(x: Any) -> float:
        v = x.get("value") if isinstance(x, Mapping) else x
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise ValueError(f"resolved value {v!r} is not a finite number")
        return float(v)

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            if "value_from" in node:
                ref = str(node["value_from"])
                try:
                    v = _num(_lookup(ref))
                    if "value" in node and node["value"] is not None:
                        if abs(float(node["value"]) - v) > tol:
                            issues.append(f"{path}: value {node['value']} != {ref} = {v}")
                    else:
                        node["value"] = v
                    log.append({"path": path, "value_from": ref, "value": v})
                except (KeyError, ValueError, TypeError) as exc:
                    issues.append(f"{path}: cannot resolve value_from {ref!r}: {exc}")
            if "value_ref" in node:
                ref = str(node["value_ref"])
                key = ref.split("#", 1)[1] if "#" in ref else ref
                if restricted_values is None:
                    log.append({"path": path, "value_ref": ref, "value": None})
                elif key not in restricted_values:
                    issues.append(f"{path}: value_ref key {key!r} not in the supplied restricted values")
                else:
                    try:
                        v = _num(restricted_values[key])
                        if "value" in node and node["value"] is not None and abs(float(node["value"]) - v) > tol:
                            issues.append(f"{path}: value {node['value']} != {ref}")
                        node["value"] = v
                        log.append({"path": path, "value_ref": ref, "value": v})
                    except (ValueError, TypeError) as exc:
                        issues.append(f"{path}: value_ref {ref!r}: {exc}")
            for k, v in node.items():
                if k not in ("value_from", "value_ref"):
                    _walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                _walk(v, f"{path}[{i}]")

    _walk(out, "")
    if issues:
        raise InvalidProblemError("unresolved value references: " + "; ".join(issues))
    return out, log


__all__ += ["DMI_FUNCTIONS", "RequirementVersion", "requirement_version_from_config", "ReferenceCowRequirements",
            "requirements_from_config", "mineral_constraint_specs", "resolve_value_refs"]
