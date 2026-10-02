"""Fixed-DMI linearisation of the NASEM (2021) Chapter 3 diet NEL supply, and the NEL requirement.

Status (contract section 2.3): ``implemented``, ``unit_passed``
(``tests/numerical/test_energy_linearization.py``; hand-computed synthetic examples only).  Passing
the tests shows that the Chapter 3 equations are computed as transcribed and that the linear row is
exact at its declared reference point -- not that any diet meets any cow's energy needs.

Why a linearisation (review round 2, R4 item 4)
-----------------------------------------------
NASEM (2021) Chapter 3 computes diet energy on a whole-diet basis (p.22: feed NEL values are "no
longer provided ... as diet NEL supply must be based on the whole diet").  The chain is nonlinear in
the ration because (a) NDF and starch digestibility are discounted by the intake level
``DMI/BW`` and NDF digestibility also by the *diet* starch concentration (Eq 3-5a/3-5b, p.25-26),
and (b) gas energy (Eq 3-9, p.28) uses diet concentrations of FA and digested NDF.  With the DMI
fixed at the scenario value ``DMI_scn`` (the planned DM supply of ``SH-DM-PLAN``) and the diet starch
in Eq 3-5a fixed at a declared reference ``S_ref``, every step becomes affine in the per-feed
composition and the diet NEL supply is linear in the realised DM amounts ``x_i = q_i d_i``::

    NEL(q, theta) = sum_i x_i(theta) * e_i(theta) + C0                                (Mcal/d)

    e_i = 0.66 * [ DE_i - GasE_i - UE_i ]                                             (Mcal/kg DM)
    DE_i   = 0.042 NDF_i dNDF_i + 0.0423 St_i dSt_i + 0.094 FA_i dFA_i
             + 0.0565 (RDP_i + dRUP_i) + 0.040 x 0.96 ROM_i
             - 0.00565 (11.62 + 0.134 NDF_i) - 0.00565 fMCP - 0.0040 efROM            Eq 3-8, 3-6a (p.27)
    dNDF_i = dNDF_base_i - 0.0059 (S_ref - 26) - 1.1 (DMI_scn/BW - 0.035)              Eq 3-3a (p.24), 3-5a (p.25)
    dSt_i  = dSt_base_i  - 1.0 (DMI_scn/BW - 0.035)                                    Table 3-1 (p.25), Eq 3-5b (p.26)
    ROM_i  = 100 - Ash_i - NDF_i - St_i - FA_i/1.06 - CP_i                            Eq 3-1 (p.22)
    GasE_i = 0.294 - 0.347 FA_i / DMI_scn + 0.0409 NDF_i dNDF_i / DMI_scn              Eq 3-9 (p.28)
    UE_i   = 0.0146 (1000/6.25) [ (RDP_i + dRUP_i)/100 - fMCP/1000
                                  - (11.62 + 0.134 NDF_i)/1000 ]                       Eq 3-10a/b, 3-7b (p.27-28)
    C0     = 0.66 x 0.0146 (1000/6.25) (Milk CP + body-gain CP)                        Eq 3-10a, 3-12 (p.28)

(composition in % of DM inside the brackets).  ``fMCP`` = 16.5 g/kg DMI and ``efROM`` = 34.3 g/kg DMI
are the values the book uses for the Table 19-1 DE values (p.27); the model proper computes fMCP from
predicted microbial CP (Eq 3-6b), which is nonlinear -- keeping 16.5 g/kg DMI is part of this
linearisation and is declared as such.

Where the linear row is exact and where it is not
-------------------------------------------------
* Exact (to floating point) when ``D = sum_i x_i = DMI_scn``, the diet starch equals ``S_ref``, and
  fMCP = 16.5 g/kg DMI: :func:`nonlinear_diet_nel` (the full Chapter 3 chain) and the linear row
  coincide (unit-tested).
* ``S_ref`` is a declared research setting (``S_ref`` = the Table 5-1 starch maximum, the bound of
  constraint PN-T4).  The row is *not above the chain* only at its reference composition: when every
  feed's composition equals the value the linearisation was built at (table means), ``D = DMI_scn``
  and the diet starch is ``<= S_ref`` (a lower diet starch raises NDF digestibility, Eq 3-5a).
  **With drawn composition this guarantee does not hold** (review round 2, red team B; corrected by
  FIX_B): the row keeps ``dNDF_base`` (Eq 3-3a) at each feed's *mean* NDF and lignin and the Eq 3-9
  concentrations at ``DMI_scn``, whereas the chain recomputes Eq 3-3a at the *drawn* NDF.  A draw
  with NDF below the mean has a lower true NDF digestibility, so the linear row can lie above the
  chain even when the diet starch is ``<= S_ref`` and ``D = DMI_scn``.  The bias understates energy
  violations; :func:`nonlinear_diet_nel` quantifies it per state and development runs report the
  residual grouped by ``starch <= S_ref`` (``reports/dev_case_v1_report.md``).  A first-order Eq 3-3a
  term in NDF (or its adverse end over the uncertainty set) would restore a draw-wise bound; it is a
  registered sensitivity option, not implemented.  A scenario whose starch exceeds ``S_ref`` already
  violates PN-T4 and so is in the joint-violation event.
* Scenario DM deviations (``D != DMI_scn``) are handled exactly for everything proportional to DM
  mass; the concentrations in Eq 3-9 and the DMI/BW discounts are evaluated at ``DMI_scn`` (the fixed
  DMI assumption).  :func:`nonlinear_diet_nel` quantifies the residual for any given state.
* Per-feed digestibility coefficients (dNDF_base from Eq 3-3a at the feed's *mean* NDF and lignin,
  dSt_base, dFA, the RUP fractions) are held at their table values; composition columns that are
  random in the uncertainty model (e.g. NDF, starch, CP) enter linearly through
  :class:`NELLinearisation` -- the columns that are not random stay at their means.
* Review round 3 (F3): the linear row is therefore a *candidate-generation* model, not a draw-wise
  lower bound.  :func:`ration_reliability.nutrition.energy_reference.reference_energy_check` scores a
  fixed ``q`` state by state with both the linear row and :func:`nonlinear_diet_nel` and reports false
  passes / false fails and the error distribution.  :func:`nonlinear_diet_nel` is the project's
  reference chain (fixed fMCP, fixed FA and lignin, no NPN / fat supplements / monensin, DMI = supplied
  DM); it is **not** the full NASEM model or software and not an animal validation
  (``reports/energy_domain_audit.md``).

Engine integration
------------------
The constraint compiler gives any expression built from mass-fraction nutrients or ``DM`` terms a
mass-rate unit, so an energy row in Mcal/d needs an ``energy_density`` column.  This module therefore
provides the column ``NEL_fixedDMI`` (Mcal/kg DM, the per-feed contribution ``e_i``; *not* a feed NEL
value in the NASEM sense, see p.22) as a deterministic function of the drawn composition:
:meth:`NELLinearisation.density`, :func:`append_energy_column` (for an existing ``DrawSet``) and
:class:`EnergyColumnModel` (wrapper around any :class:`~ration_reliability.uncertainty.base.UncertaintyModel`).
The constraint is ``kind=supply, terms={"NEL_fixedDMI": 1}, unit="Mcal/d", sense=ge`` with bound
``NEL_req - C0`` (:func:`nel_constraint_spec`); the public evaluator then uses each scenario's own DM.

Requirement (Chapter 3, p.29-35): :func:`nel_requirement_chapter3` = maintenance (Eq 3-13) + activity
(declared input) + milk (Eq 3-14b x yield) + gestation (Eq 3-15a, 3-16a/b, 3-17a/b, 3-18) + frame gain
(Eq 3-20a-e) + body-reserve gain (Eq 3-19a/3-19c).  The leftmost printed factor expression of each
equation is used (e.g. ``6.3 x 0.89`` rather than the rounded 5.6 of Eq 3-19a); the differences are
reported in the result notes.

Source: NASEM (2021) *Nutrient Requirements of Dairy Cattle*, 8th rev. ed., local PDF (source id
``SRC-C-NASEM21-T19``); printed page numbers (PDF page = printed + 20).  No feed-table value is
embedded here; feed inputs are passed in by the caller (restricted values stay in
``data/restricted_local/``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..datamodel import (
    ConstraintClass,
    ConstraintKind,
    ConstraintSpec,
    DMSource,
    NutrientSpec,
    Provenance,
    Sense,
)
from ..errors import InvalidProblemError, UnitError
from ..hashing import stable_hash
from ..normalization import units as U
from .requirements import NASEM_SOURCE_ID, EquationRef, EquationResult, Qty

__all__ = [
    "ENERGY_COLUMN_ID",
    "LINEARISATION_ID",
    "FeedEnergyInputs",
    "FixedDMISettings",
    "NELLinearisation",
    "NELRequirement",
    "dndf_base_lignin_eq3_3a",
    "rom_eq3_1",
    "feed_de_base_eq3_8",
    "feed_energy_terms",
    "linearise_nel_fixed_dmi",
    "nonlinear_diet_nel",
    "nel_requirement_chapter3",
    "nel_constraint_spec",
    "energy_nutrient_spec",
    "append_energy_column",
    "energy_world_fingerprint",
    "EnergyColumnModel",
    "energy_box_from_factory_model",
    "composition_closure_report",
    "CLOSURE_SUM_COLUMNS",
    "ANALYSIS_ANOMALY_CLASS",
    "SUPPORT_VIOLATION_CLASS",
    "SUM_OVERLAP_SUBCLASS",
    "DERIVED_QUANTITY_SUBCLASS",
    "CompositionStateClasses",
    "classify_composition_states",
]

#: Engine nutrient id of the per-feed NEL contribution at fixed DMI (energy_density, Mcal/kg DM).
ENERGY_COLUMN_ID = "NEL_fixedDMI"
#: Version tag of the linearisation (hashed into fingerprints; bump on any formula change).
LINEARISATION_ID = "NASEM2021_ch3_fixedDMI_linear_v1"

# ---------------------------------------------------------------------------------------------
# Chapter 3 constants (printed coefficients; page = printed page)
# ---------------------------------------------------------------------------------------------
_GE = {"NDF": 0.042, "St": 0.0423, "FA": 0.094, "CP": 0.0565, "ROM": 0.040}   # Mcal per %DM (Eq 3-2/3-8, p.23, p.27)
_DROM = 0.96            # true digestibility of ROM (p.25)
_FAT_FACTOR = 1.06      # Eq 3-1 (p.22), basal feeds (1 only for FA supplements / FA soaps)
_MFCP_A, _MFCP_B = 11.62, 0.134     # MFCP g/kg DMI = 11.62 + 0.134 NDF%  (Eq 3-6a, p.27)
_EN_CP = 0.00565        # Mcal/g CP (5.65 Mcal/kg; p.27)
_EN_ROM = 0.0040        # Mcal/g ROM (4.0 Mcal/kg; p.27)
_FMCP_TABLE = 16.5      # g undigested bacterial CP / kg DMI used for the Table 19-1 DE values (p.27)
_EFROM_TABLE = 34.3     # g endogenous fecal ROM / kg DMI (p.27)
_DIET_NDF_TABLE = 30.0  # diet NDF % used for MFCP in the Table 19-1 DE values (p.27)
_ST_BASE, _DMI_BW_BASE = 26.0, 0.035    # base conditions: 26 % starch, DMI 3.5 % of BW (p.22, p.24)
_K_ST_NDF, _K_DMI_NDF, _K_DMI_ST = 0.0059, 1.1, 1.0   # Eq 3-5a (p.25), Eq 3-5b (p.26)
_GAS = (0.294, 0.347, 0.0409)           # Eq 3-9 (p.28)
_UE_PER_GN = 0.0146     # Mcal / g urinary N (Eq 3-10b, p.28)
_N_PER_CP = 1000.0 / 6.25               # g N per kg CP (Eq 3-10a, p.28)
_KL = 0.66              # ME -> NEL (Eq 3-12, p.28)

_REFS = {
    "Eq 3-1": EquationRef("Eq 3-1", 22, "ROM by difference; FatFactor 1.06 for basal feeds"),
    "Eq 3-3a": EquationRef("Eq 3-3a", 24, "base NDF digestibility from lignin (model default, p.24)"),
    "Eq 3-4": EquationRef("Eq 3-4", 24, "digested CP = RDP + dRUP"),
    "Table 3-1": EquationRef("Table 3-1", 25, "base starch digestibility by source"),
    "FA/ROM": EquationRef("text p.25", 25, "base FA digestibility 0.73 (basal feeds); ROM true digestibility 0.96"),
    "Eq 3-5a": EquationRef("Eq 3-5a", 25, "NDF digestibility adjusted for diet starch and DMI/BW"),
    "Eq 3-5b": EquationRef("Eq 3-5b", 26, "starch digestibility adjusted for DMI/BW"),
    "Eq 3-6a": EquationRef("Eq 3-6a", 27, "metabolic fecal CP"),
    "p27": EquationRef("text p.27", 27, "Table 19-1 DE: diet NDF 30 %, fMCP 16.5 g/kg, efROM 34.3 g/kg DMI"),
    "Eq 3-7b": EquationRef("Eq 3-7b", 27, "apparently digested CP"),
    "Eq 3-8": EquationRef("Eq 3-8", 27, "digestible energy"),
    "Eq 3-9": EquationRef("Eq 3-9", 28, "gas energy"),
    "Eq 3-10a": EquationRef("Eq 3-10a", 28, "urinary N; body-gain CP of lactating cows can be ignored"),
    "Eq 3-10b": EquationRef("Eq 3-10b", 28, "urinary energy 0.0146 Mcal/g N"),
    "Eq 3-11": EquationRef("Eq 3-11", 28, "ME = DE - GasE - UE"),
    "Eq 3-12": EquationRef("Eq 3-12", 28, "NEL = 0.66 ME"),
    "Eq 3-13": EquationRef("Eq 3-13", 29, "NEL maintenance 0.10 BW^0.75"),
    "Eq 3-14b": EquationRef("Eq 3-14b", 30, "milk NEL from fat, true protein, lactose"),
    "activity": EquationRef("text p.30-31", 30, "maintenance covers normal activity in confinement"),
    "Eq 3-15a": EquationRef("Eq 3-15a", 31, "gravid uterus at parturition = 1.825 x calf birth weight"),
    "Eq 3-15b": EquationRef("Eq 3-15b", 31, "uterus after calving = 0.2288 x calf birth weight"),
    "Eq 3-16a": EquationRef("Eq 3-16a", 32, "gravid uterine weight during gestation"),
    "Eq 3-16b": EquationRef("Eq 3-16b", 32, "uterine involution"),
    "Eq 3-17a": EquationRef("Eq 3-17a", 32, "gravid uterine gain"),
    "Eq 3-17b": EquationRef("Eq 3-17b", 32, "gain during involution"),
    "Eq 3-18": EquationRef("Eq 3-18", 32, "gestation NEL = gain x (0.882/0.14) x 0.66"),
    "p33": EquationRef("text p.33", 33, "1 kg frame gain of a cow = 0.82 kg EBW + 0.18 kg gut fill"),
    "Eq 3-19a": EquationRef("Eq 3-19a", 34, "reserve gain (lactating): 6.3 Mcal RE/kg x 0.89"),
    "Eq 3-19c": EquationRef("Eq 3-19c", 34, "reserve loss provides 6.3 x 0.89 Mcal NEL/kg"),
    "Eq 3-20": EquationRef("Eq 3-20a-e", 35, "frame gain composition, RE, NEL = RE/0.61"),
}


def _r(*keys: str) -> tuple[EquationRef, ...]:
    return tuple(_REFS[k] for k in keys)


def _qv(x: Any, target: str, name: str) -> float:
    """``x`` (a :class:`Qty`) in ``target`` units; ``d`` for days and ``1`` for pure numbers."""
    if not isinstance(x, Qty):
        raise TypeError(f"{name}: pass Qty(value, unit); bare numbers are refused")
    v = float(x.value)
    if not math.isfinite(v):
        raise ValueError(f"{name}: value must be finite")
    if target in ("d", "1"):
        if x.unit != target:
            raise UnitError(f"{name}: needs unit {target!r}, got {x.unit!r}")
        return v
    return float(U.convert(v, x.unit, target))


# ---------------------------------------------------------------------------------------------
# feed inputs
# ---------------------------------------------------------------------------------------------

#: Composition fields of :class:`FeedEnergyInputs` (all % of DM) and the engine nutrient ids they map to.
COMPOSITION_FIELDS = ("ndf", "lignin", "starch", "fa", "cp", "ash")


@dataclass(frozen=True)
class FeedEnergyInputs:
    """Chapter 3 energy inputs of one feed (composition in % of DM, fractions as proportions).

    ``rup_pct_cp`` (RUP, % of CP) and ``drup_pct_rup`` (digestible RUP, % of RUP) are the Table 19-1 /
    feed-library values (Eq 3-4); ``dstarch_base`` is the Table 3-1 base starch digestibility of the
    source (Table 3-1 gives a default for unlisted feeds, p.25); ``dfa`` the base FA digestibility (0.73 for
    basal feeds, p.25).  ``fat_factor`` is 1.06 for basal feeds (Eq 3-1).  Supplemental NPN is not
    supported (``sNPNCPE`` = 0 is a declared scope limit; Eq 3-1/3-8 handle it separately).
    For a mineral (ash = 100 %) every organic fraction is 0 and the feed still carries its share of
    the per-kg-DMI endogenous losses and methane (Eq 3-8, 3-9).
    """

    ingredient_id: str
    ndf: float
    lignin: float
    starch: float
    fa: float
    cp: float
    ash: float
    rup_pct_cp: float
    drup_pct_rup: float
    dstarch_base: float
    dfa: float
    fat_factor: float = _FAT_FACTOR

    def __post_init__(self) -> None:
        errs = []
        for f in COMPOSITION_FIELDS:
            v = float(getattr(self, f))
            if not (math.isfinite(v) and 0.0 <= v <= 100.0):
                errs.append(f"{f}={v} must be a finite % of DM in [0, 100]")
        if self.lignin > self.ndf:
            errs.append("lignin cannot exceed NDF")
        for f in ("rup_pct_cp", "drup_pct_rup"):
            v = float(getattr(self, f))
            if not (math.isfinite(v) and 0.0 <= v <= 100.0):
                errs.append(f"{f}={v} must be in [0, 100]")
        for f in ("dstarch_base", "dfa"):
            v = float(getattr(self, f))
            if not (math.isfinite(v) and 0.0 <= v <= 1.0):
                errs.append(f"{f}={v} must be a proportion in [0, 1]")
        if self.fat_factor not in (1.0, _FAT_FACTOR):
            errs.append("fat_factor must be 1.06 (basal feeds) or 1.0 (FA supplements), Eq 3-1")
        if errs:
            raise InvalidProblemError(f"FeedEnergyInputs {self.ingredient_id}: " + "; ".join(errs))

    @property
    def digested_cp_fraction(self) -> float:
        """(RDP + dRUP) / CP = (1 - RUP) + RUP x dRUP (Eq 3-4), proportions."""
        rup = self.rup_pct_cp / 100.0
        return (1.0 - rup) + rup * self.drup_pct_rup / 100.0

    def composition(self) -> dict[str, float]:
        return {f: float(getattr(self, f)) for f in COMPOSITION_FIELDS}

    def to_dict(self) -> dict[str, Any]:
        return {"ingredient_id": self.ingredient_id, **self.composition(), "rup_pct_cp": self.rup_pct_cp,
                "drup_pct_rup": self.drup_pct_rup, "dstarch_base": self.dstarch_base, "dfa": self.dfa,
                "fat_factor": self.fat_factor}


def dndf_base_lignin_eq3_3a(ndf_pct: float, lignin_pct: float) -> float:
    """Base proportion of NDF digested (Eq 3-3a, p.24)::

        dNDF_NDF_base = 0.75 (NDF - Lg) [1 - (Lg / NDF)^0.667] / NDF

    A feed without NDF (e.g. a mineral) has no digested NDF; 0 is returned (the product
    ``NDF x dNDF`` is 0 either way).
    """
    ndf, lg = float(ndf_pct), float(lignin_pct)
    if ndf < 0 or lg < 0 or lg > ndf:
        raise ValueError("need 0 <= lignin <= NDF")
    if ndf == 0.0:
        return 0.0
    return 0.75 * (ndf - lg) * (1.0 - (lg / ndf) ** 0.667) / ndf


def rom_eq3_1(*, ndf: float, starch: float, fa: float, cp: float, ash: float, fat_factor: float = _FAT_FACTOR) -> float:
    """Residual organic matter, % of DM (Eq 3-1, p.22), without supplemental NPN."""
    return 100.0 - ash - ndf - starch - fa / fat_factor - cp


def feed_de_base_eq3_8(feed: FeedEnergyInputs) -> EquationResult:
    """Feed DE at base conditions (Mcal/kg DM) as used for the Table 19-1 "DE base" values.

    Eq 3-8 (p.27) with the base digestibilities (Eq 3-3a, Table 3-1, FA 0.73, ROM 0.96, Eq 3-4) and
    the endogenous losses the book states for the table values (p.27: diet NDF 30 % so MFCP =
    15.6 g/kg; undigested bacterial CP 16.5 g/kg; efROM 34.3 g/kg DMI).  Used to check a transcribed
    DE base value against the equations, not as a feed NEL.
    """
    c = feed.composition()
    dndf = dndf_base_lignin_eq3_3a(c["ndf"], c["lignin"])
    rom = rom_eq3_1(ndf=c["ndf"], starch=c["starch"], fa=c["fa"], cp=c["cp"], ash=c["ash"], fat_factor=feed.fat_factor)
    mfcp = _MFCP_A + _MFCP_B * _DIET_NDF_TABLE
    parts = {
        "NDF": _GE["NDF"] * c["ndf"] * dndf,
        "starch": _GE["St"] * c["starch"] * feed.dstarch_base,
        "FA": _GE["FA"] * c["fa"] * feed.dfa,
        "CP": _GE["CP"] * c["cp"] * feed.digested_cp_fraction,
        "ROM": _GE["ROM"] * rom * _DROM,
        "endogenous": -(_EN_CP * mfcp + _EN_CP * _FMCP_TABLE + _EN_ROM * _EFROM_TABLE),
    }
    return EquationResult("DE_base", sum(parts.values()), "Mcal/kg",
                          _r("Eq 3-8", "Eq 3-1", "Eq 3-3a", "Table 3-1", "FA/ROM", "Eq 3-4", "p27"),
                          {k: (float(v), "%") for k, v in c.items()},
                          {**parts, "dNDF_base": dndf, "ROM": rom, "MFCP_g_per_kg": mfcp},
                          variant="Table 19-1 base convention")


# ---------------------------------------------------------------------------------------------
# fixed-DMI settings and per-feed energy terms
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FixedDMISettings:
    """Research settings of the fixed-DMI linearisation (no defaults; each needs a source or rationale).

    ``dmi_kg_d``: the fixed scenario DMI (planned DM supply; ``SH-DM-PLAN``).
    ``body_weight_kg``: for DMI/BW in Eq 3-5a/3-5b.
    ``starch_ref_pct``: diet starch used in the Eq 3-5a starch adjustment of NDF digestibility.
    ``milk_cp_kg_d`` and ``body_gain_cp_kg_d``: the protein outputs subtracted in Eq 3-10a.
    """

    dmi_kg_d: float
    body_weight_kg: float
    starch_ref_pct: float
    milk_cp_kg_d: float
    body_gain_cp_kg_d: float

    def __post_init__(self) -> None:
        errs = []
        if not (math.isfinite(self.dmi_kg_d) and self.dmi_kg_d > 0):
            errs.append("dmi_kg_d must be > 0")
        if not (math.isfinite(self.body_weight_kg) and self.body_weight_kg > 0):
            errs.append("body_weight_kg must be > 0")
        if not (math.isfinite(self.starch_ref_pct) and 0 <= self.starch_ref_pct <= 100):
            errs.append("starch_ref_pct must be in [0, 100]")
        for f in ("milk_cp_kg_d", "body_gain_cp_kg_d"):
            v = getattr(self, f)
            if not (math.isfinite(v) and v >= 0):
                errs.append(f"{f} must be >= 0")
        if errs:
            raise InvalidProblemError("FixedDMISettings: " + "; ".join(errs))

    @property
    def dmi_bw(self) -> float:
        """DMI as a proportion of BW (kg/kg), Eq 3-5a/3-5b."""
        return self.dmi_kg_d / self.body_weight_kg

    def to_dict(self) -> dict[str, float]:
        return {"dmi_kg_d": self.dmi_kg_d, "body_weight_kg": self.body_weight_kg, "starch_ref_pct": self.starch_ref_pct,
                "milk_cp_kg_d": self.milk_cp_kg_d, "body_gain_cp_kg_d": self.body_gain_cp_kg_d,
                "dmi_bw": self.dmi_bw}


def feed_energy_terms(feed: FeedEnergyInputs, settings: FixedDMISettings,
                      composition: Optional[Mapping[str, float]] = None) -> dict[str, float]:
    """Per-kg-DM energy terms of one feed at fixed DMI (Mcal/kg DM): DE_i, GasE_i, UE_i, ME_i, NEL_i.

    ``composition`` (``% of DM`` for ``ndf``/``starch``/``fa``/``cp``/``ash``) overrides the feed's
    mean composition; the digestibility coefficients stay those of ``feed`` (dNDF_base at the
    *mean* NDF and lignin).  Every term is affine in the composition (module docstring).
    """
    c = feed.composition()
    if composition:
        unknown = set(composition) - set(COMPOSITION_FIELDS)
        if unknown:
            raise KeyError(f"unknown composition fields {sorted(unknown)}")
        c.update({k: float(v) for k, v in composition.items()})
    dmi_bw = settings.dmi_bw
    dndf = (dndf_base_lignin_eq3_3a(feed.ndf, feed.lignin)
            - _K_ST_NDF * (settings.starch_ref_pct - _ST_BASE) - _K_DMI_NDF * (dmi_bw - _DMI_BW_BASE))
    dst = feed.dstarch_base - _K_DMI_ST * (dmi_bw - _DMI_BW_BASE)
    pcp = feed.digested_cp_fraction
    rom = rom_eq3_1(ndf=c["ndf"], starch=c["starch"], fa=c["fa"], cp=c["cp"], ash=c["ash"], fat_factor=feed.fat_factor)
    mfcp = _MFCP_A + _MFCP_B * c["ndf"]                               # g / kg DM of this feed
    de = (_GE["NDF"] * c["ndf"] * dndf + _GE["St"] * c["starch"] * dst + _GE["FA"] * c["fa"] * feed.dfa
          + _GE["CP"] * c["cp"] * pcp + _GE["ROM"] * rom * _DROM
          - _EN_CP * mfcp - _EN_CP * _FMCP_TABLE - _EN_ROM * _EFROM_TABLE)
    gas = _GAS[0] - _GAS[1] * c["fa"] / settings.dmi_kg_d + _GAS[2] * c["ndf"] * dndf / settings.dmi_kg_d
    ue = _UE_PER_GN * _N_PER_CP * (c["cp"] * pcp / 100.0 - _FMCP_TABLE / 1000.0 - mfcp / 1000.0)
    me = de - gas - ue
    return {"DE": de, "GasE": gas, "UE": ue, "ME": me, "NEL": _KL * me, "dNDF": dndf, "dSt": dst,
            "ROM": rom, "MFCP_g_per_kg": mfcp, "digested_cp_fraction": pcp}


# ---------------------------------------------------------------------------------------------
# linearisation object
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class NELLinearisation:
    """Per-feed affine NEL contribution ``e_i(a) = intercept_i + sum_j slope_ij a_j`` (fixed DMI).

    ``nutrient_map`` maps engine nutrient ids (e.g. ``"NDF"``) to :data:`COMPOSITION_FIELDS`
    (e.g. ``"ndf"``); ``slopes[i, j]`` is in Mcal/kg DM per (kg/kg DM) of nutrient ``j`` (engine
    canonical fraction units); composition fields not mapped are held at the feed means and live in
    ``intercept``.  ``constant_mcal_d`` is ``C0`` (added outside the per-feed sum).
    """

    ingredient_ids: tuple[str, ...]
    nutrient_ids: tuple[str, ...]
    nutrient_map: Mapping[str, str]
    intercept: np.ndarray
    slopes: np.ndarray
    constant_mcal_d: float
    settings: FixedDMISettings
    feeds: tuple[FeedEnergyInputs, ...]
    nominal_density: np.ndarray
    equations: tuple[EquationRef, ...]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for a in (self.intercept, self.slopes, self.nominal_density):
            a.setflags(write=False)

    def fingerprint(self) -> str:
        return stable_hash(LINEARISATION_ID, self.ingredient_ids, self.nutrient_ids, tuple(sorted(self.nutrient_map.items())),
                           self.intercept, self.slopes, float(self.constant_mcal_d), self.settings.to_dict(),
                           tuple(tuple(sorted(f.to_dict().items())) for f in self.feeds))

    def density(self, theta: np.ndarray, nutrient_ids: Sequence[str]) -> np.ndarray:
        """``e`` for composition arrays ``theta [..., I, J]`` in canonical fractions (Mcal/kg DM).

        ``nutrient_ids`` labels the last axis; every id of :attr:`nutrient_ids` must be present.
        A NaN in a used column gives NaN (missing data is never replaced).
        """
        th = np.asarray(theta, dtype=float)
        if th.shape[-2] != len(self.ingredient_ids):
            raise ValueError("theta ingredient axis does not match the linearisation")
        idx = []
        for n in self.nutrient_ids:
            if n not in nutrient_ids:
                raise KeyError(f"nutrient {n!r} needed by the energy linearisation is not in the draws")
            idx.append(list(nutrient_ids).index(n))
        sub = th[..., idx] if idx else np.zeros(th.shape[:-1] + (0,))
        return self.intercept + np.einsum("...ij,ij->...i", sub, self.slopes)

    def supply(self, x_dm: np.ndarray, theta: np.ndarray, nutrient_ids: Sequence[str]) -> np.ndarray:
        """Linear NEL supply (Mcal/d) for realised DM amounts ``x_dm [..., I]``."""
        return np.einsum("...i,...i->...", np.asarray(x_dm, dtype=float), self.density(theta, nutrient_ids)) \
            + self.constant_mcal_d

    def to_record(self) -> dict[str, Any]:
        """JSON-safe description (contains feed inputs: keep it in restricted storage if they are restricted)."""
        return {"linearisation_id": LINEARISATION_ID, "fingerprint": self.fingerprint(),
                "ingredient_ids": list(self.ingredient_ids), "nutrient_ids": list(self.nutrient_ids),
                "nutrient_map": dict(self.nutrient_map), "intercept_mcal_per_kg_dm": self.intercept.tolist(),
                "slopes_mcal_per_kg_dm_per_fraction": self.slopes.tolist(), "constant_mcal_d": self.constant_mcal_d,
                "nominal_density_mcal_per_kg_dm": self.nominal_density.tolist(), "settings": self.settings.to_dict(),
                "feeds": [f.to_dict() for f in self.feeds], "equations": [r.to_dict() for r in self.equations],
                "notes": list(self.notes)}


def linearise_nel_fixed_dmi(feeds: Sequence[FeedEnergyInputs], settings: FixedDMISettings, *,
                            nutrient_map: Mapping[str, str]) -> NELLinearisation:
    """Build the fixed-DMI affine NEL contribution of every feed (module docstring).

    ``nutrient_map``: engine nutrient id -> composition field for the columns that are random in the
    uncertainty model, e.g. ``{"NDF": "ndf", "starch": "starch", "CP": "cp"}``.  The slopes are
    obtained by evaluating the (affine) per-feed terms at unit perturbations, so they are exact.
    """
    bad = {k: v for k, v in nutrient_map.items() if v not in COMPOSITION_FIELDS}
    if bad:
        raise KeyError(f"nutrient_map values must be in {COMPOSITION_FIELDS}: {bad}")
    if "lignin" in nutrient_map.values():
        raise ValueError("lignin enters only through dNDF_base (fixed at the mean); it cannot be a linear column")
    ids = tuple(f.ingredient_id for f in feeds)
    if len(set(ids)) != len(ids):
        raise InvalidProblemError("duplicate ingredient ids in energy inputs")
    nut = tuple(nutrient_map)
    inter = np.zeros(len(feeds))
    slopes = np.zeros((len(feeds), len(nut)))
    nominal = np.zeros(len(feeds))
    for i, f in enumerate(feeds):
        base = f.composition()
        zero = dict(base)
        for fld in nutrient_map.values():
            zero[fld] = 0.0
        e0 = feed_energy_terms(f, settings, {k: zero[k] for k in ("ndf", "starch", "fa", "cp", "ash")})["NEL"]
        inter[i] = e0
        for j, (nid, fld) in enumerate(nutrient_map.items()):
            pert = dict(zero)
            pert[fld] = 1.0                                           # +1 % of DM
            e1 = feed_energy_terms(f, settings, {k: pert[k] for k in ("ndf", "starch", "fa", "cp", "ash")})["NEL"]
            slopes[i, j] = (e1 - e0) * 100.0                          # per kg/kg DM (fraction)
        nominal[i] = feed_energy_terms(f, settings)["NEL"]
    c0 = _KL * _UE_PER_GN * _N_PER_CP * (settings.milk_cp_kg_d + settings.body_gain_cp_kg_d)
    notes = ("fixed DMI: DMI/BW discounts and Eq 3-9 concentrations evaluated at the scenario DMI",
             f"Eq 3-5a starch adjustment evaluated at S_ref = {settings.starch_ref_pct} % DM",
             "fMCP 16.5 g/kg DMI and efROM 34.3 g/kg DMI as for the Table 19-1 DE values (p.27)",
             "per-feed digestibility coefficients (Eq 3-3a at mean NDF/lignin, Table 3-1, FA 0.73, RUP/dRUP) fixed",
             "e_i is a marginal contribution to diet NEL at fixed DMI, not a feed NEL value (p.22)")
    lin = NELLinearisation(ids, nut, dict(nutrient_map), inter, slopes, float(c0), settings, tuple(feeds), nominal,
                           _r("Eq 3-8", "Eq 3-1", "Eq 3-3a", "Table 3-1", "FA/ROM", "Eq 3-4", "Eq 3-5a", "Eq 3-5b",
                              "Eq 3-6a", "p27", "Eq 3-7b", "Eq 3-9", "Eq 3-10a", "Eq 3-10b", "Eq 3-11", "Eq 3-12"),
                           notes)
    # self-check: the affine form must reproduce the direct evaluation at the means
    theta_mean = np.array([[f.composition()[fld] / 100.0 for fld in nutrient_map.values()] for f in feeds]) \
        if nut else np.zeros((len(feeds), 0))
    back = lin.density(theta_mean, nut)
    if not np.allclose(back, nominal, rtol=0, atol=1e-12):
        raise AssertionError("energy linearisation self-check failed (affine reconstruction != direct evaluation)")
    return lin


# ---------------------------------------------------------------------------------------------
# full (nonlinear) Chapter 3 chain for a given diet -- diagnostics of the linearisation
# ---------------------------------------------------------------------------------------------

def nonlinear_diet_nel(x_dm: Sequence[float], feeds: Sequence[FeedEnergyInputs], *, body_weight_kg: float,
                       milk_cp_kg_d: float, body_gain_cp_kg_d: float,
                       compositions: Optional[Sequence[Mapping[str, float]]] = None,
                       fmcp_g_per_kg_dmi: float = _FMCP_TABLE) -> dict[str, float]:
    """Diet NEL (Mcal/d) by the whole-diet Chapter 3 chain with the diet's *own* DMI and starch.

    DMI = sum(x_dm) (supply assumed eaten), DMI/BW and diet starch enter Eq 3-5a/3-5b, Eq 3-9 uses
    the diet concentrations, dNDF_base is Eq 3-3a at each feed's (possibly drawn) NDF and lignin.
    fMCP stays a fixed rate (default 16.5 g/kg DMI, p.27) because microbial CP is outside this model.
    Returns DE, GasE, UE, ME, NEL (Mcal/d) and the intermediates.
    """
    x = np.asarray(x_dm, dtype=float)
    if x.shape != (len(feeds),) or np.any(x < 0):
        raise ValueError("x_dm must be a non-negative vector aligned with feeds")
    dmi = float(x.sum())
    if dmi <= 0:
        raise ValueError("diet DM must be > 0")
    comps = [dict(f.composition(), **({k: float(v) for k, v in compositions[i].items()} if compositions else {}))
             for i, f in enumerate(feeds)]
    dmi_bw = dmi / body_weight_kg
    starch_pct = float(sum(x[i] * c["starch"] for i, c in enumerate(comps)) / dmi)
    fa_pct = float(sum(x[i] * c["fa"] for i, c in enumerate(comps)) / dmi)
    ndf_pct = float(sum(x[i] * c["ndf"] for i, c in enumerate(comps)) / dmi)
    de_mcal = 0.0
    dndf_mass_pct_kg = 0.0          # sum_i x_i NDF_i dNDF_i  (kg x %)
    dcp_kg = 0.0
    for i, (f, c) in enumerate(zip(feeds, comps)):
        if c["lignin"] > c["ndf"]:
            raise ValueError(f"{f.ingredient_id}: lignin > NDF in the given composition "
                             f"({SUPPORT_VIOLATION_CLASS}; Eq 3-3a undefined; not clipped)")
        dndf = (dndf_base_lignin_eq3_3a(c["ndf"], c["lignin"])
                - _K_ST_NDF * (starch_pct - _ST_BASE) - _K_DMI_NDF * (dmi_bw - _DMI_BW_BASE))
        dst = f.dstarch_base - _K_DMI_ST * (dmi_bw - _DMI_BW_BASE)
        rom = rom_eq3_1(ndf=c["ndf"], starch=c["starch"], fa=c["fa"], cp=c["cp"], ash=c["ash"], fat_factor=f.fat_factor)
        de_mcal += x[i] * (_GE["NDF"] * c["ndf"] * dndf + _GE["St"] * c["starch"] * dst + _GE["FA"] * c["fa"] * f.dfa
                           + _GE["CP"] * c["cp"] * f.digested_cp_fraction + _GE["ROM"] * rom * _DROM)
        dndf_mass_pct_kg += x[i] * c["ndf"] * dndf
        dcp_kg += x[i] * c["cp"] / 100.0 * f.digested_cp_fraction
    mfcp_kg = dmi * (_MFCP_A + _MFCP_B * ndf_pct) / 1000.0
    fmcp_kg = dmi * fmcp_g_per_kg_dmi / 1000.0
    efrom_kg = dmi * _EFROM_TABLE / 1000.0
    de_mcal -= 1000.0 * (_EN_CP * mfcp_kg + _EN_CP * fmcp_kg + _EN_ROM * efrom_kg)
    gas = _GAS[0] * dmi - _GAS[1] * fa_pct + _GAS[2] * (dndf_mass_pct_kg / dmi)
    un_g = (dcp_kg - fmcp_kg - mfcp_kg - milk_cp_kg_d - body_gain_cp_kg_d) * _N_PER_CP
    ue = _UE_PER_GN * un_g
    me = de_mcal - gas - ue
    return {"DMI": dmi, "DMI_BW": dmi_bw, "starch_pct": starch_pct, "NDF_pct": ndf_pct, "FA_pct": fa_pct,
            "DE": de_mcal, "GasE": gas, "UN_g": un_g, "UE": ue, "ME": me, "NEL": _KL * me}


# ---------------------------------------------------------------------------------------------
# requirement
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class NELRequirement:
    """NEL requirement (Mcal/d) with its components."""

    total: EquationResult
    components: Mapping[str, EquationResult]

    @property
    def value(self) -> float:
        return self.total.value

    def to_dict(self) -> dict[str, Any]:
        return {"total": self.total.to_dict(), "components": {k: v.to_dict() for k, v in self.components.items()}}


def nel_requirement_chapter3(*, body_weight: Qty, milk_energy: Qty, days_pregnant: Qty, calf_birth_weight: Qty,
                             days_in_milk: Qty, frame_gain: Qty, reserves_gain: Qty, mature_body_weight: Qty,
                             empty_body_gain_per_frame_gain: Qty, activity: Qty) -> NELRequirement:
    """NEL requirement of a lactating cow by the Chapter 3 equations (p.29-35), Mcal/d.

    * maintenance 0.10 BW^0.75 (Eq 3-13); ``activity`` is added as a declared input (0 in confinement,
      p.30-31, where maintenance covers normal activity);
    * milk: ``milk_energy`` (Mcal/d; Eq 3-14b x yield, see ``requirements.milk_energy_output_eq20_220``);
    * gestation: gravid uterine gain (Eq 3-15a, 3-16a, 3-17a) x (0.882/0.14) x 0.66 (Eq 3-18), plus the
      involution term (Eq 3-15b, 3-16b, 3-17b; ~0 late in lactation) handled the same way (a negative
      gain gives a negative value, reported as is); no pregnancy -> 0;
    * frame gain: RE/0.61 per kg (Eq 3-20a-c, 3-20e) with EBG/ADG = ``empty_body_gain_per_frame_gain``
      (0.82 for a cow, p.33);
    * body reserves: gain x 6.3 x 0.89 (Eq 3-19a); a loss provides 6.3 x 0.89 per kg (Eq 3-19c).
    """
    bw = _qv(body_weight, "kg", "body_weight")
    mat = _qv(mature_body_weight, "kg", "mature_body_weight")
    me = _qv(milk_energy, "Mcal/d", "milk_energy")
    t = _qv(days_pregnant, "d", "days_pregnant")
    dim = _qv(days_in_milk, "d", "days_in_milk")
    calf = _qv(calf_birth_weight, "kg", "calf_birth_weight")
    fg = _qv(frame_gain, "kg/d", "frame_gain")
    rg = _qv(reserves_gain, "kg/d", "reserves_gain")
    ebg = _qv(empty_body_gain_per_frame_gain, "1", "empty_body_gain_per_frame_gain")
    act = _qv(activity, "Mcal/d", "activity")
    if bw <= 0 or mat <= 0 or me < 0 or calf <= 0 or dim < 0 or act < 0 or fg < 0 or not 0 < ebg <= 1:
        raise ValueError("invalid requirement inputs")
    if t != 0 and not 12 <= t <= 280:
        raise ValueError("day of gestation must be 0 (open) or within 12-280 (Eq 3-16a validity, p.32)")
    comps: dict[str, EquationResult] = {}
    comps["maintenance"] = EquationResult("NEL_maintenance", 0.10 * bw ** 0.75, "Mcal/d", _r("Eq 3-13"),
                                          {"body_weight": body_weight.as_tuple()}, {"MBW": bw ** 0.75})
    comps["activity"] = EquationResult("NEL_activity", act, "Mcal/d", _r("activity"), {"activity": activity.as_tuple()},
                                       notes=("declared input",))
    comps["lactation"] = EquationResult("NEL_milk", me, "Mcal/d", _r("Eq 3-14b"), {"milk_energy": milk_energy.as_tuple()})
    gest_factor = 0.882 / 0.14 * 0.66
    if t > 0:
        gu_part = calf * 1.825
        rate = 0.0243 - 0.0000245 * t
        gu_wt = gu_part * math.exp(-rate * (280.0 - t))
        gu_gain = rate * gu_wt
    else:
        gu_part = gu_wt = gu_gain = 0.0
    ut_part = calf * 0.2288
    ut_wt = (ut_part - 0.204) * math.exp(-0.2 * dim) + 0.204
    inv_gain = -0.2 * (ut_wt - 0.204)
    comps["gestation"] = EquationResult(
        "NEL_gestation", (gu_gain + inv_gain) * gest_factor, "Mcal/d",
        _r("Eq 3-15a", "Eq 3-16a", "Eq 3-17a", "Eq 3-15b", "Eq 3-16b", "Eq 3-17b", "Eq 3-18"),
        {"days_pregnant": days_pregnant.as_tuple(), "calf_birth_weight": calf_birth_weight.as_tuple(),
         "days_in_milk": days_in_milk.as_tuple()},
        {"GrUter_Wt_parturition": gu_part, "GrUter_Wt": gu_wt, "GrUter_WtGain": gu_gain, "Uter_Wt": ut_wt,
         "involution_gain": inv_gain, "factor_0.882/0.14x0.66": gest_factor},
        notes=("Eq 3-18 prints the rounded factor 4.16; the product of its printed factors is used",))
    ratio = bw / mat
    fat = (0.067 + 0.375 * ratio) * ebg
    prot = (0.201 - 0.081 * ratio) * ebg
    re_f = 9.4 * fat + 5.55 * prot
    comps["frame_gain"] = EquationResult("NEL_frame_gain", fg * re_f / 0.61, "Mcal/d", _r("Eq 3-20", "p33"),
                                         {"frame_gain": frame_gain.as_tuple(), "body_weight": body_weight.as_tuple(),
                                          "mature_body_weight": mature_body_weight.as_tuple(),
                                          "empty_body_gain_per_frame_gain": empty_body_gain_per_frame_gain.as_tuple()},
                                         {"Fat_ADG": fat, "Protein_ADG": prot, "RE_FADG": re_f, "NEL_FADG": re_f / 0.61})
    rsv = 6.3 * 0.89
    comps["reserves"] = EquationResult("NEL_reserves", rg * rsv, "Mcal/d", _r("Eq 3-19a", "Eq 3-19c"),
                                       {"reserves_gain": reserves_gain.as_tuple()}, {"Mcal_NEL_per_kg": rsv},
                                       notes=("Eq 3-19a prints the rounded 5.6; 6.3 x 0.89 is used",))
    total = sum(c.value for c in comps.values())
    refs = tuple(r for c in comps.values() for r in c.equations)
    return NELRequirement(EquationResult("NEL_requirement", total, "Mcal/d", refs, {},
                                         {k: c.value for k, c in comps.items()}), comps)


# ---------------------------------------------------------------------------------------------
# engine glue
# ---------------------------------------------------------------------------------------------

def energy_nutrient_spec() -> NutrientSpec:
    """The engine nutrient column of the per-feed fixed-DMI NEL contribution (Mcal/kg DM)."""
    return NutrientSpec(ENERGY_COLUMN_ID, "energy_density",
                        "per-feed NEL contribution at fixed DMI (NASEM 2021 ch.3 linearisation; not a feed NEL)")


def nel_constraint_spec(lin: NELLinearisation, requirement: NELRequirement, *, constraint_id: str,
                        provenance: Provenance, numerical_tolerance: float,
                        claim_scope: Optional[str] = None) -> ConstraintSpec:
    """``sum_i x_i e_i >= NEL_req - C0`` as an engine supply constraint (Mcal/d, scenario DM)."""
    bound = requirement.value - lin.constant_mcal_d
    return ConstraintSpec(
        constraint_id=constraint_id, name="NEL supply (fixed-DMI linearisation) >= NEL requirement",
        kind=ConstraintKind.SUPPLY, terms={ENERGY_COLUMN_ID: 1.0}, sense=Sense.GE, bound=float(bound), unit="Mcal/d",
        constraint_class=ConstraintClass.PROBABILISTIC_NUTRITION, numerical_tolerance=float(numerical_tolerance),
        provenance=provenance, basis="none", dm_source=DMSource.SCENARIO,
        standard_version="NASEM 2021 ch.3 energy (fixed-DMI linearisation)",
        claim_scope=claim_scope or ("modelled NEL supply at the fixed scenario DMI vs the NASEM ch.3 requirement; "
                                    "not a prediction of milk yield, intake or energy balance"),
        quantity_type="model_quantity",
        notes=f"bound = NEL_req {requirement.value:.6f} - C0 {lin.constant_mcal_d:.6f} Mcal/d; {LINEARISATION_ID}")


def append_energy_column(draws, lin: NELLinearisation):
    """Copy of a :class:`~ration_reliability.uncertainty.base.DrawSet` with the ``NEL_fixedDMI`` column.

    The column is computed from the draws' own composition (same scenario, same stream), so the
    energy row sees exactly the composition the other rows see.  The new ``model_fingerprint`` hashes
    the base model fingerprint and the linearisation fingerprint.
    """
    from ..uncertainty.base import DrawSet   # local import: nutrition does not depend on uncertainty otherwise

    if ENERGY_COLUMN_ID in draws.nutrient_ids:
        raise ValueError(f"draws already contain {ENERGY_COLUMN_ID}")
    if tuple(draws.ingredient_ids) != lin.ingredient_ids:
        raise ValueError("draws and linearisation have different ingredient order")
    e = lin.density(draws.theta, draws.nutrient_ids)
    theta = np.concatenate([draws.theta, e[..., None]], axis=-1)
    return DrawSet(theta, draws.d, draws.stream, draws.stream_id, draws.model_id + "+" + ENERGY_COLUMN_ID,
                   energy_world_fingerprint(draws.model_fingerprint, lin),
                   tuple(draws.ingredient_ids), tuple(draws.nutrient_ids) + (ENERGY_COLUMN_ID,), draws.is_synthetic)


def energy_world_fingerprint(base_fingerprint: str, lin: NELLinearisation) -> str:
    """World id of draws carrying the energy column: hash of the base world id and the linearisation.

    :func:`append_energy_column` and :class:`EnergyColumnModel` give identical ids (and identical draws
    on the same named stream), so either route yields one evaluation world."""
    return stable_hash("EnergyColumn/v1", base_fingerprint, lin.fingerprint())


def _uncertainty_base():
    from ..uncertainty.base import UncertaintyModel
    return UncertaintyModel


class EnergyColumnModel(_uncertainty_base()):
    """Wrap an uncertainty model and append the ``NEL_fixedDMI`` column to every draw.

    ``model_id`` = base id + ``"+NEL_fixedDMI"``; the fingerprint hashes the base fingerprint and the
    linearisation, so every consumer that checks the evaluation world sees one consistent world.
    """

    def __init__(self, base: Any, lin: NELLinearisation):
        if tuple(base.ingredient_ids) != lin.ingredient_ids:
            raise ValueError("base model and linearisation have different ingredient order")
        missing = [n for n in lin.nutrient_ids if n not in base.nutrient_ids]
        if missing:
            raise KeyError(f"base model lacks nutrient columns {missing} used by the energy linearisation")
        if ENERGY_COLUMN_ID in base.nutrient_ids:
            raise ValueError(f"base model already has {ENERGY_COLUMN_ID}")
        self.base = base
        self.lin = lin
        self.model_id = f"{base.model_id}+{ENERGY_COLUMN_ID}"
        self.ingredient_ids = tuple(base.ingredient_ids)
        self.nutrient_ids = tuple(base.nutrient_ids) + (ENERGY_COLUMN_ID,)
        self.is_synthetic = bool(base.is_synthetic)

    def sample(self, rng: np.random.Generator, n_draws: int) -> tuple[np.ndarray, np.ndarray]:
        theta, d = self.base.sample(rng, n_draws)
        e = self.lin.density(theta, self.base.nutrient_ids)
        return np.concatenate([np.asarray(theta, dtype=float), e[..., None]], axis=-1), d

    def params_for_fingerprint(self) -> dict:
        return {"base_fingerprint": self.base.fingerprint(), "energy_linearisation": self.lin.fingerprint()}

    def fingerprint(self) -> str:
        """Same id as :func:`append_energy_column` applied to the base model's draws."""
        return energy_world_fingerprint(self.base.fingerprint(), self.lin)

    def assert_same_world(self, *draw_sets) -> None:
        """Raise ``ValueError`` unless every draw set was produced by this (wrapped) world."""
        fp = self.fingerprint()
        bad = [ds.stream_id for ds in draw_sets if ds.model_fingerprint != fp]
        if bad:
            raise ValueError(f"draws {bad} do not come from the energy-augmented world {fp[:12]}")

    def draw_world(self, streams, *, n_opt=None, n_validation=None, n_test=None) -> dict:
        """Named purpose streams (``opt`` / ``validation`` / ``test``) from this one wrapped world."""
        out = {}
        for name, n in (("opt", n_opt), ("validation", n_validation), ("test", n_test)):
            if n is not None:
                out[name] = self.draw(streams, name, int(n))
        return out

    def nominal_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Base nominal state (e.g. ``FactoryModel.nominal_state``) with the energy column appended."""
        if not hasattr(self.base, "nominal_state"):
            raise AttributeError(f"base model {type(self.base).__name__} has no nominal_state()")
        theta, d = self.base.nominal_state()
        e = self.lin.density(np.asarray(theta, dtype=float), self.base.nutrient_ids)
        return np.concatenate([np.asarray(theta, dtype=float), e[..., None]], axis=-1), np.asarray(d, dtype=float)


# ---------------------------------------------------------------------------------------------
# review round 2, FIX_B: M3 box with the energy column; composition-closure diagnostic
# ---------------------------------------------------------------------------------------------

def energy_box_from_factory_model(model: Any, lin: Optional[NELLinearisation], k: float, *,
                                  set_id: Optional[str] = None, d_min: float = 1e-6):
    """M3 box for a factory world that carries the ``NEL_fixedDMI`` column (review round 2, FIX_B / B-K7-1).

    Composition and DM intervals are :meth:`BoxUncertaintySet.from_factory_model` (achieved moments of
    the sampled marginals +/- ``k`` SD, clipped to each cell's physical range).  The energy interval of
    feed ``i`` is the **exact image** of its composition box under the affine map
    ``e_i = intercept_i + sum_j slope_ij a_ij``::

        e_lo_i = intercept_i + sum_j min(slope_ij lo_ij, slope_ij hi_ij)
        e_hi_i = intercept_i + sum_j max(slope_ij lo_ij, slope_ij hi_ij)

    It is never clipped at 0 (the contribution of a mineral is negative), so the nominal energy
    density lies inside the box whenever the nominal composition does.  ``model`` is a
    :class:`~ration_reliability.uncertainty.factory.FactoryModel` (then ``lin`` is required) or an
    :class:`EnergyColumnModel` wrapping one (then ``lin`` may be ``None`` or must be the wrapper's
    linearisation).  Returns a :class:`~ration_reliability.optimization.robust.BoxUncertaintySet` whose
    nutrient axis is the base axis + ``NEL_fixedDMI`` (the order of :class:`EnergyColumnModel`).
    """
    from ..optimization.robust import BoxUncertaintySet   # local import: nutrition does not import optimization

    if isinstance(model, EnergyColumnModel):
        if lin is not None and lin.fingerprint() != model.lin.fingerprint():
            raise ValueError("energy_box_from_factory_model: lin differs from the wrapper's linearisation")
        lin, base_model = model.lin, model.base
    else:
        if lin is None:
            raise ValueError("energy_box_from_factory_model: lin is required for a bare factory model")
        base_model = model
    base = BoxUncertaintySet.from_factory_model(base_model, k, d_min=d_min)
    if tuple(base.ingredient_ids) != tuple(lin.ingredient_ids):
        raise ValueError("energy_box_from_factory_model: model and linearisation have different ingredient order")
    missing = [n for n in lin.nutrient_ids if n not in base.nutrient_ids]
    if missing:
        raise KeyError(f"energy_box_from_factory_model: model lacks columns {missing} used by the linearisation")
    idx = [list(base.nutrient_ids).index(n) for n in lin.nutrient_ids]
    lo, hi = base.theta_lo[:, idx], base.theta_hi[:, idx]
    if np.any(np.isnan(lo)) or np.any(np.isnan(hi)):
        raise ValueError("energy_box_from_factory_model: a composition column used by the energy map is missing")
    S = np.asarray(lin.slopes, dtype=float)
    icpt = np.asarray(lin.intercept, dtype=float)
    e_lo = icpt + np.sum(np.minimum(S * lo, S * hi), axis=1)
    e_hi = icpt + np.sum(np.maximum(S * lo, S * hi), axis=1)
    th_lo = np.concatenate([base.theta_lo, e_lo[:, None]], axis=1)
    th_hi = np.concatenate([base.theta_hi, e_hi[:, None]], axis=1)
    params = dict(base.construction_params)
    params.update({"energy_column": ENERGY_COLUMN_ID,
                   "energy_interval": "exact image of the composition box under the affine map "
                                      "e_i = intercept_i + sum_j slope_ij a_ij (never clipped)",
                   "energy_linearisation_fingerprint": lin.fingerprint(), "k": float(k)})
    return BoxUncertaintySet(set_id or f"{base.set_id}+{ENERGY_COLUMN_ID}", base.ingredient_ids,
                             tuple(base.nutrient_ids) + (ENERGY_COLUMN_ID,), th_lo, th_hi, base.d_lo, base.d_hi,
                             "factory_model_moment_pm_k_sd+induced_energy_interval", params, bool(base.is_synthetic))


#: engine columns whose drawn values are summed in the closure diagnostic (kg/kg DM).
#:
#: Review round 3 (F4) correction: these analytical fractions are **not** disjoint.  NDF is itself a
#: heterogeneous residue that contains part of the nitrogen-containing compounds (neutral-detergent
#: insoluble CP) and part of the ash; because CP and ash are also counted as fractions of their own, the
#: Eq 3-1 difference ``ROM = 100 - ash - NDF - starch - FA/FatFactor - CP`` subtracts them twice, which
#: underestimates ROM and overestimates NDF by the same amount (NASEM 2021, p.22).  NASEM states that this
#: double subtraction is *incorrect*, and nevertheless -- knowingly, not because of it -- uses NDF rather
#: than ash- and CP-free NDF in the energy supply equations, for five practical reasons (p.23): data on
#: ash- and CP-free NDF are limited; published in vivo NDF digestibilities did not subtract CP or ash, so
#: the model's coefficients stay comparable with them; analytical precision is likely lower for ash- and
#: CP-free NDF; true ROM digestibility and endogenous fecal ROM are estimated more precisely with NDF; and
#: the errors largely cancel for most diets.  In addition EE is not FA (EE carries non-FA material that
#: belongs to ROM; Eq 3-1 uses FA), and independently drawn marginals add sampling and assay error.  A sum
#: above 1, or a computed ``ROM < 0``, is therefore classified :data:`ANALYSIS_ANOMALY_CLASS`
#: (``analysis_overlap_or_measurement_anomaly``), with the sub-classes :data:`SUM_OVERLAP_SUBCLASS` (sum
#: above 1: definition overlap or sampling / assay error) and :data:`DERIVED_QUANTITY_SUBCLASS` (ROM < 0:
#: an anomaly of a derived quantity) -- it triggers a check but is not by itself a proof that the state is
#: physically impossible.  It is never forced closed, normalised to 1, clipped or dropped.  A violation of
#: the physical support of a declared variable (e.g. lignin > NDF, a mass fraction outside [0, 1]) is a
#: different class, :data:`SUPPORT_VIOLATION_CLASS` (see :func:`classify_composition_states`).
CLOSURE_SUM_COLUMNS = ("CP", "NDF", "starch", "EE", "ash")

#: Class of "CP + NDF + starch + EE + ash > 100 % DM" and "Eq 3-1 ROM < 0" (review round 3, F4).
ANALYSIS_ANOMALY_CLASS = "analysis_overlap_or_measurement_anomaly"
#: Class of a violation of the physical support of the declared variables (lignin > NDF; fraction < 0 or > 1).
SUPPORT_VIOLATION_CLASS = "support_violation"
#: Sub-class of :data:`ANALYSIS_ANOMALY_CLASS` for "CP + NDF + starch + EE + ash > 100 % DM" (red team D7/F4):
#: overlapping analytical definitions or sampling / assay error.
SUM_OVERLAP_SUBCLASS = "definition_overlap_or_measurement_error"
#: Sub-class of :data:`ANALYSIS_ANOMALY_CLASS` for "Eq 3-1 ROM < 0": an anomaly of a derived quantity (red team
#: D7/F4); the reference chain still computes with it (nothing clipped).
DERIVED_QUANTITY_SUBCLASS = "derived_quantity_anomaly"


def composition_closure_report(theta: np.ndarray, nutrient_ids: Sequence[str], ingredient_ids: Sequence[str], *,
                               lin: Optional[NELLinearisation] = None,
                               sum_columns: Sequence[str] = CLOSURE_SUM_COLUMNS,
                               tol: float = 1e-9) -> dict[str, dict[str, Any]]:
    """Per-ingredient share of drawn states whose composition does not close (review round 2, FIX_B).

    Under independent marginals the drawn CP + NDF + starch + EE + ash of one feed can exceed 100 % of
    DM, and the Eq 3-1 residual organic matter ``ROM = 100 - ash - NDF - starch - FA/FatFactor - CP``
    can then be negative.  Both are classified ``analysis_overlap_or_measurement_anomaly`` (the analytical
    fractions overlap by definition -- NDF contains part of the CP and of the ash, p.22-23 -- and drawn
    marginals carry sampling / assay error; see :data:`CLOSURE_SUM_COLUMNS`), not as physically impossible.
    Neither the linear energy row nor :func:`nonlinear_diet_nel` rejects such a state; this function only
    *counts* them -- it never clips, normalises or drops a draw (that would change the target moments of
    the evaluation world).  Support violations (lignin > NDF, fractions outside [0, 1]) are a separate
    class, counted by :func:`classify_composition_states` and reported here as ``share_support_violation``.

    ``theta``: ``[S, I, J]`` (or ``[I, J]``) in engine units (kg/kg DM).  With ``lin`` the ROM check uses
    the drawn columns of ``lin.nutrient_map`` and holds the other Eq 3-1 inputs (FA, and any composition
    field that is not an engine column) at the feed values of the linearisation.  Returns
    ``{ingredient_id: {n_draws, share_sum_gt_100pct, share_rom_lt_0 (None without lin), max_sum_pct,
    min_rom_pct, classes, share_support_violation, share_lignin_gt_ndf (None without lin)}}``; NaN cells
    are left out of the counts.  ``classes`` names the class of each share (review round 3, F4).
    """
    th = np.asarray(theta, dtype=float)
    if th.ndim == 2:
        th = th[None]
    nut = list(nutrient_ids)
    ids = list(ingredient_ids)
    if th.shape[1:] != (len(ids), len(nut)):
        raise ValueError("composition_closure_report: theta shape does not match the labels")
    cols = [c for c in sum_columns if c in nut]
    if not cols:
        raise ValueError("composition_closure_report: none of the sum columns is present")
    ssum = np.sum(th[:, :, [nut.index(c) for c in cols]], axis=2)            # [S, I], NaN if any missing
    rom = None
    if lin is not None:
        if tuple(lin.ingredient_ids) != tuple(ids):
            raise ValueError("composition_closure_report: linearisation has another ingredient order")
        field_of = {v: k for k, v in dict(lin.nutrient_map).items()}          # composition field -> engine id
        rom = np.empty(ssum.shape)
        for i, f in enumerate(lin.feeds):
            comp = {fld: (th[:, i, nut.index(field_of[fld])] * 100.0 if fld in field_of and field_of[fld] in nut
                          else np.full(th.shape[0], float(getattr(f, fld))))
                    for fld in ("ndf", "starch", "fa", "cp", "ash")}
            rom[:, i] = (100.0 - comp["ash"] - comp["ndf"] - comp["starch"] - comp["fa"] / float(f.fat_factor)
                         - comp["cp"])
    cls = classify_composition_states(th, nut, ids, lin=lin, sum_columns=sum_columns, tol=tol)
    out: dict[str, dict[str, Any]] = {}
    for i, iid in enumerate(ids):
        v = ssum[:, i]
        ok = np.isfinite(v)
        rec: dict[str, Any] = {"n_draws": int(ok.sum()),
                               "share_sum_gt_100pct": float(np.mean(v[ok] > 1.0 + tol)) if ok.any() else None,
                               "max_sum_pct": float(np.max(v[ok]) * 100.0) if ok.any() else None,
                               "sum_columns": list(cols), "share_rom_lt_0": None, "min_rom_pct": None}
        if rom is not None:
            r = rom[:, i]
            okr = np.isfinite(r)
            rec["share_rom_lt_0"] = float(np.mean(r[okr] < -100.0 * tol)) if okr.any() else None
            rec["min_rom_pct"] = float(np.min(r[okr])) if okr.any() else None
        sv = cls.support_violation[:, i]
        rec["share_support_violation"] = float(np.mean(sv))
        rec["share_lignin_gt_ndf"] = float(np.mean(cls.lignin_gt_ndf[:, i])) if lin is not None else None
        rec["classes"] = {"share_sum_gt_100pct": ANALYSIS_ANOMALY_CLASS, "share_rom_lt_0": ANALYSIS_ANOMALY_CLASS,
                          "share_support_violation": SUPPORT_VIOLATION_CLASS,
                          "share_lignin_gt_ndf": SUPPORT_VIOLATION_CLASS}
        rec["subclasses"] = {"share_sum_gt_100pct": SUM_OVERLAP_SUBCLASS, "share_rom_lt_0": DERIVED_QUANTITY_SUBCLASS}
        out[str(iid)] = rec
    return out


@dataclass(frozen=True)
class CompositionStateClasses:
    """Per-state, per-ingredient composition classes (review round 3, F4); arrays ``[S, I]``, read-only.

    * ``sum_gt_100pct`` / ``rom_lt_0`` -> :data:`ANALYSIS_ANOMALY_CLASS` (``analysis_anomaly`` is their OR),
      sub-classes :data:`SUM_OVERLAP_SUBCLASS` (analytical definition overlap -- NDF contains part of the CP
      and of the ash; EE is not FA -- or sampling / assay error) and :data:`DERIVED_QUANTITY_SUBCLASS` (an
      anomaly of the derived Eq 3-1 ROM).  Not a proof of physical impossibility.
    * ``lignin_gt_ndf`` / ``fraction_outside_unit_interval`` -> :data:`SUPPORT_VIOLATION_CLASS`
      (``support_violation`` is their OR): the state lies outside the physical support of the declared
      variables (lignin is a part of the NDF residue; a mass fraction lies in [0, 1]).  The Chapter 3 chain
      is undefined there (Eq 3-3a).
    * ``defined``: the cells needed for the checks are finite.  A NaN cell never counts as a class member.

    Nothing is clipped, normalised or dropped; the classes only label states.
    """

    ingredient_ids: tuple[str, ...]
    sum_gt_100pct: np.ndarray
    rom_lt_0: np.ndarray
    lignin_gt_ndf: np.ndarray
    fraction_outside_unit_interval: np.ndarray
    defined: np.ndarray
    rom_available: bool

    def __post_init__(self) -> None:
        for f in ("sum_gt_100pct", "rom_lt_0", "lignin_gt_ndf", "fraction_outside_unit_interval", "defined"):
            a = np.ascontiguousarray(np.asarray(getattr(self, f), dtype=bool))
            a.setflags(write=False)
            object.__setattr__(self, f, a)

    @property
    def analysis_anomaly(self) -> np.ndarray:
        return self.sum_gt_100pct | self.rom_lt_0

    @property
    def support_violation(self) -> np.ndarray:
        return self.lignin_gt_ndf | self.fraction_outside_unit_interval

    def summary(self) -> dict[str, Any]:
        """Per-ingredient counts (denominator = all states; nothing removed)."""
        S = int(self.defined.shape[0])
        return {"n_states": S, "classes": {"analysis_anomaly": ANALYSIS_ANOMALY_CLASS,
                                           "support_violation": SUPPORT_VIOLATION_CLASS},
                "subclasses": {"sum_gt_100pct": SUM_OVERLAP_SUBCLASS, "rom_lt_0": DERIVED_QUANTITY_SUBCLASS},
                "per_ingredient": {iid: {"n_sum_gt_100pct": int(self.sum_gt_100pct[:, i].sum()),
                                         "n_rom_lt_0": int(self.rom_lt_0[:, i].sum()) if self.rom_available else None,
                                         "n_analysis_anomaly": int(self.analysis_anomaly[:, i].sum()),
                                         "n_lignin_gt_ndf": int(self.lignin_gt_ndf[:, i].sum()),
                                         "n_fraction_outside_unit_interval":
                                             int(self.fraction_outside_unit_interval[:, i].sum()),
                                         "n_support_violation": int(self.support_violation[:, i].sum()),
                                         "n_undefined": int((~self.defined[:, i]).sum())}
                                   for i, iid in enumerate(self.ingredient_ids)}}


def classify_composition_states(theta: np.ndarray, nutrient_ids: Sequence[str], ingredient_ids: Sequence[str], *,
                                lin: Optional[NELLinearisation] = None,
                                sum_columns: Sequence[str] = CLOSURE_SUM_COLUMNS,
                                fraction_columns: Optional[Sequence[str]] = None,
                                tol: float = 1e-9) -> CompositionStateClasses:
    """Label each drawn composition state (review round 3, F4); never modifies ``theta``.

    ``theta``: ``[S, I, J]`` (or ``[I, J]``) in engine units (kg/kg DM).  ``sum_columns`` as in
    :func:`composition_closure_report` (the columns present are summed).  ``fraction_columns``: the mass
    fraction columns whose physical support is [0, 1] (default: the ``sum_columns`` present plus the
    columns of ``lin.nutrient_map``).  With ``lin`` the Eq 3-1 ROM and the lignin check use the drawn columns
    of ``lin.nutrient_map`` and hold the other fields (FA, lignin) at the feed values of the linearisation;
    without ``lin`` ``rom_lt_0`` and ``lignin_gt_ndf`` are all False and ``rom_available`` is False.
    """
    th = np.asarray(theta, dtype=float)
    if th.ndim == 2:
        th = th[None]
    nut = list(nutrient_ids)
    ids = tuple(str(i) for i in ingredient_ids)
    if th.ndim != 3 or th.shape[1:] != (len(ids), len(nut)):
        raise ValueError("classify_composition_states: theta shape does not match the labels")
    S, I = th.shape[0], th.shape[1]
    cols = [c for c in sum_columns if c in nut]
    if not cols:
        raise ValueError("classify_composition_states: none of the sum columns is present")
    ssum = np.sum(th[:, :, [nut.index(c) for c in cols]], axis=2)
    defined = np.isfinite(ssum)
    with np.errstate(invalid="ignore"):
        sum_gt = defined & (ssum > 1.0 + tol)
    rom_lt = np.zeros((S, I), dtype=bool)
    lig_gt = np.zeros((S, I), dtype=bool)
    fcols = set(cols)
    if lin is not None:
        if tuple(lin.ingredient_ids) != ids:
            raise ValueError("classify_composition_states: linearisation has another ingredient order")
        field_of = {v: k for k, v in dict(lin.nutrient_map).items()}          # composition field -> engine id
        fcols |= {n for n in dict(lin.nutrient_map) if n in nut}
        for i, f in enumerate(lin.feeds):
            comp = {fld: (th[:, i, nut.index(field_of[fld])] * 100.0 if fld in field_of and field_of[fld] in nut
                          else np.full(S, float(getattr(f, fld))))
                    for fld in ("ndf", "starch", "fa", "cp", "ash")}
            rom = (100.0 - comp["ash"] - comp["ndf"] - comp["starch"] - comp["fa"] / float(f.fat_factor) - comp["cp"])
            ok = np.isfinite(rom)
            with np.errstate(invalid="ignore"):
                rom_lt[:, i] = ok & (rom < -100.0 * tol)
                ndf = comp["ndf"]
                lig_gt[:, i] = np.isfinite(ndf) & (float(f.lignin) > ndf + 100.0 * tol)
            defined[:, i] &= ok
    if fraction_columns is not None:
        fcols = {c for c in fraction_columns if c in nut}
    out_rng = np.zeros((S, I), dtype=bool)
    for c in sorted(fcols):
        v = th[:, :, nut.index(c)]
        with np.errstate(invalid="ignore"):
            out_rng |= np.isfinite(v) & ((v < -tol) | (v > 1.0 + tol))
    return CompositionStateClasses(ids, sum_gt, rom_lt, lig_gt, out_rng, defined, lin is not None)
