"""Unit registry, unit checks and basis conversions.

Design rules
------------
* Every numeric input carries a unit symbol from this registry.  An unknown symbol raises
  :class:`~ration_reliability.errors.UnitError` (contract T9: "单位未定义" must be rejected).
* The *basis* (``"DM"`` vs ``"as_fed"``) is a separate attribute from the unit.  ``%`` on a DM
  basis and ``%`` on an as-fed basis have the same unit but different meaning; converting between
  bases always needs an explicit DM fraction (never guessed).
* Internal canonical units (used by nutrition / evaluation / optimisation):

  ======================  ==============  ==============================================
  dimension               canonical       used for
  ======================  ==============  ==============================================
  ``mass_fraction``       ``fraction``    composition a_ij (kg/kg DM), DM fraction d_i
  ``energy_density``      ``Mcal/kg``     energy composition (per kg DM)
  ``mass``                ``kg``          batch sizes, inventories
  ``energy``              ``Mcal``        energy amounts
  ``mass_rate``           ``kg/d``        q (kg as-fed/head/d), D and N_j (kg/head/d)
  ``energy_rate``         ``Mcal/d``      energy supply per head per day
  ``price_per_mass``      ``<CUR>/kg``    p_i (currency per kg as-fed)
  ``dimensionless``       ``1``           group weights, coefficients
  ======================  ==============  ==============================================

Exact constants (sources)
-------------------------
* 1 lb (avoirdupois) = 0.45359237 kg exactly; 1 cal_th = 4.184 J exactly, hence 1 Mcal = 4.184 MJ.
  Source: NIST Special Publication 811 (2008 ed.), Appendix B.8,
  https://www.nist.gov/pml/special-publication-811 .  The same two conversions are listed as the
  project's required checks in ``audit/phase1_survey_20260924/DATA_FEASIBILITY_AUDIT.md`` §5.3
  (Dairy One NEL in Mcal/lb -> Mcal/kg; NY/T NEL in MJ/kg).
* %, g/kg, mg/kg and t/kg relations are SI prefix definitions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union

import numpy as np

from ..errors import UnitError

__all__ = [
    "LB_IN_KG",
    "MJ_PER_MCAL",
    "BASES",
    "UnitDef",
    "UNITS",
    "get_unit",
    "is_defined",
    "check_unit",
    "check_basis",
    "convert",
    "to_canonical",
    "from_canonical",
    "percent_to_g_per_kg",
    "g_per_kg_to_percent",
    "percent_to_fraction",
    "fraction_to_percent",
    "dm_to_as_fed_concentration",
    "as_fed_to_dm_concentration",
    "as_fed_mass_to_dm",
    "dm_mass_to_as_fed",
    "price_per_tonne_to_per_kg",
    "price_dm_to_as_fed",
    "price_as_fed_to_dm",
    "mcal_to_mj",
    "mj_to_mcal",
    "mcal_per_lb_to_mcal_per_kg",
    "mcal_per_kg_to_mcal_per_lb",
]

ArrayLike = Union[float, int, np.ndarray]

#: 1 lb (avoirdupois) in kg, exact (NIST SP 811, App. B.8).
LB_IN_KG: float = 0.45359237
#: MJ per Mcal via the thermochemical calorie 4.184 J, exact (NIST SP 811, App. B.8).
MJ_PER_MCAL: float = 4.184

#: Allowed basis labels.
BASES = frozenset({"DM", "as_fed", "none"})

_CURRENCY_RE = re.compile(r"^(?P<cur>[A-Z]{3})/(?P<mass>kg|g|t|lb)$")


@dataclass(frozen=True)
class UnitDef:
    """One registered unit.

    ``value_in_canonical = value * factor``.
    """

    symbol: str
    dimension: str
    factor: float
    canonical: str
    note: str = ""


def _mk(symbol: str, dimension: str, factor: float, canonical: str, note: str = "") -> UnitDef:
    return UnitDef(symbol, dimension, float(factor), canonical, note)


#: Registry of fixed (non-currency) units.
UNITS: dict[str, UnitDef] = {
    u.symbol: u
    for u in [
        # composition / DM fraction
        _mk("fraction", "mass_fraction", 1.0, "fraction", "kg/kg"),
        _mk("kg/kg", "mass_fraction", 1.0, "fraction"),
        _mk("%", "mass_fraction", 0.01, "fraction", "percent (basis carried separately)"),
        _mk("g/kg", "mass_fraction", 1e-3, "fraction"),
        _mk("mg/kg", "mass_fraction", 1e-6, "fraction"),
        # energy density
        _mk("Mcal/kg", "energy_density", 1.0, "Mcal/kg"),
        _mk("MJ/kg", "energy_density", 1.0 / MJ_PER_MCAL, "Mcal/kg"),
        _mk("kcal/kg", "energy_density", 1e-3, "Mcal/kg"),
        _mk("Mcal/lb", "energy_density", 1.0 / LB_IN_KG, "Mcal/kg", "x 2.20462262 -> Mcal/kg"),
        # mass
        _mk("kg", "mass", 1.0, "kg"),
        _mk("g", "mass", 1e-3, "kg"),
        _mk("t", "mass", 1e3, "kg", "metric tonne"),
        _mk("lb", "mass", LB_IN_KG, "kg"),
        # energy
        _mk("Mcal", "energy", 1.0, "Mcal"),
        _mk("MJ", "energy", 1.0 / MJ_PER_MCAL, "Mcal"),
        _mk("kcal", "energy", 1e-3, "Mcal"),
        # per head per day rates
        _mk("kg/d", "mass_rate", 1.0, "kg/d", "per head per day"),
        _mk("g/d", "mass_rate", 1e-3, "kg/d", "per head per day"),
        _mk("Mcal/d", "energy_rate", 1.0, "Mcal/d", "per head per day"),
        _mk("MJ/d", "energy_rate", 1.0 / MJ_PER_MCAL, "Mcal/d", "per head per day"),
        # dimensionless
        _mk("1", "dimensionless", 1.0, "1"),
    ]
}


def get_unit(symbol: str) -> UnitDef:
    """Return the :class:`UnitDef` for ``symbol``.

    Price units are parsed as ``<CUR>/<mass>`` where ``<CUR>`` is a 3-letter upper-case
    currency code (ISO 4217 style; ``XXX`` = "no currency", used for synthetic data) and
    ``<mass>`` one of ``kg, g, t, lb``.

    Raises
    ------
    UnitError
        If the symbol is not registered.
    """
    if not isinstance(symbol, str):
        raise UnitError(f"unit must be a string, got {symbol!r}")
    if symbol in UNITS:
        return UNITS[symbol]
    m = _CURRENCY_RE.match(symbol)
    if m:
        cur, mass = m.group("cur"), m.group("mass")
        # value per `mass`  ->  value per kg : divide by kg-per-`mass`
        return UnitDef(symbol, "price_per_mass", 1.0 / UNITS[mass].factor, f"{cur}/kg", cur)
    raise UnitError(f"undefined unit {symbol!r}; register it in normalization.units before use")


def is_defined(symbol: str) -> bool:
    """True if ``symbol`` is a registered unit (including ``<CUR>/<mass>`` price units)."""
    try:
        get_unit(symbol)
        return True
    except UnitError:
        return False


def check_unit(symbol: str, dimension: str | None = None) -> UnitDef:
    """Validate a unit symbol and (optionally) its dimension."""
    u = get_unit(symbol)
    if dimension is not None and u.dimension != dimension:
        raise UnitError(f"unit {symbol!r} has dimension {u.dimension!r}, expected {dimension!r}")
    return u


def check_basis(basis: str) -> str:
    """Validate a basis label (``DM``, ``as_fed`` or ``none``)."""
    if basis not in BASES:
        raise UnitError(f"undefined basis {basis!r}; allowed: {sorted(BASES)}")
    return basis


def _currency(u: UnitDef) -> str | None:
    return u.note if u.dimension == "price_per_mass" else None


def convert(value: ArrayLike, from_unit: str, to_unit: str) -> ArrayLike:
    """Convert ``value`` between two units of the same dimension (same currency for prices).

    Basis is *not* changed by this function; use the explicit DM/as-fed helpers for that.
    """
    uf, ut = get_unit(from_unit), get_unit(to_unit)
    if uf.dimension != ut.dimension:
        raise UnitError(f"cannot convert {from_unit!r} ({uf.dimension}) to {to_unit!r} ({ut.dimension})")
    if uf.dimension == "price_per_mass" and _currency(uf) != _currency(ut):
        raise UnitError(f"currency conversion {from_unit!r} -> {to_unit!r} is not supported (needs a dated FX source)")
    factor = uf.factor / ut.factor
    if isinstance(value, np.ndarray):
        return value * factor
    return float(value) * factor


def to_canonical(value: ArrayLike, unit: str) -> tuple[ArrayLike, str]:
    """Return ``(value_in_canonical_unit, canonical_symbol)``."""
    u = get_unit(unit)
    return convert(value, unit, u.canonical), u.canonical


def from_canonical(value: ArrayLike, unit: str) -> ArrayLike:
    """Convert a value expressed in the canonical unit of ``unit``'s dimension into ``unit``."""
    u = get_unit(unit)
    return convert(value, u.canonical, unit)


# --- explicit helpers ---------------------------------------------------------------------

def percent_to_g_per_kg(x: ArrayLike) -> ArrayLike:
    """% (same basis) -> g/kg (same basis)."""
    return convert(x, "%", "g/kg")


def g_per_kg_to_percent(x: ArrayLike) -> ArrayLike:
    """g/kg -> % (same basis)."""
    return convert(x, "g/kg", "%")


def percent_to_fraction(x: ArrayLike) -> ArrayLike:
    """% -> kg/kg (same basis)."""
    return convert(x, "%", "fraction")


def fraction_to_percent(x: ArrayLike) -> ArrayLike:
    """kg/kg -> % (same basis)."""
    return convert(x, "fraction", "%")


def _check_dm(dm_fraction: ArrayLike) -> np.ndarray:
    dm = np.asarray(dm_fraction, dtype=float)
    if not np.all(np.isfinite(dm)) or np.any(dm <= 0.0) or np.any(dm > 1.0):
        raise UnitError("DM fraction must be finite and in (0, 1] (kg DM / kg as-fed)")
    return dm


def _ret(x: np.ndarray, like: ArrayLike) -> ArrayLike:
    return x if isinstance(like, np.ndarray) or np.ndim(x) > 0 else float(x)


def dm_to_as_fed_concentration(c_dm: ArrayLike, dm_fraction: ArrayLike) -> ArrayLike:
    """Concentration on DM basis -> as-fed basis: ``c_af = c_dm * DM``.

    ``dm_fraction`` must be the DM of the same material the concentration refers to.
    """
    dm = _check_dm(dm_fraction)
    return _ret(np.asarray(c_dm, dtype=float) * dm, c_dm)


def as_fed_to_dm_concentration(c_af: ArrayLike, dm_fraction: ArrayLike) -> ArrayLike:
    """Concentration on as-fed basis -> DM basis: ``c_dm = c_af / DM``."""
    dm = _check_dm(dm_fraction)
    return _ret(np.asarray(c_af, dtype=float) / dm, c_af)


def as_fed_mass_to_dm(q_as_fed: ArrayLike, dm_fraction: ArrayLike) -> ArrayLike:
    """kg as-fed -> kg DM: ``x = q * d``."""
    dm = _check_dm(dm_fraction)
    return _ret(np.asarray(q_as_fed, dtype=float) * dm, q_as_fed)


def dm_mass_to_as_fed(x_dm: ArrayLike, dm_fraction: ArrayLike) -> ArrayLike:
    """kg DM -> kg as-fed: ``q = x / d``  (T2.1: d must be the *decision-time* estimate d_hat)."""
    dm = _check_dm(dm_fraction)
    return _ret(np.asarray(x_dm, dtype=float) / dm, x_dm)


def price_per_tonne_to_per_kg(p: ArrayLike) -> ArrayLike:
    """Currency per tonne -> currency per kg (same basis): divide by 1000 (T2.2)."""
    return _ret(np.asarray(p, dtype=float) / 1000.0, p)


def price_dm_to_as_fed(p_dm: ArrayLike, dm_fraction: ArrayLike) -> ArrayLike:
    """Price per kg DM -> price per kg as-fed: ``p_af = p_dm * DM``.

    Only legitimate when the quote's DM semantics are explicit (T2.2).  The DM used must be
    fixed at quotation/decision time; it must never be re-drawn per test scenario.
    """
    dm = _check_dm(dm_fraction)
    return _ret(np.asarray(p_dm, dtype=float) * dm, p_dm)


def price_as_fed_to_dm(p_af: ArrayLike, dm_fraction: ArrayLike) -> ArrayLike:
    """Price per kg as-fed -> price per kg DM: ``p_dm = p_af / DM``."""
    dm = _check_dm(dm_fraction)
    return _ret(np.asarray(p_af, dtype=float) / dm, p_af)


def mcal_to_mj(x: ArrayLike) -> ArrayLike:
    """Mcal -> MJ (x 4.184)."""
    return convert(x, "Mcal", "MJ")


def mj_to_mcal(x: ArrayLike) -> ArrayLike:
    """MJ -> Mcal (/ 4.184)."""
    return convert(x, "MJ", "Mcal")


def mcal_per_lb_to_mcal_per_kg(x: ArrayLike) -> ArrayLike:
    """Mcal/lb -> Mcal/kg (x 1/0.45359237 = x 2.20462262...)."""
    return convert(x, "Mcal/lb", "Mcal/kg")


def mcal_per_kg_to_mcal_per_lb(x: ArrayLike) -> ArrayLike:
    """Mcal/kg -> Mcal/lb."""
    return convert(x, "Mcal/kg", "Mcal/lb")
