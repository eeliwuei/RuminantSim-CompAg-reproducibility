"""SD scaling (sensitivity dimension a) and mean-shift scenarios (dimension g).

``configs/uncertainty.yaml``:

* ``sensitivity_dimensions.a_sd_scale_factor`` -- the P0 SD of every stochastic cell is multiplied
  by a factor ``s`` (all cells, or per ingredient group).  The grid in the configuration is a
  development starting point (``research_scenario_assumption``, decision D-17), not frozen.
* ``sensitivity_dimensions.g_mean_shift`` -- the *true* distribution mean of the target nutrients
  is moved by ``k`` SDs while the decision-time nominal value (M0 table value, ``d_hat``) stays the
  P0 mean.  ``k_grid`` is ``null`` / ``pending_user_decision``; only the development proposal is
  available and it is returned with that label.

Rules implemented here:

* ``s`` scales the dispersion only; means are unchanged.  ``s`` must be finite and ``> 0``.
* A mean shift is applied to the listed target nutrients only.  A shifted mean outside the open
  physical range is **not clipped**: the cell is reported as infeasible for that ``k`` and its
  mean is left NaN in :attr:`ScenarioTargets.theta_mean_true` only when ``on_infeasible="nan"``;
  the default raises.
* The SD used for the shift is declared explicitly (``shift_sd_basis``): ``"P0"`` (the unscaled
  table SD) or ``"scaled"`` (``s`` times the table SD).  The configuration does not say which
  (open question for D-17); the choice is recorded.
* Scaling and shifting never touch the decision-time nominal arrays.

No parameter values are embedded in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "SHIFT_SD_BASES",
    "scale_sd",
    "scale_sd_by_group",
    "mean_shift",
    "MeanShiftResult",
    "ScenarioTargets",
    "build_scenario_targets",
    "grid_from_config",
]

SHIFT_SD_BASES = ("P0", "scaled")


def _check_factor(s: float) -> float:
    s = float(s)
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"SD scale factor must be finite and > 0, got {s}")
    return s


def scale_sd(sd: np.ndarray, s: float) -> np.ndarray:
    """``s * sd`` (NaN preserved, a missing SD is never turned into a value)."""
    return np.asarray(sd, dtype=float) * _check_factor(s)


def scale_sd_by_group(sd: np.ndarray, groups: Sequence[str], factors: Mapping[str, float]) -> np.ndarray:
    """Scale the rows (ingredients) of ``sd`` by group-specific factors.

    ``groups[i]`` is the group label of ingredient ``i`` (axis 0 of ``sd``); every label must have
    a factor (a missing factor raises instead of defaulting to 1).
    """
    sd = np.asarray(sd, dtype=float)
    if len(groups) != sd.shape[0]:
        raise ValueError("one group label per ingredient (axis 0) is required")
    missing = sorted({g for g in groups if g not in factors})
    if missing:
        raise ValueError(f"no SD scale factor for group(s) {missing}")
    f = np.array([_check_factor(factors[g]) for g in groups])
    return sd * f.reshape((-1,) + (1,) * (sd.ndim - 1))


@dataclass(frozen=True)
class MeanShiftResult:
    """Shifted means and the cells where the shift is not physically possible."""

    mean: np.ndarray
    shifted: np.ndarray          # bool [I, J]
    infeasible: np.ndarray       # bool [I, J]
    k: float
    targets: tuple


def mean_shift(mean: np.ndarray, sd: np.ndarray, k: float, nutrient_ids: Sequence[str],
               targets: Sequence[str], lower: float = 0.0, upper: float = 1.0,
               on_infeasible: str = "raise") -> MeanShiftResult:
    """``mean + k * sd`` for the columns ``targets`` of ``[I, J]`` arrays.

    Cells with a missing mean or SD stay missing.  A shifted mean outside ``(lower, upper)`` is
    infeasible: ``on_infeasible="raise"`` (default) raises ``ValueError``; ``"nan"`` sets it NaN
    and flags it (never clipped).
    """
    if on_infeasible not in ("raise", "nan"):
        raise ValueError("on_infeasible must be 'raise' or 'nan'")
    mean = np.array(mean, dtype=float)
    sd = np.asarray(sd, dtype=float)
    k = float(k)
    if not np.isfinite(k):
        raise ValueError("k must be finite")
    nut = list(nutrient_ids)
    unknown = [t for t in targets if t not in nut]
    if unknown:
        raise ValueError(f"shift targets not among the nutrient columns: {unknown}")
    cols = np.zeros(mean.shape, dtype=bool)
    for t in targets:
        cols[:, nut.index(t)] = True
    live = cols & np.isfinite(mean) & np.isfinite(sd)
    new = np.where(live, mean + k * np.where(np.isfinite(sd), sd, 0.0), mean)
    bad = live & ~((new > lower) & (new < upper))
    if np.any(bad):
        if on_infeasible == "raise":
            idx = [tuple(int(v) for v in x) for x in np.argwhere(bad)]
            raise ValueError(f"mean shift k={k} moves {len(idx)} cell(s) outside ({lower}, {upper}): {idx}")
        new = np.where(bad, np.nan, new)
    return MeanShiftResult(new, live, bad, k, tuple(targets))


@dataclass(frozen=True)
class ScenarioTargets:
    """Target moments of the *true* distribution plus the unchanged decision-time nominal values."""

    theta_mean_true: np.ndarray
    theta_sd_true: np.ndarray
    d_mean_true: np.ndarray
    d_sd_true: np.ndarray
    theta_nominal: np.ndarray
    d_nominal: np.ndarray
    s: float
    k: float
    shift_targets: tuple
    shift_sd_basis: str
    sd_scale_applies_to_dm: bool
    infeasible_cells: tuple = field(default_factory=tuple)
    log: tuple = field(default_factory=tuple)


def build_scenario_targets(theta_mean: np.ndarray, theta_sd: np.ndarray, d_mean: np.ndarray, d_sd: np.ndarray,
                           nutrient_ids: Sequence[str], *, s: float, k: float, shift_targets: Sequence[str],
                           shift_sd_basis: str, sd_scale_applies_to_dm: bool, lower: float = 0.0,
                           upper: float = 1.0, on_infeasible: str = "raise") -> ScenarioTargets:
    """Apply SD scaling ``s`` and mean shift ``k`` (both explicit; no defaults for the choices).

    The nominal arrays returned are copies of the inputs (M0 and ``d_hat`` keep the P0 means).
    """
    if shift_sd_basis not in SHIFT_SD_BASES:
        raise ValueError(f"shift_sd_basis must be one of {SHIFT_SD_BASES}")
    theta_mean = np.asarray(theta_mean, dtype=float)
    theta_sd = np.asarray(theta_sd, dtype=float)
    d_mean = np.asarray(d_mean, dtype=float)
    d_sd = np.asarray(d_sd, dtype=float)
    s = _check_factor(s)
    sd_true = scale_sd(theta_sd, s)
    dsd_true = scale_sd(d_sd, s) if sd_scale_applies_to_dm else d_sd.copy()
    shift_sd = theta_sd if shift_sd_basis == "P0" else sd_true
    log = [f"s={s!r} applied to composition SD" + (" and DM SD" if sd_scale_applies_to_dm else " (DM SD unchanged)"),
           f"k={float(k)!r} on {list(shift_targets)} using {shift_sd_basis} SD; nominal (decision) means unchanged"]
    if float(k) != 0.0 and len(shift_targets):
        res = mean_shift(theta_mean, shift_sd, k, nutrient_ids, shift_targets, lower, upper, on_infeasible)
        mean_true, bad = res.mean, res.infeasible
    else:
        mean_true, bad = theta_mean.copy(), np.zeros(theta_mean.shape, dtype=bool)
    infeasible = tuple((int(i), str(nutrient_ids[j])) for i, j in np.argwhere(bad))
    if infeasible:
        log.append(f"{len(infeasible)} cell(s) infeasible for this k (set NaN, not clipped)")
    return ScenarioTargets(mean_true, sd_true, d_mean.copy(), dsd_true, theta_mean.copy(), d_mean.copy(), s,
                           float(k), tuple(shift_targets), shift_sd_basis, bool(sd_scale_applies_to_dm),
                           infeasible, tuple(log))


def grid_from_config(section: Mapping, *, grid_key: str, status_key: str, proposal_key: Optional[str] = None,
                     proposal_status_key: Optional[str] = None) -> tuple[Optional[list[float]], str, str]:
    """Read a candidate grid from a configuration section.

    Returns ``(grid, status, origin)``.  If the chosen grid is ``null``, the development proposal is
    returned (when present) with its own status and ``origin = "dev_start_proposal"``; otherwise
    ``(None, status, "null")``.  Nothing is frozen by reading.
    """
    grid = section.get(grid_key)
    status = str(section.get(status_key))
    if grid is not None:
        return [float(x) for x in grid], status, grid_key
    if proposal_key and section.get(proposal_key) is not None:
        pst = str(section.get(proposal_status_key)) if proposal_status_key else "research_scenario_assumption"
        return [float(x) for x in section[proposal_key]], pst, "dev_start_proposal"
    return None, status, "null"
