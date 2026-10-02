"""Reference uncertainty models (mechanisms only; no parameters are provided here).

* :class:`PointMassModel`       zero uncertainty (all draws equal the given state).
* :class:`IndependentNormalModel` independent (optionally truncated) normal marginals.
* :class:`ScenarioSetModel`     resampling of a fixed finite set of joint states.

These classes are sampling *mechanisms*.  Using one of them in research requires that its
parameters come from a registered source and that the family/independence choice is declared as
a research assumption (contract 7.4: independence is a baseline, not a fact).  They are used by
the engine tests with synthetic parameters only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import numpy as np
from scipy import stats

from .base import UncertaintyModel

__all__ = ["PointMassModel", "IndependentNormalModel", "ScenarioSetModel"]

Bound = Union[float, np.ndarray]


def _ro(a) -> np.ndarray:
    arr = np.array(a, dtype=float)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True, eq=False)
class PointMassModel(UncertaintyModel):
    """Zero uncertainty: every draw equals ``(theta [I, J], d [I])``."""

    model_id: str
    ingredient_ids: tuple[str, ...]
    nutrient_ids: tuple[str, ...]
    theta: np.ndarray
    d: np.ndarray
    is_synthetic: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "theta", _ro(self.theta))
        object.__setattr__(self, "d", _ro(self.d))
        object.__setattr__(self, "ingredient_ids", tuple(self.ingredient_ids))
        object.__setattr__(self, "nutrient_ids", tuple(self.nutrient_ids))
        if self.theta.shape != (len(self.ingredient_ids), len(self.nutrient_ids)) or \
                self.d.shape != (len(self.ingredient_ids),):
            raise ValueError("PointMassModel: shapes do not match labels")

    def sample(self, rng: np.random.Generator, n_draws: int) -> tuple[np.ndarray, np.ndarray]:
        n = int(n_draws)
        return (np.broadcast_to(self.theta, (n,) + self.theta.shape).copy(),
                np.broadcast_to(self.d, (n,) + self.d.shape).copy())

    def params_for_fingerprint(self) -> dict:
        return {"theta": self.theta, "d": self.d}


@dataclass(frozen=True, eq=False)
class IndependentNormalModel(UncertaintyModel):
    """Independent normal marginals for every ``a_ij`` and ``d_i``.

    With ``truncate=True`` each marginal is a normal truncated to ``[lower, upper]`` (by
    :func:`scipy.stats.truncnorm`); truncation shifts the mean away from ``*_mean`` when the
    bound is near, which must be reported if used.  ``sd == 0`` gives a constant.  ``NaN`` mean
    gives ``NaN`` draws (missing stays missing).  Draw order: theta first, then d.
    """

    model_id: str
    ingredient_ids: tuple[str, ...]
    nutrient_ids: tuple[str, ...]
    theta_mean: np.ndarray
    theta_sd: np.ndarray
    d_mean: np.ndarray
    d_sd: np.ndarray
    truncate: bool = True
    theta_lower: Bound = 0.0
    theta_upper: Bound = np.inf
    d_lower: float = 1e-6
    d_upper: float = 1.0
    is_synthetic: bool = False

    def __post_init__(self) -> None:
        for name in ("theta_mean", "theta_sd", "d_mean", "d_sd"):
            object.__setattr__(self, name, _ro(getattr(self, name)))
        object.__setattr__(self, "theta_lower", _ro(np.broadcast_to(self.theta_lower, self.theta_mean.shape)))
        object.__setattr__(self, "theta_upper", _ro(np.broadcast_to(self.theta_upper, self.theta_mean.shape)))
        object.__setattr__(self, "ingredient_ids", tuple(self.ingredient_ids))
        object.__setattr__(self, "nutrient_ids", tuple(self.nutrient_ids))
        I, J = len(self.ingredient_ids), len(self.nutrient_ids)
        if self.theta_mean.shape != (I, J) or self.theta_sd.shape != (I, J) or \
                self.d_mean.shape != (I,) or self.d_sd.shape != (I,):
            raise ValueError("IndependentNormalModel: shapes do not match labels")
        sd_ok = np.where(np.isnan(self.theta_mean), True, self.theta_sd >= 0)
        if not np.all(sd_ok) or np.any(self.d_sd < 0) or np.any(np.isnan(self.d_mean)):
            raise ValueError("IndependentNormalModel: sd must be >= 0 and d_mean finite")

    def _draw(self, rng, mean, sd, lo, hi, n) -> np.ndarray:
        shape = (n,) + mean.shape
        out = np.broadcast_to(mean, shape).astype(float).copy()
        live = (sd > 0) & np.isfinite(mean)
        if not np.any(live):
            return out
        m = np.where(live, mean, 0.0)
        s = np.where(live, sd, 1.0)
        if self.truncate:
            a = (lo - m) / s
            b = (hi - m) / s
            z = stats.truncnorm.rvs(np.broadcast_to(a, shape), np.broadcast_to(b, shape),
                                    size=shape, random_state=rng)
        else:
            z = rng.standard_normal(size=shape)
        vals = m + s * z
        return np.where(np.broadcast_to(live, shape), vals, out)

    def sample(self, rng: np.random.Generator, n_draws: int) -> tuple[np.ndarray, np.ndarray]:
        n = int(n_draws)
        theta = self._draw(rng, self.theta_mean, self.theta_sd, self.theta_lower, self.theta_upper, n)
        d = self._draw(rng, self.d_mean, self.d_sd, np.full(self.d_mean.shape, self.d_lower),
                       np.full(self.d_mean.shape, self.d_upper), n)
        return theta, d

    def params_for_fingerprint(self) -> dict:
        return {"theta_mean": self.theta_mean, "theta_sd": self.theta_sd, "d_mean": self.d_mean,
                "d_sd": self.d_sd, "truncate": self.truncate, "theta_lower": self.theta_lower,
                "theta_upper": self.theta_upper, "d_lower": self.d_lower, "d_upper": self.d_upper}


@dataclass(frozen=True, eq=False)
class ScenarioSetModel(UncertaintyModel):
    """Resample (with replacement) from a fixed set of joint states ``[N, I, J]``, ``[N, I]``.

    A finite scenario set is *not* a continuous distribution; reuse of the same records does not
    increase the empirical sample size (T5).  :meth:`all_scenarios` returns the set itself.
    """

    model_id: str
    ingredient_ids: tuple[str, ...]
    nutrient_ids: tuple[str, ...]
    theta_set: np.ndarray
    d_set: np.ndarray
    is_synthetic: bool = False
    labels: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        object.__setattr__(self, "theta_set", _ro(self.theta_set))
        object.__setattr__(self, "d_set", _ro(self.d_set))
        object.__setattr__(self, "ingredient_ids", tuple(self.ingredient_ids))
        object.__setattr__(self, "nutrient_ids", tuple(self.nutrient_ids))
        N = self.theta_set.shape[0]
        if self.theta_set.shape[1:] != (len(self.ingredient_ids), len(self.nutrient_ids)) or \
                self.d_set.shape != (N, len(self.ingredient_ids)):
            raise ValueError("ScenarioSetModel: shapes do not match labels")

    def sample(self, rng: np.random.Generator, n_draws: int) -> tuple[np.ndarray, np.ndarray]:
        idx = rng.integers(0, self.theta_set.shape[0], size=int(n_draws))
        return self.theta_set[idx].copy(), self.d_set[idx].copy()

    def all_scenarios(self) -> tuple[np.ndarray, np.ndarray]:
        """The full scenario set (copies)."""
        return self.theta_set.copy(), self.d_set.copy()

    def params_for_fingerprint(self) -> dict:
        return {"theta_set": self.theta_set, "d_set": self.d_set, "labels": tuple(self.labels)}
