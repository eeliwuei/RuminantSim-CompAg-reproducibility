"""Uncertainty-model interface and draw containers.

An :class:`UncertaintyModel` produces joint draws of the unknown coefficients::

    theta : [S, I, J]   composition, canonical units, DM basis (NaN = missing)
    d     : [S, I]      DM fraction, kg DM / kg as-fed

from an **explicit** :class:`numpy.random.Generator` (no global RNG state).  Draws are wrapped in
a :class:`DrawSet` that remembers the purpose stream (``opt`` / ``validation`` / ``test`` /
``outer``), so decision steps can refuse draws they must not see (:func:`require_stream`).

Implementations must document the source of every parameter; a sampling family chosen for
convenience is a *research assumption* and must be declared as such in the protocol.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Collection, Sequence

import numpy as np

from ..errors import LeakageError
from ..hashing import stable_hash
from .streams import RandomStreams

__all__ = ["DrawSet", "UncertaintyModel", "require_stream"]


@dataclass(frozen=True)
class DrawSet:
    """Read-only draws with their provenance.

    Attributes
    ----------
    theta, d : arrays (see module doc).
    stream : purpose stream name (``opt``, ``validation``, ``test``, ``outer`` or custom).
    stream_id : exact stream id, e.g. ``root=1103/test/0``.
    model_id, model_fingerprint : which model produced the draws.
    ingredient_ids, nutrient_ids : axis labels.
    is_synthetic : True if the generating parameters are synthetic/test-only.
    """

    theta: np.ndarray
    d: np.ndarray
    stream: str
    stream_id: str
    model_id: str
    model_fingerprint: str
    ingredient_ids: tuple[str, ...]
    nutrient_ids: tuple[str, ...]
    is_synthetic: bool

    def __post_init__(self) -> None:
        th = np.array(self.theta, dtype=float)
        d = np.array(self.d, dtype=float)
        if th.ndim != 3 or d.ndim != 2 or th.shape[:2] != d.shape:
            raise ValueError("DrawSet: theta must be [S,I,J] and d [S,I] with matching S, I")
        if th.shape[1:] != (len(self.ingredient_ids), len(self.nutrient_ids)):
            raise ValueError("DrawSet: axis labels do not match array shapes")
        th.setflags(write=False)
        d.setflags(write=False)
        object.__setattr__(self, "theta", th)
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "ingredient_ids", tuple(self.ingredient_ids))
        object.__setattr__(self, "nutrient_ids", tuple(self.nutrient_ids))

    @property
    def n_draws(self) -> int:
        """Number of draws ``S``."""
        return int(self.d.shape[0])

    @property
    def fingerprint(self) -> str:
        """Content hash of the draws and their labels."""
        return stable_hash("DrawSet/v1", self.stream, self.stream_id, self.model_fingerprint,
                           self.ingredient_ids, self.nutrient_ids, self.theta, self.d)

    def reordered(self, ingredient_ids: Sequence[str] | None = None,
                  nutrient_ids: Sequence[str] | None = None,
                  scenario_order: Sequence[int] | None = None) -> "DrawSet":
        """Relabelled copy (permutation of ingredients / nutrients / scenarios)."""
        ii = list(range(len(self.ingredient_ids))) if ingredient_ids is None else \
            [self.ingredient_ids.index(i) for i in ingredient_ids]
        jj = list(range(len(self.nutrient_ids))) if nutrient_ids is None else \
            [self.nutrient_ids.index(j) for j in nutrient_ids]
        ss = list(range(self.n_draws)) if scenario_order is None else [int(s) for s in scenario_order]
        th = self.theta[ss][:, ii][:, :, jj]
        d = self.d[ss][:, ii]
        return DrawSet(th, d, self.stream, self.stream_id, self.model_id, self.model_fingerprint,
                       tuple(self.ingredient_ids[i] for i in ii), tuple(self.nutrient_ids[j] for j in jj),
                       self.is_synthetic)


def require_stream(draws: DrawSet, allowed: Collection[str], consumer: str) -> None:
    """Raise :class:`LeakageError` if ``draws`` come from a stream ``consumer`` may not use."""
    if draws.stream not in set(allowed):
        raise LeakageError(f"{consumer} may only use streams {sorted(allowed)}, got {draws.stream!r} "
                           f"({draws.stream_id})")


class UncertaintyModel(abc.ABC):
    """Abstract sampling interface.

    Subclasses define ``model_id``, ``ingredient_ids``, ``nutrient_ids``, ``is_synthetic`` and
    implement :meth:`sample` and :meth:`params_for_fingerprint`.
    """

    model_id: str
    ingredient_ids: tuple[str, ...]
    nutrient_ids: tuple[str, ...]
    is_synthetic: bool

    @abc.abstractmethod
    def sample(self, rng: np.random.Generator, n_draws: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(theta [S, I, J], d [S, I])`` drawn with ``rng``."""

    @abc.abstractmethod
    def params_for_fingerprint(self) -> dict:
        """All parameters that determine the distribution (hashed into the fingerprint)."""

    def fingerprint(self) -> str:
        """Stable content hash of the model class and parameters."""
        return stable_hash(type(self).__name__, self.model_id, tuple(self.ingredient_ids),
                           tuple(self.nutrient_ids), self.is_synthetic, self.params_for_fingerprint())

    def draw(self, streams: RandomStreams, stream: str, n_draws: int, *sub: int) -> DrawSet:
        """Sample ``n_draws`` from the named stream and wrap them in a :class:`DrawSet`."""
        if int(n_draws) < 1:
            raise ValueError("n_draws must be >= 1")
        rng = streams.generator(stream, *sub)
        theta, d = self.sample(rng, int(n_draws))
        return DrawSet(theta, d, stream, streams.stream_id(stream, *sub), self.model_id, self.fingerprint(),
                       tuple(self.ingredient_ids), tuple(self.nutrient_ids), bool(self.is_synthetic))
