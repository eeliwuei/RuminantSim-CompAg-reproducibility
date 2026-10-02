"""Signal bins fixed during development (contract T7.3, E02, F09).

A :class:`SignalBinning` partitions the signal space of the observed components into a product
grid of intervals.  Per component the interior edges ``e_1 < ... < e_{n-1}`` define ``n`` bins::

    bin 0 = (-inf, e_1),  bin b = [e_b, e_{b+1}),  bin n-1 = [e_{n-1}, +inf)

The flat bin index over components uses C order (last component fastest).  Edges are chosen from
development data only (e.g. :func:`quantile_edges` on ``opt``/``validation`` states) and then
frozen; a frozen policy is later evaluated on independent streams without re-binning.

Exact grouping (:func:`exact_value_groups`) is only valid when the prior of the observed
components has discrete support.  On continuous Monte Carlo draws every state becomes its own
group, so "exact" partial perfect information silently turns into full perfect information
(finite-sample upward bias); :func:`exact_value_groups` reports that degeneracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..errors import InvalidProblemError
from ..hashing import stable_hash
from .prior import ObservedComponent

__all__ = ["SignalBinning", "quantile_edges", "exact_value_groups", "ExactGrouping", "MAX_BINS_DEFAULT"]

#: Default cap on the number of product-grid bins (MILP size is bins x candidates).
MAX_BINS_DEFAULT = 4096


@dataclass(frozen=True)
class SignalBinning:
    """Product-grid bins over observed components (edges in canonical units).

    ``frozen_from`` records where the edges came from (e.g. ``"quantiles of root=1103/opt"``).
    """

    components: tuple[ObservedComponent, ...]
    interior_edges: tuple[tuple[float, ...], ...]
    frozen_from: str = ""
    max_bins: int = MAX_BINS_DEFAULT

    def __post_init__(self) -> None:
        comps = tuple(self.components)
        edges = tuple(tuple(float(e) for e in es) for es in self.interior_edges)
        object.__setattr__(self, "components", comps)
        object.__setattr__(self, "interior_edges", edges)
        if len(comps) != len(edges):
            raise InvalidProblemError("SignalBinning: one edge list per component is required")
        if len(set(comps)) != len(comps):
            raise InvalidProblemError("SignalBinning: duplicate component")
        for c, es in zip(comps, edges):
            a = np.asarray(es, dtype=float)
            if not np.all(np.isfinite(a)) or np.any(np.diff(a) <= 0):
                raise InvalidProblemError(f"SignalBinning: edges of {c.label()} must be finite and strictly increasing")
        if self.n_bins > int(self.max_bins):
            raise InvalidProblemError(f"SignalBinning: {self.n_bins} bins exceed max_bins={self.max_bins}; "
                                      "use fewer edges or a reduced signal")

    @property
    def n_bins_per_component(self) -> tuple[int, ...]:
        """Bins per component."""
        return tuple(len(es) + 1 for es in self.interior_edges)

    @property
    def n_bins(self) -> int:
        """Total number of product-grid bins."""
        return int(np.prod(self.n_bins_per_component)) if self.components else 1

    def bin_index(self, values: np.ndarray) -> np.ndarray:
        """Flat bin index ``[S]`` of signal values ``[S, m]``."""
        v = np.asarray(values, dtype=float)
        if v.ndim != 2 or v.shape[1] != len(self.components):
            raise ValueError("bin_index: values must be [S, m]")
        if np.any(np.isnan(v)):
            raise ValueError("bin_index: NaN signal")
        per = [np.searchsorted(np.asarray(es), v[:, c], side="right") for c, es in enumerate(self.interior_edges)]
        if not per:
            return np.zeros(v.shape[0], dtype=int)
        return np.ravel_multi_index(tuple(per), self.n_bins_per_component).astype(int)

    def intervals(self, c: int) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper interval limits of the bins of component ``c`` (``+-inf`` at the ends)."""
        es = np.asarray(self.interior_edges[c], dtype=float)
        lo = np.concatenate([[-np.inf], es])
        hi = np.concatenate([es, [np.inf]])
        return lo, hi

    def bin_labels(self) -> tuple[str, ...]:
        """Readable labels of the flat bins."""
        labs = []
        for flat in range(self.n_bins):
            multi = np.unravel_index(flat, self.n_bins_per_component) if self.components else ()
            parts = []
            for c, b in enumerate(multi):
                lo, hi = self.intervals(c)
                parts.append(f"{self.components[c].label()}∈[{lo[b]:.6g},{hi[b]:.6g})")
            labs.append("&".join(parts) if parts else "all")
        return tuple(labs)

    def refine_nested(self, extra_edges: Sequence[Sequence[float]]) -> "SignalBinning":
        """A finer binning that contains all current edges (for bin-refinement checks)."""
        if len(extra_edges) != len(self.components):
            raise InvalidProblemError("refine_nested: one extra edge list per component")
        new = tuple(tuple(sorted(set(es) | set(float(x) for x in ex)))
                    for es, ex in zip(self.interior_edges, extra_edges))
        return SignalBinning(self.components, new, self.frozen_from + " +refined", self.max_bins)

    @property
    def fingerprint(self) -> str:
        """Content hash of components and edges."""
        return stable_hash("SignalBinning/v1", tuple((c.ingredient_id, c.component) for c in self.components),
                           self.interior_edges)


def quantile_edges(values: np.ndarray, n_bins: int, weights: Optional[np.ndarray] = None) -> tuple[float, ...]:
    """Interior edges at the ``1/n, ..., (n-1)/n`` weighted quantiles of development values.

    Duplicate edges (discrete values) are removed, so fewer bins may result.
    """
    v = np.asarray(values, dtype=float).ravel()
    n = int(n_bins)
    if n < 1:
        raise ValueError("n_bins must be >= 1")
    if n == 1:
        return ()
    w = np.full(v.shape, 1.0 / v.size) if weights is None else np.asarray(weights, dtype=float).ravel()
    order = np.argsort(v, kind="stable")
    cw = np.cumsum(w[order]) / w.sum()
    edges = []
    for q in np.arange(1, n) / n:
        # first sorted value whose cumulative weight exceeds q -> the bin below the edge holds mass ~q
        k = int(np.searchsorted(cw, q + 1e-12, side="left"))
        k = min(k, v.size - 1)
        edges.append(float(v[order][k]))
    es = sorted(set(edges))
    # an edge equal to the minimum value would create an empty first bin
    return tuple(e for e in es if e > v.min())


@dataclass(frozen=True)
class ExactGrouping:
    """Grouping of states by exact values of the observed components."""

    group_of_state: np.ndarray
    n_groups: int
    degenerate: bool
    note: str


def exact_value_groups(values: np.ndarray, decimals: int = 12) -> ExactGrouping:
    """Group states with identical observed values (after rounding to ``decimals``).

    ``degenerate`` is True when every state forms its own group although there are at least 2
    states: then "exact" conditioning on these components is indistinguishable from knowing the
    whole state in this finite sample (upward-biased partial-perfect value).
    """
    v = np.round(np.asarray(values, dtype=float), int(decimals))
    if v.ndim != 2:
        raise ValueError("exact_value_groups: values must be [S, m]")
    if v.shape[1] == 0:
        return ExactGrouping(np.zeros(v.shape[0], dtype=int), 1, False, "no observed component")
    _, inv = np.unique(v, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1).astype(int)
    ng = int(inv.max()) + 1
    degenerate = bool(v.shape[0] >= 2 and ng == v.shape[0])
    note = ("every state is its own group: exact conditioning collapses to full perfect information in this "
            "finite sample; use development bins instead") if degenerate else ""
    return ExactGrouping(inv, ng, degenerate, note)
