"""Signal likelihood matrices ``L[s, z] = P(signal bin z | state s)`` (contract T7.3, T7.4).

Every information structure used by the policy model is represented by a row-stochastic matrix
over the *same* prior states, so that the no-information and with-information problems share
prior, action space and risk definition (T7.1):

=============================  =====================================================================
kind                           meaning
=============================  =====================================================================
``perfect_full``               observe every stochastic component of the modelled state (bins =
                               distinct states).  Not knowledge of unmodelled animal responses.
``perfect_partial_exact``      observe the listed components exactly (bins = distinct values;
                               valid for discrete priors; degenerate on continuous draws).
``perfect_partial_binned``     observe in which development bin the listed components fall
                               (a coarsening of exact partial perfect information).
``sample_analytic_gaussian``   ``Z = theta + bias_mean + N(0, total_sd^2)`` per component, bin
                               probabilities by the normal CDF (independent components).
``sample_monte_carlo``         bin frequencies of simulated signals (component-wise error draws,
                               named stream; approximates the analytic case).
``uninformative``              ``L[s, z] = p_z`` independent of the state (benchmark: pure
                               randomisation device with the same bin probabilities).
=============================  =====================================================================

Other stochastic components are *not* removed from the prior: within a bin the remaining
uncertainty is represented by the prior states themselves, so correlations are kept (T7.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy import stats

from ..errors import InvalidProblemError
from ..hashing import stable_hash
from ..uncertainty.streams import RandomStreams
from .binning import SignalBinning, exact_value_groups
from .prior import ObservedComponent, PriorStates
from .signal import SignalModel

__all__ = [
    "LIKELIHOOD_KINDS",
    "SignalLikelihood",
    "perfect_full_likelihood",
    "perfect_partial_likelihood",
    "sample_likelihood_analytic",
    "sample_likelihood_monte_carlo",
    "uninformative_likelihood",
    "bin_occupancy",
]

LIKELIHOOD_KINDS = ("perfect_full", "perfect_partial_exact", "perfect_partial_binned", "sample_analytic_gaussian",
                    "sample_monte_carlo", "uninformative")


@dataclass(frozen=True)
class SignalLikelihood:
    """Row-stochastic ``L [S, Z]`` with labels and provenance."""

    kind: str
    L: np.ndarray
    bin_labels: tuple[str, ...]
    components: tuple[ObservedComponent, ...]
    prior_fingerprint: str
    detail_fingerprint: str
    warnings: tuple[str, ...] = ()
    stream_id: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in LIKELIHOOD_KINDS:
            raise InvalidProblemError(f"unknown likelihood kind {self.kind!r}")
        L = np.array(self.L, dtype=float)
        if L.ndim != 2 or L.shape[1] != len(self.bin_labels):
            raise InvalidProblemError("SignalLikelihood: L must be [S, Z] with one label per bin")
        if np.any(~np.isfinite(L)) or np.any(L < -1e-15) or np.any(np.abs(L.sum(axis=1) - 1.0) > 1e-9):
            raise InvalidProblemError("SignalLikelihood: rows must be probability vectors")
        L = np.clip(L, 0.0, None)
        L.setflags(write=False)
        object.__setattr__(self, "L", L)

    @property
    def n_bins(self) -> int:
        """Number of signal bins ``Z``."""
        return int(self.L.shape[1])

    @property
    def is_perfect(self) -> bool:
        """True for noise-free (partition) structures."""
        return self.kind.startswith("perfect")

    @property
    def fingerprint(self) -> str:
        """Content hash."""
        return stable_hash("SignalLikelihood/v1", self.kind, self.prior_fingerprint, self.detail_fingerprint, self.L)


def _check_prior(prior: PriorStates) -> None:
    if not isinstance(prior, PriorStates):
        raise TypeError("prior must be PriorStates")


def perfect_full_likelihood(prior: PriorStates) -> SignalLikelihood:
    """Full perfect information: each distinct modelled state is its own signal bin."""
    _check_prior(prior)
    comps = prior.stochastic_components()
    vals = prior.component_values(comps) if comps else np.zeros((prior.n_states, 0))
    g = exact_value_groups(vals)
    L = np.zeros((prior.n_states, g.n_groups))
    L[np.arange(prior.n_states), g.group_of_state] = 1.0
    labels = tuple(f"state_group_{k}" for k in range(g.n_groups))
    return SignalLikelihood("perfect_full", L, labels, comps, prior.fingerprint,
                            stable_hash("perfect_full/v1", tuple(c.label() for c in comps)))


def perfect_partial_likelihood(prior: PriorStates, components: Sequence[ObservedComponent],
                               binning: Optional[SignalBinning] = None) -> SignalLikelihood:
    """Partial perfect information on ``components``: exact (``binning=None``) or binned."""
    _check_prior(prior)
    comps = tuple(components)
    if not comps:
        raise InvalidProblemError("perfect_partial_likelihood: at least one component")
    vals = prior.component_values(comps)
    warnings: list[str] = []
    if binning is None:
        g = exact_value_groups(vals)
        others = [c for c in prior.stochastic_components() if c not in comps]
        if g.degenerate and others:
            warnings.append("exact partial perfect information is degenerate on these states (each state its own "
                            "group while other components remain uncertain): the value equals full perfect "
                            "information in this finite sample. Use development bins.")
        L = np.zeros((prior.n_states, g.n_groups))
        L[np.arange(prior.n_states), g.group_of_state] = 1.0
        labels = tuple(f"value_group_{k}" for k in range(g.n_groups))
        return SignalLikelihood("perfect_partial_exact", L, labels, comps, prior.fingerprint,
                                stable_hash("perfect_partial_exact/v1", tuple(c.label() for c in comps)),
                                tuple(warnings))
    if tuple(binning.components) != comps:
        raise InvalidProblemError("perfect_partial_likelihood: binning components differ from observed components")
    idx = binning.bin_index(vals)
    L = np.zeros((prior.n_states, binning.n_bins))
    L[np.arange(prior.n_states), idx] = 1.0
    return SignalLikelihood("perfect_partial_binned", L, binning.bin_labels(), comps, prior.fingerprint,
                            stable_hash("perfect_partial_binned/v1", binning.fingerprint), tuple(warnings))


def sample_likelihood_analytic(prior: PriorStates, signal: SignalModel, binning: SignalBinning) -> SignalLikelihood:
    """Bin probabilities of ``Z = theta + bias_mean + e`` with independent normal total errors.

    ``P(z | s) = prod_c [Phi((hi_c - t_sc - mu_c)/sd_c) - Phi((lo_c - t_sc - mu_c)/sd_c)]``; a
    component with ``sd_c = 0`` contributes the indicator of its (shifted) bin.
    """
    _check_prior(prior)
    comps = signal.components
    if tuple(binning.components) != comps:
        raise InvalidProblemError("sample_likelihood_analytic: binning components must equal signal components")
    t = prior.component_values(comps)
    sd = signal.total_sd()
    mu = signal.bias_means()
    per_comp = []
    for c in range(len(comps)):
        lo, hi = binning.intervals(c)
        x = t[:, c] + mu[c]
        if sd[c] == 0.0:
            b = np.searchsorted(np.asarray(binning.interior_edges[c]), x, side="right")
            P = np.zeros((prior.n_states, lo.size))
            P[np.arange(prior.n_states), b] = 1.0
        else:
            # survival-function difference is more accurate in the upper tail than cdf differences
            P = stats.norm.sf((lo[None, :] - x[:, None]) / sd[c]) - stats.norm.sf((hi[None, :] - x[:, None]) / sd[c])
            P = np.clip(P, 0.0, None)
            P = P / P.sum(axis=1, keepdims=True)
        per_comp.append(P)
    L = per_comp[0]
    for P in per_comp[1:]:
        L = (L[:, :, None] * P[:, None, :]).reshape(prior.n_states, -1)
    warnings = tuple(signal.provenance_issues())
    return SignalLikelihood("sample_analytic_gaussian", L, binning.bin_labels(), comps, prior.fingerprint,
                            stable_hash("sample_analytic/v1", signal.fingerprint(), binning.fingerprint),
                            warnings, extra={"total_sd": sd.tolist(), "bias_mean": mu.tolist()})


def sample_likelihood_monte_carlo(prior: PriorStates, signal: SignalModel, binning: SignalBinning,
                                  streams: RandomStreams, n_replicates: int,
                                  stream: str = "assay_signal_dev") -> SignalLikelihood:
    """Monte Carlo bin frequencies: ``n_replicates`` simulated signals per state.

    Uses the named stream ``stream`` (not ``test``); replicate ``r`` uses sub-key ``r``.
    """
    _check_prior(prior)
    if stream == "test":
        raise InvalidProblemError("development likelihoods must not use the 'test' stream")
    comps = signal.components
    if tuple(binning.components) != comps:
        raise InvalidProblemError("sample_likelihood_monte_carlo: binning components must equal signal components")
    n = int(n_replicates)
    if n < 1:
        raise ValueError("n_replicates must be >= 1")
    t = prior.component_values(comps)
    counts = np.zeros((prior.n_states, binning.n_bins))
    rows = np.arange(prior.n_states)
    for r in range(n):
        z = signal.simulate(t, streams.generator(stream, r))
        np.add.at(counts, (rows, binning.bin_index(z)), 1.0)
    L = counts / n
    return SignalLikelihood("sample_monte_carlo", L, binning.bin_labels(), comps, prior.fingerprint,
                            stable_hash("sample_mc/v1", signal.fingerprint(), binning.fingerprint, n,
                                        streams.stream_id(stream)),
                            tuple(signal.provenance_issues()), stream_id=streams.stream_id(stream),
                            extra={"n_replicates": n})


def uninformative_likelihood(prior: PriorStates, bin_probabilities: Sequence[float],
                             labels: Optional[Sequence[str]] = None) -> SignalLikelihood:
    """State-independent signal with the given bin probabilities (randomisation benchmark)."""
    _check_prior(prior)
    p = np.asarray(bin_probabilities, dtype=float)
    if p.ndim != 1 or np.any(p < 0) or abs(p.sum() - 1.0) > 1e-9:
        raise InvalidProblemError("uninformative_likelihood: bin probabilities must be a probability vector")
    L = np.broadcast_to(p / p.sum(), (prior.n_states, p.size)).copy()
    labs = tuple(labels) if labels is not None else tuple(f"noise_bin_{k}" for k in range(p.size))
    return SignalLikelihood("uninformative", L, labs, (), prior.fingerprint,
                            stable_hash("uninformative/v1", p))


def bin_occupancy(prior: PriorStates, lik: SignalLikelihood) -> dict:
    """Per-bin probability and effective number of development states.

    The effective count is the Kish size ``(sum_s v_s)^2 / sum_s v_s^2`` of the posterior weights
    ``v_s = pi_s L[s, z]`` rescaled to the equal-weight state count (for equally weighted states
    and a perfect partition it is the number of states in the bin).  Small counts mean the
    conditional distribution inside the bin is poorly represented (T8.4, F09).
    """
    if lik.prior_fingerprint != prior.fingerprint:
        raise InvalidProblemError("bin_occupancy: likelihood was built on another prior")
    v = prior.weights[:, None] * lik.L
    pz = v.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        kish = np.where(pz > 0, pz ** 2 / (v ** 2).sum(axis=0), 0.0)
        # rescale: with uniform weights 1/S the Kish size is already a state count
    return {"p_bin": pz, "effective_states": kish, "n_bins": lik.n_bins,
            "n_active_bins": int(np.sum(pz > 0)), "min_effective_states_active": float(kish[pz > 0].min())
            if np.any(pz > 0) else 0.0}
