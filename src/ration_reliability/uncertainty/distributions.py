"""Moment-matched marginal families for bounded composition cells (contract 01 §7.4; B-137).

Every stochastic cell ``(ingredient, component)`` has a *target* mean and SD (for example a
NASEM Table 19-1 mean and SD after unit conversion, possibly SD-scaled or mean-shifted by
:mod:`.scaling`) and a physical range ``[lower, upper]`` (composition fractions ``[0, 1]`` kg/kg
DM; DM fraction ``[0, 1]`` kg DM/kg as-fed).  A family is *moment matched* when the distribution
actually sampled has exactly the target mean and SD **inside** the physical range.

Families (all research-scenario assumptions; the primary family is pending, decision D-19):

``TN_MM``
    Truncated normal whose parent ``N(mu, sigma^2)`` is solved numerically so that the
    *truncated* mean and SD equal the target.  This removes the mean drift of the naive truncated
    normal (smoke S-5: 20/119 cells drifted by more than 1 %).  A truncated normal on a half line
    ``[a, inf)`` has ``SD / (mean - a) < 1``; targets with ``SD >= mean - a`` (or so close to it
    that the parent would be more than ``TN_ALPHA_MAX`` parent SDs beyond the bound) are reported
    as ``not_matchable`` and never forced.
``LN_MM``
    Shifted lognormal on ``(lower, upper]`` (the sampler always truncates at ``upper``).  The
    closed-form parameters are used when the truncated moments they imply already equal the target
    within ``MATCH_TOL``; otherwise the upper-truncated lognormal is moment matched numerically
    (heavy tails make even a tiny mass above ``upper`` matter for the SD).
``BETA_MM``
    Scaled beta on ``[lower, upper]``, moment matched in closed form.  Shape flags record a density
    that is unbounded at a bound (``a < 1`` or ``b < 1``).

Diagnostics only (not candidates for the primary analysis):

``TN_NAIVE``
    Parent ``N(target mean, target SD)`` truncated to the range (the smoke behaviour); its mean
    and SD differ from the target and the drift is reported.
``N_UNTRUNCATED``
    Untruncated normal; only the probability mass outside the physical range is reported.

Any distribution on ``[a, b]`` with mean ``m`` has ``Var <= (m - a)(b - m)`` (Bhatia-Davis); a
target violating it is ``infeasible_any_family``.

No parameter values are provided in this module; they are always passed in by the caller.
Sampling is by the inverse CDF (:meth:`MarginalFit.ppf`) so the same fits serve independent draws
and the Gaussian copula in :mod:`.correlation`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy import optimize, special, stats

__all__ = [
    "MOMENT_MATCHED_FAMILIES",
    "DIAGNOSTIC_FAMILIES",
    "ALL_FAMILIES",
    "FIT_STATUSES",
    "SAMPLEABLE_STATUSES",
    "MATCH_TOL",
    "TN_ALPHA_MAX",
    "LN_TAIL_TOL",
    "MarginalFit",
    "fit_marginal",
    "fit_truncnorm_moments",
    "fit_lognormal_moments",
    "fit_beta_moments",
    "naive_truncnorm",
    "untruncated_normal",
    "fit_array",
]

#: Families whose sampled mean and SD equal the target (when ``status == "matched"``).
MOMENT_MATCHED_FAMILIES: tuple[str, ...] = ("TN_MM", "LN_MM", "BETA_MM")
#: Families kept only for diagnostics (mean drift, mass outside the range).
DIAGNOSTIC_FAMILIES: tuple[str, ...] = ("TN_NAIVE", "N_UNTRUNCATED")
ALL_FAMILIES: tuple[str, ...] = MOMENT_MATCHED_FAMILIES + DIAGNOSTIC_FAMILIES

#: ``matched``: target reproduced within :data:`MATCH_TOL`; ``drifted``: diagnostic family whose
#: moments differ from the target (``TN_NAIVE``); ``point``: SD == 0 (constant); ``missing``: no
#: mean (sampled as NaN, never 0); ``missing_sd``: a mean without SD -- not sampleable, the
#: point-value policy is pending (``configs/uncertainty.yaml`` ``point_value_policy``; data
#: dictionary §5 rule 10: NaN must not enter the draws silently); ``not_matchable``: this family cannot reproduce the target; ``infeasible_any_family``:
#: no distribution on the range can (Bhatia-Davis); ``invalid_target``: mean outside the open range
#: or SD < 0; ``diagnostic_only``: ``N_UNTRUNCATED`` (not sampleable here).
FIT_STATUSES: tuple[str, ...] = ("matched", "drifted", "point", "missing", "missing_sd", "not_matchable",
                                 "infeasible_any_family", "invalid_target", "diagnostic_only")
#: Statuses that :meth:`MarginalFit.ppf` can sample.
SAMPLEABLE_STATUSES: tuple[str, ...] = ("matched", "drifted", "point", "missing")

#: Moment-matching tolerance: ``max(|mean error|, |SD error|) / target SD``.
MATCH_TOL = 1e-8
#: Largest standardised distance ``(lower - mu) / sigma`` of the TN parent beyond the lower bound
#: accepted by the solver (beyond it the truncated normal is numerically an exponential tail).
TN_ALPHA_MAX = 50.0
#: Kept for reference: parent lognormal mass above ``upper`` considered negligible in reports.
LN_TAIL_TOL = 1e-12
#: Standardised distance beyond which a bound is irrelevant for the TN solver.
_FAR = 8.5


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------

_SQRT2 = math.sqrt(2.0)
_SQRT2_OVER_PI = math.sqrt(2.0 / math.pi)
#: Standardised distance beyond which the other bound is treated as absent in the closed forms.
_IRRELEVANT = 9.0


def _mills(alpha: float) -> float:
    """Inverse Mills ratio ``phi(alpha) / (1 - Phi(alpha))`` via ``erfcx`` (no cancellation)."""
    with np.errstate(over="ignore"):
        return float(_SQRT2_OVER_PI / special.erfcx(alpha / _SQRT2))


def _tn_lower_std(alpha: float) -> tuple[float, float]:
    """Mean and variance of a standard normal truncated below at ``alpha`` (closed form)."""
    lam = _mills(alpha)
    return lam, max(1.0 - lam * (lam - alpha), 0.0)


def _tn_lower_cv(alpha: float) -> float:
    """``SD / (mean - a)`` of a normal truncated below at ``a`` with standardised bound ``alpha``."""
    lam, var = _tn_lower_std(alpha)
    return math.sqrt(var) / (lam - alpha)


def _tn_moments(mu: float, sigma: float, lower: float, upper: float) -> tuple[float, float, float]:
    """Mean, SD and parent mass outside ``[lower, upper]`` of a truncated normal.

    One-sided cases (the other bound more than ``_IRRELEVANT`` parent SDs away) use the closed form
    with an ``erfcx``-based Mills ratio; genuinely two-sided cases use :func:`scipy.stats.truncnorm`.
    """
    a, b = (lower - mu) / sigma, (upper - mu) / sigma
    mass_out = float(special.ndtr(a) + special.ndtr(-b))
    if b >= _IRRELEVANT:
        lam, var = _tn_lower_std(a)
        return mu + sigma * lam, sigma * math.sqrt(var), mass_out
    if a <= -_IRRELEVANT:
        lam, var = _tn_lower_std(-b)
        return mu - sigma * lam, sigma * math.sqrt(var), mass_out
    m, v = stats.truncnorm.stats(a, b, loc=mu, scale=sigma, moments="mv")
    return float(m), float(math.sqrt(max(float(v), 0.0))), mass_out


def _tln_moments(mu: float, sig: float, lower: float, upper: float) -> tuple[float, float, float]:
    """Mean, SD and parent mass above ``upper`` of ``lower + LN(mu, sig)`` truncated at ``upper``."""
    c = upper - lower
    if not np.isfinite(c):
        m1 = math.exp(mu + sig * sig / 2.0)
        var = (math.exp(sig * sig) - 1.0) * math.exp(2.0 * mu + sig * sig)
        return lower + m1, math.sqrt(var), 0.0
    z0 = (math.log(c) - mu) / sig
    lf = float(special.log_ndtr(z0))
    lm1 = mu + sig * sig / 2.0 + float(special.log_ndtr(z0 - sig)) - lf
    lm2 = 2.0 * mu + 2.0 * sig * sig + float(special.log_ndtr(z0 - 2.0 * sig)) - lf
    m1 = math.exp(lm1)
    var = math.exp(lm2) - m1 * m1
    return lower + m1, math.sqrt(max(var, 0.0)), float(special.ndtr(-z0))


def _rel_errors(target_mean: float, target_sd: float, mean: float, sd: float) -> tuple[float, float]:
    return (mean - target_mean) / target_sd, (sd - target_sd) / target_sd


# ----------------------------------------------------------------------------------------------
# fit container
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class MarginalFit:
    """One fitted marginal.

    Attributes
    ----------
    family : one of :data:`ALL_FAMILIES` (``POINT``/``MISSING`` are expressed through ``status``).
    target_mean, target_sd, lower, upper : the requested moments and physical range.
    status : one of :data:`FIT_STATUSES`.
    params : family parameters (``TN_*``: ``mu, sigma``; ``LN_MM``: ``mu_log, sigma_log``;
        ``BETA_MM``: ``a, b``; empty otherwise).
    achieved_mean, achieved_sd : analytic moments of the distribution that is sampled (NaN when not
        sampleable).
    mean_error_sd : ``(achieved_mean - target_mean) / target_sd``.
    sd_rel_error : ``(achieved_sd - target_sd) / target_sd``.
    mean_rel_shift : ``(achieved_mean - target_mean) / target_mean`` (the smoke S-5 quantity).
    outside_mass : parent probability mass outside ``[lower, upper]`` (TN: both sides; LN: above
        ``upper``; beta: 0; ``N_UNTRUNCATED``: mass a plain normal puts outside the range).
    flags : shape / feasibility notes.
    message : human-readable reason for a non-``matched`` status.
    """

    family: str
    target_mean: float
    target_sd: float
    lower: float
    upper: float
    status: str
    params: tuple[float, ...] = ()
    achieved_mean: float = float("nan")
    achieved_sd: float = float("nan")
    mean_error_sd: float = float("nan")
    sd_rel_error: float = float("nan")
    mean_rel_shift: float = float("nan")
    outside_mass: float = float("nan")
    flags: tuple[str, ...] = field(default_factory=tuple)
    message: str = ""

    def __post_init__(self) -> None:
        if self.family not in ALL_FAMILIES:
            raise ValueError(f"unknown family {self.family!r}; allowed {ALL_FAMILIES}")
        if self.status not in FIT_STATUSES:
            raise ValueError(f"unknown fit status {self.status!r}")

    # ------------------------------------------------------------------------------------------
    @property
    def sampleable(self) -> bool:
        """True if :meth:`ppf` can produce draws."""
        return self.status in SAMPLEABLE_STATUSES

    @property
    def is_stochastic(self) -> bool:
        """True for a non-degenerate sampled marginal."""
        return self.status in ("matched", "drifted")

    def ppf(self, u: np.ndarray) -> np.ndarray:
        """Inverse CDF evaluated at ``u`` in (0, 1)."""
        u = np.asarray(u, dtype=float)
        if self.status == "missing":
            return np.full(u.shape, np.nan)
        if self.status == "point":
            return np.full(u.shape, float(self.target_mean))
        if not self.sampleable:
            raise ValueError(f"marginal with status {self.status!r} cannot be sampled ({self.message})")
        lo, hi = float(self.lower), float(self.upper)
        if self.family in ("TN_MM", "TN_NAIVE"):
            mu, sigma = self.params
            a, b = (lo - mu) / sigma, (hi - mu) / sigma
            x = stats.truncnorm.ppf(u, a, b, loc=mu, scale=sigma)
        elif self.family == "LN_MM":
            mu, sig = self.params
            fb = 1.0 if not np.isfinite(hi) else float(special.ndtr((math.log(hi - lo) - mu) / sig))
            x = lo + np.exp(mu + sig * special.ndtri(u * fb))
        elif self.family == "BETA_MM":
            a, b = self.params
            x = lo + (hi - lo) * stats.beta.ppf(u, a, b)
        else:  # pragma: no cover - guarded by sampleable
            raise ValueError(f"family {self.family} is diagnostic only")
        return np.clip(x, lo, hi)

    def kurtosis(self) -> float:
        """Non-excess kurtosis ``E[(X - m)^4] / Var^2`` of the sampled distribution (NaN if constant).

        Used to judge how many draws are needed before a sample SD is informative
        (``SE(s)/s ~ sqrt((kurtosis - 1) / (4 N))``).  For ``LN_MM`` the truncation at ``upper`` is
        always included: with heavy tails most of the untruncated fourth moment can lie above
        ``upper``.
        """
        if not self.is_stochastic:
            return float("nan")
        lo, hi = float(self.lower), float(self.upper)
        if self.family in ("TN_MM", "TN_NAIVE"):
            mu, sigma = self.params
            k = stats.truncnorm.stats((lo - mu) / sigma, (hi - mu) / sigma, moments="k")
            return float(k) + 3.0
        if self.family == "BETA_MM":
            a, b = self.params
            return float(stats.beta.stats(a, b, moments="k")) + 3.0
        if self.family == "LN_MM":
            mu, sig = self.params
            if not np.isfinite(hi) or sig < 1e-3:
                s2 = sig * sig
                return float(math.exp(4 * s2) + 2 * math.exp(3 * s2) + 3 * math.exp(2 * s2) - 3)
            z0 = (math.log(hi - lo) - mu) / sig
            lf = float(special.log_ndtr(z0))
            logm = [k * mu + k * k * sig * sig / 2.0 + float(special.log_ndtr(z0 - k * sig)) - lf for k in (1, 2, 3, 4)]
            r = [math.exp(logm[k - 1] - k * logm[0]) for k in (1, 2, 3, 4)]   # E[Y^k] / E[Y]^k
            mu2 = r[1] - 1.0
            mu4 = r[3] - 4 * r[2] + 6 * r[1] - 3.0
            return float(mu4 / (mu2 * mu2)) if mu2 > 0 else float("nan")
        return float("nan")

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """``n`` independent draws with the explicit generator ``rng``."""
        return self.ppf(rng.random(int(n)))

    def fingerprint_params(self) -> tuple:
        """Everything that determines the sampled distribution."""
        return (self.family, self.status, float(self.target_mean), float(self.target_sd),
                float(self.lower), float(self.upper), tuple(float(p) for p in self.params))

    def as_record(self) -> dict:
        """Flat dictionary for diagnostics tables."""
        p = list(self.params) + [float("nan")] * (2 - len(self.params))
        return {"family": self.family, "status": self.status, "target_mean": self.target_mean,
                "target_sd": self.target_sd, "lower": self.lower, "upper": self.upper,
                "param_1": p[0], "param_2": p[1], "achieved_mean": self.achieved_mean,
                "achieved_sd": self.achieved_sd, "mean_error_sd": self.mean_error_sd,
                "sd_rel_error": self.sd_rel_error, "mean_rel_shift": self.mean_rel_shift,
                "outside_mass": self.outside_mass, "flags": ";".join(self.flags), "message": self.message}


def _base(family: str, mean: float, sd: float, lower: float, upper: float):
    """Common validation.  Returns a finished :class:`MarginalFit` or ``None`` to continue."""
    m, s, lo, hi = float(mean), float(sd), float(lower), float(upper)
    if not (np.isfinite(lo) and lo < hi):
        raise ValueError(f"need finite lower < upper, got [{lo}, {hi}]")
    if np.isnan(m):
        return MarginalFit(family, m, s, lo, hi, "missing", message="mean missing (null is never 0)")
    if np.isnan(s):
        return MarginalFit(family, m, s, lo, hi, "missing_sd",
                           message="SD missing: not sampleable; point-value policy is pending_user_decision "
                                   "(pass SD = 0 explicitly for a deterministic cell)")
    if s < 0 or not np.isfinite(m) or not np.isfinite(s):
        return MarginalFit(family, m, s, lo, hi, "invalid_target", message="SD < 0 or non-finite target")
    if s == 0.0:
        if not (lo <= m <= hi):
            return MarginalFit(family, m, s, lo, hi, "invalid_target", message="point value outside the range")
        return MarginalFit(family, m, s, lo, hi, "point", achieved_mean=m, achieved_sd=0.0,
                           outside_mass=0.0, message="SD == 0: constant")
    if not (lo < m < hi):
        return MarginalFit(family, m, s, lo, hi, "invalid_target",
                           message="mean must lie strictly inside (lower, upper) when SD > 0")
    if np.isfinite(hi) and s * s >= (m - lo) * (hi - m):
        return MarginalFit(family, m, s, lo, hi, "infeasible_any_family",
                           flags=("bhatia_davis_violated",),
                           message="SD^2 >= (mean-lower)(upper-mean): no distribution on the range has these moments")
    return None


def _finish(family: str, m: float, s: float, lo: float, hi: float, params: tuple[float, ...],
            am: float, asd: float, mass: float, flags: tuple[str, ...], tol: float,
            matched_status: str = "matched") -> MarginalFit:
    me, se = _rel_errors(m, s, am, asd)
    ok = max(abs(me), abs(se)) <= tol
    status = matched_status if (ok or matched_status == "drifted") else "not_matchable"
    msg = "" if status != "not_matchable" else (
        f"solver residual {max(abs(me), abs(se)):.2e} > tol {tol:.0e}")
    return MarginalFit(family, m, s, lo, hi, status, tuple(float(p) for p in params), am, asd, me, se,
                       (am - m) / m if m != 0 else float("nan"), mass, flags, msg)


# ----------------------------------------------------------------------------------------------
# truncated normal, moment matched
# ----------------------------------------------------------------------------------------------

def _tn_solve_lower(m: float, s: float, lo: float) -> tuple[float, float] | str:
    """Parent (mu, sigma) of a lower-truncated normal with mean m and SD s; or a failure flag."""
    cv = s / (m - lo)
    if cv >= 1.0:
        return "tn_cv_ge_1"
    alpha_min = -_FAR - 0.5
    if cv <= _tn_lower_cv(alpha_min):
        return m, s  # lower bound irrelevant
    if cv >= _tn_lower_cv(TN_ALPHA_MAX):
        return "tn_cv_too_close_to_1"
    alpha = optimize.brentq(lambda x: _tn_lower_cv(x) - cv, alpha_min, TN_ALPHA_MAX, xtol=1e-14,
                            rtol=4 * np.finfo(float).eps, maxiter=500)
    sigma = (m - lo) / (_mills(alpha) - alpha)
    return lo - alpha * sigma, sigma


def _tn_polish(m: float, s: float, lo: float, hi: float, starts: Sequence[tuple[float, float]]
               ) -> tuple[float, float] | None:
    best, best_r = None, np.inf

    def resid(x):
        mu, lsig = x
        sig = math.exp(lsig)
        am, asd, _ = _tn_moments(mu, sig, lo, hi)
        return np.array([(am - m) / s, (asd - s) / s])

    for mu0, sig0 in starts:
        try:
            res = optimize.least_squares(resid, np.array([mu0, math.log(sig0)]), xtol=1e-15, ftol=1e-15,
                                         gtol=1e-15, max_nfev=2000, method="lm")
        except (ValueError, FloatingPointError):  # pragma: no cover - numerical guard
            continue
        r = float(np.max(np.abs(res.fun)))
        if np.isfinite(r) and r < best_r:
            best, best_r = (float(res.x[0]), float(math.exp(res.x[1]))), r
    return best


def fit_truncnorm_moments(mean: float, sd: float, lower: float, upper: float,
                          tol: float = MATCH_TOL) -> MarginalFit:
    """``TN_MM``: truncated normal on ``[lower, upper]`` with truncated mean/SD equal to the target."""
    fam = "TN_MM"
    pre = _base(fam, mean, sd, lower, upper)
    if pre is not None:
        return pre
    m, s, lo, hi = float(mean), float(sd), float(lower), float(upper)
    z_lo, z_hi = (m - lo) / s, (hi - m) / s
    flags: list[str] = []
    starts: list[tuple[float, float]] = []
    if z_lo > _FAR and z_hi > _FAR:
        sol = (m, s)
    elif z_hi > _FAR:
        sol = _tn_solve_lower(m, s, lo)
    elif z_lo > _FAR:
        r = _tn_solve_lower(hi - m, s, 0.0)  # reflect x -> hi - x
        sol = r if isinstance(r, str) else (hi - r[0], r[1])
    else:
        sol = None
        rl = _tn_solve_lower(m, s, lo)
        if not isinstance(rl, str):
            starts.append(rl)
        ru = _tn_solve_lower(hi - m, s, 0.0)
        if not isinstance(ru, str):
            starts.append((hi - ru[0], ru[1]))
        starts.append((m, s))
    if isinstance(sol, str):
        return MarginalFit(fam, m, s, lo, hi, "not_matchable", flags=(sol,),
                           message=("SD >= mean - bound: a truncated normal cannot have CV >= 1 relative to "
                                    "the bound" if sol == "tn_cv_ge_1" else
                                    f"parent would lie more than {TN_ALPHA_MAX:g} SD beyond the bound "
                                    "(CV relative to the bound too close to 1)"))
    if sol is not None:
        mu, sigma = sol
        am, asd, mass = _tn_moments(mu, sigma, lo, hi)
        me, se = _rel_errors(m, s, am, asd)
        if max(abs(me), abs(se)) <= tol:
            if mass > 1e-12:
                flags.append("truncation_active")
            return _finish(fam, m, s, lo, hi, (mu, sigma), am, asd, mass, tuple(flags), tol)
        starts = [sol, (m, s)]
    pol = _tn_polish(m, s, lo, hi, starts)
    if pol is None:
        return MarginalFit(fam, m, s, lo, hi, "not_matchable", flags=("tn_solver_failed",),
                           message="two-sided truncated-normal moment matching did not converge")
    mu, sigma = pol
    am, asd, mass = _tn_moments(mu, sigma, lo, hi)
    flags.append("two_sided_solve")
    if mass > 1e-12:
        flags.append("truncation_active")
    return _finish(fam, m, s, lo, hi, (mu, sigma), am, asd, mass, tuple(flags), tol)


def naive_truncnorm(mean: float, sd: float, lower: float, upper: float) -> MarginalFit:
    """``TN_NAIVE`` (diagnostic): parent = target, truncated; mean and SD drift."""
    fam = "TN_NAIVE"
    pre = _base(fam, mean, sd, lower, upper)
    if pre is not None and pre.status != "infeasible_any_family":
        return pre
    m, s, lo, hi = float(mean), float(sd), float(lower), float(upper)
    am, asd, mass = _tn_moments(m, s, lo, hi)
    flags = ("truncation_active",) if mass > 1e-12 else ()
    if pre is not None:
        flags = flags + pre.flags
    return _finish(fam, m, s, lo, hi, (m, s), am, asd, mass, flags, MATCH_TOL, matched_status="drifted")


def untruncated_normal(mean: float, sd: float, lower: float, upper: float) -> MarginalFit:
    """``N_UNTRUNCATED`` (diagnostic): probability mass a plain normal puts outside the range."""
    fam = "N_UNTRUNCATED"
    m, s, lo, hi = float(mean), float(sd), float(lower), float(upper)
    if np.isnan(m):
        return MarginalFit(fam, m, s, lo, hi, "missing", message="mean missing")
    if np.isnan(s):
        return MarginalFit(fam, m, s, lo, hi, "missing_sd", message="SD missing")
    if s <= 0:
        return MarginalFit(fam, m, s, lo, hi, "point" if s == 0 else "invalid_target")
    mass = float(stats.norm.cdf((lo - m) / s) + stats.norm.sf((hi - m) / s))
    return MarginalFit(fam, m, s, lo, hi, "diagnostic_only", (m, s), m, s, 0.0, 0.0, 0.0, mass,
                       ("negative_values_possible",) if stats.norm.cdf((lo - m) / s) > 0 else (),
                       "diagnostic only: may produce values outside the physical range")


# ----------------------------------------------------------------------------------------------
# lognormal, moment matched
# ----------------------------------------------------------------------------------------------

def fit_lognormal_moments(mean: float, sd: float, lower: float, upper: float,
                          tol: float = MATCH_TOL) -> MarginalFit:
    """``LN_MM``: ``lower + LN(mu, sigma)``, upper-truncated at ``upper`` and moment matched."""
    fam = "LN_MM"
    pre = _base(fam, mean, sd, lower, upper)
    if pre is not None:
        return pre
    m, s, lo, hi = float(mean), float(sd), float(lower), float(upper)
    y = m - lo
    sig2 = math.log1p((s / y) ** 2)
    sig = math.sqrt(sig2)
    mu = math.log(y) - sig2 / 2.0
    tail = 0.0 if not np.isfinite(hi) else float(special.ndtr(-(math.log(hi - lo) - mu) / sig))
    flags: list[str] = []
    am, asd, mass = _tln_moments(mu, sig, lo, hi)
    me, se = _rel_errors(m, s, am, asd)
    if max(abs(me), abs(se)) <= tol:
        # closed form is exact within tol even though the sampler truncates at ``upper``
        if tail > 0:
            flags.append("upper_tail_negligible")
        return _finish(fam, m, s, lo, hi, (mu, sig), am, asd, mass, tuple(flags), tol)

    def resid(x):
        am_, asd_, _ = _tln_moments(x[0], math.exp(x[1]), lo, hi)
        return np.array([(am_ - m) / s, (asd_ - s) / s])

    try:
        res = optimize.least_squares(resid, np.array([mu, math.log(sig)]), xtol=1e-15, ftol=1e-15, gtol=1e-15,
                                     max_nfev=4000, method="lm")
        mu2, sig2_ = float(res.x[0]), float(math.exp(res.x[1]))
    except (ValueError, FloatingPointError, OverflowError):  # pragma: no cover - numerical guard
        return MarginalFit(fam, m, s, lo, hi, "not_matchable", flags=("ln_solver_failed",),
                           message="upper-truncated lognormal moment matching failed")
    am, asd, mass = _tln_moments(mu2, sig2_, lo, hi)
    flags.append("upper_truncated")
    return _finish(fam, m, s, lo, hi, (mu2, sig2_), am, asd, mass, tuple(flags), tol)


# ----------------------------------------------------------------------------------------------
# beta, moment matched
# ----------------------------------------------------------------------------------------------

def fit_beta_moments(mean: float, sd: float, lower: float, upper: float,
                     tol: float = MATCH_TOL) -> MarginalFit:
    """``BETA_MM``: ``lower + (upper - lower) * Beta(a, b)`` with the target mean and SD."""
    fam = "BETA_MM"
    if not np.isfinite(float(upper)):
        raise ValueError("BETA_MM needs a finite upper bound")
    pre = _base(fam, mean, sd, lower, upper)
    if pre is not None:
        return pre
    m, s, lo, hi = float(mean), float(sd), float(lower), float(upper)
    w = hi - lo
    my, vy = (m - lo) / w, (s / w) ** 2
    k = my * (1.0 - my) / vy - 1.0
    a, b = my * k, (1.0 - my) * k
    am_y, av_y = stats.beta.stats(a, b, moments="mv")
    am, asd = lo + w * float(am_y), w * math.sqrt(float(av_y))
    flags = []
    if a < 1.0:
        flags.append("beta_density_unbounded_at_lower")
    if b < 1.0:
        flags.append("beta_density_unbounded_at_upper")
    return _finish(fam, m, s, lo, hi, (a, b), am, asd, 0.0, tuple(flags), tol)


# ----------------------------------------------------------------------------------------------
# dispatch
# ----------------------------------------------------------------------------------------------

_DISPATCH = {"TN_MM": fit_truncnorm_moments, "LN_MM": fit_lognormal_moments, "BETA_MM": fit_beta_moments,
             "TN_NAIVE": lambda m, s, lo, hi, tol=MATCH_TOL: naive_truncnorm(m, s, lo, hi),
             "N_UNTRUNCATED": lambda m, s, lo, hi, tol=MATCH_TOL: untruncated_normal(m, s, lo, hi)}


def fit_marginal(family: str, mean: float, sd: float, lower: float, upper: float,
                 tol: float = MATCH_TOL) -> MarginalFit:
    """Fit one marginal of ``family`` (one of :data:`ALL_FAMILIES`)."""
    if family not in _DISPATCH:
        raise ValueError(f"unknown family {family!r}; allowed {ALL_FAMILIES}")
    return _DISPATCH[family](mean, sd, lower, upper, tol=tol)


def fit_array(family: str | Mapping[tuple[int, ...], str], mean: np.ndarray, sd: np.ndarray,
              lower, upper, tol: float = MATCH_TOL) -> np.ndarray:
    """Fit every cell of ``mean``/``sd`` arrays; returns an object array of :class:`MarginalFit`.

    ``family`` is a family name, or a mapping ``index tuple -> family`` whose missing entries raise.
    ``lower``/``upper`` broadcast to the array shape.
    """
    mean = np.asarray(mean, dtype=float)
    sd = np.asarray(sd, dtype=float)
    if mean.shape != sd.shape:
        raise ValueError("mean and sd must have the same shape")
    lo = np.broadcast_to(np.asarray(lower, dtype=float), mean.shape)
    hi = np.broadcast_to(np.asarray(upper, dtype=float), mean.shape)
    out = np.empty(mean.shape, dtype=object)
    for idx in np.ndindex(mean.shape):
        fam = family if isinstance(family, str) else family[idx]
        out[idx] = fit_marginal(fam, mean[idx], sd[idx], lo[idx], hi[idx], tol)
    return out
