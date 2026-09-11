r"""Per-pixel calibrated precipitation distributions -- the alert-facing
product (PASO 7 of docs/decisions/0002-implementacion-v2.md), following
Avila 2026 §6.4.

For each pixel, all ensemble-generative samples (Candidate A/B draws, and in
Phase 4 also operational-forecast-ensemble draws) are pooled into a value
set. Unlike Avila's wind speed (always positive, continuously distributed),
precipitation has a **point mass at zero** (probability of no rain), so this
module fits a **zero-inflated / hurdle model**: a Bernoulli "did it rain"
gate, and, conditional on rain, one of several candidate continuous
families for the positive amounts -- exactly the caveat PASO 7 calls out
explicitly.

**Candidate continuous families** (Avila 2026 §4/§6.4): Weibull, Gamma,
Log-normal, Burr XII, and a two-component Weibull mixture. "No single
family describes every point ... the preferred family tracks the regime
rather than holding across the storm" -- so family selection is per pixel,
not fixed a priori.

**Selection procedure** (Avila 2026 §6.4, quoted almost verbatim):
"the Akaike and Bayesian information criteria rank the families against one
another while penalising the number of fitted parameters ... The
Anderson-Darling and Kolmogorov-Smirnov statistics instead measure the
distance between the sample and each fitted distribution, and so judge
whether a family is adequate in absolute terms rather than merely better
than the alternatives. Because the parameters are estimated from the same
data the test is applied to, the published critical values do not apply,
and the significance of each statistic is obtained instead by a parametric
bootstrap: repeated samples are drawn from the fitted distribution, the
family is refitted to each, and the observed statistic is compared against
that simulated null distribution." This module implements exactly that
bootstrap and does **not** accept the lowest-AIC family blindly -- the
paper explicitly found cases where the best-AIC family was still rejected
by the goodness-of-fit test (§4, eye centre: Burr XII "has the lowest AIC of
any family and is nonetheless rejected, at p = 0.030").

**Products derived once a distribution is selected**: quantile maps
(:func:`quantile_map`, e.g. p90/p99) and exceedance probability of an
operational "atypical rain" threshold (:func:`exceedance_probability`;
spateGAN-ERA5 uses 25 mm/h as its severe-warning threshold -- a starting
reference, not a value validated for this project's regions, see
docs/decisions/0002-implementacion-v2.md OQ-5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import optimize, stats

SCIPY_FAMILIES = ("weibull_min", "gamma", "lognorm", "burr12")
ALL_FAMILIES = SCIPY_FAMILIES + ("weibull_mixture2",)

_SCIPY_DIST = {
    "weibull_min": stats.weibull_min,
    "gamma": stats.gamma,
    "lognorm": stats.lognorm,
    "burr12": stats.burr12,
}
# Number of *free* shape/scale parameters, with location fixed at 0 (a
# physical constraint: precipitation amounts are non-negative and start at
# the origin, not at some fitted offset).
_SCIPY_N_PARAMS = {"weibull_min": 2, "gamma": 2, "lognorm": 2, "burr12": 3}


# --------------------------------------------------------------------------
# Two-component Weibull mixture (EM)
# --------------------------------------------------------------------------


class WeibullMixture2:
    """Two-component Weibull mixture: ``pi * Weibull(c1,scale1) + (1-pi) *
    Weibull(c2,scale2)``, location fixed at 0 for both components."""

    n_params = 5  # 1 free mixing weight + 2 shapes + 2 scales

    def __init__(self, weight: float, shapes: tuple[float, float], scales: tuple[float, float]):
        self.weight = float(weight)
        self.shapes = (float(shapes[0]), float(shapes[1]))
        self.scales = (float(scales[0]), float(scales[1]))

    def _components(self):
        return (
            (self.weight, stats.weibull_min(self.shapes[0], scale=self.scales[0])),
            (1.0 - self.weight, stats.weibull_min(self.shapes[1], scale=self.scales[1])),
        )

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        return sum(w * dist.pdf(x) for w, dist in self._components())

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return sum(w * dist.cdf(x) for w, dist in self._components())

    def logpdf(self, x):
        return np.log(np.clip(self.pdf(x), 1e-300, None))

    def ppf(self, q):
        q = np.atleast_1d(np.asarray(q, dtype=float))
        out = np.empty_like(q)
        hi = max(self.scales) * 50.0 + 1.0
        for idx, level in enumerate(q):
            if level <= 0.0:
                out[idx] = 0.0
            elif level >= 1.0:
                out[idx] = hi
            else:
                out[idx] = optimize.brentq(lambda x: self.cdf(x) - level, 0.0, hi, xtol=1e-8)
        return out if out.size > 1 else float(out[0])


def _weighted_weibull_nll(log_params: np.ndarray, data: np.ndarray, weights: np.ndarray) -> float:
    shape, scale = np.exp(log_params)
    dist = stats.weibull_min(shape, scale=scale)
    logpdf = dist.logpdf(data)
    if not np.all(np.isfinite(logpdf)):
        return 1e12
    return -np.sum(weights * logpdf)


def _fit_weighted_weibull(data: np.ndarray, weights: np.ndarray, init: tuple[float, float]) -> tuple[float, float]:
    x0 = np.log(np.asarray(init, dtype=float))
    result = optimize.minimize(
        _weighted_weibull_nll, x0, args=(data, weights), method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 500},
    )
    shape, scale = np.exp(result.x)
    return float(shape), float(scale)


def fit_weibull_mixture2(
    data: np.ndarray, *, max_iter: int = 50, tol: float = 1e-5, random_state=None
) -> tuple[WeibullMixture2, float]:
    """EM fit of a two-component Weibull mixture. Returns ``(mixture, loglik)``."""
    data = np.asarray(data, dtype=float)
    if np.any(data <= 0):
        raise ValueError("Weibull mixture requires strictly positive data")
    rng = np.random.default_rng(random_state)

    order = np.argsort(data)
    sorted_data = data[order]
    split = len(data) // 2
    low, high = sorted_data[: max(split, 2)], sorted_data[max(split, 2):] if split < len(data) else sorted_data
    if len(high) < 2:
        high = sorted_data[-2:]

    def _moment_init(sample):
        mean, std = sample.mean(), sample.std() + 1e-6
        cv = std / mean
        shape = max(0.5, 1.0 / cv) if cv > 0 else 1.5
        scale = mean / max(1e-6, _gamma_ratio(shape))
        return shape, scale

    shape1, scale1 = _moment_init(low)
    shape2, scale2 = _moment_init(high)
    weight = 0.5

    prev_loglik = -np.inf
    for _ in range(max_iter):
        mixture = WeibullMixture2(weight, (shape1, shape2), (scale1, scale2))
        w1, comp1 = weight, stats.weibull_min(shape1, scale=scale1)
        w2, comp2 = 1.0 - weight, stats.weibull_min(shape2, scale=scale2)
        p1 = w1 * comp1.pdf(data)
        p2 = w2 * comp2.pdf(data)
        denom = np.clip(p1 + p2, 1e-300, None)
        r1 = p1 / denom
        r2 = 1.0 - r1

        loglik = float(np.sum(np.log(denom)))
        if abs(loglik - prev_loglik) < tol:
            prev_loglik = loglik
            break
        prev_loglik = loglik

        weight = float(np.clip(r1.mean(), 0.02, 0.98))
        shape1, scale1 = _fit_weighted_weibull(data, r1, (shape1, scale1))
        shape2, scale2 = _fit_weighted_weibull(data, r2, (shape2, scale2))

    mixture = WeibullMixture2(weight, (shape1, shape2), (scale1, scale2))
    final_loglik = float(np.sum(mixture.logpdf(data)))
    return mixture, final_loglik


def _gamma_ratio(shape: float) -> float:
    """E[Weibull(shape, scale=1)] = Gamma(1 + 1/shape)."""
    from math import gamma

    return gamma(1.0 + 1.0 / shape)


# --------------------------------------------------------------------------
# Goodness of fit: Anderson-Darling with parametric bootstrap significance
# --------------------------------------------------------------------------


def anderson_darling_statistic(data: np.ndarray, dist) -> float:
    """Anderson-Darling A^2 against an arbitrary fitted distribution.

    A^2 = -n - (1/n) * sum_{i=1}^{n} (2i-1) * [ln F(x_(i)) + ln(1 - F(x_(n+1-i)))]

    with ``x_(i)`` the ascending order statistics and ``F`` = ``dist.cdf``
    (a scipy frozen distribution or :class:`WeibullMixture2`, both expose
    ``.cdf``). This statistic is inherently tail-weighted (the ``F(1-F)``
    normalization in its derivation gives more weight to the distribution's
    extremes), which is why it is preferred here over a plain
    Kolmogorov-Smirnov distance for a hazard product (Avila 2026 §5:
    "Anderson-Darling is weighted toward the tails ... the region that
    matters for a hazard product").
    """
    x = np.sort(np.asarray(data, dtype=float))
    n = x.size
    u = np.clip(np.asarray(dist.cdf(x), dtype=float), 1e-12, 1.0 - 1e-12)
    i = np.arange(1, n + 1)
    s = np.sum((2 * i - 1) * (np.log(u) + np.log(1.0 - u[::-1])))
    return float(-n - s / n)


def bootstrap_ad_pvalue(
    data: np.ndarray,
    fit_fn,
    *,
    n_bootstrap: int = 200,
    random_state=None,
) -> tuple[float, float]:
    """Parametric-bootstrap significance for the Anderson-Darling statistic.

    ``fit_fn(sample) -> dist`` fits the family under test to ``sample`` and
    returns an object exposing ``.cdf``/``.ppf`` (so both the observed fit
    and every bootstrap refit use the exact same fitting procedure).
    Returns ``(observed_statistic, p_value)`` where the p-value is the
    fraction of bootstrap statistics at least as extreme as the observed
    one.
    """
    rng = np.random.default_rng(random_state)
    n = len(data)
    dist = fit_fn(data)
    observed = anderson_darling_statistic(data, dist)

    null_stats = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        # Sample from the *observed* fit, refit the family to the
        # bootstrap sample, and score that sample against its own refit.
        u = rng.uniform(1e-6, 1.0 - 1e-6, size=n)
        sim = np.asarray(_invert_cdf(dist, u))
        boot_dist = fit_fn(sim)
        null_stats[b] = anderson_darling_statistic(sim, boot_dist)

    p_value = float(np.mean(null_stats >= observed))
    return observed, p_value


def _invert_cdf(dist, u: np.ndarray, hi_bound: float = 1e6) -> np.ndarray:
    """Numerically invert an arbitrary monotone CDF via bisection, for
    sampling within the bootstrap (works for both scipy frozen dists, which
    have a faster ``.ppf``, and :class:`WeibullMixture2`)."""
    if hasattr(dist, "ppf"):
        return np.atleast_1d(dist.ppf(u))
    out = np.empty_like(u)
    for idx, level in enumerate(u):
        out[idx] = optimize.brentq(lambda x: dist.cdf(np.array([x]))[0] - level, 0.0, hi_bound, xtol=1e-6)
    return out


# --------------------------------------------------------------------------
# Family fitting + selection
# --------------------------------------------------------------------------


@dataclass
class FittedFamily:
    name: str
    params: tuple
    n_params: int
    loglik: float
    aic: float
    bic: float
    ad_statistic: float
    ad_pvalue: float
    dist: object  # supports .cdf/.ppf/.pdf

    def cdf(self, x):
        return self.dist.cdf(x)

    def ppf(self, q):
        return self.dist.ppf(q)

    def pdf(self, x):
        return self.dist.pdf(x)


def _fit_scipy_family(name: str, data: np.ndarray):
    dist_cls = _SCIPY_DIST[name]
    params = dist_cls.fit(data, floc=0)
    frozen = dist_cls(*params)
    return frozen, params


def fit_family(
    name: str, data: np.ndarray, *, n_bootstrap: int = 200, random_state=None
) -> FittedFamily:
    """Fit one candidate family and score it (loglik, AIC, BIC, bootstrap AD)."""
    data = np.asarray(data, dtype=float)
    if name == "weibull_mixture2":
        def fit_fn(sample):
            mixture, _ = fit_weibull_mixture2(sample, random_state=random_state)
            return mixture

        dist, loglik = fit_weibull_mixture2(data, random_state=random_state)
        n_params = WeibullMixture2.n_params
        params = (dist.weight, *dist.shapes, *dist.scales)
    elif name in _SCIPY_DIST:
        def fit_fn(sample):
            frozen, _ = _fit_scipy_family(name, sample)
            return frozen

        dist, params = _fit_scipy_family(name, data)
        loglik = float(np.sum(dist.logpdf(data)))
        n_params = _SCIPY_N_PARAMS[name]
    else:
        raise ValueError(f"Unknown family {name!r}; choose from {ALL_FAMILIES}")

    ad_stat, ad_pvalue = bootstrap_ad_pvalue(
        data, fit_fn, n_bootstrap=n_bootstrap, random_state=random_state
    )
    n = len(data)
    aic = 2 * n_params - 2 * loglik
    bic = n_params * np.log(n) - 2 * loglik
    return FittedFamily(
        name=name, params=tuple(params), n_params=n_params,
        loglik=loglik, aic=aic, bic=bic, ad_statistic=ad_stat, ad_pvalue=ad_pvalue, dist=dist,
    )


def fit_all_families(
    data: np.ndarray,
    families: Sequence[str] = ALL_FAMILIES,
    *,
    n_bootstrap: int = 200,
    random_state=None,
) -> dict[str, FittedFamily]:
    return {
        name: fit_family(name, data, n_bootstrap=n_bootstrap, random_state=random_state)
        for name in families
    }


def select_family(
    fitted: dict[str, FittedFamily], *, alpha: float = 0.05, criterion: str = "aic"
) -> tuple[str, bool]:
    """Select the best family: lowest ``criterion`` (AIC/BIC) **among those
    not rejected** by the bootstrap Anderson-Darling test at level ``alpha``.

    Does not accept the lowest-AIC family blindly (Avila 2026 §4/§6.4: a
    family can lead the AIC ranking while still failing the goodness-of-fit
    test). Returns ``(family_name, goodness_of_fit_passed)``; if every
    candidate is rejected, falls back to the lowest-criterion family and
    reports ``goodness_of_fit_passed=False`` so callers can flag the pixel.
    """
    score = (lambda f: f.aic) if criterion == "aic" else (lambda f: f.bic)
    passing = {name: f for name, f in fitted.items() if f.ad_pvalue >= alpha}
    pool = passing or fitted
    best_name = min(pool, key=lambda name: score(pool[name]))
    return best_name, bool(passing)


# --------------------------------------------------------------------------
# Zero-inflated (hurdle) pixel model
# --------------------------------------------------------------------------


@dataclass
class ZeroInflatedFit:
    """Hurdle model: ``P(X <= x) = p_zero + (1 - p_zero) * F_continuous(x)``
    for ``x > 0``, and ``P(X <= 0) = p_zero``. Precipitation-specific
    handling of the point mass at zero, called out explicitly by PASO 7
    (unlike Avila's always-positive wind speed)."""

    p_zero: float
    family_name: str
    family: FittedFamily
    goodness_of_fit_passed: bool
    all_candidates: dict[str, FittedFamily] = field(repr=False, default_factory=dict)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        out = np.where(x <= 0.0, self.p_zero, self.p_zero + (1.0 - self.p_zero) * self.family.cdf(np.clip(x, 1e-12, None)))
        return out

    def ppf(self, q):
        q = np.atleast_1d(np.asarray(q, dtype=float))
        out = np.zeros_like(q)
        above = q > self.p_zero
        if above.any():
            rescaled = (q[above] - self.p_zero) / (1.0 - self.p_zero)
            out[above] = np.atleast_1d(self.family.ppf(rescaled))
        return out if out.size > 1 else float(out[0])

    def exceedance_probability(self, threshold: float) -> float:
        """P(X > threshold) -- the "atypical rain" alert product (PASO 7;
        spateGAN-ERA5 uses 25 mm/h as a starting reference threshold, see
        docs/decisions/0002-implementacion-v2.md OQ-5)."""
        return float(1.0 - self.cdf(np.asarray([threshold]))[0])


def fit_pixel_distribution(
    values: np.ndarray,
    *,
    zero_threshold: float = 0.0,
    families: Sequence[str] = ALL_FAMILIES,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    criterion: str = "aic",
    random_state=None,
) -> ZeroInflatedFit:
    """Fit the full zero-inflated per-pixel model (PASO 7 end-to-end)."""
    values = np.asarray(values, dtype=float)
    n = values.size
    p_zero = float(np.mean(values <= zero_threshold))
    positive = values[values > zero_threshold]
    if positive.size < 5:
        raise ValueError("Not enough positive (rainy) samples to fit a continuous family")

    fitted = fit_all_families(positive, families, n_bootstrap=n_bootstrap, random_state=random_state)
    best_name, passed = select_family(fitted, alpha=alpha, criterion=criterion)
    return ZeroInflatedFit(
        p_zero=p_zero, family_name=best_name, family=fitted[best_name],
        goodness_of_fit_passed=passed, all_candidates=fitted,
    )


def quantile_map(fit: ZeroInflatedFit, quantiles: Sequence[float]) -> dict[float, float]:
    """Quantile-map product (e.g. p90/p99) from a fitted pixel distribution."""
    return {q: float(np.atleast_1d(fit.ppf(q))[0]) for q in quantiles}
