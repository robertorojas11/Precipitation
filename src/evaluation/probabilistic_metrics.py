r"""Probabilistic verification metrics for ensemble precipitation downscaling
(PASO 8 of docs/decisions/0002-implementacion-v2.md).

Equations are quoted from **P1** (Glawion et al. 2025, spateGAN-ERA5),
"Evaluation", and **P2** (Harder/Schmidt et al. 2026, RainShift),
"Quantitative evaluation". Every metric operates on a boolean ``mask`` of valid
pixels; invalid observations are never counted (repo rule).

Ensemble convention: ``ensemble`` has shape ``(M, ...)`` where axis 0 indexes
the M members and the trailing axes match ``observation``.

Implemented
-----------
* ``crps_ensemble``      -- P1 Eq. (6)-(7), P2 Eq. (1). Reduces **exactly** to
                            MAE for M = 1 (P1 M181, P2 M247) -- asserted in tests.
* ``fractions_skill_score`` / ``mean_fss`` / ``mean_ensemble_fss``
                         -- P1 Eq. (9), scales/thresholds M184.
* ``percentile_mae``     -- P2 Eq. (2), percentiles 95 and 99 (M246).
* ``rapsd`` / ``mean_rapsd``
                         -- P1 M186 (pySTEPS implementation, numpy fallback).
* ``rank_histogram``     -- P1 M188.
* ``linear_eccentricity`` -- P1 Eq. (10)-(12), M189. EXPERIMENTAL (see docstring).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# P1 M184: scales (km) and rain-rate thresholds (mm/h) for mFSS / meFSS.
FSS_SCALES_KM = (0, 4, 8, 16, 32, 64, 128, 256)
FSS_THRESHOLDS_MM_H = (0.1, 1.0, 3.0, 5.0)
PERCENTILES = (95.0, 99.0)  # P2 M246


def _apply_mask(array: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    if mask is None:
        return array[np.isfinite(array)]
    mask = np.asarray(mask, dtype=bool)
    return array[mask & np.isfinite(array)]


# ---------------------------------------------------------------------------
# CRPS -- P1 Eq. (6)-(7), P2 Eq. (1)
# ---------------------------------------------------------------------------
def crps_ensemble(
    ensemble: np.ndarray,
    observation: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    r"""Mean CRPS of a finite ensemble against point observations.

    P1 Eq. (6):  CRPS(F, y) = \int (F(x) - 1{x >= y})^2 dx
    Using the exact identity for a finite sample (Gneiting & Raftery 2007,
    P1 ref. 57):

        CRPS = E|X - y|  -  (1/2) E|X - X'|

    where X, X' are independent draws from the ensemble. For M = 1 the second
    term is 0 and CRPS == |x - y| == MAE (P1: "for deterministic methods ...
    this score reduces to the mean absolute error"; P2 M247).

    ``ensemble``: (M, *obs.shape). Returns the mask-mean CRPS.
    """
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observation = np.asarray(observation, dtype=np.float64)
    if ensemble.ndim != observation.ndim + 1:
        raise ValueError("ensemble must be (M, *observation.shape)")
    m = ensemble.shape[0]

    skill = np.mean(np.abs(ensemble - observation[None, ...]), axis=0)
    if m == 1:
        spread = np.zeros_like(skill)
    else:
        # (1/M^2) sum_{i,j} |X_i - X_j|  == unbiased-ish energy spread term.
        diff = np.abs(ensemble[:, None, ...] - ensemble[None, :, ...])
        spread = diff.sum(axis=(0, 1)) / (m * m)
    cell_crps = skill - 0.5 * spread

    valid = _apply_mask_pair(cell_crps, observation, mask)
    return float(valid.mean()) if valid.size else float("nan")


def _apply_mask_pair(values: np.ndarray, reference: np.ndarray,
                      mask: np.ndarray | None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(reference)
    if mask is not None:
        finite &= np.asarray(mask, dtype=bool)
    return values[finite]


# ---------------------------------------------------------------------------
# Fractions Skill Score -- P1 Eq. (9)
# ---------------------------------------------------------------------------
def _fraction_field(binary: np.ndarray, scale_px: int) -> np.ndarray:
    """Fraction of exceedances in a (2*scale_px+1) square neighbourhood."""
    if scale_px <= 0:
        return binary.astype(np.float64)
    from scipy.ndimage import uniform_filter

    return uniform_filter(binary.astype(np.float64), size=2 * scale_px + 1, mode="nearest")


def fractions_skill_score(
    prediction: np.ndarray,
    observation: np.ndarray,
    threshold: float,
    scale_px: int,
    mask: np.ndarray | None = None,
) -> float:
    r"""P1 Eq. (9):  FSS = 1 - (f_pred - f_obs)^2 / (f_pred^2 + f_obs^2)

    ``f`` is the fraction of pixels within the spatial neighbourhood that exceed
    ``threshold``. Averaging is over all valid locations (M184).
    """
    pred_bin = (np.asarray(prediction) >= threshold).astype(np.float64)
    obs_bin = (np.asarray(observation) >= threshold).astype(np.float64)
    f_pred = _fraction_field(pred_bin, scale_px)
    f_obs = _fraction_field(obs_bin, scale_px)

    sel = np.isfinite(prediction) & np.isfinite(observation)
    if mask is not None:
        sel &= np.asarray(mask, dtype=bool)
    num = np.mean((f_pred[sel] - f_obs[sel]) ** 2)
    den = np.mean(f_pred[sel] ** 2 + f_obs[sel] ** 2)
    if den == 0:
        return 1.0  # both fields empty everywhere -> perfect by convention
    return float(1.0 - num / den)


def ensemble_fraction_exceeding(ensemble: np.ndarray, threshold: float) -> np.ndarray:
    """Fraction of ensemble members exceeding ``threshold`` at each cell (P1 M183)."""
    return np.mean(np.asarray(ensemble) >= threshold, axis=0)


def _km_to_px(scale_km: float, pixel_km: float) -> int:
    return int(round(scale_km / (2.0 * pixel_km))) if scale_km > 0 else 0


def mean_fss(
    prediction: np.ndarray,
    observation: np.ndarray,
    pixel_km: float,
    mask: np.ndarray | None = None,
    scales_km=FSS_SCALES_KM,
    thresholds=FSS_THRESHOLDS_MM_H,
) -> float:
    """mFSS: mean of FSS over the M184 grid of scales x thresholds."""
    scores = [
        fractions_skill_score(prediction, observation, thr, _km_to_px(s, pixel_km), mask)
        for thr in thresholds
        for s in scales_km
    ]
    return float(np.mean(scores))


def mean_ensemble_fss(
    ensemble: np.ndarray,
    observation: np.ndarray,
    pixel_km: float,
    mask: np.ndarray | None = None,
    scales_km=FSS_SCALES_KM,
    thresholds=FSS_THRESHOLDS_MM_H,
) -> float:
    """meFSS: like mFSS but the prediction field is the ensemble exceedance
    fraction compared against the observed 0/1 exceedance (P1 M183/M184)."""
    scores = []
    for thr in thresholds:
        pred_frac = ensemble_fraction_exceeding(ensemble, thr)
        obs_bin = (np.asarray(observation) >= thr).astype(np.float64)
        for s in scales_km:
            scale_px = _km_to_px(s, pixel_km)
            f_pred = _fraction_field(pred_frac, scale_px)
            f_obs = _fraction_field(obs_bin, scale_px)
            sel = np.isfinite(observation)
            if mask is not None:
                sel &= np.asarray(mask, dtype=bool)
            num = np.mean((f_pred[sel] - f_obs[sel]) ** 2)
            den = np.mean(f_pred[sel] ** 2 + f_obs[sel] ** 2)
            scores.append(1.0 if den == 0 else 1.0 - num / den)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Percentile MAE -- P2 Eq. (2)
# ---------------------------------------------------------------------------
def percentile_mae(
    prediction: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray | None = None,
    percentiles=PERCENTILES,
) -> dict[float, float]:
    r"""P2 Eq. (2):  MAE_p = (1/D) sum_d | x_pred_(floor(pN), d) - x_ref_(floor(pN), d) |

    ``x_(i),d`` is the i-th order statistic of dimension ``d``. Here "dimensions"
    are treated as a single pooled sample of valid pixels (D = 1), which is the
    common single-region reduction; the formula generalises if callers pass
    per-region arrays. Returns ``{percentile: MAE}``.
    """
    pred = _apply_mask_pair(np.asarray(prediction, dtype=np.float64), reference, mask)
    ref = _apply_mask_pair(np.asarray(reference, dtype=np.float64), prediction, mask)
    out: dict[float, float] = {}
    for p in percentiles:
        out[p] = float(abs(np.percentile(pred, p) - np.percentile(ref, p)))
    return out


# ---------------------------------------------------------------------------
# RAPSD -- P1 M186 (pySTEPS, numpy fallback)
# ---------------------------------------------------------------------------
def rapsd(field: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Radially averaged power spectral density of a single 2D field.

    Uses ``pysteps.utils.rapsd`` when available (P1 M186), otherwise an
    equivalent numpy implementation. ``normalize`` -> spectrum sums to 1 (M187).
    """
    field = np.asarray(field, dtype=np.float64)
    field = np.nan_to_num(field, nan=0.0)
    try:
        from pysteps.utils.spectral import rapsd as _ps_rapsd

        spectrum = _ps_rapsd(field, fft_method=np.fft, normalize=False)
    except Exception:
        spectrum = _numpy_rapsd(field)
    spectrum = np.asarray(spectrum, dtype=np.float64)
    if normalize:
        total = spectrum.sum()
        if total > 0:
            spectrum = spectrum / total
    return spectrum


def _numpy_rapsd(field: np.ndarray) -> np.ndarray:
    h, w = field.shape
    power = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    nbins = min(cy, cx)
    return np.array([power[r == k].mean() for k in range(1, nbins)])


def mean_rapsd(fields: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Mean RAPSD over a stack of 2D fields (P1: averaged per evaluation set)."""
    spectra = [rapsd(f, normalize=normalize) for f in np.asarray(fields)]
    length = min(len(s) for s in spectra)
    return np.mean([s[:length] for s in spectra], axis=0)


# ---------------------------------------------------------------------------
# Rank histogram -- P1 M188
# ---------------------------------------------------------------------------
def rank_histogram(
    ensemble: np.ndarray,
    observation: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Talagrand rank histogram: counts of the observation's rank among M members.

    Returns an integer array of length M+1. A calibrated ensemble gives a flat
    histogram (P1 M188). Ties are broken by random assignment within the tie
    block, following Hamill (2001, P1 ref. 64).
    """
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observation = np.asarray(observation, dtype=np.float64)
    m = ensemble.shape[0]
    sel = np.isfinite(observation)
    if mask is not None:
        sel &= np.asarray(mask, dtype=bool)
    for member in ensemble:
        sel &= np.isfinite(member)

    obs = observation[sel]
    members = ensemble[:, sel]  # (M, K)
    below = np.sum(members < obs[None, :], axis=0)
    equal = np.sum(members == obs[None, :], axis=0)
    rng = np.random.default_rng(0)
    ranks = below + (rng.random(below.shape) * (equal + 1)).astype(int)
    ranks = np.clip(ranks, 0, m)
    return np.bincount(ranks, minlength=m + 1)


# ---------------------------------------------------------------------------
# Linear eccentricity -- P1 Eq. (10)-(12), M189  (EXPERIMENTAL)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EccentricityResult:
    linear_eccentricity: float  # ecc_l = sqrt(a^2 - b^2)   P1 Eq. (10)
    eccentricity: float         # ecc   = sqrt(1 - b^2/a^2)  P1 Eq. (11)
    size: float                 # size  = sqrt(a * b)        P1 Eq. (12)
    major_axis: float
    minor_axis: float
    orientation_deg: float


def linear_eccentricity(field: np.ndarray, max_lag_px: int = 15) -> EccentricityResult:
    r"""P1 Eq. (10)-(12): anisotropy of a rainfield via spatial autocorrelation.

    EXPERIMENTAL / first-iteration implementation (P1 M189 was marked optional
    for the first pass). Steps, following P1 "Evaluation":

    1. 2D autocorrelation of the field for lags 0..``max_lag_px`` in x and y.
    2. Fit an ellipse to the 0.5 Pearson-correlation contour (here: the
       second-moment ellipse of the region where autocorr >= 0.5).
    3. a, b = major / minor axis lengths -> Eqs. (10)-(12).

    The contour step in P1 uses an explicit iso-line; the moment-ellipse used
    here is a robust approximation. Flagged in docs/EVALUATION_PROTOCOL.md.
    """
    field = np.asarray(field, dtype=np.float64)
    field = np.nan_to_num(field - np.nanmean(field), nan=0.0)
    fft = np.fft.fft2(field)
    acf = np.fft.ifft2(fft * np.conj(fft)).real
    acf = np.fft.fftshift(acf)
    acf /= acf.max()

    cy, cx = np.array(acf.shape) // 2
    window = acf[cy - max_lag_px: cy + max_lag_px + 1, cx - max_lag_px: cx + max_lag_px + 1]
    yy, xx = np.indices(window.shape)
    y0, x0 = np.array(window.shape) / 2.0
    region = window >= 0.5
    if region.sum() < 3:
        return EccentricityResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    ys = (yy[region] - y0)
    xs = (xx[region] - x0)
    cov = np.cov(np.stack([xs, ys]))
    evals, evecs = np.linalg.eigh(cov)
    minor, major = np.sqrt(np.maximum(evals, 0.0))
    if major < minor:
        major, minor = minor, major
    a, b = float(major), float(minor)
    ecc_l = float(np.sqrt(max(a * a - b * b, 0.0)))
    ecc = float(np.sqrt(max(1.0 - (b * b) / (a * a), 0.0))) if a > 0 else 0.0
    size = float(np.sqrt(a * b))
    major_vec = evecs[:, int(np.argmax(evals))]
    orientation = float(np.degrees(np.arctan2(major_vec[1], major_vec[0])))
    return EccentricityResult(ecc_l, ecc, size, a, b, orientation)
