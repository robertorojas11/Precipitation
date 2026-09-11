r"""Continuity metrics carried over from the deterministic pipeline, plus the
full-field RMSE/SSIM diagnostic PASO 8 requires to price the eta-learning
regularizer.

Per docs/decisions/0002-implementacion-v2.md PASO 8: keep R2, RMSE, MAE and
block bootstrap as *complementary* metrics alongside the new probabilistic
ones -- do not drop them. These mirror
``legacy/deterministic_unet/evaluate.py``'s ``PooledMetrics`` and reuse
``src.data_preprocessing.quality.masked_r2``.
"""

from __future__ import annotations

import numpy as np

from src.data_preprocessing.quality import masked_r2


def _valid(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray | None):
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    sel = np.isfinite(pred) & np.isfinite(obs)
    if mask is not None:
        sel &= np.asarray(mask, dtype=bool)
    return pred[sel], obs[sel]


def rmse(pred, obs, mask=None) -> float:
    p, o = _valid(pred, obs, mask)
    return float(np.sqrt(np.mean((p - o) ** 2))) if p.size else float("nan")


def mae(pred, obs, mask=None) -> float:
    p, o = _valid(pred, obs, mask)
    return float(np.mean(np.abs(p - o))) if p.size else float("nan")


def bias(pred, obs, mask=None) -> float:
    r"""P1 Eq. (13):  BIAS = (Y - X) / Y

    with X = mean predicted amount, Y = mean observed amount (P1 M190 flags the
    sign convention: numerator is observed - predicted, denominator observed).
    """
    p, o = _valid(pred, obs, mask)
    if o.size == 0 or o.mean() == 0:
        return float("nan")
    return float((o.mean() - p.mean()) / o.mean())


def pooled_scores(pred, obs, mask=None) -> dict[str, float]:
    p, o = _valid(pred, obs, mask)
    return {
        "r2": masked_r2(p, o, np.ones_like(p, dtype=bool)) if p.size >= 2 else float("nan"),
        "rmse": rmse(p, o),
        "mae": mae(p, o),
        "bias": bias(p, o),
        "valid_pixels": int(p.size),
    }


def block_bootstrap_r2(
    pred: np.ndarray,
    obs: np.ndarray,
    block_id: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Month/time-block bootstrap CI for pooled R2 (legacy protocol).

    ``block_id`` labels each pixel with its resampling block (e.g. year-month).
    Blocks are resampled with replacement; R2 is recomputed on the pooled
    resample. Returns point estimate and the 2.5 / 97.5 percentiles.
    """
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    block_id = np.asarray(block_id)
    sel = np.isfinite(pred) & np.isfinite(obs)
    if mask is not None:
        sel &= np.asarray(mask, dtype=bool)
    pred, obs, block_id = pred[sel], obs[sel], block_id[sel]

    blocks = np.unique(block_id)
    index_by_block = {b: np.where(block_id == b)[0] for b in blocks}
    rng = np.random.default_rng(seed)

    def _r2(idx):
        o, p = obs[idx], pred[idx]
        denom = np.sum((o - o.mean()) ** 2)
        return np.nan if denom <= 0 else 1.0 - np.sum((o - p) ** 2) / denom

    point = float(_r2(np.arange(obs.size)))
    samples = []
    for _ in range(n_resamples):
        chosen = rng.choice(blocks, size=blocks.size, replace=True)
        idx = np.concatenate([index_by_block[b] for b in chosen])
        samples.append(_r2(idx))
    samples = np.array(samples, dtype=np.float64)
    return {
        "r2": point,
        "r2_lower95": float(np.nanpercentile(samples, 2.5)),
        "r2_upper95": float(np.nanpercentile(samples, 97.5)),
        "n_resamples": n_resamples,
    }


# ---------------------------------------------------------------------------
# Full-field RMSE/SSIM -- PASO 8 secondary diagnostic for the eta-learning cost
# ---------------------------------------------------------------------------
def structural_similarity(field_a: np.ndarray, field_b: np.ndarray, *, data_range: float | None = None) -> float:
    """SSIM between two 2D fields (single global window, i.e. the
    "full-field" variant PASO 8 asks for, not a sliding-window map).

    Standard single-window formula (Wang et al. 2004):

        SSIM = (2*mu_a*mu_b + c1)(2*cov_ab + c2) / ((mu_a^2+mu_b^2+c1)(var_a+var_b+c2))

    with ``c1 = (0.01*L)^2``, ``c2 = (0.03*L)^2``, ``L = data_range``. Only
    finite, jointly-valid pixels are used.
    """
    a = np.asarray(field_a, dtype=np.float64)
    b = np.asarray(field_b, dtype=np.float64)
    sel = np.isfinite(a) & np.isfinite(b)
    a, b = a[sel], b[sel]
    if a.size < 2:
        return float("nan")
    if data_range is None:
        data_range = float(max(a.max(), b.max()) - min(a.min(), b.min()) or 1.0)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov_ab = np.mean((a - mu_a) * (b - mu_b))
    numerator = (2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)
    denominator = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float(numerator / denominator)


def full_field_rmse_ssim(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """RMSE + SSIM over the full field, the pair PASO 8 uses to price the
    eta-learning regularizer's point-fidelity cost against its tail benefit."""
    p, o = _valid(pred, obs, mask)
    return {"rmse": rmse(pred, obs, mask), "ssim": structural_similarity(p, o)}


def eta_learning_cost_report(
    baseline_scores: dict[str, float],
    eta_scores: dict[str, float],
    *,
    expected_rmse_increase_pct: float = 8.0,
    flag_multiplier: float = 2.0,
) -> dict[str, float | bool]:
    """Quantify the RMSE/SSIM cost of turning on the eta-learning
    regularizer (PASO 8): "se espera un incremento modesto, en linea con el
    ~8% de RMSE reportado en el paper de referencia -- si el incremento
    observado es mucho mayor, es senal de que algo no esta bien calibrado."

    ``baseline_scores``/``eta_scores`` are ``{"rmse": ..., "ssim": ...}``
    dicts (e.g. from :func:`full_field_rmse_ssim`), with vs. without the
    eta-learning term. Flags ``miscalibrated=True`` when the observed RMSE
    increase exceeds ``flag_multiplier`` times the paper's own reference
    (Chang & Sapsis 2026: full-grid RMSE rose 8.13% when the eta-learning
    term was activated, SI Table 1).
    """
    base_rmse = baseline_scores["rmse"]
    eta_rmse = eta_scores["rmse"]
    rmse_increase_pct = 100.0 * (eta_rmse - base_rmse) / base_rmse if base_rmse else float("nan")
    ssim_decrease = baseline_scores.get("ssim", float("nan")) - eta_scores.get("ssim", float("nan"))
    miscalibrated = rmse_increase_pct > flag_multiplier * expected_rmse_increase_pct
    return {
        "rmse_increase_pct": rmse_increase_pct,
        "ssim_decrease": ssim_decrease,
        "expected_rmse_increase_pct": expected_rmse_increase_pct,
        "miscalibrated": bool(miscalibrated),
    }
