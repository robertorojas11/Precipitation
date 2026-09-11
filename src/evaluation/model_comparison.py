"""Cross-model comparison harness (PASO 8 of
docs/decisions/0002-implementacion-v2.md): "Comparacion explicita de los
tres modelos (determinista, GAN, regresion+difusion residual), con y sin
regularizacion eta-learning activada, para poder responder: cuanto aporta
cada pieza por separado?"

This module does not run training itself (no prepared dataset exists yet,
see docs/architecture_v3.md); it aggregates whatever metrics a training/
evaluation run already produced (e.g. from
``src.evaluation.probabilistic_metrics`` / ``deterministic_metrics``) into
the comparison PASO 8 asks for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MODEL_NAMES = ("deterministic", "gan", "regression_diffusion")
# Metrics where a smaller value is better (used to sign "improvement").
LOWER_IS_BETTER = frozenset({"crps", "rmse", "mae", "fss_gap", "percentile_mae_95", "percentile_mae_99"})


@dataclass(frozen=True)
class ModelRunResult:
    model_name: str
    eta_enabled: bool
    metrics: dict[str, float] = field(default_factory=dict)


def _key(model_name: str, eta_enabled: bool) -> tuple[str, bool]:
    return (model_name, eta_enabled)


def build_comparison_table(results: list[ModelRunResult]) -> list[dict]:
    """One row per (model, eta_enabled), sorted by model name then eta flag."""
    rows = [
        {"model": r.model_name, "eta_enabled": r.eta_enabled, **r.metrics}
        for r in sorted(results, key=lambda r: (r.model_name, r.eta_enabled))
    ]
    return rows


def eta_learning_contribution(
    results: list[ModelRunResult], metrics: list[str]
) -> dict[str, dict[str, float]]:
    """Per model, per metric: ``value(eta=False) - value(eta=True)``, signed
    so a **positive** number always means "eta-learning helped" (accounting
    for whether the metric is lower-is-better or higher-is-better).

    Answers PASO 8's "how much does each piece contribute separately" by
    isolating the eta-learning term's effect for a fixed model architecture.
    """
    by_key = {_key(r.model_name, r.eta_enabled): r.metrics for r in results}
    out: dict[str, dict[str, float]] = {}
    for model_name in {r.model_name for r in results}:
        without = by_key.get(_key(model_name, False))
        with_ = by_key.get(_key(model_name, True))
        if without is None or with_ is None:
            continue
        deltas: dict[str, float] = {}
        for metric in metrics:
            if metric not in without or metric not in with_:
                continue
            raw_delta = without[metric] - with_[metric]
            deltas[metric] = raw_delta if metric in LOWER_IS_BETTER else -raw_delta
        out[model_name] = deltas
    return out


def render_markdown_table(rows: list[dict]) -> str:
    """Minimal Markdown table renderer for the PASO 12 report stage."""
    if not rows:
        return ""
    columns = list(rows[0].keys())
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, separator]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)
