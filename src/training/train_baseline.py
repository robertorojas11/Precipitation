r"""Stage 6a -- train the deterministic ResNet baseline (PASO 5.4 of
docs/decisions/0002-implementacion-v2.md), the "third comparison point"
alongside Candidate A (GAN, ``train_gan.py``) and Candidate B (regression +
residual diffusion, ``train_diffusion.py``). Full narrative:
docs/stages_v2/06-train.md.

**Loss.** ``src/models/deterministic_resnet.py``'s own docstring: "this is
the deterministic reference point for which CRPS reduces to MAE" (P1 M181,
P2 M247) -- so the ERM objective here is a plain masked L1 (MAE) in
physical mm/hr, not the legacy pipeline's heavier multi-term loss
(occurrence/amount/gradient/multiscale), which was tuned for the daily
CHIRPS/Oya task and is not part of this project's P1/P2/P3/P4 contract.

**Optional eta-learning** (``src/training/eta_integration.py``, PASO 1.2):
disabled by default (``--eta-lambda 0``); when enabled, adds the W1 tail
term to this same masked-L1 loss, exactly as the eta-learning paper
validates for a plain deterministic estimator, not only generative models.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.models.deterministic_resnet import DeterministicResNet, ResNetConfig
from src.training.common import (
    PooledMetrics,
    append_history,
    denormalize_to_mm,
    init_run_dir,
    input_moments,
    load_norm_stats,
    loss_mask,
    masked_mean,
    seed_everything,
    target_moments,
    to_device,
    total_precip_input_mm,
    write_run_metadata,
)
from src.training.dataset import DayGroupedSampler, PreparedHourlyDataset
from src.training.eta_integration import EtaConfig, EtaRegularizer, build_eta_pools
from src.utils.logging import configure_logging

logger = configure_logging(name="precipitation.train_baseline")


@dataclass
class BaselineTrainingConfig:
    region: str
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    base_width: int = 64
    depth: int = 8
    patience: int = 10
    seed: int = 42
    num_workers: int = 0
    device: str = "cuda"
    run_name: str | None = None
    preload_dataset: bool = True
    use_amp: bool = True
    eta_lambda: float = 0.0
    eta_tau: float = 0.95
    eta_n_q: int = 100
    eta_pretrain_steps: int = 0
    eta_refresh_every: int = 200
    eta_pool_size: int = 512
    rain_weight: float = 1.0
    wet_threshold_mm: float = 0.1


def _forward(
    model: DeterministicResNet,
    inputs: torch.Tensor,
    invariants: torch.Tensor,
    total_precip_input: torch.Tensor | None = None,
) -> torch.Tensor:
    return model(inputs, invariants, total_precip_input)


def _erm_loss(
    prediction_mm: torch.Tensor,
    target_mm: torch.Tensor,
    mask: torch.Tensor,
    *,
    rain_weight: float = 1.0,
    wet_threshold_mm: float = 0.1,
) -> torch.Tensor:
    """Masked L1, optionally up-weighting wet pixels (``target_mm >
    wet_threshold_mm``) by ``rain_weight``.

    **Why.** Measured on real training batches: ~94-95% of pixels are dry
    (pacifico wet_frac=0.0565, atlantico-golfo wet_frac=0.0910, threshold
    0.1mm/hr -- the same lowest meFSS threshold already used elsewhere in
    this project, ``FSS_THRESHOLDS_MM_H[0]``). Plain unweighted L1 lets the
    dry majority's constant-sign gradient dominate every step, and since
    AdamW normalises each parameter's step by its own gradient RMS, even a
    tiny-but-consistent gradient produces a near-full-size step regardless
    of magnitude -- so the network converges within 1-2 epochs to a
    near-input-invariant, near-zero-everywhere prediction (confirmed via
    direct instrumentation on ``baseline_run5``: trunk output varied only
    4.59e-5..4.66e-5 mm/hr across wildly different random inputs) and stays
    there, since escaping costs more loss on the dry majority than it
    recovers on the rare wet pixels. Up-weighting wet pixels by
    ``(1-wet_frac)/wet_frac`` (the ``rain_weight`` CLI default at launch)
    rebalances the two populations' aggregate gradient contribution instead
    of changing the loss family -- still a masked L1, so the "CRPS reduces
    to MAE" P1/P2 reference-baseline contract is preserved for dry and wet
    pixels taken separately, just not pooled at their raw (imbalanced)
    frequencies.
    """
    weight = mask.to(prediction_mm.dtype)
    if weight.dim() == prediction_mm.dim() - 1:
        weight = weight.unsqueeze(1)
    if rain_weight != 1.0:
        is_wet = (target_mm > wet_threshold_mm).to(prediction_mm.dtype)
        weight = weight * (1.0 + (rain_weight - 1.0) * is_wet)
    return masked_mean(torch.abs(prediction_mm - target_mm), weight)


def validate(model, loader, device, moments, cp_moments, lsp_moments, *, use_amp: bool = False) -> dict[str, float]:
    model.eval()
    metrics = PooledMetrics()
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            total_precip_input = total_precip_input_mm(batch["inputs"], cp_moments, lsp_moments)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                prediction_mm = _forward(model, batch["inputs"], batch["invariants"], total_precip_input)
            target_mm = denormalize_to_mm(batch["target"], moments)
            metrics.update(prediction_mm.float(), target_mm, loss_mask(batch))
    return metrics.compute()


def train(
    config: BaselineTrainingConfig, *, prepared_root: Path | None = None, output_root: Path | None = None,
) -> Path:
    seed_everything(config.seed)
    device = torch.device(config.device if config.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    stats = load_norm_stats(prepared_root)
    moments = target_moments(stats)
    cp_moments, lsp_moments = input_moments(stats)

    train_dataset = PreparedHourlyDataset(
        config.region, "train", prepared_root=prepared_root, preload=config.preload_dataset,
    )
    val_dataset = PreparedHourlyDataset(
        config.region, "val", prepared_root=prepared_root, preload=config.preload_dataset,
    )
    train_sampler = DayGroupedSampler(train_dataset, shuffle=True, seed=config.seed)
    pin_memory = device.type == "cuda"
    persistent_workers = config.num_workers > 0
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, sampler=train_sampler, num_workers=config.num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers,
    )

    model_config = ResNetConfig(base_width=config.base_width, depth=config.depth)
    model = DeterministicResNet(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    eta_config = EtaConfig(
        lambda_=config.eta_lambda, tau=config.eta_tau, n_q=config.eta_n_q,
        pretrain_steps=config.eta_pretrain_steps, refresh_every=config.eta_refresh_every,
        reference_pool_size=config.eta_pool_size, auxiliary_pool_size=config.eta_pool_size,
    )
    eta = EtaRegularizer(eta_config)
    if eta_config.enabled:
        reference_pool, aux_inputs, aux_invariants = build_eta_pools(
            train_dataset, eta_config, device=device,
            denormalize_target_fn=lambda t: denormalize_to_mm(t, moments),
            dataloader_generator=torch.Generator().manual_seed(config.seed),
        )
        eta.set_reference_pool(reference_pool)
        eta.set_auxiliary_inputs(aux_inputs, aux_invariants)

    run_dir = init_run_dir("deterministic_resnet", config.region, run_name=config.run_name, output_root=output_root)
    write_run_metadata(run_dir, config, stats=stats)

    best_r2 = -float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, config.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        batches = 0
        for batch in train_loader:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            def forward_fn(inp: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
                return _forward(model, inp, inv, total_precip_input_mm(inp, cp_moments, lsp_moments))

            eta.maybe_refresh_tail_index(forward_fn)
            total_precip_input = total_precip_input_mm(batch["inputs"], cp_moments, lsp_moments)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config.use_amp):
                prediction_mm = _forward(model, batch["inputs"], batch["invariants"], total_precip_input)
                target_mm = denormalize_to_mm(batch["target"], moments)
                loss = _erm_loss(
                    prediction_mm, target_mm, loss_mask(batch),
                    rain_weight=config.rain_weight, wet_threshold_mm=config.wet_threshold_mm,
                )
            eta_term = eta.term(forward_fn)
            if eta_term is not None:
                loss = loss + eta_config.lambda_ * eta_term
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            eta.step()
            epoch_loss += loss.item()
            batches += 1

        validation = validate(model, val_loader, device, moments, cp_moments, lsp_moments, use_amp=config.use_amp)
        row = {"epoch": epoch, "train_loss": epoch_loss / max(batches, 1), **{f"val_{k}": v for k, v in validation.items()}}
        append_history(run_dir, row)
        logger.info("baseline epoch=%d train_loss=%.4f val_r2=%.4f val_mae=%.4f val_log_mae=%.4f val_clipped=%d",
                    epoch, row["train_loss"], validation["r2"], validation["mae"],
                    validation["log_mae"], validation["clipped_count"])

        checkpoint = {
            "model_state": model.state_dict(), "model_config": asdict(model_config),
            "optimizer_state": optimizer.state_dict(), "epoch": epoch,
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if validation["r2"] > best_r2:
            best_r2 = validation["r2"]
            epochs_without_improvement = 0
            torch.save(checkpoint, run_dir / "best.pt")
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break
    return run_dir


def main() -> None:
    from src.data_preprocessing import regions as region_defs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, choices=region_defs.list_regions())
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-preload", dest="preload_dataset", action="store_false", default=True)
    parser.add_argument("--no-amp", dest="use_amp", action="store_false", default=True)
    parser.add_argument("--eta-lambda", type=float, default=0.0)
    parser.add_argument("--rain-weight", type=float, default=1.0)
    parser.add_argument("--wet-threshold-mm", type=float, default=0.1)
    args = parser.parse_args()
    config = BaselineTrainingConfig(
        region=args.region, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, patience=args.patience, seed=args.seed,
        device=args.device, run_name=args.run_name, num_workers=args.num_workers,
        preload_dataset=args.preload_dataset, use_amp=args.use_amp, eta_lambda=args.eta_lambda,
        rain_weight=args.rain_weight, wet_threshold_mm=args.wet_threshold_mm,
    )
    print(train(config))


if __name__ == "__main__":
    main()
