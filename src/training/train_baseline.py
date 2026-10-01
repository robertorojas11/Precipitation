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
    load_norm_stats,
    loss_mask,
    masked_mean,
    seed_everything,
    target_moments,
    to_device,
    write_run_metadata,
)
from src.training.dataset import PreparedHourlyDataset
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
    eta_lambda: float = 0.0
    eta_tau: float = 0.95
    eta_n_q: int = 100
    eta_pretrain_steps: int = 0
    eta_refresh_every: int = 200
    eta_pool_size: int = 512


def _forward(model: DeterministicResNet, inputs: torch.Tensor, invariants: torch.Tensor) -> torch.Tensor:
    return model(inputs, invariants)


def _erm_loss(prediction_mm: torch.Tensor, target_mm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mean(torch.abs(prediction_mm - target_mm), mask)


def validate(model, loader, device, moments) -> dict[str, float]:
    model.eval()
    metrics = PooledMetrics()
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            prediction_mm = _forward(model, batch["inputs"], batch["invariants"])
            target_mm = denormalize_to_mm(batch["target"], moments)
            metrics.update(prediction_mm, target_mm, loss_mask(batch))
    return metrics.compute()


def train(
    config: BaselineTrainingConfig, *, prepared_root: Path | None = None, output_root: Path | None = None,
) -> Path:
    seed_everything(config.seed)
    device = torch.device(config.device if config.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    stats = load_norm_stats(prepared_root)
    moments = target_moments(stats)

    train_dataset = PreparedHourlyDataset(config.region, "train", prepared_root=prepared_root)
    val_dataset = PreparedHourlyDataset(config.region, "val", prepared_root=prepared_root)
    data_generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, generator=data_generator,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

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
        model.train()
        epoch_loss = 0.0
        batches = 0
        for batch in train_loader:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            def forward_fn(inp: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
                return _forward(model, inp, inv)

            eta.maybe_refresh_tail_index(forward_fn)
            prediction_mm = _forward(model, batch["inputs"], batch["invariants"])
            target_mm = denormalize_to_mm(batch["target"], moments)
            loss = _erm_loss(prediction_mm, target_mm, loss_mask(batch))
            eta_term = eta.term(forward_fn)
            if eta_term is not None:
                loss = loss + eta_config.lambda_ * eta_term
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            eta.step()
            epoch_loss += loss.item()
            batches += 1

        validation = validate(model, val_loader, device, moments)
        row = {"epoch": epoch, "train_loss": epoch_loss / max(batches, 1), **{f"val_{k}": v for k, v in validation.items()}}
        append_history(run_dir, row)
        logger.info("baseline epoch=%d train_loss=%.4f val_r2=%.4f val_mae=%.4f",
                    epoch, row["train_loss"], validation["r2"], validation["mae"])

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
    parser.add_argument("--eta-lambda", type=float, default=0.0)
    args = parser.parse_args()
    config = BaselineTrainingConfig(
        region=args.region, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, patience=args.patience, seed=args.seed,
        device=args.device, run_name=args.run_name, eta_lambda=args.eta_lambda,
    )
    print(train(config))


if __name__ == "__main__":
    main()
