r"""Stage 6c -- train Candidate B, regression + EDM residual diffusion
(``src/models/{regression_stage,diffusion_residual}.py``; PASO 5.2 of
docs/decisions/0002-implementacion-v2.md). Full narrative:
docs/stages_v2/06-train.md.

Two sequential sub-stages, run as two separate invocations of this module
(``--stage regression`` then ``--stage diffusion --stage1-checkpoint
<path/to/best.pt>``), mirroring Avila (2026) Eq. (1)'s decomposition
``y = E[y|x] + (y - E[y|x])``: stage 1 must finish training (and be frozen)
before stage 2 has a conditional mean to compute residuals against.

**Representation choice, documented explicitly.** Both sub-stages operate
in the *same normalized log-space* as ``target_norm`` (stage 4's output),
not physical mm/hr -- unlike the deterministic baseline and the GAN
generator, whose Softplus output heads are architecturally committed to a
non-negative physical representation. ``RegressionUNet`` is built here with
``nonnegative_output=False`` precisely because this residual-diffusion
decomposition needs a *signed* conditional mean and a *signed* residual
(the docstring in ``regression_stage.py`` anticipates exactly this case:
"a future residual target (e.g. a signed log-anomaly)"). Physical units are
recovered only for validation metrics and for the final evaluation stage,
via ``denormalize_to_mm``.

**Masked EDM loss, a documented deviation from
``src.models.diffusion_residual.edm_loss``.** That function's own
``.mean()`` reduction is unmasked; stage 4 leaves invalid pixels as NaN
(sanitized to 0 only at the network-input boundary, per
``src/training/dataset.py``), so training the denoiser to reconstruct a
residual computed from a fabricated zero-filled target at invalid pixels
would silently use invalid data in a loss. ``masked_edm_loss`` below
reproduces ``edm_loss``'s exact per-pixel weighting formula (Avila 2026 Eq.
5) and only changes the final reduction to ``masked_mean`` -- the
preconditioned denoiser (``EDMPrecond``) itself is reused unmodified.

**Eta-learning is wired onto stage 1 only.** Stage 2's observable would
require running the full Heun sampler (18 network evaluations by default)
inside the training loop just to get one sample to regularize -- a real
cost, not a design objection -- so it is documented here as a known
limitation (docs/stages_v2/06-train.md) rather than silently skipped.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.models.diffusion_residual import EDMConfig, EDMPrecond, HeunSamplerConfig, heun_sample
from src.models.regression_stage import RegressionStageConfig, RegressionUNet
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

logger = configure_logging(name="precipitation.train_diffusion")

REGRESSION_STAGE_IN_CHANNELS = 4  # era5 cp+lsp (2) + invariants (2)
DIFFUSION_CONDITIONING_CHANNELS = 5  # era5 cp+lsp (2) + invariants (2) + stage-1 mean (1)


@dataclass
class RegressionStageTrainingConfig:
    region: str
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 2e-4    # Avila 2026 Table 5
    weight_decay: float = 1e-5     # Avila 2026 Table 5
    base_width: int = 64
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


@dataclass
class DiffusionStageTrainingConfig:
    region: str
    stage1_checkpoint: str
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    base_width: int = 64
    patience: int = 10
    seed: int = 42
    num_workers: int = 0
    device: str = "cuda"
    run_name: str | None = None
    sampler_steps_for_validation: int = 18


def _regression_config(base_width: int) -> RegressionStageConfig:
    return RegressionStageConfig(
        in_channels=REGRESSION_STAGE_IN_CHANNELS, out_channels=1, base_width=base_width,
        nonnegative_output=False,
    )


def _stage1_forward(model: RegressionUNet, inputs: torch.Tensor, invariants: torch.Tensor) -> torch.Tensor:
    return model(torch.cat([inputs, invariants], dim=1))


def validate_regression(model, loader, device, moments) -> dict[str, float]:
    model.eval()
    metrics = PooledMetrics()
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            prediction_norm = _stage1_forward(model, batch["inputs"], batch["invariants"])
            metrics.update(
                denormalize_to_mm(prediction_norm, moments), denormalize_to_mm(batch["target"], moments),
                loss_mask(batch),
            )
    return metrics.compute()


def train_regression_stage(
    config: RegressionStageTrainingConfig, *, prepared_root: Path | None = None, output_root: Path | None = None,
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

    model_config = _regression_config(config.base_width)
    model = RegressionUNet(model_config).to(device)
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

    run_dir = init_run_dir("diffusion_stage1_regression", config.region, run_name=config.run_name, output_root=output_root)
    write_run_metadata(run_dir, config, stats=stats, extra={"model_config": asdict(model_config)})

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
                return denormalize_to_mm(_stage1_forward(model, inp, inv), moments)

            eta.maybe_refresh_tail_index(forward_fn)
            prediction_norm = _stage1_forward(model, batch["inputs"], batch["invariants"])
            mask = loss_mask(batch)
            loss = masked_mean((prediction_norm - batch["target"]) ** 2, mask)
            eta_term = eta.term(forward_fn)
            if eta_term is not None:
                loss = loss + eta_config.lambda_ * eta_term
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            eta.step()
            epoch_loss += loss.item()
            batches += 1

        validation = validate_regression(model, val_loader, device, moments)
        row = {"epoch": epoch, "train_loss": epoch_loss / max(batches, 1), **{f"val_{k}": v for k, v in validation.items()}}
        append_history(run_dir, row)
        logger.info("diffusion-stage1 epoch=%d train_loss=%.4f val_r2=%.4f", epoch, row["train_loss"], validation["r2"])

        checkpoint = {"model_state": model.state_dict(), "model_config": asdict(model_config), "epoch": epoch}
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


def masked_edm_loss(
    denoiser: EDMPrecond, clean_residual: torch.Tensor, conditioning: torch.Tensor, mask: torch.Tensor,
    *, generator: torch.Generator | None = None,
) -> torch.Tensor:
    """``edm_loss`` (Avila 2026 Eq. 5) with a masked reduction -- see module
    docstring for why. ``mask``: (B, H, W), broadcast over the channel axis.
    """
    config = denoiser.config
    batch = clean_residual.shape[0]
    device, dtype = clean_residual.device, clean_residual.dtype
    log_sigma = config.p_mean + config.p_std * torch.randn(batch, device=device, dtype=dtype, generator=generator)
    sigma = torch.exp(log_sigma)
    noise = torch.randn(clean_residual.shape, device=device, dtype=dtype, generator=generator)
    sigma_b = sigma.reshape(-1, 1, 1, 1)
    noisy_residual = clean_residual + sigma_b * noise
    weight = (sigma_b ** 2 + config.sigma_data ** 2) / (sigma_b * config.sigma_data) ** 2
    prediction = denoiser(noisy_residual, sigma, conditioning)
    per_pixel = weight * (prediction - clean_residual) ** 2
    return masked_mean(per_pixel, mask.unsqueeze(1))


def _load_stage1(checkpoint_path: str, device: torch.device) -> RegressionUNet:
    payload = torch.load(checkpoint_path, map_location=device)
    model = RegressionUNet(RegressionStageConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _denoiser_raw_network_config(base_width: int) -> RegressionStageConfig:
    # SimpleNoiseConditionedUNet's `in_channels` must count EDMPrecond's
    # *entire* network_input -- the scaled noisy residual (1 channel) plus
    # the conditioning tensor -- not the conditioning tensor alone (it adds
    # its own +1 internally only for the noise-level embedding channel).
    return RegressionStageConfig(
        in_channels=1 + DIFFUSION_CONDITIONING_CHANNELS, out_channels=1, base_width=base_width,
        nonnegative_output=False,
    )


def _build_denoiser(base_width: int, device: torch.device) -> tuple[EDMPrecond, RegressionStageConfig]:
    from src.models.diffusion_residual import SimpleNoiseConditionedUNet

    raw_config = _denoiser_raw_network_config(base_width)
    raw = SimpleNoiseConditionedUNet(raw_config)
    return EDMPrecond(raw, EDMConfig()).to(device), raw_config


def validate_diffusion(denoiser, stage1, loader, device, moments, sampler_config) -> dict[str, float]:
    denoiser.eval()
    metrics = PooledMetrics()
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            stage1_mean = _stage1_forward(stage1, batch["inputs"], batch["invariants"])
            conditioning = torch.cat([batch["inputs"], batch["invariants"], stage1_mean], dim=1)
            residual_sample = heun_sample(denoiser, conditioning, stage1_mean.shape, sampler_config)
            prediction_norm = stage1_mean + residual_sample
            metrics.update(
                denormalize_to_mm(prediction_norm, moments), denormalize_to_mm(batch["target"], moments),
                loss_mask(batch),
            )
    return metrics.compute()


def train_diffusion_stage(
    config: DiffusionStageTrainingConfig, *, prepared_root: Path | None = None, output_root: Path | None = None,
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

    stage1 = _load_stage1(config.stage1_checkpoint, device)
    denoiser, denoiser_config = _build_denoiser(config.base_width, device)
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    sampler_config = HeunSamplerConfig(num_steps=config.sampler_steps_for_validation)

    run_dir = init_run_dir("diffusion_stage2_residual", config.region, run_name=config.run_name, output_root=output_root)
    write_run_metadata(run_dir, config, stats=stats, extra={"stage1_checkpoint": config.stage1_checkpoint})

    best_r2 = -float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, config.epochs + 1):
        denoiser.train()
        epoch_loss = 0.0
        batches = 0
        for batch in train_loader:
            batch = to_device(batch, device)
            with torch.no_grad():
                stage1_mean = _stage1_forward(stage1, batch["inputs"], batch["invariants"])
            clean_residual = batch["target"] - stage1_mean
            conditioning = torch.cat([batch["inputs"], batch["invariants"], stage1_mean], dim=1)

            optimizer.zero_grad(set_to_none=True)
            loss = masked_edm_loss(denoiser, clean_residual, conditioning, loss_mask(batch))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1

        validation = validate_diffusion(denoiser, stage1, val_loader, device, moments, sampler_config)
        row = {"epoch": epoch, "train_loss": epoch_loss / max(batches, 1), **{f"val_{k}": v for k, v in validation.items()}}
        append_history(run_dir, row)
        logger.info("diffusion-stage2 epoch=%d train_loss=%.4f val_r2=%.4f", epoch, row["train_loss"], validation["r2"])

        checkpoint = {
            "denoiser_state": denoiser.state_dict(), "denoiser_config": asdict(denoiser_config), "epoch": epoch,
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
    parser.add_argument("--stage", required=True, choices=["regression", "diffusion"])
    parser.add_argument("--region", required=True, choices=region_defs.list_regions())
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name")
    parser.add_argument("--eta-lambda", type=float, default=0.0)
    parser.add_argument("--stage1-checkpoint")
    args = parser.parse_args()

    if args.stage == "regression":
        config = RegressionStageTrainingConfig(
            region=args.region, epochs=args.epochs, batch_size=args.batch_size,
            seed=args.seed, device=args.device, run_name=args.run_name, eta_lambda=args.eta_lambda,
        )
        print(train_regression_stage(config))
    else:
        if not args.stage1_checkpoint:
            raise SystemExit("--stage1-checkpoint is required for --stage diffusion")
        config = DiffusionStageTrainingConfig(
            region=args.region, stage1_checkpoint=args.stage1_checkpoint, epochs=args.epochs,
            batch_size=args.batch_size, seed=args.seed, device=args.device, run_name=args.run_name,
        )
        print(train_diffusion_stage(config))


if __name__ == "__main__":
    main()
