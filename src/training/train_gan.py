r"""Stage 6b -- train Candidate A, the P1-inspired conditional GAN
(``src/models/{generator,discriminator}.py`` +
``src/training/losses_gan.py``; PASO 5.1 of
docs/decisions/0002-implementacion-v2.md). Full narrative:
docs/stages_v2/06-train.md.

**Optimizer hyperparameters are paper values, not tuned defaults** (matrix
``docs/audit/matriz_trazabilidad_metodologica.md`` M155/M156/M158, P1
"Training and model selection"): generator AdamW lr=1e-4, betas=(0.0,
0.999); discriminator AdamW lr=2e-4, betas=(0.0, 0.5); batch size 9.

**Masking the discriminator's real/fake fields -- a documented deviation.**
P1's discriminator operates on full continuous HR grids; this project's
grids include out-of-basin/invalid pixels that must never be *used* as if
they were real information in a loss (docs/stages_v2/README.md's shared
experimental contract). A per-pixel mask cannot simply be applied only to
"real" or only to "fake" without biasing the discriminator toward using the
mask pattern itself as a tell -- so both the real and the generated field
are multiplied by the identical ``loss_mask(batch)`` before either reaches
the discriminator. Since the exact same zero pattern appears in both real
and fake at the same spatial locations, the discriminator cannot learn
anything about rain from those pixels; it can only ever compare the
genuinely-valid ones. The ensemble L1 term (Eq. 3) uses the same mask
directly on valid pixels, as everywhere else in this codebase.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.models.discriminator import ConditionalDiscriminator, DiscriminatorConfig
from src.models.generator import ConditionalGenerator, GeneratorConfig
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
from src.training.dataset import DayGroupedSampler, PreparedHourlyDataset
from src.training.eta_integration import EtaConfig, EtaRegularizer, build_eta_pools
from src.training.losses_gan import discriminator_loss, ensemble_l1_loss, generator_adversarial_loss
from src.utils.logging import configure_logging

logger = configure_logging(name="precipitation.train_gan")

N_ENSEMBLE_MEMBERS = 3  # P1 Eq. 3: N >= 3, paper uses exactly 3


@dataclass
class GanTrainingConfig:
    region: str
    epochs: int = 50
    batch_size: int = 9        # P1 M158
    generator_lr: float = 1e-4  # P1 M155
    discriminator_lr: float = 2e-4  # P1 M155
    base_width: int = 64
    depth: int = 6
    seed: int = 42
    num_workers: int = 0
    device: str = "cuda"
    run_name: str | None = None
    preload_dataset: bool = True
    use_amp: bool = True
    label_smoothing: float = 0.1
    eta_lambda: float = 0.0
    eta_tau: float = 0.95
    eta_n_q: int = 100
    eta_pretrain_steps: int = 0
    eta_refresh_every: int = 200
    eta_pool_size: int = 512


def _apply_mask(field: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return field * mask.unsqueeze(1).to(field.dtype)


def validate(generator, loader, device, moments, *, use_amp: bool = False) -> dict[str, float]:
    generator.eval()
    metrics = PooledMetrics()
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                prediction_mm = generator(batch["inputs"], batch["invariants"])
            target_mm = denormalize_to_mm(batch["target"], moments)
            metrics.update(prediction_mm.float(), target_mm, loss_mask(batch))
    return metrics.compute()


def train(
    config: GanTrainingConfig, *, prepared_root: Path | None = None, output_root: Path | None = None,
) -> Path:
    seed_everything(config.seed)
    device = torch.device(config.device if config.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    stats = load_norm_stats(prepared_root)
    moments = target_moments(stats)

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

    generator_config = GeneratorConfig(base_width=config.base_width, depth=config.depth)
    discriminator_config = DiscriminatorConfig()
    generator = ConditionalGenerator(generator_config).to(device)
    discriminator = ConditionalDiscriminator(discriminator_config).to(device)
    g_optimizer = torch.optim.AdamW(generator.parameters(), lr=config.generator_lr, betas=(0.0, 0.999))
    d_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=config.discriminator_lr, betas=(0.0, 0.5))

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

    run_dir = init_run_dir("gan", config.region, run_name=config.run_name, output_root=output_root)
    write_run_metadata(run_dir, config, stats=stats)

    best_r2 = -float("inf")
    for epoch in range(1, config.epochs + 1):
        train_sampler.set_epoch(epoch)
        generator.train()
        discriminator.train()
        g_loss_sum = d_loss_sum = 0.0
        batches = 0
        for batch in train_loader:
            batch = to_device(batch, device)
            mask = loss_mask(batch)
            target_mm = denormalize_to_mm(batch["target"], moments)
            real_masked = _apply_mask(target_mm, mask)

            # -- Discriminator step (relativistic P1 Eq. 1, see losses_gan.py) --
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config.use_amp):
                fake_sample = generator(batch["inputs"], batch["invariants"])
            d_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config.use_amp):
                real_logits = discriminator(real_masked, batch["inputs"], batch["invariants"])
                fake_logits = discriminator(_apply_mask(fake_sample, mask), batch["inputs"], batch["invariants"])
                d_loss = discriminator_loss(real_logits, fake_logits, label_smoothing=config.label_smoothing)
            d_loss.backward()
            d_optimizer.step()

            # -- Generator step (relativistic P1 Eq. 4 = Eq. 2 + Eq. 3) --------
            # D's weights just changed (d_optimizer.step() above), so the
            # relativistic margin needs a fresh D(real) under the *current*
            # weights -- reusing the one computed before the D step would
            # compare the generator against a stale discriminator.
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config.use_amp):
                real_logits_for_g = discriminator(real_masked, batch["inputs"], batch["invariants"])
            g_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config.use_amp):
                members = torch.stack(
                    [generator(batch["inputs"], batch["invariants"]) for _ in range(N_ENSEMBLE_MEMBERS)], dim=0,
                )
                fake_logits_for_g = discriminator(_apply_mask(members[0], mask), batch["inputs"], batch["invariants"])
                adversarial = generator_adversarial_loss(
                    fake_logits_for_g, real_logits_for_g, label_smoothing=config.label_smoothing,
                )
                l1 = ensemble_l1_loss(members, target_mm, mask.unsqueeze(1))
                g_loss = adversarial + l1

            def forward_fn(inp: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
                return generator(inp, inv)

            eta.maybe_refresh_tail_index(forward_fn)
            eta_term = eta.term(forward_fn)
            if eta_term is not None:
                g_loss = g_loss + eta_config.lambda_ * eta_term
            g_loss.backward()
            g_optimizer.step()
            eta.step()

            g_loss_sum += g_loss.item()
            d_loss_sum += d_loss.item()
            batches += 1

        validation = validate(generator, val_loader, device, moments, use_amp=config.use_amp)
        row = {
            "epoch": epoch, "generator_loss": g_loss_sum / max(batches, 1),
            "discriminator_loss": d_loss_sum / max(batches, 1),
            **{f"val_{k}": v for k, v in validation.items()},
        }
        append_history(run_dir, row)
        logger.info("gan epoch=%d g_loss=%.4f d_loss=%.4f val_r2=%.4f val_log_mae=%.4f val_clipped=%d",
                    epoch, row["generator_loss"], row["discriminator_loss"], validation["r2"],
                    validation["log_mae"], validation["clipped_count"])

        checkpoint = {
            "generator_state": generator.state_dict(), "discriminator_state": discriminator.state_dict(),
            "generator_config": asdict(generator_config), "discriminator_config": asdict(discriminator_config),
            "g_optimizer_state": g_optimizer.state_dict(), "d_optimizer_state": d_optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if validation["r2"] > best_r2:
            best_r2 = validation["r2"]
            torch.save(checkpoint, run_dir / "best.pt")
    return run_dir


def main() -> None:
    from src.data_preprocessing import regions as region_defs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, choices=region_defs.list_regions())
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-preload", dest="preload_dataset", action="store_false", default=True)
    parser.add_argument("--no-amp", dest="use_amp", action="store_false", default=True)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--eta-lambda", type=float, default=0.0)
    args = parser.parse_args()
    config = GanTrainingConfig(
        region=args.region, epochs=args.epochs, batch_size=args.batch_size,
        seed=args.seed, device=args.device, run_name=args.run_name, num_workers=args.num_workers,
        preload_dataset=args.preload_dataset, use_amp=args.use_amp, label_smoothing=args.label_smoothing,
        eta_lambda=args.eta_lambda,
    )
    print(train(config))


if __name__ == "__main__":
    main()
