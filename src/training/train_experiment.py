import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.config import Config
from src.data_preprocessing.dataset import PrecipDataset, FastPrecipDataset
from src.models.rrdb_gan import GeneratorGAN1, GeneratorGAN2, PatchGANDiscriminator
from src.utils.land_mask import get_land_mask

logger = Config.get_logger()

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def spectral_loss(pred, target):
    """L1 loss between the amplitude spectra of pred and target.

    Penalizes differences in spatial frequency content, which captures
    precipitation texture better than pixel-wise losses alone.

    Args:
        pred (torch.Tensor): Predicted field, shape (B, 1, H, W), normalized.
        target (torch.Tensor): Target field, shape (B, 1, H, W), normalized.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    fft_pred = torch.fft.rfft2(pred)
    fft_target = torch.fft.rfft2(target)
    return F.l1_loss(torch.abs(fft_pred), torch.abs(fft_target))

def denormalize_precip(tensor, mean, std, log_transform=True):
    # tensor: (B, 1, H, W)
    # mean, std: (1, 1, 1) tensors
    mean = mean.to(tensor.device)
    std = std.to(tensor.device)
    
    # Calculate normalized value corresponding to 300.0 mm/day physical limit
    import math
    if log_transform:
        max_norm = (math.log1p(300.0) - mean) / std
    else:
        max_norm = (300.0 - mean) / std
        
    # Clamp to prevent exponential explosion/outliers on unconstrained outputs
    clamped_tensor = torch.clamp(tensor, min=-10.0, max=max_norm)
    denorm = clamped_tensor * std + mean
    if log_transform:
        denorm = torch.expm1(denorm)
    return torch.clamp(denorm, min=0.0)

def run_training_experiment(
    target_name, epochs, batch_size, save_interval, device_name, dry_run=False,
    model_size='small', pixel_weight=5.0, adv_weight=0.2
):
    device = torch.device(device_name if torch.cuda.is_available() and device_name.startswith('cuda') else 'cpu')
    logger.info(
        f"Running training experiment on device: {device} for target: {target_name} "
        f"(Dry run: {dry_run}, model_size: {model_size}, pixel_weight: {pixel_weight}, adv_weight: {adv_weight})"
    )

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        logger.info("cuDNN benchmarking enabled for faster static-input convolutions.")

    # Setup directories
    checkpoints_dir = os.path.join(Config.LOCAL_DATA_DIR, "checkpoints", target_name)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # 1. Load Datasets and DataLoaders
    logger.info("Initializing datasets...")
    train_ds = FastPrecipDataset(target_name, 'train')
    val_ds = FastPrecipDataset(target_name, 'val')

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    stats_path = os.path.join(Config.LOCAL_DATA_DIR, "metadata", f"norm_stats_{target_name}.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    mean_val = torch.tensor(stats['target_mean'], dtype=torch.float32)
    std_val = torch.tensor(stats['target_std'], dtype=torch.float32)
    log_transform = stats.get('log_transform_precip', True)

    logger.info(f"Train dataset: {len(train_ds)} samples. Validation dataset: {len(val_ds)} samples.")

    # 2. Instantiate Models
    logger.info("Instantiating models...")
    nf = 64 if model_size == 'large' else 32
    nb = 8  if model_size == 'large' else 4
    gc = 32 if model_size == 'large' else 16
    ndf = 64 if model_size == 'large' else 32
    logger.info(f"Model capacity: nf={nf}, nb={nb}, gc={gc} (model_size='{model_size}')")

    netG1 = GeneratorGAN1(in_nc=18, nf=nf, nb=nb, gc=gc).to(device)
    netD1 = PatchGANDiscriminator(in_nc=1, ndf=ndf).to(device)

    netG2 = GeneratorGAN2(in_nc=4, nf=nf, nb=nb, gc=gc).to(device)
    netD2 = PatchGANDiscriminator(in_nc=1, ndf=ndf).to(device)

    # 3. Setup Optimizers
    opt_G1 = torch.optim.Adam(netG1.parameters(), lr=1e-4, betas=(0.9, 0.999))
    opt_D1 = torch.optim.Adam(netD1.parameters(), lr=1e-4, betas=(0.9, 0.999))

    opt_G2 = torch.optim.Adam(netG2.parameters(), lr=1e-4, betas=(0.9, 0.999))
    opt_D2 = torch.optim.Adam(netD2.parameters(), lr=1e-4, betas=(0.9, 0.999))

    # 4a. LR Schedulers — cosine annealing; lr decays from 1e-4 to 1e-6 over all epochs
    scheduler_G1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G1, T_max=epochs, eta_min=1e-6)
    scheduler_D1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D1, T_max=epochs, eta_min=1e-6)
    scheduler_G2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G2, T_max=epochs, eta_min=1e-6)
    scheduler_D2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D2, T_max=epochs, eta_min=1e-6)

    # 4b. Setup Losses and Weights
    criterion_pixel = nn.L1Loss()
    criterion_gan = nn.BCEWithLogitsLoss()

    # Spectral loss replaces VGG perceptual loss — domain-appropriate for precipitation fields
    spectral_weight = 0.5
    # Occurrence loss encourages correct wet/dry discrimination
    occurrence_weight = 1.0

    logger.info(
        f"Loss weights — pixel: {pixel_weight}, adv: {adv_weight}, "
        f"spectral: {spectral_weight}, occurrence (Stage-2 only): {occurrence_weight}"
    )

    # 4c. Land mask — used to exclude ocean pixels from pixel loss
    land_mask_5km = get_land_mask(device=device)  # (1, 1, 460, 720)
    land_mask_10km = F.interpolate(land_mask_5km, size=(230, 360), mode='nearest')  # (1, 1, 230, 360)

    # 4d. Precipitation occurrence threshold in normalized space (corresponds to 1 mm/day)
    import math
    if log_transform:
        threshold_norm = (math.log1p(1.0) - mean_val.item()) / std_val.item()
    else:
        threshold_norm = (1.0 - mean_val.item()) / std_val.item()

    # ----------------------------------------------------
    # STAGE 1: Train GAN-1 (ERA5 25km -> Intermediate 10km)
    # ----------------------------------------------------
    logger.info("=== STAGE 1: Training GAN-1 ===")
    best_val_mae_g1 = float('inf')
    patience_g1 = 0
    PATIENCE = 3
    best_g1_path = os.path.join(checkpoints_dir, f"gan1_{target_name}_best.pt")
    for epoch in range(1, epochs + 1):
        netG1.train()
        netD1.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        for batch_idx, (inputs_25km, phys_dem_10km, real_10km, real_5km) in enumerate(train_loader):
            if dry_run and batch_idx >= 2:
                break
            inputs_25km = inputs_25km.to(device)
            real_10km = real_10km.to(device)

            # Generate fake 10km
            fake_10km = netG1(inputs_25km)

            # (A) Update Discriminator D1
            opt_D1.zero_grad()
            pred_real = netD1(real_10km)
            loss_D_real = criterion_gan(pred_real, torch.ones_like(pred_real))
            pred_fake = netD1(fake_10km.detach())
            loss_D_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            opt_D1.step()
            epoch_d_loss += loss_D.item()

            # (B) Update Generator G1
            opt_G1.zero_grad()
            pred_fake_g = netD1(fake_10km)
            loss_G_adv = criterion_gan(pred_fake_g, torch.ones_like(pred_fake_g))
            # Land-masked pixel loss — excludes ocean fill values from gradient
            loss_G_pixel = criterion_pixel(fake_10km * land_mask_10km, real_10km * land_mask_10km)
            # Spectral consistency loss — penalizes power-spectrum differences
            loss_G_spectral = spectral_loss(fake_10km, real_10km)

            loss_G = adv_weight * loss_G_adv + pixel_weight * loss_G_pixel + spectral_weight * loss_G_spectral

            loss_G.backward()
            opt_G1.step()
            epoch_g_loss += loss_G.item()

            if batch_idx % 100 == 0:
                logger.info(
                    f"GAN-1 Epoch [{epoch}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] | G Loss: {loss_G.item():.4f} | D Loss: {loss_D.item():.4f}"
                )

        # Validation GAN-1
        netG1.eval()
        val_mae = 0.0
        val_rmse = 0.0
        val_r2 = 0.0
        val_corr = 0.0
        with torch.no_grad():
            for batch_idx, (inputs_25km, phys_dem_10km, real_10km, real_5km) in enumerate(val_loader):
                if dry_run and batch_idx >= 2:
                    break
                inputs_25km = inputs_25km.to(device)
                real_10km = real_10km.to(device)
                fake_10km = netG1(inputs_25km)

                # Denormalize to mm/day scale
                real_10km_phys = denormalize_precip(real_10km, mean_val, std_val, log_transform)
                fake_10km_phys = denormalize_precip(fake_10km, mean_val, std_val, log_transform)

                # Compute batch metrics on physical scale
                mae = torch.mean(torch.abs(fake_10km_phys - real_10km_phys))
                rmse = torch.sqrt(torch.mean((fake_10km_phys - real_10km_phys) ** 2))

                ss_res = torch.sum((real_10km_phys - fake_10km_phys) ** 2)
                ss_tot = torch.sum((real_10km_phys - torch.mean(real_10km_phys)) ** 2)
                r2 = 1.0 - (ss_res / torch.clamp(ss_tot, min=1e-8))

                vx = fake_10km_phys - torch.mean(fake_10km_phys)
                vy = real_10km_phys - torch.mean(real_10km_phys)
                corr = torch.sum(vx * vy) / torch.clamp(torch.sqrt(torch.sum(vx ** 2) * torch.sum(vy ** 2)), min=1e-8)

                val_mae += mae.item()
                val_rmse += rmse.item()
                val_r2 += r2.item()
                val_corr += corr.item()
        
        num_batches_val = 2 if dry_run else len(val_loader)
        num_batches_train = 2 if dry_run else len(train_loader)
        
        avg_mae = val_mae / num_batches_val
        avg_rmse = val_rmse / num_batches_val
        avg_r2 = val_r2 / num_batches_val
        avg_corr = val_corr / num_batches_val
        
        avg_g_loss = epoch_g_loss / num_batches_train
        avg_d_loss = epoch_d_loss / num_batches_train

        logger.info(
            f"GAN-1 Epoch [{epoch}/{epochs}] | G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f} | "
            f"MAE: {avg_mae:.4f} mm/d | RMSE: {avg_rmse:.4f} mm/d | R^2: {avg_r2:.4f} | Corr: {avg_corr:.4f} | "
            f"LR: {scheduler_G1.get_last_lr()[0]:.2e}"
        )

        # Early stopping on validation MAE
        if avg_mae < best_val_mae_g1:
            best_val_mae_g1 = avg_mae
            patience_g1 = 0
            torch.save(netG1.state_dict(), best_g1_path)
            logger.info(f"GAN-1 new best MAE: {avg_mae:.4f} mm/d — best checkpoint saved.")
        else:
            patience_g1 += 1
            if patience_g1 >= PATIENCE:
                logger.info(f"GAN-1 early stopping triggered at epoch {epoch} (patience={PATIENCE}).")
                netG1.load_state_dict(torch.load(best_g1_path, map_location=device))
                break

        # Step LR schedulers at the end of each epoch
        scheduler_G1.step()
        scheduler_D1.step()

        if epoch % save_interval == 0 or epoch == epochs:
            g1_path = os.path.join(checkpoints_dir, f"gan1_gen_epoch_{epoch}.pt")
            torch.save(netG1.state_dict(), g1_path)
            logger.info(f"Saved GAN-1 checkpoint: {g1_path}")

    # ----------------------------------------------------
    # STAGE 2: Train GAN-2 (Intermediate 10km + Physics -> High-res 5km)
    # ----------------------------------------------------
    logger.info("=== STAGE 2: Training GAN-2 ===")
    netG1.eval()  # Freeze GAN-1 Generator
    best_val_mae_g2 = float('inf')
    patience_g2 = 0
    best_g2_path = os.path.join(checkpoints_dir, f"gan2_{target_name}_best.pt")

    for epoch in range(1, epochs + 1):
        netG2.train()
        netD2.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        for batch_idx, (inputs_25km, phys_dem_10km, real_10km, real_5km) in enumerate(train_loader):
            if dry_run and batch_idx >= 2:
                break
            inputs_25km = inputs_25km.to(device)
            phys_dem_10km = phys_dem_10km.to(device)
            real_5km = real_5km.to(device)

            with torch.no_grad():
                # 1. Run GAN-1 to generate intermediate 10km precip
                gan1_out = netG1(inputs_25km).detach() # Shape: (B, 1, 230, 360)
                
                # 2. Concatenate to construct input for GAN-2: (B, 4, 230, 360)
                inputs_gan2 = torch.cat([gan1_out, phys_dem_10km], dim=1)

            # Generate high-res 5km precip
            fake_5km = netG2(inputs_gan2)

            # (A) Update Discriminator D2
            opt_D2.zero_grad()
            pred_real = netD2(real_5km)
            loss_D_real = criterion_gan(pred_real, torch.ones_like(pred_real))
            pred_fake = netD2(fake_5km.detach())
            loss_D_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            opt_D2.step()
            epoch_d_loss += loss_D.item()

            # (B) Update Generator G2
            opt_G2.zero_grad()
            pred_fake_g = netD2(fake_5km)
            loss_G_adv = criterion_gan(pred_fake_g, torch.ones_like(pred_fake_g))
            # Land-masked pixel loss
            loss_G_pixel = criterion_pixel(fake_5km * land_mask_5km, real_5km * land_mask_5km)
            # Spectral consistency loss
            loss_G_spectral = spectral_loss(fake_5km, real_5km)
            # Precipitation occurrence loss — soft BCE on wet/dry threshold
            pred_occur = torch.sigmoid((fake_5km - threshold_norm) * 5.0)
            true_occur = (real_5km >= threshold_norm).float()
            loss_G_occur = F.binary_cross_entropy(pred_occur, true_occur)

            loss_G = (
                adv_weight * loss_G_adv
                + pixel_weight * loss_G_pixel
                + spectral_weight * loss_G_spectral
                + occurrence_weight * loss_G_occur
            )

            loss_G.backward()
            opt_G2.step()
            epoch_g_loss += loss_G.item()

            if batch_idx % 100 == 0:
                logger.info(
                    f"GAN-2 Epoch [{epoch}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] | G Loss: {loss_G.item():.4f} | D Loss: {loss_D.item():.4f}"
                )

        # Validation GAN-2
        netG2.eval()
        val_mae = 0.0
        val_rmse = 0.0
        val_r2 = 0.0
        val_corr = 0.0
        with torch.no_grad():
            for batch_idx, (inputs_25km, phys_dem_10km, real_10km, real_5km) in enumerate(val_loader):
                if dry_run and batch_idx >= 2:
                    break
                inputs_25km = inputs_25km.to(device)
                phys_dem_10km = phys_dem_10km.to(device)
                real_5km = real_5km.to(device)
                
                gan1_out = netG1(inputs_25km)
                inputs_gan2 = torch.cat([gan1_out, phys_dem_10km], dim=1)
                
                fake_5km = netG2(inputs_gan2)

                # Denormalize to mm/day scale
                real_5km_phys = denormalize_precip(real_5km, mean_val, std_val, log_transform)
                fake_5km_phys = denormalize_precip(fake_5km, mean_val, std_val, log_transform)

                # Compute batch metrics on physical scale
                mae = torch.mean(torch.abs(fake_5km_phys - real_5km_phys))
                rmse = torch.sqrt(torch.mean((fake_5km_phys - real_5km_phys) ** 2))

                ss_res = torch.sum((real_5km_phys - fake_5km_phys) ** 2)
                ss_tot = torch.sum((real_5km_phys - torch.mean(real_5km_phys)) ** 2)
                r2 = 1.0 - (ss_res / torch.clamp(ss_tot, min=1e-8))

                vx = fake_5km_phys - torch.mean(fake_5km_phys)
                vy = real_5km_phys - torch.mean(real_5km_phys)
                corr = torch.sum(vx * vy) / torch.clamp(torch.sqrt(torch.sum(vx ** 2) * torch.sum(vy ** 2)), min=1e-8)

                val_mae += mae.item()
                val_rmse += rmse.item()
                val_r2 += r2.item()
                val_corr += corr.item()
        
        num_batches_val = 2 if dry_run else len(val_loader)
        num_batches_train = 2 if dry_run else len(train_loader)
        
        avg_mae = val_mae / num_batches_val
        avg_rmse = val_rmse / num_batches_val
        avg_r2 = val_r2 / num_batches_val
        avg_corr = val_corr / num_batches_val
        
        avg_g_loss = epoch_g_loss / num_batches_train
        avg_d_loss = epoch_d_loss / num_batches_train

        logger.info(
            f"GAN-2 Epoch [{epoch}/{epochs}] | G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f} | "
            f"MAE: {avg_mae:.4f} mm/d | RMSE: {avg_rmse:.4f} mm/d | R^2: {avg_r2:.4f} | Corr: {avg_corr:.4f} | "
            f"LR: {scheduler_G2.get_last_lr()[0]:.2e}"
        )

        # Early stopping on validation MAE
        if avg_mae < best_val_mae_g2:
            best_val_mae_g2 = avg_mae
            patience_g2 = 0
            torch.save(netG2.state_dict(), best_g2_path)
            logger.info(f"GAN-2 new best MAE: {avg_mae:.4f} mm/d — best checkpoint saved.")
        else:
            patience_g2 += 1
            if patience_g2 >= PATIENCE:
                logger.info(f"GAN-2 early stopping triggered at epoch {epoch} (patience={PATIENCE}).")
                netG2.load_state_dict(torch.load(best_g2_path, map_location=device))
                break

        # Step LR schedulers at the end of each epoch
        scheduler_G2.step()
        scheduler_D2.step()

        if epoch % save_interval == 0 or epoch == epochs:
            g2_path = os.path.join(checkpoints_dir, f"gan2_gen_epoch_{epoch}.pt")
            torch.save(netG2.state_dict(), g2_path)
            logger.info(f"Saved GAN-2 checkpoint: {g2_path}")

    # Save final model checkpoints to final path names
    final_g1_path = os.path.join(checkpoints_dir, f"gan1_{target_name}.pt")
    final_g2_path = os.path.join(checkpoints_dir, f"gan2_{target_name}.pt")
    torch.save(netG1.state_dict(), final_g1_path)
    torch.save(netG2.state_dict(), final_g2_path)
    logger.info(f"Training completed successfully! Saved final checkpoints:\n- {final_g1_path}\n- {final_g2_path}")

def main():
    parser = argparse.ArgumentParser(description="Train downscaling GAN-1 and GAN-2 models.")
    parser.add_argument("--target", type=str, choices=["chirps", "oya"], required=True, help="Target pipeline")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs per stage")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--save_interval", type=int, default=5, help="Save checkpoints interval in epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Training execution device (cuda or cpu)")
    parser.add_argument("--dry-run", action="store_true", help="Execute in dry-run mode (runs 2 batches per epoch)")
    parser.add_argument(
        "--model_size", type=str, default="small", choices=["small", "large"],
        help="Model capacity: small=nb=4,nf=32 | large=nb=8,nf=64"
    )
    parser.add_argument(
        "--pixel_weight", type=float, default=5.0,
        help="Weight for the L1 pixel loss term (default: 5.0)"
    )
    parser.add_argument(
        "--adv_weight", type=float, default=0.2,
        help="Weight for the adversarial GAN loss term (default: 0.2)"
    )
    args = parser.parse_args()

    run_training_experiment(
        target_name=args.target,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_interval=args.save_interval,
        device_name=args.device,
        dry_run=args.dry_run,
        model_size=args.model_size,
        pixel_weight=args.pixel_weight,
        adv_weight=args.adv_weight,
    )

if __name__ == "__main__":
    main()
