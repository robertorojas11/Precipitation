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
from src.data_preprocessing.dataset import PrecipDataset
from src.models.rrdb_gan import GeneratorGAN1, GeneratorGAN2, PatchGANDiscriminator

logger = Config.get_logger()

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class VGGFeatureExtractor(nn.Module):
    """VGG19 feature extractor for perceptual loss calculation."""

    def __init__(self):
        super().__init__()
        import torchvision.models as models
        # Attempt to load pretrained VGG19 features
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        # Extract features up to conv4_4 layer
        self.features = nn.Sequential(*list(vgg.features.children())[:16]).eval()
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Replicate grayscale channel to 3 channels for RGB VGG
        x_rgb = torch.cat([x, x, x], dim=1)
        # Standard Imagenet normalizations applied inside
        return self.features(x_rgb)

def run_training_experiment(target_name, epochs, batch_size, save_interval, device_name, dry_run=False):
    device = torch.device(device_name if torch.cuda.is_available() and device_name.startswith('cuda') else 'cpu')
    logger.info(f"Running training experiment on device: {device} for target: {target_name} (Dry run: {dry_run})")

    # Setup directories
    checkpoints_dir = os.path.join(Config.LOCAL_DATA_DIR, "checkpoints", target_name)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # 1. Load Datasets and DataLoaders
    logger.info("Initializing datasets...")
    train_ds = PrecipDataset(target_name, 'train')
    val_ds = PrecipDataset(target_name, 'val')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    logger.info(f"Train dataset: {len(train_ds)} samples. Validation dataset: {len(val_ds)} samples.")

    # 2. Instantiate Models
    logger.info("Instantiating models...")
    netG1 = GeneratorGAN1(in_nc=18).to(device)
    netD1 = PatchGANDiscriminator(in_nc=1).to(device)
    
    netG2 = GeneratorGAN2(in_nc=4).to(device)
    netD2 = PatchGANDiscriminator(in_nc=1).to(device)

    # 3. Setup Optimizers
    opt_G1 = torch.optim.Adam(netG1.parameters(), lr=1e-4, betas=(0.9, 0.999))
    opt_D1 = torch.optim.Adam(netD1.parameters(), lr=1e-4, betas=(0.9, 0.999))
    
    opt_G2 = torch.optim.Adam(netG2.parameters(), lr=1e-4, betas=(0.9, 0.999))
    opt_D2 = torch.optim.Adam(netD2.parameters(), lr=1e-4, betas=(0.9, 0.999))

    # 4. Setup Losses and Weights
    criterion_pixel = nn.L1Loss()
    criterion_gan = nn.BCEWithLogitsLoss()
    
    # Load VGG with fallback if offline
    try:
        criterion_perceptual = VGGFeatureExtractor().to(device)
        perceptual_weight = 1.0
        logger.info("Perceptual loss enabled using VGG19.")
    except Exception as e:
        criterion_perceptual = None
        perceptual_weight = 0.0
        logger.warning(f"Perceptual loss disabled because VGG19 could not be downloaded/loaded: {e}")

    pixel_weight = 10.0
    adv_weight = 0.1

    # ----------------------------------------------------
    # STAGE 1: Train GAN-1 (ERA5 25km -> Intermediate 10km)
    # ----------------------------------------------------
    logger.info("=== STAGE 1: Training GAN-1 ===")
    for epoch in range(1, epochs + 1):
        netG1.train()
        netD1.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        for batch_idx, (inputs_t, target_t) in enumerate(train_loader):
            if dry_run and batch_idx >= 2:
                break
            inputs_t = inputs_t.to(device)
            target_t = target_t.to(device)

            # Downsample inputs to 25km (92x144) for GAN-1
            inputs_25km = F.interpolate(inputs_t[:, :18, :, :], size=(92, 144), mode='bilinear', align_corners=False)
            # Downsample target to 10km (230x360) for GAN-1 ground truth
            real_10km = F.interpolate(target_t, size=(230, 360), mode='bilinear', align_corners=False)

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
            loss_G_pixel = criterion_pixel(fake_10km, real_10km)
            
            loss_G = adv_weight * loss_G_adv + pixel_weight * loss_G_pixel
            
            if criterion_perceptual is not None:
                feat_fake = criterion_perceptual(fake_10km)
                feat_real = criterion_perceptual(real_10km)
                loss_G_percep = criterion_pixel(feat_fake, feat_real)
                loss_G += perceptual_weight * loss_G_percep

            loss_G.backward()
            opt_G1.step()
            epoch_g_loss += loss_G.item()

        # Validation GAN-1
        netG1.eval()
        val_l1_10km = 0.0
        with torch.no_grad():
            for batch_idx, (inputs_t, target_t) in enumerate(val_loader):
                if dry_run and batch_idx >= 2:
                    break
                inputs_t = inputs_t.to(device)
                target_t = target_t.to(device)
                inputs_25km = F.interpolate(inputs_t[:, :18, :, :], size=(92, 144), mode='bilinear', align_corners=False)
                real_10km = F.interpolate(target_t, size=(230, 360), mode='bilinear', align_corners=False)
                fake_10km = netG1(inputs_25km)
                val_l1_10km += criterion_pixel(fake_10km, real_10km).item()
        
        num_batches_val = 2 if dry_run else len(val_loader)
        num_batches_train = 2 if dry_run else len(train_loader)
        val_l1_10km /= num_batches_val
        avg_g_loss = epoch_g_loss / num_batches_train
        avg_d_loss = epoch_d_loss / num_batches_train

        logger.info(
            f"GAN-1 Epoch [{epoch}/{epochs}] | G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f} | Val L1: {val_l1_10km:.4f}"
        )

        if epoch % save_interval == 0 or epoch == epochs:
            g1_path = os.path.join(checkpoints_dir, f"gan1_gen_epoch_{epoch}.pt")
            torch.save(netG1.state_dict(), g1_path)
            logger.info(f"Saved GAN-1 checkpoint: {g1_path}")

    # ----------------------------------------------------
    # STAGE 2: Train GAN-2 (Intermediate 10km + Physics -> High-res 5km)
    # ----------------------------------------------------
    logger.info("=== STAGE 2: Training GAN-2 ===")
    netG1.eval() # Freeze GAN-1 Generator

    for epoch in range(1, epochs + 1):
        netG2.train()
        netD2.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        for batch_idx, (inputs_t, target_t) in enumerate(train_loader):
            if dry_run and batch_idx >= 2:
                break
            inputs_t = inputs_t.to(device)
            target_t = target_t.to(device)

            with torch.no_grad():
                # 1. Run GAN-1 to generate intermediate 10km precip
                inputs_25km = F.interpolate(inputs_t[:, :18, :, :], size=(92, 144), mode='bilinear', align_corners=False)
                gan1_out = netG1(inputs_25km).detach() # Shape: (B, 1, 230, 360)

                # 2. Extract and downsample Physics/DEM bands from inputs_t (channels 18, 19, 20)
                phys_dem = inputs_t[:, 18:21, :, :] # (B, 3, 460, 720)
                phys_dem_10km = F.interpolate(phys_dem, size=(230, 360), mode='bilinear', align_corners=False)
                
                # 3. Concatenate to construct input for GAN-2: (B, 4, 230, 360)
                inputs_gan2 = torch.cat([gan1_out, phys_dem_10km], dim=1)

            # Generate high-res 5km precip
            fake_5km = netG2(inputs_gan2)
            real_5km = target_t # Ground-truth (5km)

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
            loss_G_pixel = criterion_pixel(fake_5km, real_5km)
            
            loss_G = adv_weight * loss_G_adv + pixel_weight * loss_G_pixel
            
            if criterion_perceptual is not None:
                feat_fake = criterion_perceptual(fake_5km)
                feat_real = criterion_perceptual(real_5km)
                loss_G_percep = criterion_pixel(feat_fake, feat_real)
                loss_G += perceptual_weight * loss_G_percep

            loss_G.backward()
            opt_G2.step()
            epoch_g_loss += loss_G.item()

        # Validation GAN-2
        netG2.eval()
        val_l1_5km = 0.0
        with torch.no_grad():
            for batch_idx, (inputs_t, target_t) in enumerate(val_loader):
                if dry_run and batch_idx >= 2:
                    break
                inputs_t = inputs_t.to(device)
                target_t = target_t.to(device)
                
                inputs_25km = F.interpolate(inputs_t[:, :18, :, :], size=(92, 144), mode='bilinear', align_corners=False)
                gan1_out = netG1(inputs_25km)
                phys_dem = inputs_t[:, 18:21, :, :]
                phys_dem_10km = F.interpolate(phys_dem, size=(230, 360), mode='bilinear', align_corners=False)
                inputs_gan2 = torch.cat([gan1_out, phys_dem_10km], dim=1)
                
                fake_5km = netG2(inputs_gan2)
                val_l1_5km += criterion_pixel(fake_5km, target_t).item()
        
        num_batches_val = 2 if dry_run else len(val_loader)
        num_batches_train = 2 if dry_run else len(train_loader)
        val_l1_5km /= num_batches_val
        avg_g_loss = epoch_g_loss / num_batches_train
        avg_d_loss = epoch_d_loss / num_batches_train

        logger.info(
            f"GAN-2 Epoch [{epoch}/{epochs}] | G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f} | Val L1: {val_l1_5km:.4f}"
        )

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
    args = parser.parse_args()

    run_training_experiment(
        target_name=args.target,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_interval=args.save_interval,
        device_name=args.device,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
