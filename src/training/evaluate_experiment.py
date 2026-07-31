"""Evaluation script for the precipitation downscaling pipeline.

Fits the bias corrector on validation data, runs inference on the test split,
computes global and Sierra Madre Occidental metrics, and saves plots/metrics.
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.config import Config
from src.data_preprocessing.dataset import FastPrecipDataset
from src.models.rrdb_gan import GeneratorGAN1, GeneratorGAN2, load_generator_from_checkpoint
from src.training.bias_correction import PrecipBiasCorrector

def denormalize(tensor, mean, std, log_transform=True):
    """Denormalizes normalized values back to physical scale."""
    denorm = tensor * std + mean
    if log_transform:
        denorm = torch.expm1(denorm)
    return torch.clamp(denorm, min=0.0, max=300.0)

def crps_ensemble(ensemble, obs, batch_size=100):
    """Computes the Continuous Ranked Probability Score (CRPS) for an ensemble.

    Args:
        ensemble (np.ndarray): Ensemble predictions, shape (N, M, H, W).
        obs (np.ndarray): Observations, shape (N, H, W).
        batch_size (int): Size of batches to split along N to bound memory.
    Returns:
        np.ndarray: CRPS values per pixel, shape (N, H, W).
    """
    N, M, H, W = ensemble.shape
    crps = np.zeros((N, H, W), dtype=np.float32)
    
    i = np.arange(M) + 1
    weights = 2 * (2 * i - M - 1)
    weights = weights.reshape(1, M, 1, 1)
    
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ens_batch = ensemble[start:end]
        obs_batch = obs[start:end, np.newaxis, :, :]
        
        # Sort ensemble members along the member dimension
        ens_sorted = np.sort(ens_batch, axis=1)
        
        # Compute mean absolute difference vs observations
        diff_obs = np.mean(np.abs(ens_batch - obs_batch), axis=1)
        
        # Compute pairwise differences
        pairwise_diff = np.sum(ens_sorted * weights, axis=1) / (M * M)
        
        crps[start:end] = diff_obs - 0.5 * pairwise_diff
        
    return crps

def compute_csi(pred, obs, threshold):
    """Computes Critical Success Index (CSI) at a given threshold."""
    pred_mask = pred >= threshold
    obs_mask = obs >= threshold
    
    tp = np.sum(pred_mask & obs_mask)
    fp = np.sum(pred_mask & ~obs_mask)
    fn = np.sum(~pred_mask & obs_mask)
    
    if (tp + fp + fn) == 0:
        return 1.0
    return tp / (tp + fp + fn)

def compute_metrics(pred, obs, prefix=""):
    """Computes standard deterministic metrics."""
    mae = np.mean(np.abs(pred - obs))
    rmse = np.sqrt(np.mean((pred - obs) ** 2))
    
    # Correlation coefficient
    pred_flat = pred.ravel()
    obs_flat = obs.ravel()
    corr = np.corrcoef(pred_flat, obs_flat)[0, 1] if np.std(pred_flat) > 0 and np.std(obs_flat) > 0 else 0.0
    
    # R2 Score
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        f"{prefix}mae": float(mae),
        f"{prefix}rmse": float(rmse),
        f"{prefix}r2": float(r2),
        f"{prefix}corr": float(corr)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True, choices=["oya", "chirps"], help="Target name")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick verification on 2 samples")
    parser.add_argument("--model_size", type=str, default="auto", choices=["auto", "small", "large"], help="Model capacity (auto detects from checkpoint)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on {device} for target {args.target} (Dry-run: {args.dry_run}, model_size: {args.model_size})")

    # Load statistics
    metadata_dir = os.path.join(Config.LOCAL_DATA_DIR, "metadata")
    stats_path = os.path.join(metadata_dir, f"norm_stats_{args.target}.json")
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    input_mean_0 = stats['input_mean'][0]
    input_std_0 = stats['input_std'][0]
    target_mean = stats['target_mean']
    target_std = stats['target_std']

    # Load checkpoints
    checkpoint_g1 = os.path.join(Config.LOCAL_DATA_DIR, "checkpoints", args.target, f"gan1_{args.target}.pt")
    checkpoint_g2 = os.path.join(Config.LOCAL_DATA_DIR, "checkpoints", args.target, f"gan2_{args.target}.pt")

    # Instantiate and load models
    netG1 = load_generator_from_checkpoint(checkpoint_g1, GeneratorGAN1, in_nc=18, device=device, model_size=args.model_size)
    netG2 = load_generator_from_checkpoint(checkpoint_g2, GeneratorGAN2, in_nc=4, device=device, model_size=args.model_size)
    netG1.eval()
    netG2.eval()

    # Load validation split to fit/load bias corrector
    corrector_path = os.path.join(Config.LOCAL_DATA_DIR, "checkpoints", args.target, f"bias_corrector_{args.target}.npz")
    corrector = PrecipBiasCorrector()
    
    if os.path.exists(corrector_path) and not args.dry_run:
        print(f"Loading pre-fitted bias corrector parameters from: {corrector_path}")
        corrector.load(corrector_path)
    else:
        print("Fitting bias corrector on validation split...")
        val_dataset = FastPrecipDataset(args.target, "val")
        val_len = 2 if args.dry_run else len(val_dataset)
        
        val_preds = []
        val_obs = []
        
        with torch.no_grad():
            for i in range(val_len):
                inputs_25km, phys_dem_10km, _, real_5km = val_dataset[i]
                
                in_25 = inputs_25km.unsqueeze(0).to(device)
                fake_10_norm = netG1(in_25)
                
                phys_dem_10 = phys_dem_10km.unsqueeze(0).to(device)
                gan2_in = torch.cat([fake_10_norm, phys_dem_10], dim=1)
                fake_5_norm = netG2(gan2_in)
                
                fake_5_phys = denormalize(fake_5_norm.squeeze(0).squeeze(0), target_mean, target_std, log_transform=True).cpu().numpy()
                real_5_phys = denormalize(real_5km.squeeze(0), target_mean, target_std, log_transform=True).cpu().numpy()
                
                val_preds.append(fake_5_phys)
                val_obs.append(real_5_phys)
                
        corrector.fit(np.array(val_preds), np.array(val_obs))
        if not args.dry_run:
            corrector.save(corrector_path)

    # Load test split for final evaluation
    print("Running inference on test split...")
    test_dataset = FastPrecipDataset(args.target, "test")
    test_len = 2 if args.dry_run else len(test_dataset)

    test_preds_raw = []
    test_preds_det = []
    test_preds_stoch = []
    test_obs = []

    with torch.no_grad():
        for i in range(test_len):
            inputs_25km, phys_dem_10km, _, real_5km = test_dataset[i]
            
            in_25 = inputs_25km.unsqueeze(0).to(device)
            fake_10_norm = netG1(in_25)
            
            phys_dem_10 = phys_dem_10km.unsqueeze(0).to(device)
            gan2_in = torch.cat([fake_10_norm, phys_dem_10], dim=1)
            fake_5_norm = netG2(gan2_in)
            
            fake_5_phys = denormalize(fake_5_norm.squeeze(0).squeeze(0), target_mean, target_std, log_transform=True).cpu().numpy()
            real_5_phys = denormalize(real_5km.squeeze(0), target_mean, target_std, log_transform=True).cpu().numpy()
            
            # Apply bias correction
            det_corrected = corrector.apply_deterministic(fake_5_phys)
            stoch_members = corrector.apply_stochastic(fake_5_phys, num_members=10)
            
            test_preds_raw.append(fake_5_phys)
            test_preds_det.append(det_corrected)
            test_preds_stoch.append(stoch_members)
            test_obs.append(real_5_phys)

    test_preds_raw = np.array(test_preds_raw)
    test_preds_det = np.array(test_preds_det)
    test_preds_stoch = np.array(test_preds_stoch)
    test_obs = np.array(test_obs)

    # 1. Global Metrics Calculation
    print("Calculating global evaluation metrics...")
    metrics_raw = compute_metrics(test_preds_raw, test_obs, "raw_")
    metrics_det = compute_metrics(test_preds_det, test_obs, "det_")
    metrics_stoch_mean = compute_metrics(np.mean(test_preds_stoch, axis=1), test_obs, "stoch_mean_")
    
    # CSI metrics
    csi_raw_1 = compute_csi(test_preds_raw, test_obs, 1.0)
    csi_raw_10 = compute_csi(test_preds_raw, test_obs, 10.0)
    csi_raw_25 = compute_csi(test_preds_raw, test_obs, 25.0)

    csi_det_1 = compute_csi(test_preds_det, test_obs, 1.0)
    csi_det_10 = compute_csi(test_preds_det, test_obs, 10.0)
    csi_det_25 = compute_csi(test_preds_det, test_obs, 25.0)

    # Kolmogorov-Smirnov statistic (sample 100,000 pixels for speed)
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(test_obs.size, size=min(100000, test_obs.size), replace=False)
    ks_raw = ks_2samp(test_preds_raw.ravel()[sample_indices], test_obs.ravel()[sample_indices]).statistic
    ks_det = ks_2samp(test_preds_det.ravel()[sample_indices], test_obs.ravel()[sample_indices]).statistic

    # CRPS
    crps_vals = crps_ensemble(test_preds_stoch, test_obs)
    mean_crps = float(np.mean(crps_vals))

    # 2. Sierra Madre Occidental (SMO) Subregion Metrics
    print("Calculating subregional metrics (Sierra Madre Occidental)...")
    # Row range: [100, 300], Column range: [220, 340]
    sub_raw = test_preds_raw[:, 100:300, 220:340]
    sub_det = test_preds_det[:, 100:300, 220:340]
    sub_stoch = test_preds_stoch[:, :, 100:300, 220:340]
    sub_obs = test_obs[:, 100:300, 220:340]

    metrics_raw_sub = compute_metrics(sub_raw, sub_obs, "sub_raw_")
    metrics_det_sub = compute_metrics(sub_det, sub_obs, "sub_det_")
    
    csi_det_sub_1 = compute_csi(sub_det, sub_obs, 1.0)
    csi_det_sub_10 = compute_csi(sub_det, sub_obs, 10.0)
    csi_det_sub_25 = compute_csi(sub_det, sub_obs, 25.0)

    sub_sample_indices = rng.choice(sub_obs.size, size=min(100000, sub_obs.size), replace=False)
    ks_det_sub = ks_2samp(sub_det.ravel()[sub_sample_indices], sub_obs.ravel()[sub_sample_indices]).statistic
    crps_vals_sub = crps_ensemble(sub_stoch, sub_obs)
    mean_crps_sub = float(np.mean(crps_vals_sub))

    # Package all metrics
    final_metrics = {
        "target": args.target,
        "global": {
            **metrics_raw,
            **metrics_det,
            **metrics_stoch_mean,
            "csi_raw_1": float(csi_raw_1),
            "csi_raw_10": float(csi_raw_10),
            "csi_raw_25": float(csi_raw_25),
            "csi_det_1": float(csi_det_1),
            "csi_det_10": float(csi_det_10),
            "csi_det_25": float(csi_det_25),
            "ks_raw": float(ks_raw),
            "ks_det": float(ks_det),
            "mean_crps": float(mean_crps)
        },
        "smo": {
            **metrics_raw_sub,
            **metrics_det_sub,
            "csi_det_1": float(csi_det_sub_1),
            "csi_det_10": float(csi_det_sub_10),
            "csi_det_25": float(csi_det_sub_25),
            "ks_det": float(ks_det_sub),
            "mean_crps": float(mean_crps_sub)
        }
    }

    # Save metrics JSON file
    output_dir = os.path.join("outputs", args.target)
    os.makedirs(output_dir, exist_ok=True)
    metrics_output = os.path.join(output_dir, f"metrics_{args.target}.json")
    with open(metrics_output, 'w') as f:
        json.dump(final_metrics, f, indent=4)
    print(f"Metrics saved successfully to: {metrics_output}")

    # Generate Spatial Relative Bias Map
    print("Generating Spatial Relative Bias plot...")
    mean_pred = np.mean(test_preds_det, axis=0)
    mean_obs = np.mean(test_obs, axis=0)
    epsilon = 0.1 # prevent division by zero in dry zones
    relative_bias = (mean_pred - mean_obs) / (mean_obs + epsilon)

    plt.figure(figsize=(10, 8))
    # Relative bias plotted in percentages (-100% to +100%)
    im = plt.imshow(relative_bias * 100.0, cmap="RdBu_r", vmin=-100, vmax=100)
    plt.title(f"Spatial Relative Bias (%) - Target: {args.target.upper()} (Test Period)", fontsize=14, fontweight="bold")
    plt.colorbar(im, label="Relative Bias (%)")
    plt.axis("off")
    
    bias_map_path = os.path.join(output_dir, f"downscaling_bias_{args.target}.png")
    plt.savefig(bias_map_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Bias map saved to: {bias_map_path}")

    # Print out results as Markdown
    print("\n================ EVALUATION SUMMARY ================")
    print(f"Target: {args.target.upper()}")
    print("----------------------------------------------------")
    print(f"Global Deterministic Corrector MAE: {final_metrics['global']['det_mae']:.4f} mm/d (Raw: {final_metrics['global']['raw_mae']:.4f} mm/d)")
    print(f"Global Deterministic Corrector RMSE: {final_metrics['global']['det_rmse']:.4f} mm/d (Raw: {final_metrics['global']['raw_rmse']:.4f} mm/d)")
    print(f"Global Correlation: {final_metrics['global']['det_corr']:.4f}")
    print(f"Global KS Statistic (vs Target): {final_metrics['global']['ks_det']:.4f} (Raw: {final_metrics['global']['ks_raw']:.4f})")
    print(f"Global CSI (>1mm): {final_metrics['global']['csi_det_1']:.4f} | (>10mm): {final_metrics['global']['csi_det_10']:.4f} | (>25mm): {final_metrics['global']['csi_det_25']:.4f}")
    print(f"Global Mean CRPS (Stochastic): {final_metrics['global']['mean_crps']:.4f}")
    print("------------------------- SMO ----------------------")
    print(f"SMO Deterministic Corrector MAE: {final_metrics['smo']['sub_det_mae']:.4f} mm/d")
    print(f"SMO Correlation: {final_metrics['smo']['sub_det_corr']:.4f}")
    print(f"SMO KS Statistic: {final_metrics['smo']['ks_det']:.4f}")
    print(f"SMO CSI (>1mm): {final_metrics['smo']['csi_det_1']:.4f} | (>10mm): {final_metrics['smo']['csi_det_10']:.4f} | (>25mm): {final_metrics['smo']['csi_det_25']:.4f}")
    print("====================================================\n")

if __name__ == "__main__":
    main()
