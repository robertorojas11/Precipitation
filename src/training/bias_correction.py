"""Bias Correction Module for Precipitation Downscaling.

Implements Quantile Mapping (CDF matching), Generalized Pareto Distribution (GPD)
tail correction for extremes (>95th percentile), and stochastic noise injection
using the spectral synthesis method.
"""

import os
import json
import numpy as np
from scipy.stats import genpareto

class PrecipBiasCorrector:
    """Performs deterministic bias correction and stochastic noise generation."""

    def __init__(self, num_quantiles=1000, noise_std=2.0):
        self.num_quantiles = num_quantiles
        # Controls the physical amplitude of stochastic noise perturbations (mm/day)
        self.noise_std = noise_std
        self.quantiles = np.linspace(0.0, 100.0, num_quantiles)
        
        # Parameters to fit
        self.V_pred = None
        self.V_obs = None
        self.u_pred = None
        self.u_obs = None
        self.gpd_c = None
        self.gpd_scale = None
        self.spatial_filter = None

    def fit(self, pred_samples, obs_samples):
        """Fits the bias correction parameters.

        Args:
            pred_samples (np.ndarray): Predicted samples, shape (N, H, W).
            obs_samples (np.ndarray): Ground truth samples, shape (N, H, W).
        """
        print("Fitting bias correction parameters...")
        N, H, W = pred_samples.shape

        # Flatten arrays for quantile mapping fitting
        pred_flat = pred_samples.ravel()
        obs_flat = obs_samples.ravel()

        # 1. Compute empirical quantiles
        self.V_pred = np.percentile(pred_flat, self.quantiles)
        self.V_obs = np.percentile(obs_flat, self.quantiles)

        # 2. Identify 95th percentile threshold
        self.u_pred = np.percentile(pred_flat, 95.0)
        self.u_obs = np.percentile(obs_flat, 95.0)

        # 3. Fit GPD on target observations excesses exceeding 95th percentile
        obs_excesses = obs_flat[obs_flat > self.u_obs] - self.u_obs
        if len(obs_excesses) > 10:
            # Sample at most 100,000 points to speed up GPD fitting
            if len(obs_excesses) > 100000:
                rng = np.random.default_rng(42)
                obs_excesses = rng.choice(obs_excesses, size=100000, replace=False)
            # Fit GPD with location parameter fixed to 0
            self.gpd_c, _, self.gpd_scale = genpareto.fit(obs_excesses, floc=0)
            print(f"Fitted GPD: shape (c) = {self.gpd_c:.4f}, scale = {self.gpd_scale:.4f}")
        else:
            print("Warning: Too few excesses to fit GPD. Falling back to empirical QM tail.")
            self.gpd_c = 0.0
            self.gpd_scale = 1.0

        # 4. Compute residuals and estimate spatial correlation structure
        print("Computing residuals and spectral spatial filter...")
        residuals = []
        for i in range(N):
            # Apply deterministic bias correction to this sample
            corr_deterministic = self.apply_deterministic(pred_samples[i])
            res = obs_samples[i] - corr_deterministic
            residuals.append(res)

        # Compute empirical power spectrum of residuals
        power_spectrum = np.zeros((H, W))
        for res in residuals:
            fft_res = np.fft.fft2(res)
            power_spectrum += np.abs(fft_res) ** 2
        power_spectrum /= N
        
        # Spatial filter is square root of power spectrum (amplitude)
        self.spatial_filter = np.sqrt(power_spectrum)
        print("Bias corrector fitted successfully.")

    def apply_deterministic(self, pred):
        """Applies deterministic bias correction (QM + GPD) to a single sample.

        Args:
            pred (np.ndarray): Raw prediction of shape (H, W) or (N, H, W).
        Returns:
            np.ndarray: Deterministic corrected prediction.
        """
        # Determine empirical ranks/quantiles of input predictions
        p = np.interp(pred, self.V_pred, self.quantiles) / 100.0
        p = np.clip(p, 0.0, 1.0)

        # Apply Quantile Mapping for the body (p <= 0.95)
        qm_corrected = np.interp(pred, self.V_pred, self.V_obs)

        # Apply GPD for the tail (p > 0.95)
        if self.gpd_scale > 0:
            gpd_q = np.clip((p - 0.95) / 0.05, 0.0, 0.9999)
            tail_corrected = self.u_obs + genpareto.ppf(gpd_q, self.gpd_c, scale=self.gpd_scale)
            # Replace tail values
            corrected = np.where(p > 0.95, tail_corrected, qm_corrected)
        else:
            corrected = qm_corrected

        # Clip values to physically possible ranges
        return np.clip(corrected, 0.0, 300.0)

    def apply_stochastic(self, pred, num_members=10):
        """Generates stochastic ensemble members with spatially correlated residuals.

        Args:
            pred (np.ndarray): Raw prediction of shape (H, W) or (N, H, W).
            num_members (int): Number of ensemble members to generate.
        Returns:
            np.ndarray: Stochastic ensemble, shape (num_members, H, W) or (N, num_members, H, W).
        """
        is_batched = len(pred.shape) == 3
        if is_batched:
            N, H, W = pred.shape
            ensemble = np.zeros((N, num_members, H, W))
            # Normalize spatial filter to unit std to prevent amplitude explosion
            norm_filter = self.spatial_filter / (np.std(self.spatial_filter) + 1e-8)
            for i in range(N):
                det = self.apply_deterministic(pred[i])
                # Vectorized generation of spatially correlated Gaussian noise
                white_noise = np.random.normal(size=(num_members, H, W))
                fft_noise = np.fft.fft2(white_noise, axes=(-2, -1))
                corr_noise = np.fft.ifft2(fft_noise * norm_filter[np.newaxis, :, :], axes=(-2, -1)).real
                # Scale by noise_std to produce physically meaningful perturbations
                ensemble[i] = np.clip(det[np.newaxis, :, :] + self.noise_std * corr_noise, 0.0, 300.0)
            return ensemble
        else:
            H, W = pred.shape
            det = self.apply_deterministic(pred)
            # Normalize spatial filter to unit std to prevent amplitude explosion
            norm_filter = self.spatial_filter / (np.std(self.spatial_filter) + 1e-8)
            # Vectorized generation of spatially correlated Gaussian noise
            white_noise = np.random.normal(size=(num_members, H, W))
            fft_noise = np.fft.fft2(white_noise, axes=(-2, -1))
            corr_noise = np.fft.ifft2(fft_noise * norm_filter[np.newaxis, :, :], axes=(-2, -1)).real
            # Scale by noise_std to produce physically meaningful perturbations
            ensemble = np.clip(det[np.newaxis, :, :] + self.noise_std * corr_noise, 0.0, 300.0)
            return ensemble

    def save(self, file_path):
        """Saves fitted parameters to a .npz file."""
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        np.savez_compressed(
            file_path,
            V_pred=self.V_pred,
            V_obs=self.V_obs,
            u_pred=self.u_pred,
            u_obs=self.u_obs,
            gpd_c=self.gpd_c,
            gpd_scale=self.gpd_scale,
            spatial_filter=self.spatial_filter
        )
        print(f"Bias correction parameters saved to {file_path}")

    def load(self, file_path):
        """Loads fitted parameters from a .npz file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Parameters file not found: {file_path}")
        data = np.load(file_path)
        self.V_pred = data['V_pred']
        self.V_obs = data['V_obs']
        self.u_pred = data['u_pred'].item()
        self.u_obs = data['u_obs'].item()
        self.gpd_c = data['gpd_c'].item()
        self.gpd_scale = data['gpd_scale'].item()
        self.spatial_filter = data['spatial_filter']
        print(f"Bias correction parameters successfully loaded from {file_path}")
