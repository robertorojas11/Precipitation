"""Orographic feature calculations for precipitation downscaling."""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin

def resample_dem(dem_path, output_height=460, output_width=720):
    """Loads and resamples NASADEM to the target grid coordinates.

    Args:
        dem_path (str): Path to the NASADEM GeoTIFF.
        output_height (int): Height of target grid.
        output_width (int): Width of target grid.

    Returns:
        np.ndarray: Resampled elevation array of shape (output_height, output_width).
    """
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)
        resampled = np.empty((output_height, output_width), dtype=np.float32)
        # Mexico bounding box: Lon -120 to -84, Lat 12 to 35
        dst_transform = from_origin(-120.0, 35.0, 0.05, 0.05)
        
        reproject(
            source=dem_data,
            destination=resampled,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear
        )
        return resampled

def compute_terrain_gradients(elevation):
    """Computes spatial derivatives of terrain elevation in meters.

    Args:
        elevation (np.ndarray): Elevation array of shape (H, W).

    Returns:
        tuple: (dz_dx, dz_dy) of shape (H, W) in meters/meter.
    """
    elevation_clean = np.nan_to_num(elevation, nan=0.0)
    height, width = elevation_clean.shape
    
    # y-axis (latitude) decreases downwards (35.0 to 12.0)
    # x-axis (longitude) increases rightwards (-120.0 to -84.0)
    # Spacing is 0.05 degrees
    
    # Compute basic row/column gradients (element differences)
    dz_dy_row, dz_dx_col = np.gradient(elevation_clean)
    
    # Convert to degree differences (row step is -0.05, col step is +0.05)
    dz_dy_deg = dz_dy_row / (-0.05)
    dz_dx_deg = dz_dx_col / 0.05
    
    # Generate latitude grid to scale longitude spacing
    lats = 35.0 - np.arange(height)[:, np.newaxis] * 0.05
    lats_rad = np.radians(lats)
    
    # 1 degree of latitude = 111,120 meters
    dz_dy = dz_dy_deg / 111120.0
    dz_dx = dz_dx_deg / (111120.0 * np.cos(lats_rad))
    
    return dz_dx, dz_dy

def compute_upslope_model(u, v, rh, dz_dx, dz_dy):
    """Computes the Upslope moisture flux orographic lifting field.

    Args:
        u (np.ndarray): Zonal wind component at 850 hPa.
        v (np.ndarray): Meridional wind component at 850 hPa.
        rh (np.ndarray): Relative humidity component at 850 hPa.
        dz_dx (np.ndarray): Terrain gradient in x-direction.
        dz_dy (np.ndarray): Terrain gradient in y-direction.

    Returns:
        np.ndarray: Upslope physical field of same shape as inputs.
    """
    # Clean inputs of NaNs
    u_clean = np.nan_to_num(u, nan=0.0)
    v_clean = np.nan_to_num(v, nan=0.0)
    rh_clean = np.nan_to_num(rh, nan=0.0)
    
    # w_orographic = u * dz_dx + v * dz_dy (dot product of velocity and terrain gradient)
    w_orographic = u_clean * dz_dx + v_clean * dz_dy
    
    # Scale by relative humidity (fraction)
    upslope = w_orographic * (rh_clean / 100.0)
    
    # Clip negative values (downslope flows do not generate rain in this prior)
    return np.maximum(upslope, 0.0)

def compute_spectral_model(u, v, elevation, tau_c=1000.0, tau_f=1000.0, C_w=0.005):
    """Computes the Smith & Barstad (2004) linear model in the Fourier domain.

    Args:
        u (np.ndarray): Zonal wind component at 850 hPa.
        v (np.ndarray): Meridional wind component at 850 hPa.
        elevation (np.ndarray): Elevation array of shape (H, W).
        tau_c (float): Condensation time scale in seconds.
        tau_f (float): Fallout/advection time scale in seconds.
        C_w (float): Condensation sensitivity factor (uplift factor) in kg/m^3.

    Returns:
        np.ndarray: Spectral physical precipitation field of shape (H, W).
    """
    # Clean inputs of NaNs
    u_clean = np.nan_to_num(u, nan=0.0)
    v_clean = np.nan_to_num(v, nan=0.0)
    elevation_clean = np.nan_to_num(elevation, nan=0.0)
    
    height, width = elevation_clean.shape
    
    # Calculate domain average winds
    U = np.mean(u_clean)
    V = np.mean(v_clean)
    
    # Spacing in meters (using average latitude of Mexico ~ 23.5 degrees)
    dy_meters = 0.05 * 111120.0
    dx_meters_avg = 0.05 * 111120.0 * np.cos(np.radians(23.5))
    
    # Zonal and meridional wave numbers (radians per meter)
    ks = np.fft.fftfreq(width, d=dx_meters_avg) * 2.0 * np.pi
    ls = np.fft.fftfreq(height, d=dy_meters) * 2.0 * np.pi
    
    l_grid, k_grid = np.meshgrid(ls, ks, indexing='ij')
    
    # Intrinsic frequency omega = k * U + l * V
    omega = k_grid * U + l_grid * V
    
    # Fourier transform of elevation
    h_hat = np.fft.fft2(elevation_clean)
    
    # Smith & Barstad transfer function
    # P_hat = C_w * i * omega * h_hat / ((1 - i * omega * tau_c) * (1 - i * omega * tau_f))
    num = C_w * 1j * omega * h_hat
    denom = (1.0 - 1j * omega * tau_c) * (1.0 - 1j * omega * tau_f)
    
    # Avoid division by zero at zero frequency (omega = 0)
    # Add a small epsilon to denominator to prevent NaN
    eps = 1e-12
    P_hat = num / (denom + eps)
    
    # Inverse FFT to return to spatial domain
    P_spatial = np.real(np.fft.ifft2(P_hat))
    
    # Clip negative values
    return np.maximum(P_spatial, 0.0)
