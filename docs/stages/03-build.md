# Stage 3 — Build the versioned dataset

## Purpose and inputs

This stage converts each valid source-index date into one geospatially aligned,
mask-aware sample. Inputs are ERA5 surface GeoTIFFs, ERA5 pressure-level
GeoTIFFs, CHIRPS or cleaned Oya target GeoTIFFs, NASADEM, and the source index.
Outputs never overwrite the historical dataset; they live under
`$LOCAL_DATA_DIR/v2_clean`.

## Common grid and resampling

All arrays use the 460 × 720 EPSG:4326 grid described in raw validation.
Continuous atmospheric fields and DEM use bilinear resampling. Accumulated
target precipitation uses average resampling with an independently propagated
coverage mask. ERA5-Land precipitation is converted from metres to mm:

\[
P_{ERA5,mm}=1000P_{ERA5,m}.
\]

Every atmospheric pixel is valid only if every feature channel is finite and
below the (10^{10}) sentinel threshold.

## Land and target masks

The land/study mask is derived from stable CHIRPS coverage intersected with
finite NASADEM elevation greater than −100 m. The final supervised mask is

\[
M_{target}=M_{precipitation\ QC}\land M_{land}.
\]

Oya additionally requires at least 30 slots. Invalid tensor values are filled
with zero only alongside their mask; those fills are never observations.

## Atmospheric and physical features

There are 18 atmospheric channels. Three derived physical/topographic channels
are added for later conditioning:

1. Upslope lifting:

   \[
   w_o=u_{850}\frac{\partial z}{\partial x}
      +v_{850}\frac{\partial z}{\partial y},
   \qquad
   U=\max\left(0,w_o\frac{RH_{850}}{100}\right).
   \]

   Terrain derivatives convert the 0.05° grid to metres using 111,120 m per
   latitude degree and a cosine correction for longitude.

2. Smith–Barstad-style spectral response. With mean wind ((U,V)), terrain
   Fourier transform \(\hat h\), wave numbers \((k,l)\), and intrinsic
   frequency \(\omega=kU+lV\):

   \[
   \hat P =
   \frac{C_w i\omega\hat h}
   {(1-i\omega\tau_c)(1-i\omega\tau_f)+\epsilon},
   \]

   using \(\tau_c=\tau_f=1000\) s, \(C_w=0.005\), and
   \(\epsilon=10^{-12}\). The inverse FFT is clipped at zero.

3. NASADEM elevation.

Existing compatible feature arrays may be referenced through
`feature_source_npz` to avoid recomputing them; target values and masks are
always rebuilt.

## Artifacts and provenance

Each accepted date produces
`v2_clean/processed/<target>/<split>/<date>.npz` containing target, target
mask, input mask, land mask, optional Oya slot counts, grid metadata, version,
and either features or a feature-source reference. The index records QC
statistics, rejection reasons, output path, and SHA-256 of the raw target.

`manifest_<target>.json` records source index, total/accepted/rejected counts,
grid definition, threshold, and a deterministic manifest hash.

## Connection to processed validation

The next gate opens every indexed artifact, verifies its required arrays and
feature references, rejects finite sentinels/nonfinite valid values, and checks
exact artifact/index parity. A manifest alone cannot authorize preparation.
