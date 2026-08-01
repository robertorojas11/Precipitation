# Stage 2 — Raw-data validation

## Purpose

Raw validation is the first scientific gate. It ensures every source-index row
marked `valid_flag=True` has a usable target raster on the exact model domain.
This distinction matters: source exports extend beyond the modeled grid and may
contain legitimate or anomalous values that must not affect the experiment.

## Geographic alignment

Each target is reprojected before QC to the same grid later used for training:

- CRS: EPSG:4326;
- upper-left origin: (−120°, 35°);
- pixel size: 0.05° × 0.05°;
- shape: 460 rows × 720 columns;
- bounds: longitude −120° to −84°, latitude 12° to 35°.

Precipitation uses average resampling. A separate source-valid field is
reprojected with average resampling, and a destination cell is covered only if
coverage is at least 0.999. This prevents partially covered boundary cells from
being treated as full observations.

## Precipitation quality contract

For value (p_i), validity is

\[
M_i = M_{source,i}
\land \operatorname{finite}(p_i)
\land |p_i| < 10^{10}
\land 0 \le p_i \le 1000\;\text{mm/day}.
\]

The 1,000 mm/day ceiling is a rejection threshold, not clipping. A file is
rejected for any finite sentinel, any above-threshold value within the model
domain, valid coverage below 5%, or no valid precipitation. Negative values are
masked and counted diagnostically. QC records minimum, maximum, median, and
95th, 99th, and 99.9th percentiles.

For Oya, the second band must exist and target coverage is additionally
intersected with `slot_count >= 30`.

## Count parity and output

The validator counts accepted train/validation/test dates and compares them
with the source index. All three splits must be nonempty. Missing files,
rejections, and mismatches are written with date, path, error type, and reasons.

```bash
python -m src.data_preprocessing.validate_dataset \
  --target chirps --stage raw \
  --output outputs/v2_clean/chirps/validation/raw.json
```

The stage exits nonzero unless `accepted=true`. Resume accepts the artifact
only when that field remains true.

## Connection to build

Successful raw validation authorizes building but does not create processed
samples. The build stage repeats alignment and masks while combining targets
with ERA5, pressure fields, DEM, and physical features. This intentional
separation keeps validation independently auditable.
