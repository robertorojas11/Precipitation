# Precipitation downscaling

Precipitation downscaling from ERA5 reanalysis. The project is **migrating** from
a deterministic 25 km → 5 km *daily* pipeline (CHIRPS / Oya) to an *hourly,
probabilistic* framework: a generative (GAN) model inspired by spateGAN-ERA5,
evaluated with probabilistic metrics (CRPS, FSS, percentile-MAE) against IMERG,
inside a RainShift-style geographic-generalization framework.

- Methodology decision: [docs/decisions/0001-methodology-choice.md](docs/decisions/0001-methodology-choice.md)
- New architecture: [docs/architecture_v3.md](docs/architecture_v3.md)
- Migration log: [docs/MIGRATION.md](docs/MIGRATION.md)
- Data sources (authoritative): [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)
- Evaluation protocol: [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md)
- Stage documentation: [docs/stages/README.md](docs/stages/README.md)

## Project history

This repository previously shipped a deterministic daily CHIRPS/Oya U-Net
(`v2_clean`) and an in-progress walk-forward CHIRPS study (`chirps_v3`). An audit
(`docs/audit/auditoria_codigo_vs_papers.md`) against the two reference papers
concluded that neither could be applied as-is. The deterministic work was
**archived, not deleted** — it is still runnable:

- [legacy/README.md](legacy/README.md) — what was archived and how to run it
- `outputs/legacy_v2_clean_deterministic/` — its checkpoints, histories, reports
- git tag `pre-migration-v2_clean` — the exact pre-migration state

## Quick checks

```bash
python -m unittest discover -s tests
```

## Credentials

The new acquisition modules need credentials and fail with an explicit message
if they are missing (see `env.example`):

- Copernicus CDS (`CDSAPI_URL`, `CDSAPI_KEY`) — pure ERA5 cp + lsp
- NASA Earthdata (`EARTHDATA_TOKEN`) — IMERG V07

Data and checkpoints under `/mnt/data-r2/RobertoRojas/downscaling/` are external
artifacts and are not deleted by repository maintenance.
