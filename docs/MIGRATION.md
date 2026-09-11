# Migration log — deterministic v2 → v2-generative framework

## Intento de migración #1 — revertido el 2026-09-07

Se ejecutó una primera versión del prompt de migración sobre la rama
`migration/v3-generative` (renombrada luego `migration-attempt-1-backup`),
completando sus 8 PASOS en 9 commits atómicos, **sin tocar `master`**:

- Auditoría reubicada a `docs/audit/` + papers P1 (spateGAN-ERA5) / P2
  (RainShift) añadidos.
- `docs/decisions/0001-methodology-choice.md` (adopta esqueleto RainShift + GAN
  inspirado en spateGAN-ERA5), `docs/audit/data_resolution_check.md`.
- Adquisición nueva: `src/data_extraction/{era5_cds,imerg}.py` (**nunca
  ejecutada** — sin credenciales CDS/Earthdata; **cero datos descargados**).
- Modelo determinista + `chirps_v3` archivados en `legacy/` (aún ejecutables).
- `outputs/v2_clean/` → `outputs/legacy_v2_clean_deterministic/` (filesystem).
- Framework nuevo: `src/framework/`, `src/models/{generator,discriminator,…}`,
  `src/training/{losses_gan,train_gan}`, `src/evaluation/*`,
  `src/utils/provenance.py`.
- 43 tests nuevos (63 en total, todos pasando).
- Docs: `architecture_v3.md`, `DATA_SOURCES.md`, `EVALUATION_PROTOCOL.md`.

**Se decidió revertir** para aplicar un prompt de implementación nuevo y
mejorado, definido tras revisar dos papers adicionales (Chang & Sapsis 2026,
*Extreme Event Aware η-Learning*; Avila 2026, *Downscaling Tropical Cyclone
Wind Speed Using Machine Learning*) antes de continuar, en vez de seguir sobre
el intento #1.

**Rollback ejecutado (2026-09-07):**
- `git checkout master` — `master` intacto en `b6de7ec` (= tag
  `pre-migration-v2_clean`).
- Rama `migration/v3-generative` renombrada a **`migration-attempt-1-backup`**.
- `outputs/legacy_v2_clean_deterministic/` restaurado a `outputs/v2_clean/`.
- Caches y directorios huérfanos (`legacy/`, `src/framework/`,
  `src/evaluation/`) eliminados del working tree de `master`.

**Estado conservado del intento #1 (NO revertido):**
- Todo el código y docs del intento #1 viven íntegros en la rama
  `migration-attempt-1-backup` + tag `pre-migration-v2_clean`.
- Ningún dato crudo se movió ni borró (`raw/era5`, `raw/era5_pl`, `raw/chirps`,
  `raw/dem` intactos; no se descargó ERA5 cp/lsp ni IMERG).
- `master` quedó 100% prístino por decisión del usuario: los papers y la
  auditoría no se copiaron a `master`; el prompt del intento #2 los
  re-aportó (ver más abajo).

## Intento de migración #2 — en curso desde 2026-09-11

Prompt de implementación nuevo, recibido el 2026-09-11, que reemplaza la
decisión de arquitectura única del intento #1 por **dos candidatos en
paralelo** (GAN inspirado en spateGAN-ERA5 + Regresión/Difusión-residual
siguiendo Avila 2026 / Mardani et al.), ambos con el regularizador de
conciencia de extremos **η-learning** (Chang & Sapsis 2026), sobre el mismo
esqueleto experimental RainShift (P2) adoptado en el intento #1.

Decisiones registradas en `docs/decisions/0002-implementacion-v2.md`.
Confirmaciones explícitas del usuario obtenidas al inicio del intento #2
(2026-09-11):

- **`src/chirps_v3/` / `pipeline_v3.py`**: archivar a `legacy/chirps_v3/`
  (misma decisión que el intento #1, re-confirmada).
- **Credenciales CDS / Earthdata**: no disponibles todavía. Igual que en el
  intento #1, los nuevos módulos de adquisición se implementan y quedan
  listos para ejecutarse, pero **no se ejecuta ninguna descarga real** en
  esta ronda; fallan explícitamente si faltan las variables de entorno.
- **Región Puerto Rico/Caribe**: bounding box arco Antillas Mayores +
  Menores, lon [-85°, -59°], lat [10°, 23°] (ver `docs/DATA_SOURCES.md`).

El estado de avance del intento #2, paso a paso, se registra en los commits
de `master` a partir de este archivo y en `docs/decisions/0002-*.md`.
