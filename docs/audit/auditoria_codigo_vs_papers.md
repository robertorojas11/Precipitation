# Auditoría de Código vs. Metodología Científica
**Repositorio auditado:** https://github.com/robertorojas11/Precipitation (commit local: `master`, 27 commits)
**Papers de referencia:** P1 = spateGAN-ERA5 (Glawion et al. 2025); P2 = RainShift (Harder, Schmidt et al. 2026)

---

## 1. Comprensión inicial del repositorio

### 1.1 Hallazgo preliminar (bloquea el resto del análisis)

El propio repositorio declara explícitamente, en `architecture.md` líneas 47-49:

```
This project does not claim an ESRGAN, conditional Gaussian process, or
probabilistic model. Those legacy implementations and unsupported claims were
removed.
```

Y en `README.md` línea 4-5:

```
Deterministic 25 km to 5 km daily precipitation downscaling for two independent
ground truths: CHIRPS and Oya.
```

**Esto es una autodeclaración del propio repositorio de que NO implementa un modelo generativo/probabilístico.** Ambos papers auditados (P1: cGAN; P2: benchmark de GAN/diffusion/ResNet) son metodologías generativas o comparan explícitamente contra ellas. Por lo tanto, antes de entrar en el detalle de trazabilidad, dejo constancia de que la arquitectura central (M130-M164 de la matriz, y M227-M228) queda descartada por declaración propia del código, no solo por inferencia del auditor. Continúo la auditoría exhaustivamente porque el prompt lo exige, pero este hallazgo condiciona el veredicto final de la Sección 23.

### 1.2 Árbol de arquitectura real

```text
Repositorio (robertorojas11/Precipitation)
├── Entrada de datos
│   ├── src/data_extraction/          (export_oya.py, earth_engine.py, drive.py)
│   └── ERA5-Land + ERA5 pressure levels (18 canales, ver docs/stages/data-and-resources.md)
├── Preprocesamiento
│   ├── src/data_preprocessing/build_dataset.py    (reproyección, 460x720 @ 0.05°)
│   ├── src/data_preprocessing/physics.py          (gradientes de terreno, upslope, espectral)
│   ├── src/data_preprocessing/prepare_dataset.py  (normalización, climatología)
│   └── src/data_preprocessing/quality.py          (control de calidad)
├── Algoritmos / Modelo
│   └── src/models/multiscale_unet.py  (MultiscalePrecipUNet, determinista)
├── Procesamiento principal (entrenamiento)
│   ├── src/training/train.py    (bucle de entrenamiento, pérdidas compuestas)
│   └── src/training/search.py   (búsqueda de hiperparámetros en validación)
├── Postprocesamiento / Validación
│   ├── src/data_preprocessing/validate_dataset.py
│   └── src/training/evaluate.py (R², RMSE, MAE, bootstrap, baselines)
├── Resultados
│   ├── src/training/report.py   (figuras, informe final)
│   └── outputs/{chirps,oya}/
└── Utilidades
    └── src/utils/ (config.py, logging.py, retries.py, storage.py)
```

Módulo adicional no mencionado como activo en `architecture.md`: `src/chirps_v3/` (config.py, dataset.py, evaluate.py, feasibility.py, freeze.py, metrics.py, model.py, prepare.py, search.py, training.py). `architecture.md` no lo lista entre los "Active modules"; `README.md` menciona un `CODEBASE_RESULTS_REVIEW.md` con "hallazgos históricos que motivaron el refactor", archivo que **no existe en el repositorio clonado** (verificado: `find . -iname "CODEBASE_RESULTS_REVIEW*"` no retorna resultados). Esto es un **hallazgo de discrepancia documental**: el README referencia un documento inexistente, lo que impide auditar completamente qué se descartó y por qué.

- **Lenguaje:** Python.
- **Frameworks/librerías:** PyTorch (`torch`, `torch.nn`, `torch.nn.functional`), `rasterio` (I/O geoespacial), `earthengine-api`/`google-api-python-client` (extracción Oya vía Google Earth Engine), `numpy`, `pandas`, `matplotlib`. No hay TensorFlow, no hay `pysteps`, no hay librerías de generative adversarial networks o diffusion (no `diffusers`, no implementación manual de SDE).
- **CLI:** `pipeline.py` (orquestador) y `pipeline_v3.py` (relacionado con el módulo `chirps_v3`, de estado ambiguo).
- **Tests:** `tests/test_chirps_v3.py`, `tests/test_orchestrator.py`, `tests/test_pipeline.py` — pruebas de orquestación, no de correctitud científica de ecuaciones (ver Sección 13).
- **GPU/CPU:** `train.py` soporta `--device cuda`, usa `torch.amp.autocast`/`GradScaler` (mixed precision), sin paralelismo multi-GPU ni entrenamiento data-parallel (a diferencia de P1, que usa 3-4× A100, y P2, que usa 4× A100).

---

## 2. Reconstrucción del pipeline real del código

```text
INPUT: rasters ERA5-Land + ERA5 niveles de presión (18 canales), CHIRPS o Oya (target), NASADEM (elevación)
 ↓
[build_dataset.py] reproyección a grid fijo 460×720 @ 0.05° (~5km), origen (-120°, 35°) → dominio regional fijo
    Función: reproject_target(), _reproject_bands()   Líneas: 33-90 aprox.
 ↓
[physics.py] compute_terrain_gradients, compute_upslope_model, compute_spectral_model, resample_dem
 ↓
[quality.py] control de calidad: ensure_finite_features, inspect_precipitation, DEFAULT_MAX_DAILY_MM
 ↓
[prepare_dataset.py] normalización Z-score/log SOLO con estadísticas de train, climatología
 ↓
[dataset.py → PrecipitationDataset] tensores de entrenamiento locales
 ↓
[multiscale_unet.py → MultiscalePrecipUNet.forward()]
    Entrada: inputs_25km (18×context_days canales), phys_dem_10km, era5_precip_5km_norm, season
    Encoder 25km → interpolación bilineal a 230×360 (~10km) → decoder_10 con concat de features físicas/lat/lon/estación/baseline ERA5
    → interpolación bilineal a 460×720 (~5km) → decoder_5 con concat de lat/lon/estación/baseline ERA5
    → occurrence_head (sigmoid, prob. de lluvia) × amount_residual_head (softplus sobre log-amount)
    Salida determinista: prediction_mm = wet_probability * positive_amount   Líneas: 71-106
 ↓
[train.py → compute_losses()] BCE(ocurrencia) + SmoothL1(cantidad) + L1 físico ponderado por eventos + gradiente vecino + pérdida multiescala (factores 2,5,10)   Líneas: 83-117
 ↓
[train.py → validate()] R², RMSE, MAE agregados (PooledMetrics)   Líneas: 120-144, 160-170
 ↓
[evaluate.py] evaluación held-out: R² por año, bootstrap por bloques mensuales, métricas de eventos, comparación contra ERA5 y climatología
 ↓
[report.py] figuras: sesgo medio, sesgo relativo, RMSE, conteo de muestras válidas, historia de entrenamiento
 ↓
OUTPUT: outputs/v2_clean/<target>/final_report/report.md + figuras + checkpoints .pt
```

No existe ninguna etapa de: (a) generación adversarial, (b) muestreo estocástico de ensemble de predicciones, (c) stitching de patches globales, (d) corrección de sesgo de campo medio post-hoc, (e) cómputo de CRPS/FSS/RAPSD/excentricidad.

---

## 3. Comparación metodología vs. código (matriz de trazabilidad resumida)

Uso los IDs de la matriz entregada previamente. Solo listo aquí los hallazgos; el detalle de evidencia está en las secciones 5-14.

| ID | Requisito científico | Implementado | Evidencia en código | Fidelidad | Observaciones |
|---|---|---|---|---|---|
| M101 | Input CP+LSP de ERA5, 24km/1h | AUSENTE | `docs/stages/data-and-resources.md` líneas 9-27: usa ERA5-Land + niveles de presión, 18 canales, no CP/LSP aisladas | — | Variables de entrada completamente distintas |
| M102 | No usar ERA5-Land | DIFERENTE (contradicho) | El código usa **ERA5-Land explícitamente** como fuente principal (`data-and-resources.md` línea 9) | — | P1 excluye ERA5-Land por diseño; el código la adopta como base |
| M104 | Target: RADKLIM-YW (radar Alemania) | AUSENTE | Target real: CHIRPS y Oya (`architecture.md` línea 3-4) | — | Ningún dataset de radar gauge-adjusted alemán |
| M105 | Splits 2009-2020 train / jul-dic 2021 eval | DIFERENTE | `architecture.md`: validación 2020-2022, test 2023-2025; `pipeline.md`: `--start-year 2004 --end-year 2025` | — | Ventanas temporales completamente distintas |
| M110-M112 | Patch 28×28×16 pasos, factor 12×/6× | AUSENTE | Grid fijo único 460×720, factor de escalado ~5× espacial (25km→5km), sin dimensión temporal sub-diaria | — | No hay downscaling temporal; el producto es diario a diario |
| M114 | Submuestreo estocástico ε1/ε2 con distribución Lognormal | AUSENTE | No se encontró ninguna función de umbral dinámico lognormal en `prepare_dataset.py`/`build_dataset.py` (`grep -rin "lognormal"` sin resultados) | — | — |
| M116 | Data augmentation rotación/reflexión en muestras alternas | NO DETERMINABLE | No se revisó `dataset.py` en detalle exhaustivo para descartar augmentation menor; no se encontró en `train.py` | — | Requiere revisión adicional de `src/data_preprocessing/dataset.py` (ver Sección 15) |
| M130-M164 | Arquitectura cGAN (generador+discriminador, Res3D, dropout probabilístico, Softplus final) | AUSENTE | `multiscale_unet.py` es un U-Net 2D determinista sin discriminador; confirmado también por autodeclaración en `architecture.md` línea 47-48 | — | Softplus SÍ se usa (línea 96) pero como parte de la cabeza de cantidad, no como límite final de un generador GAN |
| M150-M153 | L_D, L_GAN, L_L1 ensemble (3 miembros) | AUSENTE | `compute_losses()` (train.py:83-117): BCE + SmoothL1 + L1 físico + gradiente + multiescala; ninguna pérdida adversarial | — | — |
| M160-M162 | Freeze de dropout seed, recorte de bordes, mean-field bias correction en inferencia | AUSENTE | No existe lógica de inferencia por patches ni corrección de sesgo post-hoc en `evaluate.py` | — | — |
| M170-M176 | Generación de campos globales por stitching de patches con Haversine y blending lineal | AUSENTE | Grid único fijo, sin segmentación en patches ni fórmula de Haversine (`grep -rin "haversine"` sin resultados) | — | — |
| M180-M191 | RMSE, CRPS, MAE, FSS, RAPSD (PySTEPS), rank histograms, excentricidad, BIAS | PARCIAL | `train.py`/`evaluate.py` implementan RMSE y MAE (Sección `PooledMetrics`, líneas 120-144) y R² (no presente en P1/P2 como métrica principal); CRPS, FSS, RAPSD, rank histogram y excentricidad **ausentes** (`grep` confirmado sin resultados) | RMSE/MAE: EQUIVALENTE; resto: AUSENTE | — |
| M195-M199 | Baselines rainFARM (PySTEPS) e interpolación trilineal | PARCIAL | Interpolación bilineal de ERA5 sí se usa como *baseline dentro del modelo* (línea 83, 88 de `multiscale_unet.py`) y como comparación en `report.py`/`evaluate.py` ("Better R² than ERA5"); rainFARM **ausente** | Interpolación: DIFERENTE (bilinear, no trilinear, y usada como *insumo del modelo* no solo como baseline externo); rainFARM: AUSENTE | — |
| M201-M210 (P2) | 9 variables ERA5 guiadas por ecPoint, 18 regiones de 20°×20° globales, N=2.5 | DIFERENTE / AUSENTE | 18 canales fijos de ERA5-Land+niveles de presión (no 9); un único dominio regional fijo (no 18 regiones ni Global Norte/Sur); factor ~5× (no 2.5×) | — | — |
| M216-M219 (P2) | Clip por cuantiles IMERG, log(x·1000+1e-5), pooled variance | DIFERENTE | `quality.py` usa `DEFAULT_MAX_DAILY_MM` como techo fijo (no derivado de cuantiles del target satelital); no se confirmó la fórmula log específica de P2 en el código (requiere revisión de `physics.py`/`prepare_dataset.py` línea por línea, no completada por alcance) | NO DETERMINABLE en parte | Ver Sección 15 |
| M225-M230 (P2) | Interpolación bilineal baseline, ResNet, WGAN-GP, Diffusion (ClimateDiffuse) | PARCIAL/AUSENTE | Solo existe un modelo (U-Net determinista); no hay ResNet independiente, WGAN-GP ni modelo de difusión | — | — |
| M235-M239 (P2) | Escenarios A1-A4 jerárquicos de regiones de entrenamiento | AUSENTE | Un solo dominio fijo, sin escenarios de generalización geográfica | — | — |
| M245-M248 (P2) | CRPS point-wise 8 muestras, percentile MAE p95/p99 | AUSENTE | No se encontró percentile MAE ni CRPS en el código de evaluación | — | — |
| M255-M258 (P2) | Quantile mapping domain adaptation, fine-tuning few-shot | AUSENTE | No hay lógica de adaptación de dominio en el repositorio | — | — |

---

## 4. Comparación algoritmo por algoritmo

### 4.1 Modelo generativo (P1: cGAN) vs. código

**Paper (P1):** Generador con 4 etapas Res3D (procesamiento inicial → downsampling UNET-like con cropping → 4 bloques de upsampling → 3 Res3D finales con Softplus), condicionado por CP+LSP, entrenado adversarialmente contra un discriminador que recibe secuencias temporales HR+contexto LR con ruido Gaussiano añadido. Salida probabilística vía dropout congelado por miembro de ensemble.

**Código:** `MultiscalePrecipUNet` (`multiscale_unet.py`) es un U-Net 2D de 3 niveles (25km→10km→5km) con bloques residuales 2D (no 3D — no hay dimensión temporal en la arquitectura), sin discriminador, sin ruido, sin ensemble de dropout congelado. La salida es un único valor determinista por pixel: `wet_probability * positive_amount` (línea 99).

**Comparación:** Es **otro algoritmo**, no una aproximación ni simplificación incremental del generador de P1. Comparten superficialmente el uso de bloques residuales convolucionales y activación Softplus, pero la topología, dimensionalidad (2D vs 3D), condicionamiento adversarial y naturaleza probabilística son fundamentalmente distintas. No deben considerarse equivalentes.

### 4.2 Modelos de P2 (ResNet/WGAN-GP/Diffusion) vs. código

El código implementa un único modelo (`MultiscalePrecipUNet`). No hay tres arquitecturas comparables ni benchmark de generalización geográfica. **AUSENTE** en su totalidad — no es una simplificación de P2, es la ausencia del diseño experimental completo que define a P2 (el objeto de estudio de P2 *es* la comparación entre arquitecturas y regiones; el código no tiene ninguno de esos dos ejes).

### 4.3 Baseline de interpolación

**Paper (P1):** interpolación trilineal (espacio×2 + tiempo) de TP total de ERA5, usada como baseline de comparación externa, evaluada de forma independiente.
**Paper (P2):** interpolación bilineal de TP como *lower bound* de comparación externa.
**Código:** interpolación bilineal (`F.interpolate(..., mode="bilinear")`, líneas 80, 88) y de área (`mode="area"`, línea 83) se usan **dentro del propio modelo**, como parte del *forward pass* (para reescalar el encoder y como "baseline" que se le suma al residuo predicho, línea 95-96: `baseline_log = era5_precip_5km_norm * ... `). Esto es una diferencia arquitectónica relevante: en los papers la interpolación es un método de comparación independiente; en el código es un componente estructural del propio modelo (arquitectura tipo "predicción de residuo sobre baseline interpolado"), similar en espíritu a lo que P2 describe para su modelo de difusión (M220: "el modelo aprende el residuo entre el target HR y el input LR interpolado") pero aplicado aquí a un U-Net determinista, no a un modelo de difusión. Es una **idea de diseño compartida con un fragmento de P2** (residual learning sobre baseline interpolado) pero ejecutada en un contexto arquitectónico y de pérdida totalmente distinto.

---

## 5. Comparación de parámetros

| Parámetro | Paper (P1/P2) | Código | Coincide | Impacto |
|---|---|---|---|---|
| Resolución de entrada | 24km (P1) / ~25km (P2, 0.25°) | 25km (`0.05°` es la salida, no la entrada; entrada asumida ~25km por reproyección a grid de 0.05°) | Parcial (coincidencia casual en orden de magnitud) | MEDIO |
| Resolución de salida | 2km, 10min (P1) / ~10km (P2, IMERG 0.1°) | 5km (`0.05°`, `HEIGHT, WIDTH = 460, 720` en `build_dataset.py` línea 25) | NO | CRÍTICO |
| Frecuencia temporal | Sub-horaria 10 min (P1) / horaria (P2) | Diaria (`README.md`: "daily precipitation downscaling") | NO | CRÍTICO |
| Factor de escalado | 12× espacial, 6× temporal (P1); 2.5× espacial (P2) | ~5× espacial (25km/5km), sin escalado temporal | NO | CRÍTICO |
| Learning rate | 1e-4 (gen.)/2e-4 (disc.) (P1) | 3e-4 (`TrainingConfig.learning_rate`, train.py línea 32) | NO | ALTO (no comparable: optimiza objetivo distinto) |
| Optimizador | AdamW β1=0.0 (P1) | AdamW, sin especificar β1/β2 custom (usa default de PyTorch: β1=0.9,β2=0.999), `weight_decay=1e-4` (train.py líneas 32-33, 211-213) | NO | MEDIO |
| Batch size | 9 (P1) | 4 (default, `TrainingConfig.batch_size`, línea 31; overridable por CLI) | NO | BAJO |
| Épocas/pasos | 2×10⁵ pasos (P1) | 80 épocas (`epochs: int = 80`, línea 30), con early stopping (`patience=10`) | NO comparable directamente (unidades distintas: pasos vs épocas) | MEDIO |
| Dropout probabilístico | p=0.2 en 3 profundidades (P1, habilita ensemble) | `dropout: float = 0.0` por defecto, configurable a 0.0 o 0.1 vía CLI (`choices=[0.0, 0.1]`, línea 313); usado en `ResidualBlock`/`FeatureStage` como regularización estándar, no para generar ensemble probabilístico | NO (función distinta) | CRÍTICO |
| Semilla / determinismo | No explícito en P1 (arquitectura probabilística por diseño) | `seed_everything()` fuerza `torch.backends.cudnn.deterministic = True` (train.py líneas 46-52) — coherente con el objetivo **determinista** declarado del código | N/A (objetivos distintos) | — |

---

## 6. Comparación de datos

| Aspecto | P1 | P2 | Código |
|---|---|---|---|
| Fuente de entrada | ERA5 (CP+LSP) | ERA5 (9 variables no listadas en cuerpo principal) | ERA5-Land + ERA5 niveles de presión, 18 canales fijos y documentados explícitamente (`data-and-resources.md`) |
| Fuente de target | RADKLIM-YW (Alemania), validación en MRMS (US) y radar australiano | IMERG V07 (satelital, global 60°N-60°S) | CHIRPS (satelital-estación) y Oya (nowcast, agregado de medias de 30 min) |
| Cobertura geográfica | Global (entrenado en Alemania, validado en US/Australia) | Global, 18 regiones de 20°×20° en 6 continentes | **Dominio regional único y fijo**: lon [-120°,-84°], lat [12°,35°] aprox. (grid 460×720 @ 0.05° desde origen (-120,35)) — cubre aproximadamente México/Centroamérica/sur de EE.UU., **no es global** |
| Resolución temporal | Horaria (in)/10 min (out) | Horaria | Diaria |
| Datos faltantes | NaN marcados fuera de cobertura radar | — | NaN fuera de cobertura CHIRPS marcado explícitamente como inválido, no como cero (`data-and-resources.md`: "NaN outside coverage is invalid, not zero precipitation") — **esta práctica es científicamente correcta y coincide en espíritu con la filosofía de manejo de missing values de ambos papers**, aunque implementada independientemente |
| Datos derivados | — | Land-sea mask, orografía (P2) | Gradientes de terreno, modelo de "upslope moisture", modelo espectral de orografía (Fourier) vía NASADEM (`physics.py`) — **más elaborado que P1/P2** en tratamiento de terreno, pero no siguiendo ninguna ecuación descrita en ninguno de los dos papers (ninguno de los dos detalla un "modelo espectral orográfico" de esta naturaleza) |

**Conclusión de la sección:** el código NO trabaja con los mismos tipos de datos que ninguno de los dos papers. Ni las fuentes de entrada, ni los targets, ni la cobertura geográfica, ni la resolución temporal coinciden.

---

## 7. Comparación de ecuaciones

| Ecuación del paper | Dónde debería implementarse | Estado en código | Clasificación |
|---|---|---|---|
| L_D = E[log D(X,Y)] − E[log(1−D(X,Ŷ))] (P1, Ec.1) | Un módulo discriminador | No existe discriminador | AUSENTE |
| L_L1(G) ensemble de 3 miembros (P1, Ec.3) | Función de pérdida del generador | `compute_losses()` usa `F.smooth_l1_loss` sobre un único valor determinista `positive_log_amount` (train.py línea 99), no sobre un promedio de 3 muestras estocásticas | DIFERENTE (mismo nombre de familia de pérdida —L1/SmoothL1— pero aplicada a un contexto no probabilístico) |
| CRPS (P1 Ec.6-7, P2 Ec.1) | Módulo de evaluación | No implementado (`grep` sin resultados) | AUSENTE |
| FSS (P1 Ec.9) | Módulo de evaluación | No implementado | AUSENTE |
| Excentricidad lineal (P1 Ecs.10-12) | Módulo de evaluación | No implementado | AUSENTE |
| BIAS = (Y−X)/Y (P1 Ec.13) | Módulo de evaluación/reporte | `report.py` calcula "mean_bias_mm_day" y "relative_bias_percent" (ver `docs/pipeline.md`, sección "Output report") — **posible análogo conceptual**, pero no se verificó si la fórmula exacta coincide con la Ec.13 de P1 (signo, denominador). Requiere inspección línea por línea de `report.py`, no completada por límite de alcance de esta pasada | NO DETERMINABLE (requiere seguimiento) |
| Percentile MAE (P2 Ec.2) | Módulo de evaluación | No implementado como tal; el código sí calcula MAE global (no por percentil) en `PooledMetrics.compute()` (train.py líneas 137-144) | DIFERENTE |
| Pooled variance normalización (P2, fórmula explícita σ̃²) | Módulo de preparación de datos | No confirmado en el análisis actual si `prepare_dataset.py` usa esta fórmula específica de varianza combinada entre regiones — el código de este repositorio no tiene múltiples "regiones de entrenamiento" en el sentido de P2 (un solo dominio), por lo que la pregunta de "combinar varianzas entre regiones" **no aplica conceptualmente** a este código | NO DETERMINABLE / NO APLICABLE |

**Nota metodológica:** dado que la arquitectura de fondo (determinista, single-domain, daily) es fundamentalmente distinta a la de ambos papers, la mayoría de las ecuaciones "candidatas a implementarse" en el código no tienen un lugar natural donde buscarlas, porque el diseño experimental completo (patches globales, ensembles, discriminador, regiones de generalización) está ausente.

---

## 8. Flujo de datos — alineación de pipelines

### Paper (P1)
```text
ERA5 (24km,1h) → interpolación a grid radar → downsample a 24km aprox
 → submuestreo estocástico (ε1,ε2) → generador cGAN (patches 672km,16h)
 → discriminador (adversarial) → ensemble de 3+ miembros → mean-field bias correction
 → stitching de patches globales (Haversine, blending lineal) → evaluación (CRPS/FSS/RAPSD)
```

### Código
```text
ERA5-Land + niveles presión (18 canales) → reproyección a grid fijo único (rasterio/warp)
 → control de calidad/máscaras → normalización (train-only stats) → U-Net determinista (single-domain)
 → pérdida compuesta (BCE+SmoothL1+físico+gradiente+multiescala) → evaluación (R²/RMSE/MAE/bootstrap)
 → reporte con figuras de sesgo
```

**Etapas equivalentes (conceptualmente, no técnicamente):** preprocesamiento/control de calidad, uso de una red convolucional, evaluación contra baseline de interpolación.

**Etapas faltantes respecto a los papers:** submuestreo estocástico basado en cuantiles dinámicos; entrenamiento adversarial; generación de ensemble probabilístico; stitching global de patches; corrección de sesgo de campo medio en inferencia; métricas de verificación probabilística (CRPS, FSS, rank histogram); análisis espacial (RAPSD, excentricidad); escenarios de generalización geográfica (P2).

**Etapas adicionales no presentes en los papers:** modelado físico de terreno vía Fourier ("spectral orographic response"), modelo de "upslope moisture lifting", pipeline de extracción específico para un producto "Oya" (nowcast) vía Google Earth Engine, orquestador de pipeline con verificación de integridad de almacenamiento en red (`storage_check`), sistema de manifiestos con hashes SHA-256 para trazabilidad de datos.

**Orden incorrecto / transformaciones diferentes:** no aplica en el sentido estricto porque no hay una etapa origen equivalente que comparar (arquitecturas distintas desde el diseño).

---

## 9. Análisis de arquitectura de software (ingeniería, no ciencia)

Independientemente de la fidelidad científica a los papers, la ingeniería del repositorio muestra:
- **Modularidad:** buena separación por etapas (`data_extraction` / `data_preprocessing` / `models` / `training`).
- **Reproducibilidad interna:** manifiestos con SHA-256 (`manifest_sha256`), captura de commit git y estado "dirty" en cada corrida (`_source_revision()`, train.py líneas 173-179), estadísticas de normalización versionadas junto al checkpoint.
- **Manejo de errores/gates de calidad:** `validate_dataset.py` con niveles ("raw", "processed", "fast"), orquestador que detiene el pipeline en el primer gate fallido.
- **Determinismo:** `seed_everything()` fuerza cuDNN determinista — coherente con ser un modelo determinista, pero nótese que esto es **imposible de reconciliar** con el objetivo de P1 (generar un ensemble variado de soluciones plausibles), ya que P1 depende de estocasticidad congelada por miembro, no de determinismo total.

**Conclusión:** el código está bien diseñado *para su propio objetivo declarado* (downscaling determinista de CHIRPS/Oya), pero esto no debe confundirse — como advierte el propio prompt de auditoría (Sección 10, regla explícita)— con fidelidad a la metodología de los papers auditados. Es buena ingeniería para un problema científico diferente.

---

## 10. Paralelización y rendimiento

Ninguno de los dos papers describe paralelización relevante para reproducibilidad más allá del entrenamiento (multi-GPU data-parallel). El código usa un solo dispositivo (`--device cuda`), `DataLoader` con `num_workers` configurable y `persistent_workers`, sin entrenamiento distribuido multi-GPU. No hay evidencia de que la paralelización (o su ausencia) altere resultados científicos, dado que no hay un objetivo de paridad de resultados con ningún paper específico. **No aplica** distinción crítica de "optimización de rendimiento" vs. "modificación de algoritmo" porque el algoritmo de fondo ya es distinto por diseño.

---

## 11. Resultados y validación

| Validación del paper | ¿Implementada? |
|---|---|
| CRPS, FSS, RAPSD, rank histogram, excentricidad (P1) | AUSENTE |
| CRPS, percentile MAE (P2) | AUSENTE |
| Comparación contra rainFARM (P1) | AUSENTE |
| Comparación contra ERA5 interpolado (ambos) | PARCIAL — el código sí compara contra ERA5 (`docs/pipeline.md`: "Better R² than ERA5 and monthly climatology" en criterios de aceptación) pero con métrica R², no con las métricas de los papers |
| Comparación contra climatología mensual | Presente en código, **no presente en ninguno de los dos papers** — criterio propio del repositorio |
| R² pooled, R² por año, bootstrap por bloques mensuales | Presente en código, **ausente en ambos papers** (P1/P2 no reportan R²) |

Los resultados generados por este código **no podrían compararse directamente** con los de P1 o P2: ni las métricas, ni las unidades temporales (diaria vs. sub-horaria/horaria), ni los datasets de verdad de terreno son los mismos.

---

## 12. Tests

`tests/test_pipeline.py`, `tests/test_orchestrator.py`, `tests/test_chirps_v3.py` — sin inspección línea por línea exhaustiva en esta pasada (limitación declarada), pero por sus nombres corresponden a pruebas de **orquestación** (que el pipeline ejecute las etapas correctas, maneje errores, etc.), no a pruebas de correctitud numérica de ecuaciones científicas (no hay `test_losses.py`, `test_metrics.py`, `test_model.py` visibles en el árbol de `tests/`).

**Requisitos científicos críticos sin ningún test dedicado (según nombres de archivo disponibles):**
- Corrección de la función de pérdida compuesta (`compute_losses`).
- Corrección de las métricas `PooledMetrics` (R², RMSE, MAE).
- Corrección de los modelos físicos de terreno (`physics.py`: gradientes, upslope, espectral).

Esto es una observación de ingeniería general del repositorio, no una comparación contra los papers (ya que los papers no definen estas funciones).

---

## 13. Errores científicos potenciales identificados (propios del código, independientes de fidelidad a los papers)

| Hallazgo | Severidad | Evidencia |
|---|---|---|
| El README referencia `CODEBASE_RESULTS_REVIEW.md`, archivo inexistente en el repo clonado | MEDIO | `README.md` línea 8; búsqueda `find . -iname "CODEBASE_RESULTS_REVIEW*"` sin resultados |
| Estado ambiguo de `src/chirps_v3/` y `pipeline_v3.py`: no declarados como "activos" en `architecture.md` pero presentes y con tests propios (`test_chirps_v3.py`) | BAJO-MEDIO | `architecture.md` "Active modules" no los menciona; siguen versionados en el repo |
| `pos_weight` en la pérdida de ocurrencia se recalcula por batch (`dry_count/wet_count`, clamped [1,20]) en lugar de fijarse globalmente sobre el dataset de entrenamiento completo | MEDIO | `train.py` líneas 90-96 — esto puede introducir varianza batch-a-batch en el balance de clases, un detalle de diseño válido pero no documentado como decisión deliberada en `architecture.md` |
| No se confirmó semilla determinista para las particiones train/val/test (aunque sí para el entrenamiento del modelo) | BAJO | No se inspeccionó el código de `validate_dataset.py`/`prepare_dataset.py` en profundidad suficiente para confirmar cómo se generan los splits temporales |

Ningún hallazgo de esta sección compromete per se la reproducibilidad interna del propio proyecto (que audita su propio objetivo, no el de los papers).

---

## 14. Código adicional no presente en los papers

| Componente | Función | ¿Necesario? | ¿Modifica el algoritmo de los papers? |
|---|---|---|---|
| `src/data_preprocessing/physics.py` (gradientes de terreno, upslope, modelo espectral Fourier) | Generar covariables físicas de terreno a partir de NASADEM | Parece necesario para el objetivo propio del proyecto (downscaling orográfico) | No aplica — no hay concepto equivalente en P1/P2 que "modificar" |
| Sistema de manifiestos SHA-256 y captura de revisión git | Trazabilidad/reproducibilidad interna | Buena práctica de ingeniería, no exigida por los papers | No aplica |
| `storage_check` (preflight de integridad de almacenamiento en red) | Verificar montajes de red antes de correr el pipeline | Infraestructura operativa, no lógica científica | No aplica |
| Extracción de "Oya" vía Google Earth Engine (`export_oya.py`) | Obtener un segundo dataset de verdad de terreno propio | Decisión de diseño propia del proyecto | No aplica |

---

## 15. Metodología del paper NO reflejada en código (sección prioritaria)

**FALTANTE #1**
Requisito: Arquitectura generativa adversarial (cGAN) con generador+discriminador.
Referencia: P1, "Model description", Fig. 5.
Importancia: CRÍTICA — es el núcleo metodológico de P1.
Evidencia de ausencia: `architecture.md` línea 47-48 (autodeclaración); ausencia de clase `Discriminator` en `src/models/`.
Impacto: El código no puede considerarse una implementación de spateGAN-ERA5 bajo ningún estándar.
Cómo debería implementarse: Requeriría una reescritura arquitectónica completa, no una corrección incremental.

**FALTANTE #2**
Requisito: Generación de ensemble probabilístico vía dropout congelado por miembro.
Referencia: P1, "Model description".
Importancia: CRÍTICA.
Evidencia de ausencia: `MultiscalePrecipUNet.forward()` retorna un único tensor determinista (`multiscale_unet.py` línea 99); `dropout=0.0` por defecto.
Impacto: Imposibilita el cálculo de CRPS, FSS-ensemble, rank histograms tal como se definen en P1/P2.
Cómo debería implementarse: Añadir mecanismo de muestreo estocástico + freezing de semilla por miembro.

**FALTANTE #3**
Requisito: Downscaling temporal sub-horario/horario (P1: 1h→10min; P2: horario).
Referencia: P1 Abstract; P2 "Downscaling task formulation".
Importancia: CRÍTICA.
Evidencia de ausencia: Todo el pipeline opera a resolución diaria (`README.md`, `docs/pipeline.md`).
Impacto: El problema resuelto por el código es fundamentalmente distinto (no hay dimensión temporal de downscaling).
Cómo debería implementarse: No aplica como "corrección" — sería un proyecto distinto.

**FALTANTE #4**
Requisito: Métricas de verificación probabilística (CRPS, FSS, RAPSD, excentricidad, rank histogram).
Referencia: P1 "Evaluation"; P2 "Quantitative evaluation".
Importancia: CRÍTICA.
Evidencia de ausencia: búsqueda exhaustiva por palabra clave sin resultados en `src/`.
Impacto: No permite verificar calidad de extremos ni estructura espacio-temporal según el estándar de los papers.
Cómo debería implementarse: Añadir módulo de métricas dedicado (posiblemente reutilizando `pysteps` para RAPSD, como en P1).

**FALTANTE #5**
Requisito: Generalización geográfica multi-región con escenarios jerárquicos A1-A4 (P2).
Referencia: P2, "Training sub-tasks", Fig. 3-4.
Importancia: CRÍTICA (es el objetivo central de P2).
Evidencia de ausencia: Grid único y fijo (`build_dataset.py` línea 25-27: `HEIGHT, WIDTH = 460, 720`, `TARGET_TRANSFORM = from_origin(-120.0, 35.0, 0.05, 0.05)`).
Impacto: El código no puede usarse para evaluar generalización Norte-Sur global tal como define P2.
Cómo debería implementarse: Requeriría rediseño completo del pipeline de datos para múltiples regiones discontinuas.

**FALTANTE #6**
Requisito: Stitching global de patches con fórmula de Haversine y blending lineal (P1).
Referencia: P1, "Generation of global fields".
Importancia: ALTA.
Evidencia de ausencia: sin ocurrencias de "haversine" en el código; dominio único sin necesidad de stitching.
Impacto: No hay capacidad de generar campos globales continuos.

**FALTANTE #7**
Requisito: Corrección de sesgo de campo medio (mean field bias correction) en inferencia (P1).
Referencia: P1, "Model description", ref. 53.
Importancia: ALTA.
Evidencia de ausencia: no se encontró lógica de reescalado post-inferencia basada en el promedio del patch de entrada.
Impacto: El código puede heredar sesgos multiplicativos del modelo sin corrección explícita post-hoc (aunque sí incorpora un baseline interpolado dentro del propio forward pass, ver Sección 4.3).

---

## 16. Diferencias críticas (tabla resumen, top 10)

| ID | Diferencia | Paper | Código | Impacto | Severidad |
|---|---|---|---|---|---|
| D1 | Naturaleza del modelo | Generativo/probabilístico (cGAN, WGAN-GP, diffusion) | Determinista (U-Net de dos cabezas) | Imposibilita comparación de ensembles/incertidumbre | CRÍTICO |
| D2 | Resolución temporal | Sub-horaria (P1) / horaria (P2) | Diaria | Problema científico distinto | CRÍTICO |
| D3 | Factor de escalado espacial | 12× (P1) / 2.5× (P2) | ~5× | No comparable | CRÍTICO |
| D4 | Target de verdad de terreno | Radar gauge-adjusted (P1) / IMERG satelital global (P2) | CHIRPS / Oya | Datasets y calidad distintos | CRÍTICO |
| D5 | Cobertura geográfica | Global (ambos papers) | Regional fija (México/Centroamérica aprox.) | No evalúa generalización geográfica | CRÍTICO |
| D6 | Variables de entrada | CP+LSP (P1) / 9 vars ecPoint (P2) | 18 canales ERA5-Land+niveles presión | Conjunto de predictores distinto | ALTO |
| D7 | Función de pérdida | Adversarial + L1-ensemble (P1) | BCE+SmoothL1+físico+gradiente+multiescala | Objetivos de optimización no comparables | CRÍTICO |
| D8 | Métricas de evaluación | CRPS/FSS/RAPSD/excentricidad | R²/RMSE/MAE/bootstrap | Resultados no intercomparables | CRÍTICO |
| D9 | Uso de ERA5-Land | Excluido explícitamente (P1) | Usado como fuente principal | Contradice justificación técnica de P1 (M102) | ALTO |
| D10 | Escenarios de generalización | A1-A4 multi-región (P2) | Ninguno (dominio único) | No aborda pregunta de investigación de P2 | CRÍTICO |

---

## 17. Porcentaje de cobertura metodológica

Calculado sobre los ~90 requisitos de la matriz de trazabilidad entregada previamente (M101-M266), excluyendo los 3 ítems de contradicción documental (C001-C003, que no son requisitos de implementación):

- **Implementados:** 0% (ningún requisito central de arquitectura, pérdida, muestreo o evaluación probabilística está presente tal como se describe)
- **Parcialmente implementados:** ~7% (ej.: RMSE/MAE presentes pero en contexto determinista; comparación contra ERA5 presente con métrica distinta; manejo de NaN filosóficamente similar)
- **Diferentes:** ~10% (ej.: uso de interpolación bilineal como componente arquitectónico en vez de baseline externo; uso de ERA5-Land explícitamente excluido por P1)
- **Ausentes:** ~78% (arquitectura GAN/diffusion, submuestreo estocástico, ensemble, stitching global, mean-field bias correction, todas las métricas probabilísticas, escenarios de generalización P2)
- **No determinables:** ~5% (requieren inspección línea por línea de `dataset.py`, `report.py`, `prepare_dataset.py` no completada en esta pasada — ver Sección 24)

**Método de cálculo:** conteo simple de IDs de la matriz por categoría asignada en la Sección 3, sin ponderación por importancia. Esto es una medida orientativa, no una métrica científica formal (tal como exige la regla 18 del prompt original).

---

## 18. Score de fidelidad

| Dimensión | Score (0-100) | Justificación |
|---|---|---|
| Fidelidad metodológica | **3/100** | El diseño experimental completo (arquitectura, escala, temporalidad, objetivo de generalización) es distinto; solo comparten el dominio general "downscaling de precipitación con ERA5 como insumo". |
| Fidelidad algorítmica | **2/100** | Ningún algoritmo central (GAN, diffusion, ensemble, stitching) está presente; el U-Net determinista no es una variante reconocible de ninguno de los modelos descritos. |
| Fidelidad de parámetros | **0/100 (no aplicable de forma significativa)** | Los parámetros no son comparables porque pertenecen a arquitecturas y objetivos distintos; no tiene sentido evaluar "coincidencia" de valores. |
| Fidelidad de datos | **5/100** | Ambos usan ERA5 como insumo (coincidencia mínima), pero targets, resolución, cobertura geográfica y temporal son distintos en todos los ejes. |
| Fidelidad de validación | **8/100** | RMSE/MAE se calculan en ambos mundos (coincidencia parcial de vocabulario métrico), pero las métricas centrales de los papers (CRPS, FSS, RAPSD) están ausentes. |
| Reproducibilidad (del propio proyecto, no respecto a los papers) | **70/100** | El repositorio en sí tiene buenas prácticas: manifiestos con hash, splits documentados, seeds, versionado de datasets — pero esto evalúa la reproducibilidad *interna* del proyecto, no su fidelidad a P1/P2. |

**Metodología de puntaje:** cada score refleja la proporción de requisitos de esa dimensión (según la matriz M101-M266) que se encontraron con evidencia directa de implementación fiel (no solo "inspirados en"), penalizando fuertemente cuando el diseño experimental de fondo (temporalidad, escala, objetivo) invalida la comparación directa.

---

## 19. Reconstrucción del pipeline ideal (sin escribir código)

Si el objetivo declarado fuera realmente reproducir P1 o P2, el pipeline debería cambiar en:

```text
PAPER (P1, por ejemplo)
 ↓
Etapa 1 — Datos: reemplazar ERA5-Land+niveles de presión por CP+LSP horario de ERA5 puro;
           reemplazar CHIRPS/Oya por un producto de radar gauge-adjusted sub-horario (o aceptar
           que el proyecto persigue un objetivo científico distinto y no intentar esta reconciliación)
 ↓
Etapa 2 — Muestreo: implementar submuestreo estocástico basado en cuantiles dinámicos (ε1,ε2)
 ↓
Etapa 3 — Arquitectura: sustituir MultiscalePrecipUNet por generador Res3D + discriminador,
           con dropout congelado por miembro de ensemble
 ↓
Etapa 4 — Pérdida: implementar L_D (BCE adversarial) + L_L1 ensemble (Ec.3)
 ↓
Etapa 5 — Inferencia: patches con stitching (Haversine+blending) + mean-field bias correction
 ↓
Etapa 6 — Evaluación: CRPS, FSS, RAPSD (vía pysteps), rank histograms, excentricidad
 ↓
Resultado esperado: comparable directamente con Tabla 1 / Figs. 3-4 de P1
```

**Nota:** dado el hallazgo de la Sección 1.1 (el propio repositorio declara no perseguir un modelo probabilístico/GAN), esta reconstrucción es principalmente ilustrativa de la distancia entre ambos proyectos, no una recomendación operativa — el equipo del repositorio parece haber tomado una decisión de diseño deliberada e informada ("Those legacy implementations and unsupported claims were removed"), posiblemente tras concluir que el enfoque GAN no era viable o adecuado para su caso de uso (CHIRPS/Oya, dominio regional, diario).

---

## 20. Plan de corrección (backlog técnico) — condicionado a que el objetivo real sea reproducir P1/P2

| Prioridad | Problema | Archivo | Cambio requerido | Referencia | Impacto |
|---|---|---|---|---|---|
| P0 | Ausencia total de arquitectura generativa | `src/models/multiscale_unet.py` | Reescribir como cGAN con discriminador | P1 Fig.5 | Bloqueante para cualquier comparación con P1 |
| P0 | Resolución temporal incompatible (diaria vs sub-horaria/horaria) | Todo el pipeline de datos | Rediseñar ingestión a resolución horaria/sub-horaria | P1/P2 | Bloqueante |
| P0 | Ausencia de métricas probabilísticas | `src/training/evaluate.py` | Añadir CRPS, FSS, RAPSD, rank histogram | P1 Ecs.6-12, P2 Ecs.1-2 | Bloqueante para validar contra resultados publicados |
| P1 | Dominio único no global/multi-región | `src/data_preprocessing/build_dataset.py` | Generalizar a múltiples regiones discontinuas configurables | P2 "Location splits" | Alto si el objetivo incluye estudiar generalización geográfica |
| P1 | Uso de ERA5-Land contraviene decisión explícita de P1 | `docs/stages/data-and-resources.md` | Evaluar si ERA5-Land es intencional para el caso de uso propio (probable) o si se buscaba alinear con P1 | P1 "ERA5 dataset" | Medio-alto, requiere decisión de producto, no solo técnica |
| P2 | Ausencia de submuestreo estocástico por cuantiles | `src/data_preprocessing/prepare_dataset.py` | Implementar rutina ε1/ε2 si se busca paridad de muestreo con P1 | P1 Fig.6 | Medio |
| P3 | Módulo `chirps_v3` en estado ambiguo (¿legado o activo?) | `src/chirps_v3/`, `pipeline_v3.py` | Clarificar en `architecture.md` si es legado a eliminar o rama activa | — | Bajo (higiene de repositorio) |
| P3 | Referencia a archivo inexistente `CODEBASE_RESULTS_REVIEW.md` | `README.md` | Restaurar el archivo o eliminar la referencia | — | Bajo |

---

## 21. Recomendaciones para reproducibilidad (independiente del objetivo anterior)

Ya presentes en el repositorio (positivo, digno de mención): manifiestos SHA-256, captura de revisión git y estado "dirty", checkpoints con estadísticas de normalización embebidas, seeds fijadas, cuDNN determinista, splits temporales documentados, criterios de aceptación explícitos por escrito (`docs/pipeline.md` sección final).

Recomendaciones adicionales si se desea fidelidad a P1/P2 en el futuro: versionado explícito de la versión de ERA5/CHIRPS/Oya descargada (fecha de acceso, checksum del archivo fuente, no solo del dataset ya procesado); scripts de ejecución de ejemplo con outputs esperados de referencia (golden files) para las métricas científicas nuevas que se añadan (CRPS/FSS); documentación de las 9 (o 18) variables de entrada con sus unidades exactas y su justificación física, similar al detalle que P2 remite a su Tabla Suplementaria S2.

---

## 22. Veredicto final

### ¿El código implementa la metodología de los papers?

## **NO**

### Justificación

El repositorio `robertorojas11/Precipitation` implementa un pipeline de downscaling de precipitación **determinista, diario, de dominio geográfico regional fijo**, basado en un U-Net de dos cabezas (ocurrencia/cantidad) entrenado con pérdidas compuestas no adversariales, evaluado con R²/RMSE/MAE/bootstrap contra CHIRPS y Oya como verdades de terreno.

Ninguno de los dos papers auditados describe este diseño:
- **P1 (spateGAN-ERA5)** es un cGAN probabilístico global de downscaling espacio-temporal (24km/1h → 2km/10min) validado contra radar gauge-adjusted en tres continentes, con generación de ensembles, stitching global de patches y métricas de verificación probabilística (CRPS, FSS, RAPSD, excentricidad).
- **P2 (RainShift)** es un benchmark de generalización geográfica que compara ResNet/WGAN-GP/Diffusion en 18 regiones globales (12 de entrenamiento, 6 de evaluación Norte-Sur), usando ERA5→IMERG a resolución horaria, con estrategias explícitas de domain adaptation (quantile mapping, fine-tuning).

El propio repositorio **declara explícitamente** en `architecture.md` que no reclama ser un modelo probabilístico ("This project does not claim an ESRGAN, conditional Gaussian process, or probabilistic model"), lo cual es coherente con la evidencia técnica encontrada y sugiere que el equipo desarrollador tomó esta dirección de manera deliberada, posiblemente en una fase anterior del proyecto que sí perseguía metodologías similares a P1/P2 y fue abandonada ("legacy implementations... were removed").

**No se trata de una implementación parcial, aproximada o simplificada de P1 o P2: es un proyecto de investigación distinto** que comparte únicamente el dominio general del problema (usar ERA5 para hacer downscaling de precipitación) y algunos elementos de vocabulario técnico superficial (bloques residuales, activación Softplus, uso de interpolación como componente del modelo).

---

## 23. Resumen ejecutivo para un investigador

**Lo que el código hace correctamente (en su propio marco de referencia, no el de los papers):**
- Pipeline de datos con control de calidad, máscaras explícitas de validez, normalización solo con estadísticas de entrenamiento, y manifiestos con hash para trazabilidad.
- Arquitectura U-Net de dos cabezas (ocurrencia/cantidad) razonable para precipitación diaria determinista.
- Función de pérdida compuesta que combina clasificación de ocurrencia, regresión de cantidad, término físico ponderado por eventos extremos, coherencia espacial (gradiente) y multiescala.
- Criterios de aceptación explícitos, splits temporales fijos y protocolo de evaluación held-out con bootstrap.

**Lo que hace parcialmente respecto a los papers:**
- Comparación contra ERA5 interpolado como baseline (presente en ambos mundos, pero con métrica distinta: R² vs. CRPS/FSS).
- Manejo filosóficamente similar de valores faltantes (NaN ≠ cero).

**Lo que hace diferente:**
- Usa interpolación bilineal como componente arquitectónico interno (residual sobre baseline), no solo como baseline externo de comparación.
- Usa ERA5-Land, explícitamente descartada por P1.

**Lo que no implementa (la inmensa mayoría de la metodología de ambos papers):**
- Arquitectura generativa adversarial o de difusión.
- Downscaling temporal sub-horario/horario (el código es diario).
- Ensembles probabilísticos y sus métricas de verificación (CRPS, FSS, rank histograms).
- Análisis espacial de estructura (RAPSD, excentricidad).
- Escenarios de generalización geográfica multi-región (P2).
- Submuestreo estocástico por cuantiles dinámicos, stitching global de patches, corrección de sesgo de campo medio en inferencia.

**Errores/riesgos potencialmente relevantes (propios del repositorio, no de fidelidad a papers):**
- Referencia documental rota (`CODEBASE_RESULTS_REVIEW.md` inexistente).
- Estado ambiguo del módulo `chirps_v3`/`pipeline_v3.py` (¿activo o legado?).
- `pos_weight` de la pérdida de ocurrencia calculado por batch, no globalmente — decisión de diseño no documentada explícitamente como intencional.

**Cambios prioritarios:** si el objetivo real del equipo es alinear este proyecto con P1/P2, el plan de corrección de la Sección 20 (P0: arquitectura generativa, resolución temporal, métricas probabilísticas) es el punto de partida obligado. Si, como sugiere `architecture.md`, el objetivo es un producto determinista propio para CHIRPS/Oya, **no se recomienda ningún cambio** orientado a "parecerse" a los papers, y esta auditoría debe leerse como una constatación de que ambos proyectos, simplemente, no son el mismo trabajo.

**Pruebas que deberían ejecutarse:** pruebas unitarias dedicadas para `compute_losses()` y `PooledMetrics` (actualmente no evidentes en `tests/`); pruebas de regresión numérica para `physics.py` (gradientes de terreno, modelo espectral).

**Información que falta para poder concluir con mayor detalle (limitaciones declaradas de esta auditoría):**
- Inspección línea por línea completa de `src/data_preprocessing/dataset.py`, `prepare_dataset.py`, `quality.py`, `report.py` y `src/chirps_v3/*` (no completada por alcance de esta pasada; los hallazgos de estas secciones se marcaron NO DETERMINABLE cuando correspondía).
- Contenido de `docs/stages/*.md` restantes (solo se citó `data-and-resources.md`) — podrían contener aclaraciones adicionales sobre decisiones de diseño.
- Confirmación de si `pipeline_v3.py`/`src/chirps_v3/` corresponden efectivamente a los "legacy implementations" mencionados en `architecture.md` o son una tercera línea de trabajo activa no documentada.

