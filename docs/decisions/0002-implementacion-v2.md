# 0002 — Implementación v2: dos candidatos generativos + η-learning

- **Estado:** Aceptada
- **Fecha:** 2026-09-11
- **Reemplaza a:** `docs/decisions/0001-methodology-choice.md` (vive en la
  rama `migration-attempt-1-backup`; revertida el 2026-09-07, ver
  `docs/MIGRATION.md`).
- **Contexto:** Segundo intento de migración desde el pipeline determinista
  diario CHIRPS/Oya (`v2_clean`) hacia un marco de downscaling probabilístico.
  A diferencia del intento #1 (una sola arquitectura GAN), este prompt exige
  dos arquitecturas candidatas en paralelo más un regularizador de
  conciencia de extremos aplicable a ambas (y opcionalmente al baseline
  determinista).
- **Papers de referencia** (`docs/audit/papers/`):
  - **P1** — Glawion et al. 2025, *spateGAN-ERA5* (`glawion2025_spateGAN-ERA5.pdf`).
  - **P2** — Harder, Schmidt et al. 2026, *RainShift* (`harder2026_RainShift.pdf`).
  - **P3** — Chang & Sapsis 2026, *Extreme Event Aware (η-) Learning*, Nature
    Communications, Article in Press (`chang_sapsis2026_eta-learning.pdf`).
  - **P4** — Avila 2026, *Downscaling Tropical Cyclone Wind Speed Using
    Machine Learning* (`avila2026_TC_windspeed_downscaling.pdf`).
- **Inputs a esta decisión:** `docs/audit/auditoria_codigo_vs_papers.md`
  (veredicto intento #1: el código no implementa ni P1 ni P2; fidelidad
  0–8/100 salvo reproducibilidad interna ≈70/100), `docs/audit/matriz_trazabilidad_metodologica.md`
  (~90 requisitos M101–M266), `docs/decisions/0001-methodology-choice.md`
  (rama backup), y una lectura completa de P3/P4 realizada en esta sesión.

---

## 1.1 Marco experimental

**Se mantiene RainShift (P2) como esqueleto experimental**, reafirmando la
decisión 0001: regiones de entrenamiento/evaluación disjuntas, splits
temporales por calendario, métricas de generalización geográfica (CRPS,
percentile-MAE), interpolación bilineal como baseline inferior. Esto no
cambia respecto al intento #1 porque el propio P2 ya es un arnés de
comparación entre clases de modelo (ResNet/WGAN-GP/Diffusion) — es
exactamente lo que permite añadir dos candidatos nuevos sin construir dos
pipelines paralelos.

**Lo que sí cambia respecto al intento #1:** en vez de una sola arquitectura
generativa (GAN inspirado en P1), se implementan **dos candidatos en
paralelo**, evaluados bajo las mismas métricas y regiones (§8):

- **Candidato A — GAN condicional** (inspirado en P1/spateGAN-ERA5):
  generador + discriminador adversarial, pérdida `L_D` (BCE, P1 Ec. 1) +
  `L_GAN(G)` (P1 Ec. 2) + `L_L1(G)` ensemble de N≥3 miembros (P1 Ec. 3),
  `L_G = L_GAN(G) + L_L1(G)` (P1 Ec. 4). Ensemble vía dropout congelado por
  miembro en inferencia (P1, "Model description": *"freeze the dropout seed
  for each produced ensemble member"*). Implementado en
  `src/models/{generator,discriminator}.py` + `src/training/losses_gan.py`,
  igual que el intento #1 (no se revive el `rrdb_gan.py` histórico —
  decisión OQ-2 del intento #1 se mantiene).
- **Candidato B — Regresión + difusión residual** (nuevo, siguiendo P4/Avila
  2026, que a su vez sigue a Mardani et al. 2024, *Residual Corrective
  Diffusion Modeling for Km-scale Atmospheric Downscaling*):
  - **Etapa 1 (regresión):** U-Net encoder-decoder con skip connections y
    auto-atención en la resolución interna más gruesa (P4 §3.1, diseño ADM
    de Dhariwal & Nichol 2021). Entrenada con MSE para predecir el campo
    promedio condicional `E[y|x]` (P4 Ec. 1: `y = E[y|x] + (y - E[y|x])`).
    Hiperparámetros de partida (P4 Tabla 5, a ajustar a nuestra escala):
    ancho base 64 canales, atención en la resolución interna más gruesa,
    AdamW LR 2×10⁻⁴, weight decay 1×10⁻⁵.
  - **Etapa 2 (difusión residual):** modelo de difusión sobre el framework
    EDM (Karras et al. 2022), cuya entrada es la concatenación (por canal)
    del residuo ruidoso, los canales de condicionamiento, y el campo medio
    de la etapa 1. Preacondicionamiento EDM (P4 Ecs. 3–4):
    `D(z;σ) = c_skip(σ)·z + c_out(σ)·F(c_in(σ)·z, c_noise(σ))`, con
    `c_skip = σ_data²/(σ²+σ_data²)`, `c_out = σ·σ_data/√(σ²+σ_data²)`,
    `c_in = 1/√(σ_data²+σ²)`, `c_noise = ¼·ln(σ)`, `σ_data = 0.5`.
    Pérdida de entrenamiento (P4 Ec. 5):
    `L = E_σ,n[λ(σ)·‖D(r'+σn; σ) − r'‖²]`, `λ(σ) = (σ²+σ_data²)/(σ·σ_data)²`,
    con `σ` muestreado vía `ln σ ~ N(P_mean, P_std²)`, `P_mean=-1.2`,
    `P_std=1.2` (valores recomendados por Karras et al., reutilizados sin
    reajustar salvo evidencia en contra).
  - **Muestreo:** esquema de Heun de segundo orden (P4 §3.3, Ec. 6:
    `σ_i = (σ_max^(1/ρ) + i/(N-1)·(σ_min^(1/ρ) − σ_max^(1/ρ)))^ρ`) con
    parámetro de churn estocástico `S_churn` (P4 Ec. 7:
    `γ = min(churn/N, √2−1)`, `ŝ = s(1+γ)`,
    `z ← z + √(ŝ²−s²)·S_noise·ε`). Documentado explícitamente (código +
    aquí): **`S_churn` es la palanca más barata para mejorar la recuperación
    de picos extremos sin reentrenar** — P4 §6.1 reporta que subir el churn
    hasta ≈7 (a 18 pasos) mejora monótonamente la recuperación del pico en
    tormentas fuertes antes de saturar, y que el efecto es "conservador" (no
    sobreestima) en tormentas moderadas. Se debe ajustar `S_churn` antes de
    invertir en cambios de arquitectura o reentrenamiento más costosos.
  - Referencia a las ecuaciones de origen documentada en docstrings, igual
    que se exige para las pérdidas de la GAN (regla transversal del prompt).

Ambos candidatos se evalúan bajo las mismas métricas y regiones (§8 de este
documento y `docs/EVALUATION_PROTOCOL.md`), tal como el propio RainShift
compara ResNet/GAN/Diffusion entre sí.

**Candidato descartado explícitamente:** ClimateDiffuse (P2 M228, la
arquitectura de difusión con la que RainShift originalmente compara) no se
reimplementa aparte — el Candidato B (Avila/Mardani) cubre el mismo rol de
"modelo de difusión" en la comparación de P2, con una receta más detallada
y con evidencia directa de funcionar en un dominio geográfico equivalente al
nuestro (México/Pacífico Oriental, aunque para viento, no lluvia).

## 1.2 Regularización de conciencia de extremos (η-learning)

Ambos candidatos (A y B), y opcionalmente el baseline determinista (§5.4),
incorporan como capa adicional de pérdida el regularizador de η-learning
(P3, Chang & Sapsis 2026): un término de distancia de Wasserstein-1 entre la
distribución del observable **"máximo espacial de precipitación del campo
generado"** y una distribución de referencia `ν0`.

Formulación adoptada (P3 Ec. 11, caso de aproximar `y = g∘u` directamente,
que es nuestro caso porque el observable de interés — máximo espacial — se
computa sobre la salida del modelo, no sobre un estado interno):

```
y_η := argmin_φ  (1/n)Σᵢ ℓ(φ, xᵢ, yᵢ)  +  λ · W1(φ#μ, ν0)
```

con `W1` estimado vía su **representación por cuantiles en 1-D** (P3 Ec. 13):

```
W1(ν1, ν2) = ∫₀¹ |F⁻¹_ν1(q) − F⁻¹_ν2(q)| dq
```

aproximada por cuadratura de Monte Carlo sobre un conjunto auxiliar de
muestras (P3 Ec. 14), model-agnostic y sin rediseñar ninguna de las dos
arquitecturas — se añade a la función de pérdida existente, exactamente como
exige el prompt.

**Justificación** (P3, Teorema 4, "Fundamental Limitation of Data
Scarcity"): un estimador *data-consistent* (uno que ajusta bien donde hay
datos) tiene, con probabilidad ≥p, una cota inferior demostrable de error
`W1` en la región extrema `E` cuando el tamaño de entrenamiento cae bajo un
umbral — es decir, es preciso donde hay datos y provablemente pobre en la
cola cuando esa cola está subrepresentada. Ambas regiones de interés (y
especialmente Puerto Rico/Caribe, con historial corto de eventos extremos
verificados) están en ese régimen. El paper de viento (P4/Avila 2026)
ilustra el fallo en la práctica sin este regularizador: su difusión residual,
diseñada específicamente para recuperar extremos, aun así subestima el pico
del huracán más intenso del set de prueba por casi la mitad (44.5 m/s
predicho vs. 90.3 m/s real, P4 Tabla 3, fila 4).

**Procedimiento de entrenamiento** (implementado en
`src/training/eta_learning.py`, PASO 5.3 de este documento):

1. **Pre-entrenamiento ERM** antes de activar el término η (P3, sección
   "Effective training strategies" → "Gradient conflict and ERM
   pre-training"): se entrena primero solo con la pérdida supervisada, y el
   estimador ERM resultante inicializa `φ_θ` y el índice de muestras de cola
   `I` (Algoritmo 1/2 de P3). El propio paper identifica esto como la
   estrategia de estabilización crítica para evitar conflictos de gradiente
   al inicio del entrenamiento (las muestras "de cola" identificadas por un
   modelo aún no entrenado no son fiables).
2. **Muestreo de cuantiles sesgado hacia la cola** con parámetro `τ`
   (típicamente 0.95–0.975, P3 fórmula tras Ec. 14): en vez de muestreo
   uniforme de niveles de cuantil, se concentra el cómputo en
   `[τ, 1]` para maximizar la señal de gradiente relevante a extremos.
3. **Refresco periódico del índice de cola** (Inference-Informed Continual
   Training, IICT — P3 Algoritmo 1/2): cada `ω` iteraciones se recalcula qué
   muestras del conjunto auxiliar corresponden a los cuantiles de interés,
   porque la distribución inducida por el modelo cambia conforme se
   entrena.
4. **`λ` configurable y auto-ajustable** por norma de gradiente (P3, "Choice
   of λ"): `λ = ‖∇_θ L_ERM‖₂ / (‖∇_θ L_W1‖₂ + ε)`, para que el término η
   tenga una magnitud de gradiente comparable al ERM durante las primeras
   iteraciones. El valor final usado (fijo o el resultado de este balanceo
   adaptativo) debe documentarse en `docs/ETA_LEARNING_NOTES.md` una vez se
   corra un entrenamiento real.

## 1.3 Fuentes de `ν0` para el regularizador

Documentado en detalle en `docs/ETA_LEARNING_NOTES.md` (PASO 10). Resumen de
la decisión:

- **México-Pacífico y México-Atlántico/Golfo:** si el historial IMERG/Oya
  acumulado en la región es suficiente (a determinar empíricamente al
  preparar cada región — regla operativa provisional: ≥10 años de datos
  horarios válidos), `ν0` se estima empíricamente de esos datos —
  `ν0 = y#μ`, el caso ideal de P3 (`ν0 = y_#μ` en la notación del paper,
  sección "η-Learning framework").
- **Puerto Rico/Caribe:** historial insuficiente (nueva región, sin cascada
  IMERG/Oya histórica construida aún). `ν0` se hipotetiza como una
  **distribución generalizada de valores extremos (GEVD)** ajustada a partir
  de literatura de climatología regional de huracanes/tormentas tropicales
  del Caribe — siguiendo el mismo patrón que el experimento final de P3
  ("η-Downscaling with hypothesized heavier-tailed distributions"). Esto
  **se documenta explícitamente como una hipótesis, no como un hecho
  verificado**: P3 advierte que un `ν0` mal especificado produce extremos
  "consistentes pero físicamente equivocados" — en su experimento de
  sensibilidad (perturbación del cuantil de cola por un factor `α`), al
  mover el sesgo del cuantil de α=-1 a α=2 el RMSE de campo completo subió
  de 2.926 a 3.527 (**+20.55%**), mientras el SSIM se mantuvo casi
  invariante (0.941→0.947) — es decir, el costo de un `ν0` mal especificado
  aparece en la fidelidad puntual (RMSE), no necesariamente en la
  estructura espacial (SSIM). Esta cifra concreta ("hasta 20% más de error
  si el sesgo del `ν0` asumido es grande") es la que cita el prompt y queda
  registrada aquí con su procedencia exacta.

## 1.4 Datos de entrada y verdad de terreno (reafirmación)

- **Entrada:** `cp` (*convective_precipitation*) + `lsp`
  (*large_scale_precipitation*) de ERA5 puro, horario, extendiendo el
  mecanismo de adquisición existente (`src/data_extraction/`,
  `src/utils/config.py`) — no se construye un módulo nuevo desde cero (PASO
  3.1).
- **Verdad de terreno:** cascada ERA5 → IMERG → Oya (§3.2 de este
  documento). CHIRPS queda **excluido** de esta cascada por ser diario
  únicamente (P2/RainShift y P1/spateGAN-ERA5 operan a resolución horaria o
  sub-horaria; un target diario no permite evaluar downscaling temporal).
  CHIRPS y ERA5-Land permanecen como fuentes **heredadas**, documentadas y
  sin modificar (siguen sirviendo al pipeline determinista archivado en
  `legacy/`, §4).
- **Regiones:** `ATLANTICO_SHP_PATH` / `PACIFICO_SHP_PATH` recuperados y
  documentados (`docs/DATA_SOURCES.md`) — son polígonos de cuenca oceánica
  completa (Atlántico tropical y Pacífico Oriental), no límites
  costeros/administrativos. Se definen regiones de trabajo como bounding
  boxes dentro de/superpuestos a estas cuencas: México-Pacífico,
  México-Atlántico/Golfo, y una nueva región Puerto Rico/Caribe (arco de
  Antillas Mayores y Menores, lon [-85°, -59°], lat [10°, 23°] — confirmado
  con el usuario 2026-09-11). Implementado en
  `src/data_preprocessing/regions.py` (PASO 2).

## 1.5 Ensemble operativo con pronósticos de IA (Fase 4 — no bloqueante)

Documentado como trabajo futuro, no implementado en esta ronda:

- Correr el modelo entrenado (Candidato A o B) sobre cada miembro de un
  ensemble de pronóstico operativo (WeatherNext 2 u otro), siguiendo el
  patrón validado por P4/Avila 2026: un ensemble de 50–64 miembros con
  múltiples draws generativos por miembro corre en ≈3 horas en una sola GPU
  (P4 §5, "Inference cost is modest enough that a full probabilistic
  product can be produced overnight on a single card... approximately 3
  hours on one RTX A6000" para 64 miembros × 4 draws × 18 pasos de
  sampler).
- Antes de esta fase deben resolverse: (a) qué variables de precipitación
  expone realmente el pronóstico elegido (P4 no downscala precipitación,
  downscala viento — la analogía no es automática); (b) un paso de
  **calibración estadística (quantile mapping)** entre la distribución de
  ERA5 (entrenamiento) y la del producto de pronóstico (operación). P4 **no
  implementa este paso** — se deja aquí como una mejora deliberada sobre su
  metodología, no como omisión, y debe construirse antes de considerar la
  Fase 4 lista para producción. RainShift (P2, M255–M258) ya da la receta
  de quantile mapping multiplicativo (1000 cuantiles, aplicado antes de
  normalizar) que se reutilizará para este paso cuando llegue el momento.

---

## Qué preservar y qué archivar (PASO 4, reafirmación sin cambios de
principio respecto al intento #1)

- **Reutilizar y extender:** `pipeline.py`, `src/utils/` (config, logging,
  retries, storage), `src/data_preprocessing/quality.py` (máscaras de
  validez, manifiestos), sistema de manifiestos SHA-256, `physics.py`
  (covariables de terreno, ofrecidas como condicionamiento opcional para
  ambos candidatos — extensión del proyecto, no exigida por ningún paper).
- **Archivar sin borrar:** el modelo determinista original
  (`multiscale_unet.py` + `train/search/evaluate/report.py` de
  `src/training/`) y sus resultados (`outputs/v2_clean/`) → `legacy/`. Igual
  que en el intento #1: nunca quedó archivado en `master` porque el rollback
  restauró el estado pre-migración íntegro (confirmado contra
  `docs/MIGRATION.md`, sección "Intento de migración #1").
- **`src/chirps_v3/` / `pipeline_v3.py`:** confirmado con el usuario
  2026-09-11 — archivar a `legacy/chirps_v3/` (misma decisión que OQ-1 del
  intento #1).

---

## Deviations / open questions

- **OQ-1 (heredada del intento #1, re-confirmada 2026-09-11).**
  `src/chirps_v3/` y `pipeline_v3.py` → `legacy/chirps_v3/`.
- **OQ-2 (heredada).** No se revive `rrdb_gan.py` del historial de git; el
  Candidato A se escribe fresco contra P1.
- **OQ-3 — Credenciales CDS/Earthdata.** No disponibles (confirmado con el
  usuario 2026-09-11). Los módulos de adquisición nuevos (`era5_cds.py`
  extendido, `imerg.py`) se implementan completos pero **no se ejecutan**
  esta ronda — fallan explícitamente ante variables de entorno faltantes, sin
  fallback sintético.
- **OQ-4 — Umbral de "historial suficiente" para `ν0` empírico.** Se fija
  provisionalmente en ≥10 años de datos horarios válidos por región (§1.3).
  Es una regla operativa, no un valor de ningún paper — debe revisarse
  cuando se conozca la cobertura temporal real de IMERG/Oya por región.
- **OQ-5 — Umbral operativo de "lluvia atípica" (PASO 7).** Pendiente de
  definir con el equipo de dominio; P1/spateGAN-ERA5 usa 25 mm/h como umbral
  de advertencia severa (referencia de partida, no un valor ya validado
  para nuestras regiones).

## Git checkpoint

El estado pre-intento-#2 es el mismo `pre-migration-v2_clean`
(`b6de7ec`) heredado del intento #1 — no se crea un tag nuevo porque
`master` no se ha desviado de ese punto hasta este commit.

## Traceability

El estado de implementación de cada pieza de este documento se registra en
`docs/architecture_v3.md` (tabla de cobertura M101–M266 + nuevas filas para
P3/P4), actualizada después de cada paso, no antes.
