# Matriz de Trazabilidad Metodológica
## Papers fuente
- **P1**: Glawion et al. 2025, *"Global spatio-temporal ERA5 precipitation downscaling to km and sub-hourly scale using generative AI"* (spateGAN-ERA5), npj Climate and Atmospheric Science 8:219.
- **P2**: Harder, Schmidt et al. 2026, *"Benchmarking the geographic generalization of deep learning models for precipitation downscaling"* (RainShift), Scientific Reports 16:3733.

Repos referenciados (a auditar cuando estén disponibles):
- P1 → https://github.com/LGlawion/spateGAN_ERA5
- P2 → https://github.com/RolnickLab/downscaling-transferability

Convenciones de columnas:
- **ID**: identificador único del requisito.
- **Ref.**: sección/página/figura/ecuación del paper.
- **Prioridad de auditoría**: CRÍTICO / ALTO / MEDIO / BAJO (impacto potencial sobre reproducibilidad científica si está mal implementado).

---

## BLOQUE A — spateGAN-ERA5 (P1)

### A.1 Datos de entrada y variables

| ID | Requisito científico | Ref. (P1) | Prioridad |
|---|---|---|---|
| M101 | Input: ERA5 *convective precipitation* (CP) y *large-scale precipitation* (LSP), variables horarias, resolución 24 km (~0.25°) | "Model description", Fig. 1b | CRÍTICO |
| M102 | NO usar ERA5-Land (excluida explícitamente por falta de info de escala, exclusión de océanos, mayor latencia) | Sección "ERA5 dataset" | ALTO |
| M103 | Variables adicionales (viento, temperatura, presión) probadas pero **no incluidas** en el modelo final por no mejorar desempeño | Sección "ERA5 dataset" | MEDIO |
| M104 | Target de entrenamiento: RADKLIM-YW (radar alemán gauge-adjusted y climatology-corrected), 1 km, 5 min nativo, coarsened a 2 km/10 min | "RADKLIM-YW Germany" | CRÍTICO |
| M105 | Periodo de entrenamiento: 2009–2020; selección de modelo: enero–junio 2021; evaluación: julio–diciembre 2021 (+ enero–marzo 2021 para trópicos Australia) | Tabla 1, "Data preparation" | CRÍTICO |
| M106 | Datasets de validación adicionales: MRMS (US, no gauge-adjusted, 0.01°, 2 min) y radar australiano (gauge-adjusted, 0.5 km, 5-10 min) | "MRMS United States", "Australian Radar Network" | ALTO |
| M107 | Grillas NaN donde no hay cobertura radar (>150 km de radio) deben excluirse/marcarse correctamente | "RADKLIM-YW Germany" | MEDIO |

### A.2 Preprocesamiento y muestreo de entrenamiento

| ID | Requisito científico | Ref. (P1) | Prioridad |
|---|---|---|---|
| M110 | Patch de entrada: 28×28 celdas × 16 pasos temporales (672 km × 672 km × 16 h) a 24 km/1h | "Model description" | CRÍTICO |
| M111 | Dominio de salida (downscaled): 168×168 px × 48 pasos (336 km × 336 km × 8 h) a 2 km/10 min | "Methods" | CRÍTICO |
| M112 | Factor de escalado: 12× espacial, 6× temporal | Abstract, "Results" | CRÍTICO |
| M113 | ERA5 interpolado al grid del target (radar) mediante *nearest neighbor* antes de downsamplear a 24km/1h aproximando resolución inicial | "Training and model selection dataset" | ALTO |
| M114 | Rutina de submuestreo estocástica con 2 condiciones basadas en cuantiles dinámicos: <br>• Cond. 1: ambos X,y sin missing values <br>• Cond. 2: percentil 66 de X e y > ε₁, donde ε₁ = \|−50ε′+500\|, ε′~Lognormal(0,1) <br>• Cond. 3: Σ(X)>ε₂ y Σ(y)>ε₂, donde ε₂=\|−450ε+4500\|, ε~Lognormal(0,1) | Ecs. sin numerar, Fig. 6, sección "Training and model selection dataset" | CRÍTICO |
| M115 | ~20,000 muestras de entrenamiento resultantes (~850 GB) | Misma sección | MEDIO |
| M116 | Data augmentation: rotación (90° o 270°) o reflexión (vertical/horizontal) aplicada a *cada muestra alterna* | Misma sección | ALTO |
| M117 | Para model selection: 1000 muestras de periodo independiente (ene-jun 2021); reescalado de precipitación promedio del target para igualar promedio de ERA5 de entrada (vía multiplicación escalar) | "Evaluation dataset" | ALTO |
| M118 | Evaluación: dataset temporal y espacialmente independiente; ERA5 proyectado al grid de observación e interpolado a 24 km | "Evaluation dataset" | ALTO |
| M119 | Subconjunto para RAPSD/excentricidad: solo casos donde ERA5 interpolado logra mFSS > 0.2 | "Evaluation dataset" | MEDIO |
| M120 | Para comparación a resolución ERA5: promedio 2D pooling a 24 km + agregación temporal a 1h | "Evaluation dataset" | MEDIO |

### A.3 Arquitectura del modelo (generador)

| ID | Requisito científico | Ref. (P1) | Prioridad |
|---|---|---|---|
| M130 | cGAN (generador + discriminador), extiende arquitectura "spateGAN" previa (ref. 32) | "Model description" | CRÍTICO |
| M131 | Generador: 4 componentes secuenciales con bloques residuales 3D-convolucionales (Res3D) | "Model description", Fig. 5 | CRÍTICO |
| M132 | Etapa 1: procesamiento de CP+LSP en resolución inicial | Fig. 5 | ALTO |
| M133 | Etapa 2: downsampling tipo UNET con skip connection + operación de *cropping* | Fig. 5 | ALTO |
| M134 | Etapa 3: upsampling sucesivo (4 bloques) con interpolación bilineal/lineal + Res3D | Fig. 5 | ALTO |
| M135 | Etapa 4: 3 bloques Res3D finales, activación **Softplus** para limitar salida a valores positivos | "Model description" | CRÍTICO (signo/dominio físico) |
| M136 | Dropout constante temporalmente (p=0.2) en 3 profundidades distintas del generador (baja, media, alta frecuencia) → habilita ensemble probabilístico | "Model description" | CRÍTICO |
| M137 | Dimensión de features del generador: 96 | "Model details" | MEDIO |
| M138 | Padding: reflection padding 3D en todas las capas conv3D del generador | "Model details" | MEDIO |
| M139 | Kernel size = 3 en todas las capas conv excepto skip connections de Res3D (kernel=1) con instance normalization | "Model details" | MEDIO |
| M140 | Bloque residual: Conv3D→InstanceNorm→ReLU→Conv3D→InstanceNorm + skip con Conv3D(k=1)+InstanceNorm, salida con ReLU final | Fig. 5 (diagrama "Residual Block") | ALTO |

### A.4 Arquitectura del discriminador

| ID | Requisito científico | Ref. (P1) | Prioridad |
|---|---|---|---|
| M141 | Discriminador recibe: secuencia temporal HR (pred. u obs.) + contexto LR (ERA5) | "Model description" | CRÍTICO |
| M142 | Ruido Gaussiano (mean=1, std=0.05) añadido a inputs del discriminador antes de procesar (anti-quantization) | "Model description" | ALTO |
| M143 | Rutas HR y LR procesadas por separado (Res3D), luego concatenadas en etapa tardía | Fig. 5 | ALTO |
| M144 | Padding del discriminador: zero padding (≠ generador que usa reflection) | "Model details" | MEDIO |
| M145 | Features discriminador HR: 128,128,128,64; LR: 64,32; tras concat: 64→1 en última capa Conv3D | "Model details" | BAJO |
| M146 | Pérdida discriminador: binary cross-entropy | Ec. (1) | CRÍTICO |

### A.5 Función objetivo / entrenamiento

| ID | Requisito científico | Ref. (P1) | Prioridad |
|---|---|---|---|
| M150 | L_D = E[log D(X,Y)] − E[log(1−D(X,Ŷ))] (Ec. 1) | Ec. (1) | CRÍTICO |
| M151 | L_GAN(G) = E[log D(X,G(X))] (Ec. 2) | Ec. (2) | CRÍTICO |
| M152 | L_L1(G) = \|Y − (1/3)Σ Ŷᵢ\| para i=1..3, es decir **ensemble L1-loss** sobre 3 miembros (Ec. 3) | Ec. (3) | CRÍTICO — fácil de confundir con L1 simple de 1 miembro |
| M153 | L_G = L_GAN(G) + L_L1(G) (Ec. 4, sin ponderación explícita entre términos) | Ec. (4) | CRÍTICO |
| M154 | 2×10⁵ pasos de entrenamiento adversarial | "Training and model selection" | ALTO |
| M155 | LR generador = 1e-4; LR discriminador = 2e-4 | Misma sección | ALTO |
| M156 | Optimizador: AdamW; β1=0.0, β2=0.999 (generador); discriminador β1=0.0, β2=0.5 | Misma sección | ALTO |
| M157 | Entrenamiento data-parallel en 3× A100 80GB, 4 días (nota: Abstract/intro dice "4 A100 GPUs... 3 days" — **posible inconsistencia interna del paper**, ver hallazgo de contradicción) | pág. 3 vs "Training and model selection" | MEDIO (contradicción a señalar, no del código) |
| M158 | Batch size = 9 por paso | "Training and model selection" | ALTO |
| M159 | Checkpoint guardado cada 250 pasos; selección de mejor estado vía meFSS promedio (thresholds 0.1,1,3,5,8 mm/h; escalas 1,4,8,16,32,64,128 km; escala temporal 1h) | Misma sección | CRÍTICO |
| M160 | Inferencia: freeze de dropout seed por miembro de ensemble (mejora consistencia espacio-temporal vs. perturbación aleatoria) | "Model description" | ALTO |
| M161 | Inferencia: recorte de bordes externos (24 km y 1h) para eliminar efectos de frontera | "Model description" | ALTO |
| M162 | Inferencia: corrección de sesgo de campo medio (*mean field bias correction*) — multiplicar precipitación promedio predicha por un valor escalar para igualar promedio del patch ERA5 de entrada | "Model description", ref. 53 | CRÍTICO |
| M163 | Tiempo de inferencia: 0.04 s por patch en 1× NVIDIA Tesla V100 | "Training and model selection" / pág. 3 | BAJO |
| M164 | Memoria de inferencia: modelo corre en GPUs de 10 GB | "Model description" | BAJO |

### A.6 Generación de campos globales (inferencia productiva)

| ID | Requisito científico | Ref. (P1) | Prioridad |
|---|---|---|---|
| M170 | Segmentación de ERA5 en patches de 672×672 km sobre grid lat-lon original | "Generation of global fields" | ALTO |
| M171 | Cálculo de extensión de patches usando **fórmula de Haversine**; centro de latitud determina extensión longitudinal; distorsión en longitud se desprecia por tamaño pequeño de patch | Misma sección | ALTO — verificar implementación exacta de Haversine |
| M172 | Patches se solapan ~10% en lat y lon | Misma sección | MEDIO |
| M173 | Interpolación nearest-neighbor de patches a grid regular 672×672 km (~24km) | Misma sección | ALTO |
| M174 | Tras downscaling: mean field bias correction; rain rates ERA5 <0.1 mm/h (especialmente sobre océano) se fuerzan a cero **antes** del ajuste | Misma sección | CRÍTICO |
| M175 | Interpolación final a grid lat-lon global de resolución 0.018° (~2km en el ecuador) | Misma sección | ALTO |
| M176 | Combinación de patches solapados: *linear blending* con pesos decayendo de 1→0 hacia el borde | Misma sección | ALTO |
| M177 | Downscaling de patches durante evaluación: overlap temporal, se conservan las 2h centrales de cada patch para continuidad | "Evaluation" | MEDIO |
| M178 | Para videos del caso de estudio: blending lineal en periodos de solape de 1h | "Evaluation" | BAJO |
| M179 | Umbral universal: rain rates < 0.01 (de todos los datasets comparados) se fijan a cero antes de evaluar | "Evaluation" | MEDIO |

### A.7 Métricas de evaluación (ecuaciones)

| ID | Requisito científico | Ref. (P1) | Categoría de ecuación a verificar | Prioridad |
|---|---|---|---|---|
| M180 | RMSE = sqrt((Y−Ŷᵢ)²) por miembro individual (Ec. 5) | Ec. (5) | EXACTA esperada | ALTO |
| M181 | CRPS(F̂,y) = ∫(F̂(x_t) − 𝟙(x_t≥y))² dx_t (Ec. 6-7); para métodos determinísticos se reduce a MAE | Ecs. (6)-(7) | EXACTA esperada | CRÍTICO |
| M182 | MAE = \|Y−Ŷᵢ\| (Ec. 8) | Ec. (8) | EXACTA | ALTO |
| M183 | FSS = 1 − (f_Ŷ−f_Y)² / (f_Ŷ²+f_Y²) (Ec. 9); ensemble FSS = fracción de miembros que exceden umbral σ | Ec. (9) | EXACTA esperada | CRÍTICO |
| M184 | mFSS/meFSS: promedio sobre escalas s=(0,4,8,16,32,64,128,256 km, t=1h) y umbrales σ=(0.1,1,3,5 mm/h); para resolución ERA5: escalas (0,24,96,192 km), mismos umbrales | "Evaluation" | ALTO |
| M185 | ΔmFSS = desviación relativa (%) de meFSS de rainFARM/spateGAN-ERA5 respecto a mFSS de ERA5 interpolado | "Evaluation" | MEDIO |
| M186 | RAPSD calculada con implementación **PySTEPS**, promediada por dataset de evaluación; PSD temporal por pixel y por semana, promediada | "Evaluation" | ALTO — dependencia externa específica (pySTEPS) debe confirmarse en código |
| M187 | RAPSD normalizado: espectro de cada imagen/secuencia normalizado para sumar 1 | "Evaluation" | MEDIO |
| M188 | Rank histograms: 100 predicciones de ensemble por pixel/paso temporal, ordenadas, rango normalizado r de la observación | "Evaluation" | ALTO |
| M189 | Excentricidad lineal: autocorrelación espacial para lags 0–60 km en x,y; elipse desde contorno de 0.5 Pearson Correlation Coefficient (PCC); ecc_l=√(a²−b²) (Ec. 10), ecc=√(1−b²/a²) (Ec. 11), size=√(a·b) (Ec. 12) | Ecs. (10)-(12) | EXACTA esperada | ALTO |
| M190 | BIAS = (Y−X)/Y, donde X=promedio predicho, Y=promedio observado (Ec. 13) — **atención al signo/convención**: numerador es (obs−pred) pero divide por obs | Ec. (13) | verificar signo/convención exacta | ALTO |
| M191 | Evaluación probabilística completa: 100 miembros de ensemble para rank histograms y CRPS; para ensemble FSS/meFSS: solo 6 miembros (score converge con pocos miembros) | "Evaluation" | ALTO — nº de miembros difiere según métrica |

### A.8 Métodos de referencia (baselines)

| ID | Requisito científico | Ref. (P1) | Prioridad |
|---|---|---|---|
| M195 | rainFARM implementado vía paquete **PySTEPS**, usando el algoritmo espectral avanzado (D'Onofrio et al. 2014, ref. 44), seguido de interpolación temporal | "Reference methods" | CRÍTICO |
| M196 | rainFARM usa semilla aleatoria fija distinta para componente estocástica | "Reference methods" | MEDIO |
| M197 | rainFARM downscala TP total (no CP/LSP por separado — se probó y las diferencias fueron despreciables) | "Reference methods" | MEDIO |
| M198 | rainFARM opera sobre todo el dominio de entrada ERA5 (672×672 km, 16h) y luego se recorta al dominio de evaluación (≠ spateGAN-ERA5 que ya recorta internamente) | "Reference methods" | ALTO |
| M199 | Interpolación trilineal (espacio ×2 + tiempo) de TP de ERA5 como baseline simple, sobre el grid km descrito en "Data preparation" | "Reference methods" | ALTO |

---

## BLOQUE B — RainShift (P2)

### B.1 Datos y variables

| ID | Requisito científico | Ref. (P2) | Prioridad |
|---|---|---|---|
| M201 | Input LR: ERA5, 9 variables seleccionadas por relevancia meteorológica (guiadas por modelo ecPoint, ref. 29) — **el paper no lista explícitamente cuáles son las 9 en el cuerpo principal, remite a Tabla Suplementaria S2** | "Input data" | CRÍTICO — variables exactas deben confirmarse en supl./código |
| M202 | Covariables geográficas invariantes: land-sea mask (fracción de tierra) + elevación (geopotential height at surface), resolución 0.1° | "Input data" | ALTO |
| M203 | Target HR: IMERG V07 Final Run, promediado de datos semi-horarios a horario para igualar resolución temporal de ERA5 | "Target data" | CRÍTICO |
| M204 | Cobertura IMERG: 60°N–60°S, 0.1° resolución (~10 km) | "Target data" | MEDIO |
| M205 | Factor de upsampling N = 2.5 (no un entero simple); patches LR 80×80 px → HR 200×200 px | "Downscaling task formulation" | CRÍTICO — a diferencia de P1 (12×), aquí es 2.5× |
| M206 | Canales: c=9 (LR, ERA5), d=2 (HR invariantes: land-sea mask + orografía) | "Downscaling task formulation" | ALTO |
| M207 | Splits temporales: entrenamiento 2001–2020 (regiones T1-T12), evaluación 2021–2022 (regiones E1-E6); años 2019-2020 usados como validación para hyperparameter tuning | "Temporal splits", "Training details" | CRÍTICO |
| M208 | 18 regiones de 20°×20° cada una: 12 entrenamiento (T1-T12) + 6 evaluación (E1-E6: Cape Horn, Amazon Basin, West Africa, Horn of Africa, Tibetan Plateau, Melanesia) | "Location splits", Fig. 4 | CRÍTICO |
| M209 | Regiones de entrenamiento elegidas por alta disponibilidad de datos observacionales (Global Norte); evaluación en zonas de datos escasos (Global Sur, proxies) | "Location splits" | ALTO |
| M210 | Constraint: A ∩ E = ∅ (regiones de entrenamiento y evaluación disjuntas) — tarea "zero-shot" sin labels en evaluación | "Geographical generalization" | CRÍTICO |

### B.2 Preprocesamiento de datos

| ID | Requisito científico | Ref. (P2) | Prioridad |
|---|---|---|---|
| M215 | Almacenamiento en formato Zarr, chunks de 200 timesteps (~10MB/chunk ERA5, ~30MB/chunk IMERG) | "Data processing" | BAJO |
| M216 | Clip de precipitación ERA5 usando min/max de IMERG como umbrales (corrige sobreestimación de lluvia débil, falsos positivos, valores extremos irreales) | "Data processing" | CRÍTICO |
| M217 | Transformación log de tp, cp y orografía: x̃ = log(x·1000 + 1e-5) | "Data processing" (fórmula explícita) | CRÍTICO — constantes exactas (1000, 1e-5) deben coincidir |
| M218 | Land-sea mask permanece sin transformar [0,1] | "Data processing" | MEDIO |
| M219 | Normalización Z-score por variable, calculada sobre todos los timesteps/lat/lon, agregada across regiones de entrenamiento. Media global = promedio de medias regionales; varianza combinada vía pooled variance: σ̃² = (1/n)Σ(σᵢ²+µᵢ²) − ((1/n)Σµᵢ)² | Fórmula explícita en "Data processing" | CRÍTICO — fórmula específica de pooled variance debe verificarse exacta (no es la fórmula estándar simple) |
| M220 | Para el modelo de difusión: interpolación bilinear adicional de inputs ERA5 al tamaño espacial de IMERG antes de entrar al UNet; modelo aprende el **residuo** entre target HR y el input LR interpolado (no predice el campo HR directo) | "Data processing" | CRÍTICO — diferencia arquitectónica clave vs. GAN/ResNet |

### B.3 Modelos baseline

| ID | Requisito científico | Ref. (P2) | Prioridad |
|---|---|---|---|
| M225 | Interpolación bilineal de TP de ERA5 como lower-bound baseline | "Interpolation baseline" | ALTO |
| M226 | ResNet: arquitectura fully-convolutional, sigue trabajo previo (ref. 13 = Harris et al., el mismo grupo que inspira arquitectura de P1) | "ResNet" | ALTO — verificar que arquitectura realmente siga ref. 13 y no sea genérica |
| M227 | GAN: Wasserstein GAN con gradient penalty (WGAN-GP, ref. 21); condicionado tanto generador como discriminador con input LR | "Generative Adversarial Networks" | CRÍTICO |
| M228 | Diffusion Model (DM): sigue framework "ClimateDiffuse" (ref. 22); score-based bajo denoising score matching; U-Net condicional; SDE para proceso forward/reverse; usa preconditioning mejorado y esquema de integración de orden superior (ref. 46, Karras et al.) | "Diffusion-based models" | CRÍTICO |
| M229 | Entrenamiento GAN/DM: 60-168h (según tamaño de área de entrenamiento), batch efectivo 128, 4× NVIDIA A100 | "Training details" | MEDIO |
| M230 | Entrenamiento ResNet: 45-280h, 1× NVIDIA RTX8000 | "Training details" | BAJO |

### B.4 Escenarios de entrenamiento jerárquicos

| ID | Requisito científico | Ref. (P2) | Prioridad |
|---|---|---|---|
| M235 | A1 := T1 (una sola región) | "Training sub-tasks" | CRÍTICO |
| M236 | A2 := T1 ∪ T2 ∪ T5 (Norteamérica + Europa) | "Training sub-tasks" | CRÍTICO |
| M237 | A3 := A2 ∪ T10 ∪ T11 ∪ T12 (+ Asia oriental + Australia oriental) | "Training sub-tasks" | CRÍTICO |
| M238 | A4 := unión de las 12 regiones T1-T12 | "Training sub-tasks" | CRÍTICO |
| M239 | Jerarquía estricta: Aᵢ ⊆ Aᵢ₊₁ para i=1..3 | "Training sub-tasks" | ALTO |

### B.5 Métricas de evaluación

| ID | Requisito científico | Ref. (P2) | Prioridad |
|---|---|---|---|
| M245 | CRPS(F,y) = ∫[F(z)−𝟙(z≥y)]²dz (Ec. 1); calculado point-wise con 8 muestras por ubicación de las 6 regiones target | Ec. (1), "Quantitative evaluation" | CRÍTICO |
| M246 | Percentile MAE: MAE_p = (1/D)Σ\|x_pred(⌊pN⌋,d) − x_ref(⌊pN⌋,d)\| (Ec. 2), evaluado en percentiles 95 y 99, promediado sobre dimensiones | Ec. (2) | CRÍTICO |
| M247 | Para modelos determinísticos, CRPS se reduce a MAE (coherente con P1, M181) | "Quantitative evaluation" | ALTO |
| M248 | Entrenamiento directo en regiones target (Eᵢ) usado como upper-bound/no-competidor, mismo split temporal (2001-2020 train / 2021-2022 test) | "In-area training" | ALTO |

### B.6 Domain adaptation (zero-shot y few-shot)

| ID | Requisito científico | Ref. (P2) | Prioridad |
|---|---|---|---|
| M255 | Quantile mapping (QM) multiplicativo (variante no-negativa, lower-bounded en cero) sobre inputs LR de precipitación, usando 1000 cuantiles, aplicado **antes** de la normalización durante inferencia | Fórmula explícita: x̂_target,f(t) = F⁻¹_train,h(F_target,h[x_target,f(t)]) | CRÍTICO |
| M256 | Mapeo construido entre CDF histórica de train y CDF histórica de target sobre periodo histórico h; aplicado a periodo futuro f | "Quantile mapping..." | ALTO |
| M257 | Fine-tuning few-shot: 2,000 muestras aleatorias de la región target; 1 época; LR reducido a 1/5 del original (0.0001) | "Fine-tuning model to target regions..." | CRÍTICO — valores exactos (2000 muestras, 1 época, LR/5=0.0001) |
| M258 | Selección de combinaciones modelo-escenario-región para fine-tuning: basada en mayores caídas de desempeño relativas a entrenamiento in-distribution (Fig. 7) | Misma sección | MEDIO |

### B.7 Comportamiento anómalo documentado (para verificar si el código lo reproduce o corrige)

| ID | Hallazgo del paper | Ref. (P2) | Prioridad |
|---|---|---|---|
| M265 | ResNet + A1 + Tibetan Plateau: CRPS anómalo (18113.558) por inestabilidad numérica en inferencia — causa: normalización con estadísticas de train comprime inputs casi-cero de Tibetan Plateau a rango estrecho; tras exponenciación inversa, pequeñas desviaciones se amplifican en outliers extremos | "Domain adaptation resolves instabilities...", Fig. 12 | ALTO — el código debería (o no) tener manejo de este caso; es una limitación conocida, no necesariamente un bug a corregir |
| M266 | El paper marca explícitamente esos valores N/A en heatmaps de Figs. 5-8 cuando la inestabilidad hace el CRPS "no confiable" | Figs. 5-8, notas al pie | MEDIO |

---

## BLOQUE C — Contradicciones/ambigüedades a señalar entre P1 y P2 (Regla 16 del prompt)

| ID | Descripción | Ref. |
|---|---|---|
| C001 | P1 usa factor de upsampling 12× espacial / 6× temporal (24km→2km, 1h→10min); P2 usa factor de upsampling 2.5× espacial (aunque compara conceptualmente con arquitecturas tipo P1). No son directamente comparables porque son tareas distintas (P1: downscaling espacio-temporal fijo; P2: benchmark de generalización geográfica espacial). El código de P2 (ResNet/GAN) dice inspirarse en la arquitectura de P1 (ref. 13 en P2 = Harris et al., no exactamente P1, pero P2 sí cita a P1 como ref. 41/56 en su discusión) — **verificar en P2 si realmente reutiliza código/arquitectura de P1 o solo se inspira conceptualmente**. | P1 Methods; P2 "Downscaling task formulation", refs. 41,56 |
| C002 | Posible inconsistencia interna en P1 sobre número de GPUs de entrenamiento (abstract-adyacente dice "4 A100 80GB GPUs... 3 days"; sección "Training and model selection" dice "3 Nvidia A100 GPUs... 4 days"). No es contradicción entre papers sino dentro de P1 mismo — señalar como hallazgo de auditoría documental, no de código. | P1 pág. 3 vs "Training and model selection" |
| C003 | P1 evalúa con CRPS de 100 miembros de ensemble; P2 evalúa CRPS con solo 8 muestras por ubicación. Si el código de auditoría reutiliza métricas compartidas entre ambos proyectos, debe usarse el número de miembros correcto según cuál paper se está reproduciendo. | P1 "Evaluation"; P2 "Quantitative evaluation" |

---

## BLOQUE D — Notas de uso de esta matriz

1. Esta matriz cubre **~90 requisitos verificables** (M101-M266) más 3 contradicciones documentales (C001-C003). Está pensada para ser recorrida ítem por ítem contra el código real en el Prompt 2.
2. Los requisitos marcados **CRÍTICO** son prioritarios: en su mayoría corresponden a (a) ecuaciones de pérdida/arquitectura, (b) parámetros de submuestreo/normalización con constantes numéricas explícitas, y (c) splits de datos que determinan reproducibilidad y fuga de datos (data leakage).
3. Varios requisitos (M112, M201, M226-M228) dependen de material suplementario o de referencias externas (ref. 13, 22, 29, 32, 44) que el paper no reproduce en su totalidad — el código debe consultarse directamente para estos casos; no se debe asumir su contenido.
4. No se incluyen aquí los detalles de arquitectura de las referencias externas citadas (spateGAN original ref.32, ClimateDiffuse ref.22, ecPoint ref.29) — si el código las reimplementa, se recomienda una sub-auditoría independiente contra esos papers también.

