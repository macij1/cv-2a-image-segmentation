# Experimentos — Parte 4: Segmentación Semántica xBD

Dataset: xBD (xView2) — imágenes satelitales pre/post desastre natural.  
Tarea base: segmentación semántica de edificios.

---

## EXP-01 — Encoder-Decoder Binario (baseline)

**Fecha**: 2026-04-23  
**Ruta del modelo**: `results/parte-4/exp01_binary_encoder_decoder/encoder_decoder_binary_exp01_best.pth.tar`  
**Ruta de resultados**: `results/parte-4/exp01_binary_encoder_decoder/`

### Configuración

| Parámetro | Valor |
|---|---|
| Arquitectura | Encoder-Decoder custom (desde cero) |
| Encoder | 3× Conv3×3+ReLU + MaxPool2×2 |
| Bottleneck | Conv3×3+ReLU (256 ch) |
| Decoder | 3× Upsample bilinear + Conv3×3+ReLU |
| Head | Conv1×1 → num\_classes |
| Skip connections | ninguna |
| Tarea | Binaria (0=fondo, 1=edificio) |
| patch\_size | 128 |
| num\_classes | 2 |
| batch\_size | 4 |
| Optimizador | Adam lr=1e-3 |
| LR scheduler | StepLR step=5 γ=0.1 |
| Épocas | 8 |
| Augmentation | ninguna |
| Loss | CrossEntropyLoss sin pesos |
| Estadísticas | calculadas sobre el split de train de xBD |

### Resultados de entrenamiento

| Época | Train Loss | Val Loss | Val Jaccard |
|---|---|---|---|
| 1 | 0.3410 | 0.4061 | 0.262 |
| 2 | 0.1741 | 0.3609 | 0.367 |
| 3 | 0.2327 | 0.4167 | 0.336 |
| 4 | 0.3890 | 0.4021 | 0.314 |
| 5 | 0.4503 | 0.3203 | 0.336 |
| 6 | 0.4483 | 0.3909 | 0.322 |
| 7 | 0.2749 | 0.3696 | 0.330 |
| 8 | 0.2732 | 0.3643 | 0.345 |

*Nota: Train Jaccard crece monotónicamente (0.357→0.601); Val Jaccard oscila entre 0.31 y 0.37 sin converger claramente.*

### Resultados de evaluación (test)

| Clase | Recall (matriz de confusión)z |
|---|---|
| fondo | 89.24% correctamente clasificado (10.76% FP → predicho como edificio) |
| edificio | 47.27% correctamente detectado (52.73% FN → predicho como fondo) |
| **Jaccard foreground (test)** | **0.346** |
| **Val Jaccard (foreground, mejor época)** | **0.367 (época 2)** |

**Matriz de confusión**: el modelo detecta casi la mitad de los edificios (47.27% recall). Precision > recall: cuando el modelo predice edificio suele acertar; cuando duda, tiende a predecir fondo. Los falsos negativos dominan — el desbalanceo de clases es el limitante principal.

### Detalles

**Hipótesis**: una arquitectura mínima entrenada desde cero debería ser capaz de detectar edificios (binario) en parches de 128×128, validando que el pipeline completo (dataset → modelo → loss → Jaccard) funciona antes de añadir complejidad.

**Observaciones**:
- La brecha train/val es notable (Train Jaccard 0.60 vs Val 0.35 en época 8) — el modelo memoriza más de lo que generaliza.
- Val Jaccard alcanza su pico en época 2 (0.367) y no vuelve a superarlo: el modelo toca techo con esta configuración.
- **Precision > recall (cualitativamente confirmado).** El modelo sabe *que hay un edificio*, pero tiene problemas para definir *dónde acaban exactamente sus bordes*. Las predicciones son manchas difusas centradas en la región del edificio, no máscaras precisas. Esto es la firma directa de la ausencia de skip connections: el bottleneck codifica semántica global, pero el decoder no tiene acceso a los feature maps de alta resolución para delinear contornos.
- **El desbalanceo de clases limita el recall.** Con más píxeles de fondo que de edificio en cada parche, la CrossEntropyLoss uniforme penaliza más los errores en fondo. El modelo aprende a ser conservador: predice edificio solo cuando hay señal clara, sacrificando recall.

**Aprendizajes**:
- **El desbalanceo de clases penaliza el recall.** Con CrossEntropyLoss uniforme y ~85% de píxeles de fondo, el modelo predice fondo ante cualquier ambigüedad. Loss ponderada (`weight=[1.0, w]` con `w ~ ratio fondo/edificio ≈ 5–6`) o Focal Loss reequilibran la contribución por clase y deberían mejorar el recall sin sacrificar precision.
- **La ausencia de skip connections produce bordes borrosos.** El decoder reconstruye la máscara partiendo únicamente del bottleneck (resolución H/8), sin acceso a los feature maps de capas anteriores. El modelo localiza el edificio pero sus fronteras son difusas. Las skip connections (estilo U-Net) transfieren información espacial de alta frecuencia al decoder, permitiendo delinear bordes con precisión.
- **El LR schedule es demasiado agresivo para 8 épocas.** StepLR con step=5 reduce el LR ×10 en la época 5, dejando solo 3 épocas de ajuste fino. El modelo necesita más tiempo con LR estable o un schedule más suave.
- **Próximo experimento**: CrossEntropyLoss ponderada + skip connections (U-Net) + más épocas + LR schedule más suave. Además, vamos a introducir técnicas de data augmentation para ampliar el tamaño de 

---

## EXP-02 — U-Net Binario + Loss Ponderada

**Fecha**: 2026-05-05  
**Ruta del modelo**: `results/parte-4/exp02_unet_weighted_loss/unet_weighted_loss_exp02_best.pth.tar`  
**Ruta de resultados**: `results/parte-4/exp02_unet_weighted_loss/`

### Configuración

| Parámetro | Valor |
|---|---|
| Arquitectura | U-Net custom (3 niveles, skip connections) |
| Encoder | 3× Conv3×3+ReLU + MaxPool2×2 (32→64→128 ch) |
| Bottleneck | Conv3×3+ReLU (256 ch) |
| Decoder | 3× Upsample bilinear + cat(skip) + Conv3×3+ReLU |
| Head | Conv1×1 → num\_classes |
| Skip connections | sí (concat encoder↔decoder en cada nivel) |
| Tarea | Binaria (0=fondo, 1=edificio) |
| patch\_size | 128 |
| num\_classes | 2 |
| batch\_size | 4 |
| Optimizador | Adam lr=1e-3 |
| LR scheduler | CosineAnnealingLR T\_max=20 |
| Épocas | 20 |
| Augmentation | ninguna |
| Loss | CrossEntropyLoss ponderada weight=[1.0, 5.0] |
| Estadísticas | cacheadas de EXP-01 |

### Resultados de entrenamiento

| Época | Train Loss | Val Loss | Train Jaccard | Val Jaccard |
|---|---|---|---|---|
| 1  | 0.4727 | 0.7920 | 0.397 | 0.348 |
| 2  | 0.3917 | 0.6266 | 0.464 | 0.386 |
| 3  | 0.3598 | 0.6513 | 0.491 | 0.403 |
| 4  | 0.3402 | 0.6034 | 0.508 | 0.415 |
| 5  | 0.3264 | 0.6186 | 0.520 | 0.391 |
| 6  | 0.3161 | 0.6793 | 0.531 | 0.386 |
| **7**  | **0.3094** | **0.5438** | **0.537** | **0.422** ← mejor |
| 8  | 0.2977 | 0.5602 | 0.546 | 0.416 |
| 9  | 0.2905 | 0.6481 | 0.552 | 0.419 |
| 10 | 0.2847 | 0.6142 | 0.558 | 0.420 |
| 11 | 0.2744 | 0.6331 | 0.567 | 0.409 |
| 12 | 0.2662 | 0.7150 | 0.575 | 0.414 |
| 13 | 0.2592 | 0.7872 | 0.581 | 0.398 |
| 14 | 0.2510 | 0.7677 | 0.589 | 0.399 |
| 15 | 0.2444 | 0.8388 | 0.595 | 0.394 |
| 16 | 0.2385 | 0.7919 | 0.601 | 0.398 |
| 17 | 0.2327 | 0.8672 | 0.607 | 0.393 |
| 18 | 0.2286 | 0.8464 | 0.611 | 0.397 |
| 19 | 0.2258 | 0.9533 | 0.614 | 0.390 |
| 20 | 0.2245 | 0.9343 | 0.615 | 0.390 |

### Resultados de evaluación (test)

| Métrica | Valor |
|---|---|
| **Jaccard foreground (test)** | **0.427** |
| **Val Jaccard foreground (mejor época, ep. 7)** | **0.422** |

*Mejora respecto a EXP-01: +8.1pp en test Jaccard (0.346 → 0.427), +5.5pp en val Jaccard (0.367 → 0.422).*

### Detalles

**Hipótesis**: las dos limitaciones identificadas en EXP-01 (ausencia de skip connections → bordes borrosos; loss uniforme → recall bajo por desbalanceo) se abordan simultáneamente. Las skip connections transfieren feature maps de alta resolución al decoder; el peso ×5 en la clase edificio fuerza a la red a penalizar más los falsos negativos.

**Observaciones**:
- Las skip connections mejoran claramente la precisión de bordes: el Jaccard de test sube de 0.346 a 0.427.

- La loss ponderada mejora el recall de edificios respecto a EXP-01 (el modelo ya no es tan conservador).
- El modelo entra en overfitting a partir de la época 7: Val Jaccard alcanza su pico (0.422) y a partir de ahí oscila a la baja mientras Train Jaccard sigue creciendo hasta 0.615. Val Loss prácticamente se duplica entre épocas 7 y 20 (0.54 → 0.93).
- CosineAnnealingLR no previene el overfitting: reduce el LR suavemente pero sin regularización adicional el modelo memoriza el train set.
- La brecha train/val al final (0.615 vs 0.390) es similar a EXP-01, pero el punto de partida y el pico son más altos.

**Aprendizajes**:
- **Las skip connections son el cambio más impactante.** La ganancia de 8pp en test Jaccard confirma que el decoder necesita información espacial de alta frecuencia para delinear contornos de edificios con precisión.
- **La loss ponderada ayuda pero no resuelve el desbalanceo.** El peso [1.0, 5.0] mejora el recall, pero un ratio fijo puede no ser óptimo para todos los parches. Focal Loss adaptaría dinámicamente el peso según la dificultad de cada píxel.
- **El overfitting es el nuevo cuello de botella.** Con más capacidad (U-Net vs Encoder-Decoder plano), la red memoriza más rápido. Opciones para el siguiente experimento: data augmentation (flip/rotate/color jitter aplicados simultáneamente a imagen y máscara), dropout en el decoder, o weight decay.
- **Próximo experimento**: añadir data augmentation básico (flip horizontal/vertical + rotaciones 90°) para aumentar la variabilidad del training set y reducir la brecha train/val.

---

## EXP-03 — U-Net + Focal Loss + Data Augmentation

**Fecha**: 2026-05-05  
**Ruta del modelo**: `results/parte-4/exp03_unet_focal_augmentation/unet_focal_aug_exp03_best.pth.tar`  
**Ruta de resultados**: `results/parte-4/exp03_unet_focal_augmentation/`

### Configuración

| Parámetro | Valor |
|---|---|
| Arquitectura | U-Net custom (3 niveles, skip connections) — igual que EXP-02 |
| Tarea | Binaria (0=fondo, 1=edificio) |
| patch\_size | 128 |
| num\_classes | 2 |
| batch\_size | 4 |
| Optimizador | Adam lr=1e-3 |
| LR scheduler | CosineAnnealingLR T\_max=30 |
| Épocas | 30 |
| Augmentation | Flip H + Flip V + Rot {0°,90°,180°,270°} + Brillo/Contraste ±20% (solo train) |
| Loss | FocalLoss γ=2.0, α=[1.0, 4.6] (ratio calculado del training set) |
| Estadísticas | cacheadas de EXP-01 |

**Ratio de desbalanceo medido**: 50 imágenes de train muestreadas → 18% edificio / 82% fondo → ratio 4.6×.

### Resultados de entrenamiento

| Época | Train Loss | Val Loss | Train Jaccard | Val Jaccard |
|---|---|---|---|---|
| 1  | 0.4790 | 0.6611 | 0.329 | 0.349 |
| 2  | 0.3973 | 0.5773 | 0.390 | 0.355 |
| 3  | 0.3646 | 0.5183 | 0.417 | 0.359 |
| 4  | 0.3520 | 0.5885 | 0.428 | 0.378 |
| 5  | 0.3431 | 0.5470 | 0.436 | 0.383 |
| 6  | 0.3369 | 0.6212 | 0.441 | 0.389 |
| 7  | 0.3307 | 0.5397 | 0.448 | 0.387 |
| 8  | 0.3264 | 0.5582 | 0.450 | 0.399 |
| 9  | 0.3219 | 0.5851 | 0.453 | 0.418 |
| 10 | 0.3187 | 0.5431 | 0.457 | 0.394 |
| 11 | 0.3153 | 0.5161 | 0.461 | 0.384 |
| **12** | **0.3122** | **0.5501** | **0.464** | **0.426** ← mejor |
| 13 | 0.3099 | 0.5661 | 0.466 | 0.403 |
| 14 | 0.3064 | 0.5491 | 0.470 | 0.399 |
| 15 | 0.3030 | 0.5375 | 0.473 | 0.413 |
| 16 | 0.3008 | 0.5192 | 0.476 | 0.401 |
| 17 | 0.2979 | 0.5334 | 0.479 | 0.412 |
| 18 | 0.2963 | 0.5497 | 0.480 | 0.407 |
| 19 | 0.2947 | 0.5725 | 0.482 | 0.421 |
| 20 | 0.2920 | 0.5380 | 0.484 | 0.413 |
| 21 | 0.2895 | 0.5706 | 0.487 | 0.417 |
| 22 | 0.2888 | 0.5435 | 0.488 | 0.418 |
| 23 | 0.2869 | 0.5502 | 0.489 | 0.423 |
| 24 | 0.2850 | 0.5296 | 0.492 | 0.417 |
| 25 | 0.2842 | 0.5175 | 0.492 | 0.419 |
| 26 | 0.2833 | 0.5401 | 0.494 | 0.423 |
| 27 | 0.2818 | 0.5418 | 0.495 | 0.420 |
| 28 | 0.2814 | 0.5275 | 0.496 | 0.422 |
| 29 | 0.2810 | 0.5349 | 0.495 | 0.423 |
| 30 | 0.2807 | 0.5369 | 0.496 | 0.423 |

### Resultados de evaluación (test)

| Métrica | Valor |
|---|---|
| **Jaccard foreground (test)** | **0.441** |
| **Val Jaccard foreground (mejor época, ep. 12)** | **0.426** |

**Matriz de confusión (test):**

| | Pred. fondo | Pred. edificio |
|---|---|---|
| GT fondo | 65.02% | **34.98%** (FP) |
| GT edificio | **18.04%** (FN) | 81.96% |

*Mejora respecto a EXP-02: +1.4pp en test Jaccard (0.427 → 0.441). Recall de edificio: 47% (EXP-01) → 82% (EXP-03). FP de fondo: 11% → 35%.*

### Detalles

**Hipótesis**: combinando Focal Loss (que suprime el gradiente de píxeles fáciles y se concentra en los difíciles) con data augmentation geométrico y fotométrico se esperaba mejorar tanto el recall como reducir el overfitting observado en EXP-02.

**Observaciones**:

- **Data augmentation redujo el overfitting drásticamente.** La brecha train/val al final del entrenamiento pasó de 0.225 (EXP-02, ep. 20) a 0.073 (EXP-03, ep. 30). El modelo ya no memoriza: cada época ve versiones distintas de los mismos parches. Sin embargo, el techo del Val Jaccard apenas se movió (0.422 → 0.426), lo que indica que el cuello de botella ya no es el overfitting sino la capacidad representacional del modelo.

- **Focal Loss resolvió el desbalanceo de clases.** El recall de edificios saltó de 47% (EXP-01, sin pesos) a 82% (EXP-03). El precio fue un aumento de los falsos positivos: el modelo ahora "se atreve" a predecir edificio en zonas grises ambiguas (carreteras, superficies planas con textura similar a tejados). La confusion matrix confirma: 35% de los píxeles de fondo son clasificados como edificio.

- **Val Jaccard se aplana desde la época 9.** A partir de ahí oscila entre 0.38 y 0.43 durante 21 épocas sin converger. El modelo localiza correctamente *dónde* hay edificios pero genera blobs imprecisos que sobrepasan los bordes reales. La inspección visual confirma: las predicciones son manchas orgánicas sin la geometría rectangular característica de los edificios vistos desde satélite.

- **El problema residual es falta de contexto global.** Con patch\_size=128, el campo receptivo del modelo abarca una fracción pequeña de la imagen. Carreteras, parkings o cualquier superficie plana gris que aparezca en el parche se confunden con tejados porque localmente tienen texturas similares. Para distinguirlos se necesita ver el entorno más amplio — el modelo necesita saber que esa superficie gris está rodeada de carriles de tráfico, no de otras estructuras.

**Por qué el siguiente paso lógico es DeepLabV3:**

La arquitectura U-Net custom ha llegado a su techo con esta configuración. Los problemas que quedan son de naturaleza diferente a los que se pueden resolver cambiando la loss o el augmentation:

1. **Contexto multi-escala**: el ASPP (*Atrous Spatial Pyramid Pooling*) de DeepLabV3 aplica convoluciones dilatadas con tasas r=6, 12, 18 simultáneamente. Esto permite al modelo ver el contexto inmediato del píxel (r=6), el contexto del edificio (r=12) y el contexto del barrio (r=18) en una sola pasada, sin perder resolución espacial. Es la solución directa al problema de "confusión con superficies grises".

2. **Features preentrenadas**: el backbone ResNet-101 preentrenado en ImageNet llega con representaciones ricas de texturas, bordes y estructuras aprendidas de millones de imágenes. Fine-tuning sobre xBD parte de features mucho más discriminativas que las aprendidas desde cero en 6358 parches.

3. **Patch size mayor**: al tener menos muestras por época con patch\_size=512 (~400-600 vs 6358), el modelo preentrenado puede converger igualmente porque no necesita aprender desde cero. Más contexto por parche sin el coste de un entrenamiento más largo.

**Aprendizajes clave del ciclo EXP-01 → EXP-03:**
- Skip connections: imprescindibles para bordes (+8pp Jaccard, EXP-01→02)
- Focal Loss: resuelve el desbalanceo de recall (+35pp recall de edificio)
- Augmentation: controla el overfitting (gap 0.225→0.073) pero no mejora el techo
- El techo de una U-Net custom entrenada desde cero con patch\_size=128 está alrededor de Jaccard=0.44

---

## EXP-04 — DeepLabV3 R101 + Focal Loss + Augmentation

**Fecha**: 2026-05-05  
**Ruta del modelo**: `results/parte-4/exp04_deeplabv3_focal_aug/deeplabv3_focal_aug_exp04_best.pth.tar`  
**Ruta de resultados**: `results/parte-4/exp04_deeplabv3_focal_aug/`

> **Nota**: hubo un primer run con normalización incorrecta (xBD stats en lugar de ImageNet) que fue descartado. El run documentado aquí usa `stats=None` en xBDDataset y normalización ImageNet en el adaptador.

### Configuración

| Parámetro | Valor |
|---|---|
| Arquitectura | DeepLabV3 backbone ResNet-101 preentrenado (COCO) |
| Tarea | Binaria (0=fondo, 1=edificio) |
| patch\_size | 512 |
| num\_classes | 2 |
| batch\_size | 2 |
| Optimizador | Adam lr=1e-4 |
| LR scheduler | CosineAnnealingLR T\_max=25 |
| Épocas | 25 |
| Augmentation | Flip H/V + Rot90 + Brillo/Contraste ±20% |
| Loss | FocalLoss γ=2.0, α=[1.0, 4.6] |
| Normalización | ImageNet: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] |
| Parámetros | 60,986,436 |

### Resultados de entrenamiento

| Época | Train Loss | Val Loss | Train Jaccard | Val Jaccard |
|---|---|---|---|---|
| 1  | 0.3745 | 0.4246 | 0.323 | 0.336 |
| 2  | 0.2742 | 1.0392 | 0.405 | 0.280 |
| 3  | 0.2647 | 0.4144 | 0.416 | 0.368 |
| 4  | 0.2404 | 0.6398 | 0.436 | 0.405 |
| 5  | 0.2253 | 0.4005 | 0.459 | 0.413 |
| 6  | 0.2167 | 0.4099 | 0.467 | 0.409 |
| 7  | 0.2102 | 0.5687 | 0.476 | 0.428 |
| 8  | 0.2060 | 0.5360 | 0.485 | 0.410 |
| 9  | 0.2061 | 0.4990 | 0.483 | 0.449 |
| 10 | 0.1950 | 0.3823 | 0.498 | 0.443 |
| 11 | 0.1993 | 0.3621 | 0.489 | 0.409 |
| 12 | 0.1915 | 0.3796 | 0.498 | 0.446 |
| 13 | 0.1851 | 0.3949 | 0.512 | 0.446 |
| 14 | 0.1867 | 0.4605 | 0.499 | 0.467 |
| 15 | 0.1796 | 0.3826 | 0.519 | 0.464 |
| 16 | 0.1765 | 0.4200 | 0.522 | 0.460 |
| 17 | 0.1710 | 0.4128 | 0.534 | 0.478 |
| 18 | 0.1693 | 0.4239 | 0.533 | 0.473 |
| 19 | 0.1673 | 0.3414 | 0.535 | 0.461 |
| 20 | 0.1654 | 0.4535 | 0.539 | 0.481 |
| 21 | 0.1634 | 0.4039 | 0.543 | 0.478 |
| 22 | 0.1628 | 0.3735 | 0.542 | 0.476 |
| **23** | **0.1614** | **0.4185** | **0.544** | **0.483** ← mejor |
| 24 | 0.1608 | 0.4037 | 0.551 | 0.481 |
| 25 | 0.1596 | 0.4066 | 0.550 | 0.481 |

*Train/val gap al finalizar: 0.550 − 0.481 = **0.069** (mismo nivel que EXP-03, a pesar del modelo mucho más grande).*

### Resultados de evaluación (test)

| Métrica | Valor |
|---|---|
| **Jaccard foreground medio (test)** | **0.494** |
| Val Jaccard mejor época (23) | 0.483 |

*Mejora respecto a EXP-03: +5.3pp Jaccard (0.441 → 0.494). El modelo sigue sin platear al final de las 25 épocas, lo que sugiere que podría mejorar con más entrenamiento.*

### Detalles

**El problema de los bordes persiste.** A nivel cuantitativo el Jaccard sube, pero cualitativamente el modelo sigue sin capturar la geometría individual de los edificios. Las predicciones muestran manchas (*blobs*) que cubren zonas correctas de edificios pero fusionan edificios adyacentes en una sola región, sin preservar los bordes rectangulares ni separar instancias cercanas. El modelo localiza *dónde hay edificios*, pero no *dónde termina cada uno*.

Este problema tiene varias causas entrelazadas:

- **Resolución efectiva del decoder**: DeepLabV3 recupera resolución con un simple upsampling bilinear desde stride=16. Los detalles finos de borde se pierden en el encoder y no se recuperan.
- **Contexto de escena dominado por región**: ASPP capta contexto de barrio correctamente, pero ese contexto hace que píxeles de borde entre dos edificios próximos sean ambiguos — el modelo los asigna a "zona de edificio" en bloque.
- **Patch\_size y desbalanceo espacial**: con parches de 512px, las zonas de fondo son muy dominantes en área pero hay muchos edificios pequeños. El Focal Loss ayuda a nivel de píxel pero no impone ninguna prior sobre forma.

**Posibles mejoras:**

1. **DeepLabV3+ decoder**: añade un módulo de decoder con skip connections desde encoder layer1 (stride=4), recuperando detalles de borde sin coste computacional relevante.
2. **Reducir output stride de 16 a 8**: usando `replace_stride_with_dilation` en ResNet. Más costoso en memoria pero mejor resolución de features.
3. **Boundary-aware loss**: añadir un término de pérdida específico sobre los píxeles de borde extraídos del GT (laplaciano binario de la máscara) para forzar al modelo a discriminar en las fronteras.
4. **CRF como post-processing**: un CRF denso (DenseCRF) refina la segmentación usando contraste de color local, efectivo para recuperar bordes rectos en imágenes de satélite.
5. **Instance segmentation**: si el objetivo final es separar edificios individuales (vs. segmentación semántica), Mask R-CNN o similar es el enfoque correcto.

---

## Plantilla para nuevos experimentos

```
## EXP-XX — <Nombre descriptivo>

**Fecha**: YYYY-MM-DD
**Ruta del modelo**: `results/parte-4/<exp_dir>/<model_name>_best.pth.tar`
**Ruta de resultados**: `results/parte-4/<exp_dir>/`

### Configuración

| Parámetro | Valor |
|---|---|
| Arquitectura | |
| Tarea | Binaria / Multiclase (N clases) |
| patch_size | |
| num_classes | |
| batch_size | |
| Optimizador | |
| LR scheduler | |
| Épocas | |
| Augmentation | |
| Loss | |
| Estadísticas | |

### Resultados de entrenamiento

| Época | Train Loss | Val Loss | Val Jaccard |
|---|---|---|---|

### Resultados de evaluación (test)

| Clase | Jaccard / Recall |
|---|---|
| **Val Jaccard foreground (mejor época)** | |

### Detalles

**Hipótesis**:

**Observaciones**:

**Aprendizajes**:
```
