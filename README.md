# bbb: ¿cuántos genes hacen falta para distinguir GBM de LGG?

Pipeline reproducible: matriz de expresión (células endoteliales x genes) → genes variables → regresión de profundidad → escalado → PCA (ROM) → regresión logística / random forest → predecir `glioma_type` (GBM vs LGG).

**Pregunta:** ¿cuánta dimensionalidad (genes) puede quitar la PCA sin perder la capacidad de predecir el tipo de glioma? Se compara `PCA-r` (r = 2, 5, 10, 20, 50) frente a `HVG-2000` (genes más variables) y frente a `All genes` (26.666), con regresión logística y random forest. El modelo se entrena con un estudio (Wälchli) y se valida internamente (dejando un paciente fuera) y externamente (Xie).

> **Estado (30-oct-2026):** análisis interno, ajuste de hiperparámetros, validación externa, diagnóstico de estudio, análisis secundario `core`, intervalos bootstrap, significado de PC1/PC2 y dos análisis exploratorios terminados. Conclusión: señal modesta dentro de Wälchli; **no se reproduce en Xie completo** pero **sí aparece con solo muestras `core`** (6 pacientes, análisis secundario fijado de antemano); hay un efecto de estudio muy fuerte entre ambos conjuntos (sección 1).

---

## 1. Resumen de resultados

| Hallazgo | Estado |
|---|---|
| Con regresión logística, 2 componentes de PCA predicen tan bien como 2.000 o 26.666 genes (macro-F1 0,740 frente a 0,716 y 0,713) | Sólido en la dirección; las diferencias entre representaciones no están contrastadas |
| Con random forest, más genes empeoran (todos los genes: 0,578; PCA-2: 0,680) | Sólido |
| La señal supera al azar dentro de Wälchli (permutación exacta a nivel de paciente) | **Débil y dependiente de decisiones:** solo PCA-2 con regresión logística y regresión de profundidad (p ajustado = 0,033) |
| La señal es solo composición de tipos celulares | **No:** el acierto se mantiene dentro de casi todos los `ECclusters` (0,60-0,77) |
| Ajustar hiperparámetros (validación anidada) | **No cambia nada:** diferencias de ±0,02, dentro del ruido; el orden de representaciones no cambia |
| **Validación externa (Xie, 7 pacientes)** | **No se replica:** macro-F1 0,54-0,60; el modelo llama LGG a casi todo (los 4 GBM solo se aciertan 2, por poco) |
| **Validación externa, solo muestras `core` de Xie (6 pacientes; secundario fijado de antemano)** | **Mejor:** macro-F1 0,67-0,84 (PCA-5 con regresión logística 0,837); sin intervalos aún y con 6 pacientes. Sugiere que las muestras periféricas (`N`) y/o el paciente P036 explican parte del fallo; no está separado (4.7) |
| Intervalos bootstrap sobre pacientes | **Muy anchos:** PCA-2 con regresión logística 0,739 [0,557-0,889] en Wälchli; en Xie completo casi todos incluyen 0,5 (4.9) |
| Qué representan PC1 y PC2 | PC1 sigue la profundidad y la composición del paciente (rho 0,95); PC2 sigue el tipo de glioma, pero también la edad (4.8) |
| **Efecto de estudio (batch) entre Wälchli y Xie** | **Muy fuerte:** el estudio se identifica casi sin error en pacientes nuevos (AUC hasta 1,000); no lo explica la profundidad |

Línea base: predecir siempre la clase mayoritaria acierta 8/14 pacientes = 0,571 en Wälchli y 4/7 = 0,571 en Xie.

**Cómo contarlo:** señal modesta dentro de una cohorte de 14 pacientes, que no se reproduce en una segunda cohorte independiente de 7, con un efecto de estudio mucho mayor que la señal de glioma. Es un resultado negativo válido; con estos datos no se puede separar un artefacto técnico de una diferencia real entre cohortes.

---

## 2. Los dos conjuntos de datos

| | **Wälchli (entrenamiento)** | **Xie (validación externa)** |
|---|---|---|
| Artículo | Wälchli et al., *Nature* 632, 603-613 (2024) | Xie et al., *Neuron* 112(18), 3089-3105 (2024): "Single-cell dissection of the human blood-brain barrier and glioma blood-tumor barrier" |
| Datos | GEO `GSE256493`, objeto AnnData `...Combined_Glioma_Endothelial__Normalized_FullGenes_Scanpy.h5ad` (~2 GB) | GEO `GSE242044`, "Cerebrovasculature in brain malignancies from 13 surgical resected samples" (13 muestras, matrices 10x) |
| Pacientes | 14: 8 GBM y 6 LGG | 7: 4 GBM y 3 LGG |
| Células | 67.372 (GBM 49.999, LGG 17.373); entre 264 (GBM1) y 17.580 (GBM5) por paciente | 198.445 crudas → 198.205 tras control de calidad → **106.399 endoteliales** (7.492 a 24.100 por paciente) |
| Genes | 26.666 | 22.767 en común con Wälchli (85,4 %); de los 2.000 genes variables, 98,4 % presentes |
| Tipo de células | Endotelio ya aislado, 14 subtipos en `ECclusters` | Muestras enriquecidas en CD31+ (endotelio) y PDGFRβ+ (células murales); el endotelio se selecciona aquí por marcadores |
| Procesado | `log1p` de conteos normalizados a 10.000 por célula (verificado) | Conteos crudos de Cell Ranger 4.0.0/6.1.1 (GRCh38); aquí se normalizan igual que Wälchli |
| Profundidad (mediana) | 5.528 conteos y 2.358 genes por célula (percentiles 25-75: 3.096-9.134 y 1.493-3.291) | 2.963 conteos y 1.436 genes por célula |
| Control de calidad | El del objeto publicado | Aquí solo se exigen 200 genes por célula (más laxo que el del artículo, que informa de 103.230 células en total y 57.324 endoteliales) |
| Etiqueta | `glioma_type` (una por paciente) en `obs` | Diagnóstico de GEO por muestra (`configs/xie_samples.csv`) |

**Muestras de Xie.** `T` = núcleo del tumor y `N` = tejido periférico del mismo tumor (mismo diagnóstico que su `T`). El paciente 036 solo tiene muestra periférica (no hay T036). La numeración de pacientes de GEO difiere de la de las muestras; ambas están en `configs/xie_samples.csv`.

| Paciente | Diagnóstico | Muestras | Células endoteliales |
|---|---|---|---|
| P027 | LGG | N027, T027 | 18.732 |
| P037 | LGG | N037, T037 | 24.100 |
| P043 | LGG | N043, T043 | 7.492 |
| P036 | GBM | N036 | 11.596 |
| P038 | GBM | N038, T038 | 13.507 |
| P045 | GBM | N045, T045 | 19.675 |
| P046 | GBM | N046, T046 | 11.297 |

**Fracción endotelial por muestra** (tras el control de calidad): de 27 % (T038) a 81 % (N027, con solo 572 células); en conjunto 54 %, parecido al 55 % que informa el artículo (57.324 de 103.230). Tienen el doble de células que el artículo, probablemente por el control de calidad más laxo de aquí (inferencia, no comprobada).

**Independencia:** otro grupo, otra serie de GEO y otros pacientes. No se puede descartar del todo solapamiento de pacientes, pero es improbable. En Wälchli, `PATH`, `age` y `Patient` determinan por sí solas la etiqueta, y nunca se usan como características.

---

## 3. Qué se hizo, por orden

### 3.1 Comprobaciones de los datos (`src.validation`)
- Forma exacta 67.372 x 26.666: es el objeto de Wälchli, un único estudio. No hay columnas de estudio/muestra/lote en `obs`.
- Una sola etiqueta `glioma_type` por paciente.
- **Normalización:** la matriz ya es `log1p` de conteos normalizados a 10.000 por célula (suma de `expm1(X)` por célula: mediana 10.000, CV 0,0000; máximo de X = 8,75). Medido en una muestra de 500 células. No hace falta renormalizar.
- **Columnas que determinan la etiqueta por sí solas:** `PATH`, `age`, `Patient`. Nunca se usan como características (solo se usa `X`).

### 3.2 Diseño del experimento
- **Unidad independiente: el paciente.** Las células de un mismo paciente se parecen mucho entre sí. Validación dejando un paciente fuera (14 rondas); IDs en `results/<modo>/splits/`.
- **Sin fugas:** genes variables, regresión de profundidad, escalado y PCA se ajustan en cada ronda solo con las células de los pacientes de entrenamiento (test automático).
- **Desequilibrio** (GBM 49.999 células frente a LGG 17.373): tope de 1.000 células por paciente en entrenamiento, `class_weight="balanced"`, y evaluación con todas las células del paciente de test. Las métricas pesan igual a cada paciente.
- **Regresión de profundidad** (`rom.regress_out`): quita por regresión lineal, gen a gen, el efecto de `log1p(nCount_RNA)` y `log1p(nFeature_RNA)`, ajustada solo con entrenamiento. Se decidió por ser práctica habitual, antes de ver los resultados del conjunto completo. Se repitió sin ella (`full_noreg`) como análisis de sensibilidad.
- **Control:** etiquetas permutadas entre pacientes. Sale por debajo del azar (0,38-0,47 en vez de 0,5): artefacto conocido de dejar-un-paciente-fuera con pocos pacientes, no un fallo.
- **Test de permutación exacto** a nivel de paciente: C(14,6) = 3.003 asignaciones, todas recorridas. `p_adjusted_max` corrige por mirar todas las representaciones a la vez (estadístico máximo).

### 3.3 Ajuste de hiperparámetros con validación anidada (`experiments.09_tune`)
- **Rejilla** (por pareja representación-modelo): regresión logística con C ∈ {0,001; 0,01; 0,1; 1; 10}; random forest con `min_samples_leaf` ∈ {1, 20} (`max_depth` sin límite, 200 árboles). `All genes` queda con valores por defecto.
- **Anidada:** para cada paciente de test, los hiperparámetros se eligen solo con los otros 13 pacientes. La estimación resultante (`results/<modo>/tuning/results_nested_glioma_type.md`) es la honrada.
- **Modelo final:** hiperparámetros elegidos con los 14 pacientes, en `models/<modo>_tuned/bundle.joblib`. El rendimiento del LOPO interno de los 14 (`tuning_scores_all_patients.csv`) es optimista (máximo sobre la rejilla): no usarlo como estimación.
- No se ajustan la regresión de profundidad ni el tamaño de HVG: se fijan por diseño.

### 3.4 Validación externa con Xie (`10_prepare_xie`, `08_external`)
Decisiones fijadas **antes** de ver ningún resultado externo:
1. Paciente = unidad independiente; se usan las muestras `T` y `N` de cada paciente.
2. Células endoteliales = puntuación de marcadores endoteliales (PECAM1, CLDN5, VWF, FLT1, EGFL7, ESAM) por encima del percentil 5 de esa puntuación en las células de Wälchli (umbral 0,717), y mayor que la de marcadores murales/inmunes/gliales.
3. Mismo preprocesado que Wälchli: 10.000 por célula + log1p; `nCount_RNA` y `nFeature_RNA` desde los conteos crudos.
4. Se aplica el paquete entrenado con Wälchli (`models/full_tuned`) sin reajustar nada y se informan **todas** las representaciones y modelos, no solo la mejor.
5. Secundario: solo muestras `core` (`--sites core`), que deja 6 pacientes. Ejecutado (4.7).

### 3.5 Diagnóstico de estudio (batch) entre Wälchli y Xie (`experiments.11_batch`)
- **Pregunta:** ¿se puede adivinar de qué estudio es una célula de un paciente nuevo solo por su expresión?
- **Método:** dejar un paciente fuera sobre los 21 pacientes (14 + 7), con los 22.767 genes en común; mismas representaciones (PCA-2…50, HVG-2000) y modelos; con y sin regresión de profundidad; y un control que usa solo la profundidad.
- **Gráficas:** puntos coloreados por estudio (y por tipo de glioma) en el espacio PCA-2 del modelo entrenado, con su frontera de decisión (`09`), en una PCA ajustada con ambos estudios (`10`), y capacidad de identificar el estudio por representación (`11`).

### 3.6 Lo que se probó y se descartó
- **Renormalizar la matriz de Wälchli:** innecesario, ya viene normalizada.
- **Etiqueta de localización:** descartada; el objetivo final es solo `glioma_type`.
- **Permutación con "Todos los genes":** no se hizo; requiere matrices densas de 26.666 columnas y miles de ajustes.
- **Diagnóstico de batch con un solo estudio:** se quitó (no aportaba nada) y se recuperó cuando hubo dos conjuntos (3.5).
- **Corrección de C en PCA-2:** no era un fallo, la puntuación no depende de C (4.8).
- **Corte de profundidad en Xie (exploratorio, no es el análisis principal):** quedarse solo con células de ≥1.500 genes empeoró el resultado (ver 4.5). Se hizo después de ver los resultados y no debe presentarse como corrección del análisis principal.

---

## 4. Resultados

### 4.1 Rendimiento dentro de Wälchli (10 semillas, valores por defecto, regresión de profundidad activa)

Regresión logística, datos reales:

| Representación | Macro-F1 | Balanced acc. | Acierto por paciente |
|---|---|---|---|
| PCA-2 | 0,740 ± 0,001 | 0,737 | 0,857 (12/14) |
| PCA-5 | 0,704 ± 0,001 | 0,702 | 0,786 |
| PCA-10 | 0,672 ± 0,003 | 0,671 | 0,786 |
| PCA-20 | 0,654 ± 0,006 | 0,654 | 0,729 |
| PCA-50 | 0,702 ± 0,006 | 0,700 | 0,714 |
| HVG-2000 | 0,716 ± 0,005 | 0,715 | 0,786 |
| Todos los genes | 0,713 ± 0,008 | 0,712 | 0,764 |

Random forest, datos reales:

| Representación | Macro-F1 | Balanced acc. | Acierto por paciente |
|---|---|---|---|
| PCA-2 | 0,680 ± 0,003 | 0,679 | 0,793 |
| PCA-5 | 0,654 ± 0,005 | 0,653 | 0,786 |
| PCA-10 | 0,682 ± 0,007 | 0,680 | 0,786 |
| PCA-20 | 0,669 ± 0,005 | 0,668 | 0,750 |
| PCA-50 | 0,680 ± 0,002 | 0,678 | 0,729 |
| HVG-2000 | 0,622 ± 0,006 | 0,623 | 0,736 |
| Todos los genes | 0,578 ± 0,007 | 0,579 | 0,650 |

Con permutación de etiquetas (control), todas las filas quedan entre 0,38 y 0,47 de macro-F1. La desviación entre semillas mide el submuestreo de células y el azar del random forest, no la incertidumbre por tener 14 pacientes.

### 4.2 ¿Supera el azar? Permutación exacta (regresión logística)

Con regresión de profundidad (`results/full_hvgperm`, 6 representaciones):

| Representación | Macro-F1 real | Media nula | p sin ajustar | p ajustado (máx.) |
|---|---|---|---|---|
| PCA-2 | 0,739 | 0,487 | 0,010 | **0,033** |
| PCA-5 | 0,702 | 0,481 | 0,039 | 0,071 |
| PCA-10 | 0,674 | 0,474 | 0,057 | 0,116 |
| PCA-20 | 0,649 | 0,469 | 0,077 | 0,169 |
| PCA-50 | 0,699 | 0,464 | 0,034 | 0,075 |
| HVG-2000 | 0,713 | 0,460 | 0,026 | 0,060 |

Sin regresión de profundidad (`results/full_noreg`, 5 PCA):

| Representación | Macro-F1 real | p sin ajustar | p ajustado (máx.) |
|---|---|---|---|
| PCA-2 | 0,663 | 0,032 | 0,125 |
| PCA-5 | 0,681 | 0,051 | 0,090 |
| PCA-10 | 0,658 | 0,075 | 0,136 |
| PCA-20 | 0,618 | 0,116 | 0,238 |
| PCA-50 | 0,666 | 0,059 | 0,118 |

**Lectura honesta:** solo PCA-2 supera el umbral de 0,05 y solo con la regresión de profundidad. Sin ella, nada sobrevive. El ajuste corrige por el número de representaciones, **no** por haber elegido la regresión logística ni por haber probado con y sin regresión de profundidad.

### 4.3 Qué pacientes fallan y composición celular (Wälchli, PCA-2, regresión logística)
- **LGG1 y LGG6** se clasifican como GBM de forma sistemática (7 % y 7 % de sus células bien; con PCA-10, 9 % y 9 %). Es un fallo consistente, no ruido. **GBM6** es límite: bien con PCA-2 (58 % de células), mal con PCA-10 (36 %). El resto se clasifica con claridad: 12 de 14 pacientes con PCA-2, 11 de 14 con PCA-10.
- **Intervalos:** 12/14 corresponde a un intervalo de confianza del 95 % de 57 %-98 % (Clopper-Pearson); su límite inferior coincide con la línea base de 8/14 = 57 %. Con 11/14, 49 %-95 %.
- La composición de `ECclusters` difiere entre GBM y LGG (*Angiogenic capillary* 42 % frente a 27 %; *Stem-to-EC* 3,5 % frente a 0,2 %). **LGG6** tiene un 52 % de *Angiogenic capillary*, como los GBM típicos (48-55 %), lo que encaja con su error. **LGG1** tiene una composición casi idéntica a LGG2 y LGG4 (bien clasificados), así que la composición **no** lo explica. **GBM8** tiene solo un 2 % de *Angiogenic capillary* y aun así se acierta con el 82 % de sus células.
- Acierto dentro de cada clúster (PCA-2, balanced accuracy): *Artery* 0,77, *Vein* 0,77, *Proliferating cell* 0,76, *Large vein* 0,76, *Venule* 0,74, *Angiogenic capillary* 0,73, *Mitochondrial* 0,70, *Capillary* 0,69, *Large artery* 0,67, *EndoMT* 0,64, *Arteriole* 0,61, *Stem-to-EC* 0,60. No pasaron por permutación y los 14 pacientes son los mismos en todos los clústeres.

### 4.4 Ajuste de hiperparámetros: validación anidada frente a valores por defecto (macro-F1)

| Representación | Reg. logística: defecto → anidada | Random forest: defecto → anidada |
|---|---|---|
| PCA-2 | 0,740 → 0,740 | 0,680 → 0,693 |
| PCA-5 | 0,704 → 0,703 | 0,654 → 0,679 |
| PCA-10 | 0,672 → 0,681 | 0,682 → 0,692 |
| PCA-20 | 0,654 → 0,664 | 0,669 → 0,675 |
| PCA-50 | 0,702 → 0,700 | 0,680 → 0,681 |
| HVG-2000 | 0,716 → 0,695 | 0,622 → 0,630 |
| Todos los genes (sin ajuste) | 0,713 → 0,707 | 0,578 → 0,570 |

Acierto por paciente (anidada): PCA-2 0,857 (regresión logística) y 0,786 (random forest); el resto entre 0,643 y 0,857.

- **Conclusión:** el ajuste no mejora de forma apreciable. Random forest gana 0,01-0,025 con `min_samples_leaf = 20` (siempre elegido); la regresión logística en HVG-2000 pierde 0,02. El orden de representaciones no cambia: PCA-2 con regresión logística sigue siendo lo mejor.
- **Hiperparámetros finales (14 pacientes):** regresión logística C = 10 (PCA-2, HVG-2000), C = 0,001 (PCA-5, 10, 20), C = 0,01 (PCA-50); random forest `min_samples_leaf = 20` en todos.
- **Inconsistencia pendiente de revisar:** para PCA-2 con regresión logística, la validación anidada eligió C = 0,001 en 13 de 14 rondas, pero el modelo final eligió C = 10. Con solo 2 variables C influye poco. **Comprobado** (`tuning_scores_all_patients.csv`): macro-F1 0,7402 (C=0,001), 0,7392 (0,01) y 0,7391 (0,1, 1 y 10). La diferencia es de una milésima, así que la elección de C es ruido y no hay fallo.

### 4.5 Validación externa con Xie (análisis principal, `models/full_tuned`)

7 pacientes, 106.399 células. Macro-F1 ponderado por paciente:

| Representación | Reg. logística (F1 / bal. acc. / pacientes) | Random forest (F1 / bal. acc. / pacientes) |
|---|---|---|
| PCA-2 | 0,545 / 0,575 / 0,714 | 0,564 / 0,577 / 0,571 |
| PCA-5 | 0,576 / 0,637 / 0,571 | 0,581 / 0,623 / 0,714 |
| PCA-10 | 0,590 / 0,627 / 0,714 | 0,596 / 0,616 / 0,714 |
| PCA-20 | 0,546 / 0,598 / 0,571 | 0,575 / 0,596 / 0,714 |
| PCA-50 | 0,567 / 0,637 / 0,571 | 0,603 / 0,624 / 0,714 |
| HVG-2000 | 0,572 / 0,642 / 0,571 | 0,587 / 0,629 / 0,714 |
| Todos los genes | 0,536 / 0,617 / 0,571 | 0,581 / 0,616 / 0,714 |

- **Patrón por paciente (regresión logística):** los 3 LGG se aciertan (PCA-2: P027 73 %, P037 90 %, P043 62 % de sus células), pero el modelo llama LGG a casi todo. De los 4 GBM solo se aciertan 2 y por poco (PCA-2: P045 50 %, P046 52 %; P036 36 % y P038 21 %, mal clasificados). Con todos los genes los LGG salen al 95-98 % y solo se acierta P046.
- **El "5/7" (0,714) engaña:** son 3 LGG fáciles y 2 GBM justos; la línea base (4 GBM de 7) ya da 0,571. No hay evidencia de que distinga.
- **Exploratorio (corte de profundidad, solo células ≥1.500 genes, 50.114 células):** empeora: macro-F1 0,45-0,54 y acierto por paciente 3/7 en todas las filas. El modelo llama LGG a los 7 pacientes. Descarta la profundidad como causa principal del sesgo (solo se probó un corte).

### 4.6 Efecto de estudio (batch): Wälchli frente a Xie

Identificar el estudio en células de un paciente nuevo (balanced accuracy; entre paréntesis, AUC). Regresión logística:

| Representación | Con regresión de profundidad | Sin regresión de profundidad |
|---|---|---|
| PCA-2 | 0,658 (0,641) | 0,679 (0,704) |
| PCA-5 | 0,820 (0,900) | 0,820 (0,886) |
| PCA-10 | 0,876 (0,942) | 0,880 (0,950) |
| PCA-20 | 0,971 (0,996) | 0,986 (0,999) |
| PCA-50 | 0,975 (0,997) | 0,989 (0,999) |
| HVG-2000 | 0,973 (0,997) | 0,993 (1,000) |

Random forest da cifras parecidas (HVG-2000: 0,939 con y 0,966 sin regresión de profundidad). Con PCA-5 o más, casi todas las configuraciones asignan los 21 pacientes a su estudio sin fallo (las demás fallan 1 o 2 pacientes: acierto por paciente 0,905-0,952).

- **Solo con la profundidad** (log nCount y log nFeature): balanced accuracy 0,61, AUC 0,62. La profundidad casi no explica el efecto de estudio; es de expresión.
- **Más dimensiones, más se nota el estudio.** En PCA-2, el espacio donde el modelo funcionaba mejor dentro de Wälchli, el estudio es poco identificable (0,66-0,68) y el glioma sí (0,74).
- **Espacio PCA-2 del modelo** (PCA ajustada solo con Wälchli; `09_batch_model_space.png`): Xie no cae fuera de Wälchli (solo el 5,2 % de las células de Xie están fuera del rango central de Wälchli en PC1, y el 0,1 % en PC2), pero está **desplazada en PC1 hacia el lado LGG** (-1,02 desviaciones). Células en el lado LGG de la frontera:

| | Wälchli (datos de entrenamiento) | Xie |
|---|---|---|
| GBM | 14,6 % | 61,3 % |
| LGG | 80,3 % | 78,4 % |

  Es decir, las células LGG se reconocen igual (78 % frente a 80 %), pero la mayoría de los GBM de Xie caen en el lado LGG. El desplazamiento está **a lo largo del mismo eje que separa GBM de LGG**, lo peor para el modelo.
- **PCA conjunta** (`10_batch_joint_pca.png`; las 5 primeras componentes explican solo el 9,4 % de la varianza): el estudio se asocia sobre todo con PC3 (η² = 0,197) y PC5 (0,239); PC1 se asocia más con el tipo de glioma (0,294) que con el estudio (0,129). El η² del tipo de glioma mezcla ambos estudios y está confundido con el paciente (21 pacientes), así que no debe leerse como efecto causal.

### 4.7 Validación externa solo con muestras `core` de Xie (secundario, fijado de antemano)

6 pacientes (3 LGG: P027, P037, P043; 3 GBM: P038, P045, P046), 50.204 células, 6 muestras `T`. Se pierde P036 y todas las muestras `N`. Línea base: 3/6 = 0,5. Mismo paquete, sin reajustar:

| Representación | Reg. logística (F1 / bal. acc. / pacientes) | Random forest (F1 / bal. acc. / pacientes) |
|---|---|---|
| PCA-2 | 0,704 / 0,704 / 0,833 | 0,669 / 0,669 / 0,667 |
| PCA-5 | 0,837 / 0,838 / 0,833 | 0,831 / 0,831 / 1,000 |
| PCA-10 | 0,797 / 0,797 / 1,000 | 0,790 / 0,791 / 1,000 |
| PCA-20 | 0,776 / 0,777 / 0,833 | 0,764 / 0,765 / 1,000 |
| PCA-50 | 0,809 / 0,813 / 0,833 | 0,778 / 0,778 / 1,000 |
| HVG-2000 | 0,808 / 0,812 / 0,833 | 0,826 / 0,827 / 1,000 |
| Todos los genes | 0,824 / 0,827 / 1,000 | 0,804 / 0,804 / 1,000 |

- **Mejora clara respecto a Xie completo** (F1 0,54-0,60 → 0,67-0,84) en todas las representaciones y en ambos modelos, y con más de dos dimensiones, incluidos todos los genes.
- **Lo que no sabemos:** si la mejora viene de quitar las muestras periféricas `N`, de quitar a P036 (GBM mal clasificado en 4.5) o de ambas. No se ha mirado el detalle por paciente de esta ejecución (`results/external_xie_core/results_per_patient_external.csv`).
- **Cautelas:** 6 pacientes (un paciente = 0,167 de acierto por paciente), sin intervalo de confianza, 14 filas probadas sin corrección. Se ejecutó como análisis secundario fijado de antemano, pero después de ver que el principal fallaba. Es una indicación, no una replicación.

### 4.8 Qué representan PC1 y PC2 del modelo PCA-2 (`12_pc_meaning`)

Medias por paciente de Wälchli (14 pacientes). Correlaciones de Spearman sin ajustar por comparaciones múltiples (8 contrastes mostrados):

| Componente | Variable | rho | p |
|---|---|---|---|
| PC1 | mediana de nFeature_RNA | +0,95 | <0,001 |
| PC1 | mediana de nCount_RNA | +0,95 | <0,001 |
| PC1 | fracción *Angiogenic capillary* | +0,87 | <0,001 |
| PC1 | nº de células | +0,55 | 0,041 |
| PC2 | es LGG | +0,68 | 0,007 |
| PC2 | edad | -0,61 | 0,020 |
| PC2 | fracción *Stem-to-EC* | -0,47 | 0,091 |

- **PC1 sigue la profundidad y la composición celular del paciente, no el glioma**, aunque se regresó la profundidad por célula: a nivel de paciente persiste. Los GBM tienen valores de PC1 muy dispares (de -23,2 a +11,0).
- **PC2 sigue el tipo de glioma, pero también la edad.** En Wälchli edad y tipo van juntas (LGG: 26-46 años salvo LGG1, 69; GBM: 48-82), así que no se pueden separar con estos datos. El modelo podría estar aprendiendo en parte edad o rasgos asociados.
- **LGG1** (69 años, el único LGG mayor de 46) tiene PC2 = -5,4 y cae del lado GBM; los otros 5 LGG tienen PC2 entre +2,4 y +13,3. Es compatible con la confusión por edad, pero con un solo paciente no es demostración. Sus metadatos clínicos (subtipo, localización) siguen sin revisar.
- **Comprobación de C en PCA-2:** ver 4.4 (ruido, no fallo).

### 4.9 Intervalos bootstrap sobre pacientes (`13_bootstrap`)

Remuestreo de pacientes con reemplazo (5.000 veces, manteniendo el nº por clase) a partir de las predicciones ya hechas; no reentrena, así que los intervalos reales serían algo más anchos. Macro-F1 [IC 95 %]:

| Representación | Wälchli, reg. logística | Wälchli, random forest | Xie completo, reg. logística | Xie completo, random forest |
|---|---|---|---|---|
| PCA-2 | 0,739 [0,557-0,889] | 0,677 [0,503-0,817] | 0,545 [0,453-0,632] | 0,564 [0,423-0,680] |
| PCA-5 | 0,702 [0,515-0,856] | 0,658 [0,461-0,824] | 0,576 [0,447-0,688] | 0,581 [0,455-0,694] |
| PCA-10 | 0,674 [0,495-0,823] | 0,691 [0,496-0,852] | 0,590 [0,508-0,666] | 0,596 [0,484-0,705] |
| PCA-20 | 0,649 [0,456-0,812] | 0,675 [0,483-0,829] | 0,546 [0,426-0,650] | 0,575 [0,464-0,680] |
| PCA-50 | 0,699 [0,506-0,859] | 0,682 [0,488-0,837] | 0,567 [0,437-0,677] | 0,603 [0,498-0,700] |
| HVG-2000 | 0,713 [0,534-0,861] | 0,611 [0,398-0,788] | 0,572 [0,432-0,688] | 0,587 [0,462-0,700] |
| Todos los genes | 0,707 [0,515-0,861] | 0,569 [0,368-0,745] | 0,536 [0,393-0,655] | 0,581 [0,454-0,696] |

- En Wälchli, 6 de 14 filas tienen el límite inferior por encima de 0,5 (sin corregir por probar 14 filas); en Xie completo, 1 de 14 (PCA-10 con regresión logística, 0,508).
- Las diferencias entre representaciones son mucho menores que la anchura de los intervalos. No se puede afirmar que una sea mejor.
- No se calcularon aún intervalos para el análisis `core` (4.7) ni para el exploratorio (4.10).

### 4.10 Dos análisis exploratorios (declarados como tales; se hicieron tras ver los resultados anteriores)

**a) Alinear las medias por gen de Xie con las de Wälchli (`15_align_external`).** Se resta la diferencia de medias por gen (cada paciente pesa igual) antes de aplicar el modelo. Es **transductivo**: usa las células de Xie, sin etiquetas, para calcular el desplazamiento. No es validación externa limpia. Se omite "Todos los genes". Xie completo, macro-F1 sin alinear → alineado (pacientes bien clasificados alineado, de 7):

| Representación | Reg. logística | Random forest |
|---|---|---|
| PCA-2 | 0,545 → 0,559 (5/7) | 0,564 → 0,547 (5/7) |
| PCA-5 | 0,576 → 0,636 (6/7) | 0,581 → 0,600 (5/7) |
| PCA-10 | 0,590 → 0,635 (6/7) | 0,596 → 0,602 (5/7) |
| PCA-20 | 0,546 → 0,608 (4/7) | 0,575 → 0,582 (5/7) |
| PCA-50 | 0,567 → 0,716 (7/7) | 0,603 → 0,618 (6/7) |
| HVG-2000 | 0,572 → 0,725 (7/7) | 0,587 → 0,658 (7/7) |

  Mejora en casi todas las filas, más con muchas dimensiones; en PCA-2 apenas. Apunta a que parte del fallo es un desplazamiento de medias entre estudios, pero con 7 pacientes, sin intervalos y probando 12 filas, es una pista.

**b) Dejar un paciente fuera solo dentro de Xie (`14_within_xie`, 7 pacientes).** Macro-F1:

| Representación | Reg. logística | Random forest |
|---|---|---|
| PCA-2 | 0,620 | 0,517 |
| PCA-5 | 0,639 | 0,549 |
| PCA-10 | 0,609 | 0,573 |
| PCA-20 | 0,552 | 0,518 |
| PCA-50 | 0,537 | 0,515 |
| HVG-2000 | 0,524 | 0,358 |

  **No hay señal clara dentro de Xie:** cercano al azar, y baja al aumentar las dimensiones. HVG-2000 con random forest (0,358) queda por debajo del azar. PCA-2 con regresión logística acierta 7/7 pacientes pero con F1 0,62; con 7 pacientes no se interpreta como evidencia.

---

## 5. Por qué podría pasar (hipótesis, NO verificadas)

- **PCA-2 sin pérdida frente a todos los genes:** con 14 pacientes, pocas dimensiones actúan como regularización; más variables dan más ocasión de ajustar ruido de paciente o de estudio. Coherente con que random forest empeore con más genes y con que el estudio se note más al subir de dimensión (4.6).
- **Por qué no se transfiere a Xie:** el modelo puede haber aprendido en parte un eje que mezcla diagnóstico con rasgos propios de la cohorte o del procesamiento de Wälchli. Xie se desplaza justo sobre ese eje hacia el lado LGG. No sabemos si el desplazamiento es técnico (protocolo, control de calidad, selección del endotelio por marcadores, aquí menos estricta) o biológico (otra cohorte, otro tejido).
- **Posible influencia del tejido periférico en Xie:** las muestras `N` son periferia del tumor y podrían parecerse más a LGG. El análisis `core` (4.7) mejora mucho el resultado, pero también quita a P036; no se ha separado qué de las dos cosas lo causa.
- **Confusión con la edad en Wälchli:** PC2, que sigue el glioma, correlaciona con la edad (rho -0,61), y edad y tipo van juntas en esa cohorte (4.8). No se pueden separar con estos datos.
- **Desplazamiento de medias entre estudios:** alinear las medias de Xie mejora el resultado (4.10 a), compatible con un efecto de estudio que actúa sobre las medias; sin validar.
- **Composición celular:** explica en parte el caso de LGG6 (4.3), pero no el resto ni el fallo en Xie.
- **Regresión de profundidad:** mejora a PCA-2 (macro-F1 0,663 → 0,739 con una semilla) pero no sabemos por qué.

---

## 6. Limitaciones

- **14 pacientes en entrenamiento y 7 en validación.** Diferencias de uno o dos puntos entre representaciones no son distinguibles del ruido. Con 7 pacientes el test exacto tiene 35 asignaciones posibles: p < 0,05 solo se alcanza acertando los 7 (mínimo p ≈ 0,029). La validación externa es una indicación de dirección, no una prueba.
- **Las semillas no miden la incertidumbre real**: con dejar-un-paciente-fuera el reparto no cambia entre semillas.
- **Bifurcaciones del análisis:** modelo, con/sin regresión de profundidad y tamaño de PCA son decisiones; el p ajustado solo cubre una de ellas. Conviene reportar todas.
- **Estudio y paciente están confundidos:** los pacientes de cada estudio son distintos, así que el diagnóstico de batch no separa efecto técnico de diferencias reales entre cohortes. Las muestras `N` y `T` de un mismo paciente de Xie no son independientes.
- **Selección del endotelio en Xie por marcadores** (decisión propia, no validada contra una anotación celular publicada). Control de calidad más laxo que el del artículo.
- **Etiqueta confundida con el paciente:** `PATH`, `age` y `Patient` determinan `glioma_type` por sí solas; la separación es entre pacientes, no entre células.

---

## 7. Pendiente

Hecho desde la versión anterior: análisis `core` (4.7), revisión de C (4.4), significado de PC1/PC2 (4.8), bootstrap (4.9), alineación de medias y dentro de Xie (4.10).

1. **Separar qué explica la mejora en `core`:** detalle por paciente de `results/external_xie_core/results_per_patient_external.csv`; y comprobar con solo `T` incluyendo P036 (muestras `T` de todos los pacientes) frente a `core` sin P036.
2. **Bootstrap para el análisis `core`** y para la alineación (4.10 a).
3. **Metadatos clínicos de LGG1** (subtipo, localización): único fallo interno sin explicación; la edad (69) es una pista (4.8).
4. **Probar si la edad explica PC2:** p. ej. entrenar controlando por edad o comparar con un modelo solo de edad (con 14 pacientes tiene poca potencia).
5. **Si se vuelve a alinear entre estudios:** hacerlo de forma no transductiva (calcular el desplazamiento sin usar el estudio de prueba) o con un método estándar (ComBat/Harmony).
6. **Otro conjunto independiente** y, si es posible, anotación celular publicada de Xie para validar la selección de endotelio. Es lo único que puede desempatar entre artefacto técnico y señal real; hay que conseguirlo fuera de este repositorio.

---

## 8. Uso (PowerShell, sin activar el entorno)

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe -m src.validation --run-tests                          # tests con datos sintéticos
.\.venv\Scripts\python.exe -m src.validation                                      # valida el dataset real
.\.venv\Scripts\python.exe -m experiments.04_final --config configs/full.yaml     # análisis completo (dejar-un-paciente-fuera)
.\.venv\Scripts\python.exe -m experiments.05_seeds --config configs/full.yaml --n-seeds 10
.\.venv\Scripts\python.exe -m experiments.06_permutation --config configs/full.yaml
.\.venv\Scripts\python.exe -m experiments.replot --config configs/full.yaml       # regenera figuras y descripciones sin reentrenar
```

Configuraciones: `demo` (comprueba que todo funciona; no sacar conclusiones), `mini`, `full`, `full_noreg` (sin regresión de profundidad), `full_hvgperm` (permutación también con HVG-2000; se genera a partir de `full` cambiando `results_dir` y añadiendo `HVG-2000` a `permutation.representations`).

### Entrenamiento final, validación externa y diagnóstico de estudio

```powershell
# hiperparámetros con validación anidada + modelo final (reanudable, tarda horas)
.\.venv\Scripts\python.exe -m experiments.09_tune --config configs/full.yaml
# versión rápida, valores por defecto, sin ajuste: models/<modo>/bundle.joblib (fuera de GitHub)
.\.venv\Scripts\python.exe -m experiments.07_train_final --config configs/full.yaml

# Xie: descomprime, une las 13 muestras, normaliza y selecciona el endotelio (calibra el umbral con Wälchli)
.\.venv\Scripts\python.exe -m experiments.10_prepare_xie --folder "$([Environment]::GetFolderPath('Desktop'))\datos XIE" --reference-config configs/full.yaml
# aplica el paquete entrenado a Xie SIN reentrenar
.\.venv\Scripts\python.exe -m experiments.08_external --bundle models/full_tuned/bundle.joblib --external data/xie/xie_endothelial.h5ad --out results/external_xie
# análisis secundario fijado de antemano: solo muestras core
.\.venv\Scripts\python.exe -m experiments.10_prepare_xie --folder "$([Environment]::GetFolderPath('Desktop'))\datos XIE" --reference-config configs/full.yaml --sites core --out data/xie_core
.\.venv\Scripts\python.exe -m experiments.08_external --bundle models/full_tuned/bundle.joblib --external data/xie_core/xie_endothelial.h5ad --out results/external_xie_core

# qué representan PC1/PC2, bootstrap, y los dos exploratorios (4.8-4.10)
.\.venv\Scripts\python.exe -m experiments.12_pc_meaning --config configs/full.yaml --bundle models/full_tuned/bundle.joblib --out results/pc_meaning
.\.venv\Scripts\python.exe -m experiments.13_bootstrap --out results/bootstrap
.\.venv\Scripts\python.exe -m experiments.15_align_external --config configs/full.yaml --bundle models/full_tuned/bundle.joblib --out results/external_xie_aligned
.\.venv\Scripts\python.exe -m experiments.14_within_xie --config configs/full.yaml --out results/within_xie

# diagnóstico de estudio (batch): numérico + gráficas 09, 10, 11
.\.venv\Scripts\python.exe -m experiments.11_batch --config configs/full.yaml --xie data/xie/xie_endothelial.h5ad --bundle models/full_tuned/bundle.joblib --out results/batch
```

Si el estudio externo no trae `nCount_RNA` / `nFeature_RNA` ni conteos crudos, el paquete con regresión de profundidad no se puede aplicar y hay que usar un paquete entrenado sin ella (`configs/full_noreg.yaml`). Los genes ausentes se tratan como 0 y el script avisa si faltan muchos genes variables. Si las etiquetas externas se llaman distinto, usar `--label-map "Glioblastoma=GBM,..."`.

### Archivos de resultados (`results/<modo>/`, no versionados)
`01_cohort.png`, `02_pca_variance.png`, `03_confusion_*.png`, `04_components_vs_performance_*.png`, `05_per_patient_*.png`, `06_components_vs_performance_seeds_*.png`, `07_by_cluster_*.png`, `08_permutation_*.png`, `permutation_results_*.csv`, `results_*.csv`, `summary_*.md`, `summary.json`, `run_metadata.json`, `splits/`, `tuning/` (validación anidada). En `results/batch/`: `09_batch_model_space.png`, `10_batch_joint_pca.png`, `11_batch_domain_classifier.png`, `batch_domain_results.csv/.md`, `batch_domain_per_patient.csv`, `batch_pca_eta2.csv`, `batch_model_space_summary.csv`. Además: `results/pc_meaning/` (`12_pc_by_patient.png`, `patients_metadata_pcs.csv`, `pc_correlations.csv`), `results/bootstrap/` (`13_bootstrap_ci.png`, `bootstrap_ci.csv`), `results/external_xie_core/`, `results/external_xie_aligned/`, `results/within_xie/`.

**Descripciones de las gráficas:** cada `.png` tiene un `.txt` con el mismo nombre en la carpeta `descriptions/` de su resultado (p. ej. `results/full/descriptions/01_cohort.txt`). Explica qué expresa la gráfica y da sus cifras clave, calculadas de los propios datos. Se generan al crear la figura y se regeneran con `experiments.replot`.

Tiempos de referencia en el PC de Gabi: una ejecución completa dejando-un-paciente-fuera, unos 38 minutos; la permutación exacta de los PCA, unos 15 minutos; las 10 semillas, unas 6 horas.
