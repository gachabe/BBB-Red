# bbb: ¿cuántos genes hacen falta para distinguir GBM de LGG?

Pipeline reproducible: matriz de expresión (células endoteliales x genes) → genes variables → regresión de profundidad → escalado → PCA (ROM) → regresión logística / random forest → predecir `glioma_type` (GBM vs LGG).

**Pregunta:** ¿cuánta dimensionalidad (genes) puede quitar la PCA sin perder la capacidad de predecir el tipo de glioma? Se compara `PCA-r` (r = 2, 5, 10, 20, 50) frente a `HVG-2000` (genes más variables) y frente a `All genes` (26.666), con regresión logística y random forest. El modelo se entrena con un estudio (Wälchli) y se valida internamente (dejando un paciente fuera) y externamente (Xie).

> **Estado (4-nov-2026):** análisis a nivel de célula, ajuste de hiperparámetros, validación externa, diagnóstico de estudio, análisis `core`, bootstrap, significado de PC1/PC2, exploratorios y **pseudobulks de conteos crudos (fase 1 del protocolo de los coautores, sección 4.11)** terminados. Conclusión: señal modesta dentro de Wälchli; **no se reproduce en Xie completo**; **con pseudobulks y solo muestras `core` (6 pacientes) la separación es perfecta y robusta a las variantes técnicas probadas (fase 2)**, pero con 6 pacientes eso no se puede demostrar estadísticamente, la edad no se puede descartar y la periferia está desplazada; el efecto de estudio sigue siendo total (sección 1).

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
| **Validación externa, solo muestras `core` de Xie (6 pacientes; secundario fijado de antemano)** | **Mejor:** macro-F1 0,67-0,84 (PCA-5 con regresión logística 0,837 [0,700-0,946]); los 14 intervalos quedan por encima de 0,5, pero con 6 pacientes el bootstrap es poco fiable y el test exacto no puede bajar de p = 0,05 (4.9). Las células de muestras periféricas (`N`) empujan hacia LGG al menos en P045 y P046; P036 también se pierde, no está separado (4.7) |
| **Pseudobulks de conteos crudos, Xie solo `core` (6 pacientes, 100 células, modelo congelado de Wälchli)** | **Macro-F1 1,000** con PCA-2, 5, 10, 20 y HVG-2000 (todos los genes: 0,966); igual con paquetes de 50 a 500 células y 5 semillas. Los 3 GBM salen con probabilidad ≥ 0,95 y los 3 LGG ≤ 0,001 (PCA-5). **Cautelas:** con 6 pacientes el test exacto no baja de p = 0,05 y el IC exacto del 100 % (6/6) es [0,54-1,00]; con el espécimen completo PCA-2 falla (0,33); mejora a nivel de célula (0,70-0,84) y supera a la validación interna, lo cual conviene mirar con lupa (4.11) |
| **Fase 2: robustez del `core` (4.12)** | **Aguanta:** modelo, forma del paquete, profundidad, genes comunes, filtros de calidad de Xie y quitar 15.000 genes de estudio no lo rompen (PCA-5 1,000 siempre); se mantiene con subconjuntos de entrenamiento (84-91 %). La señal aparece también dentro de Xie (AUROC 1,000) y en sentido inverso (AUROC 0,90). **No descartado:** edad (las medias publicadas de Xie quedan a ambos lados de 0,5 en un modelo solo de edad) y diferencias entre cohortes; la periferia es un desplazamiento constante (AUROC 0,8-0,9), no ausencia de señal |
| Pseudobulks, Xie completo y periferia | **No funciona con 100 células** (macro-F1 0,30-0,71; periferia 0,30, todo LGG), pero el AUROC de la periferia es 0,8-0,9: ordena bien y falla el umbral. Con paquetes de 250-500 células el Xie completo mejora (hasta 0,91-1,00 con PCA-2), sin contrastar (4.11) |
| Pseudobulks, validación interna (Wälchli, 14 pacientes) | Macro-F1 0,65-0,81 (célula: 0,65-0,74). Permutación exacta ajustada: PCA-2 p = 0,016, PCA-10 y PCA-20 p = 0,036, PCA-5 p = 0,25. **PCA-2 no es robusto al tamaño del paquete** (AUROC 0,86 con 100 células, 0,71 con 250 y 0,46 con espécimen completo); HVG-2000 sí (macro-F1 0,78-0,85 y AUROC ≈ 0,9 en todos los tamaños). Bootstrap muy ancho (4.11) |
| Efecto de estudio con pseudobulks | **Total:** AUROC 1,000 con todas las representaciones, incluso PCA-2. Agregar quita ruido de célula, no la diferencia entre estudios |
| Edad y composición con pseudobulks (Wälchli) | La edad sola da AUROC 0,854, igual que la expresión (0,84-0,92); la composición sola, 0,68. La expresión no supera claramente a la edad (4.11) |
| Intervalos bootstrap sobre pacientes | **Muy anchos:** PCA-2 con regresión logística 0,739 [0,557-0,889] en Wälchli; en Xie completo casi todos incluyen 0,5 (4.9) |
| Qué representan PC1 y PC2 | PC1 sigue la profundidad y la composición del paciente (rho 0,95); PC2 sigue el tipo de glioma, pero la edad predice igual de bien (AUC 0,896 ambas); no se pueden separar (4.8) |
| **Xie solo con muestras periféricas (7 pacientes, exploratorio)** | **Azar:** macro-F1 0,41-0,55; la regresión logística llama LGG a todos (3/7). Con `core` funciona y con periferia no (4.7) |
| **Efecto de estudio (batch) entre Wälchli y Xie** | **Muy fuerte:** el estudio se identifica casi sin error en pacientes nuevos (AUC hasta 1,000); no lo explica la profundidad |

Línea base: predecir siempre la clase mayoritaria acierta 8/14 pacientes = 0,571 en Wälchli y 4/7 = 0,571 en Xie.

**Cómo contarlo:** señal modesta dentro de una cohorte de 14 pacientes; a nivel de célula no se reproduce en una segunda cohorte de 7. Con pseudobulks de conteos y solo muestras `core` la separación externa es perfecta (6 pacientes), pero no con la periferia, y el efecto de estudio sigue siendo total. Es una indicación prometedora y robusta, no una demostración: con 6 pacientes no hay potencia estadística, y no se ha podido descartar que la edad (o la cohorte) explique la separación en Xie.

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

### 3.6 Pseudobulks promediados (`experiments.17_pseudobulk`; exploratorio, secundario)
Primera versión de la idea de los coautores: promediar paquetes de células del mismo paciente. Usa la media de la expresión log-normalizada (la regla 3 de la guía, marcada allí como solo exploratoria) y paquetes disjuntos (familia D). **Queda como análisis secundario; el principal es el protocolo de conteos de 3.7.** Código listo, sin ejecutar.
- **Qué es un pseudobulk aquí:** media de la expresión log-normalizada (log1p de 10.000 por célula, que es lo que contienen ambos conjuntos) de un paquete de células elegidas al azar **dentro del mismo paciente** (en Xie, dentro de la misma muestra, para no mezclar núcleo y periferia). No se suman conteos crudos porque en Wälchli no los tenemos.
- **Sin fugas:** los pseudobulks de un paciente van siempre juntos en el mismo pliegue (dejar un paciente fuera) y las métricas se calculan con cada paciente pesando igual y, además, con **una sola predicción por paciente** (columnas `*_patient`).
- **Tres análisis:** (A) dentro de Wälchli; (B) entrenar con pseudobulks de Wälchli y aplicar a los de Xie (todo, solo `core`, solo periferia), sin reajustar nada; (C) diagnóstico de estudio: ¿se sigue identificando el estudio?
- **Fijado de antemano:** análisis principal con paquetes de **50 células** y regresión de profundidad según `configs/full.yaml` (profundidad de un pseudobulk = media de la de sus células). Se informan también 25 y 100 células y sin regresión de profundidad, como secundarios. Cada configuración se repite con 5 particiones aleatorias de los paquetes y se informa media y desviación.
- **Tope de entrenamiento:** el mismo presupuesto de células por paciente que a nivel de célula (1.000 células ≈ 20 pseudobulks de 50).

### 3.7 Pseudobulks de conteos crudos (`experiments.18`-`21`, protocolo de los coautores)
Sigue la guía de implementación de los coautores (fase 1). **Ejecutado; resultados en 4.11.**
- **Conteos crudos:** ambos conjuntos solo traen la matriz log-normalizada, pero los conteos se reconstruyen exactos: `conteos = expm1(X) × nCount_RNA / 10.000`. Comprobado en los datos reales: desviación máxima al entero 0,003 (Wälchli) y 0,001 (Xie), mínimo no nulo 1,000. El código se detiene si no salen enteros.
- **Auditoría de Xie** (`18_audit_xie`): tabla por muestra (diagnóstico, sitio, células antes y después del control de calidad, endoteliales seleccionadas, profundidad), marcadores endoteliales y puntuaciones de contaminación mural, inmune y glial. La anotación endotelial publicada no está en los datos descargados (columna vacía) y el recuento del artículo no se ha verificado.
- **Pseudobulks** (`19`): un pseudobulk de Wälchli = un paciente; de Xie = una muestra (núcleo y periferia nunca se mezclan). Paquetes de **50, 100, 250 y 500 células** con 20 réplicas (sin reemplazo dentro del paquete; pueden solaparse entre réplicas), 5 semillas maestras, y el **espécimen completo** (mínimo 100 células) como referencia. Se suman los conteos.
- **Pipeline ajustado solo con pacientes de entrenamiento:** filtro de genes (CPM > 1 en ≥ 25 % de los pseudobulks de entrenamiento), TMM frente a una referencia de entrenamiento, log2-CPM, escalado, HVG-2000 / PCA-2, 5, 10, 20 / todos los genes filtrados, y regresión logística (L2, balanceada, `liblinear`, C = 1) con peso 1/R por paciente. Random forest no se usa.
- **Una probabilidad de GBM por paciente** = media de la de sus paquetes (la mediana y la media recortada se calculan también). Métricas por paciente: macro-F1, balanced accuracy, sensibilidad, especificidad, AUROC, AUPRC. Permutación exacta entre pacientes (3.003 asignaciones, ajustada por máximo entre representaciones PCA) y bootstrap de pacientes.
- **Xie:** modelo congelado entrenado con los 14 pacientes de Wälchli; se informan núcleo (principal), periferia y todo (probabilidad del paciente = media de sus muestras). Se registran los pacientes disponibles en cada configuración (matriz de cobertura) y se repite con el mismo conjunto de pacientes para todos los tamaños.
- **Controles** (`20`): clasificador de estudio (21 pacientes), de sitio dentro de Xie, solo edad, solo composición celular y su combinación con la expresión. Edad y composición solo en Wälchli.
- **Informe** (`21`): tabla comparativa célula frente a pseudobulk y figuras con descripción.
- **Fijado de antemano:** paquetes de 100 células, media de probabilidades, Xie núcleo como análisis externo principal, 5 semillas. Los demás tamaños son sensibilidad y se informan todos.
- **No implementado todavía:** véase la sección 7.

### 3.8 Lo que se probó y se descartó
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
- **Detalle por paciente (regresión logística, % de células bien clasificadas, Xie completo → `core`):**

| Paciente | Etiqueta | PCA-2 completo → core | PCA-5 core |
|---|---|---|---|
| P027 | LGG | 73 % → 73 % | 91 % |
| P037 | LGG | 90 % → 90 % | 98 % |
| P043 | LGG | 62 % → 55 % | 87 % |
| P038 | GBM | 21 % → 46 % | 49 % |
| P045 | GBM | 50 % → 64 % | 80 % |
| P046 | GBM | 52 % → 94 % | 97 % |

  P036 (GBM, 36 % en el análisis completo) desaparece de `core`. Con los mismos pacientes, P046 pasa de 52 % a 94 % y P045 de 50 % a 64 % al quitar sus células de muestras `N`, así que las células periféricas **sí** empujan hacia LGG, al menos en estos dos. El único error de `core` es P038 (GBM, 2.236 células), a ~46-49 %: prácticamente cara o cruz. Quitar P036 también contribuye a la mejora global; no se ha cuantificado cuánto.
- **Solo muestras periféricas** (exploratorio, 7 pacientes, 56.195 células; `data/xie_periph`): macro-F1 0,41-0,55 en todas las filas (balanced accuracy 0,54-0,60). Con regresión logística el acierto por paciente es 3/7 = 0,43 en todas las representaciones, es decir, solo los 3 LGG: llama LGG a todos. Con random forest, hasta 5/7 pero con F1 ~0,5. P036 solo tiene muestra periférica (`N036`), así que no hay forma de ver a P036 con tejido de núcleo.
- **Lectura:** `core` funciona (F1 0,67-0,84) y periferia no (≈ azar), con los mismos pacientes salvo P036. Es coherente con que el tejido periférico empuje hacia LGG, además del detalle por paciente de arriba. No prueba que el tejido sea la única causa.
- **Cautelas:** 6 pacientes (un paciente = 0,167 de acierto por paciente), 14 filas probadas sin corrección. Intervalos bootstrap en 4.9. Se ejecutó como análisis secundario fijado de antemano, pero después de ver que el principal fallaba. Es una indicación, no una replicación.

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
- **PC2 sigue el tipo de glioma, pero también la edad.** En Wälchli edad y tipo van juntas (LGG: 26-46 años salvo LGG1, 69; GBM: 48-82). Comprobación (`16_age_check`, 14 pacientes, exploratorio y con muy poca potencia):

| Análisis | Resultado |
|---|---|
| Dejar un paciente fuera, solo edad | acierto 0,714 (fallan GBM7, GBM8, LGG1, LGG2) |
| Dejar un paciente fuera, solo PC2 | 0,857 (fallan LGG1, LGG6) |
| Dejar un paciente fuera, PC2 + edad | 0,929 (falla LGG1) |
| AUC para LGG: edad (invertida) / PC2 | 0,896 / 0,896 |
| Spearman PC2-edad: todos / solo GBM / solo LGG | -0,61 / +0,42 (n=8) / -0,60 (n=6) |
| Spearman parcial de PC2 con LGG quitando la edad (rangos) | 0,455 |

  La edad sola separa tan bien como PC2 por AUC, así que **no se puede descartar que PC2 sea en parte un indicador de edad**. Hay una pista de que PC2 aporta algo más (0,857 frente a 0,714; parcial 0,455; la edad dentro de cada clase va en direcciones opuestas, así que la correlación global refleja sobre todo la diferencia entre grupos), pero con 14 pacientes no es concluyente.
- **LGG1** (69 años, el único LGG mayor de 46) tiene PC2 = -5,4 y cae del lado GBM; los otros 5 LGG tienen PC2 entre +2,4 y +13,3. Es compatible con la confusión por edad, pero con un solo paciente no es demostración. Los metadatos que trae el conjunto de datos para LGG1 son solo `PATH` = LGG, mujer y 69 años; **no incluye subtipo ni localización**. Su composición celular no destaca (capilar 41 %, capilar angiogénico 24 %, 923 células). Esos datos clínicos hay que pedirlos a la coautora o al artículo.
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
**Intervalos de los análisis posteriores** (macro-F1 [IC 95 %]; regresión logística / random forest):

| Representación | Xie `core` (6 pacientes) | Xie alineado (exploratorio) | Dentro de Xie (exploratorio) |
|---|---|---|---|
| PCA-2 | 0,704 [0,566-0,842] / 0,669 [0,533-0,799] | 0,559 [0,476-0,642] / 0,547 [0,478-0,618] | 0,620 [0,576-0,664] / 0,517 [0,491-0,539] |
| PCA-5 | 0,837 [0,700-0,946] / 0,831 [0,713-0,933] | 0,636 [0,523-0,734] / 0,600 [0,507-0,690] | 0,639 [0,529-0,739] / 0,549 [0,429-0,655] |
| PCA-10 | 0,797 [0,696-0,903] / 0,790 [0,676-0,905] | 0,635 [0,566-0,707] / 0,602 [0,511-0,673] | 0,609 [0,477-0,740] / 0,573 [0,492-0,654] |
| PCA-20 | 0,776 [0,655-0,910] / 0,764 [0,650-0,878] | 0,608 [0,525-0,696] / 0,582 [0,504-0,671] | 0,552 [0,405-0,702] / 0,518 [0,415-0,624] |
| PCA-50 | 0,809 [0,684-0,959] / 0,778 [0,664-0,880] | 0,716 [0,656-0,772] / 0,618 [0,547-0,684] | 0,537 [0,381-0,686] / 0,515 [0,416-0,620] |
| HVG-2000 | 0,808 [0,674-0,962] / 0,826 [0,701-0,926] | 0,725 [0,654-0,795] / 0,658 [0,604-0,717] | 0,524 [0,397-0,659] / 0,358 [0,268-0,449] |
| Todos los genes | 0,824 [0,723-0,969] / 0,804 [0,685-0,914] | (no se aplica) | (no se aplica) |

- **`core`:** los 14 intervalos quedan por encima de 0,5, a diferencia de Xie completo (1 de 14).
- **Alineado:** 10 de 12 por encima de 0,5; dentro de Xie: 2 de 12 (PCA-2 y PCA-5 con regresión logística).
- **No fiarse de la estrechez de estos intervalos.** Con 6-7 pacientes el remuestreo tiene pocas combinaciones distintas y los intervalos salen artificialmente estrechos (p. ej. acierto por paciente [1,00-1,00], o dentro de Xie PCA-2 con regresión logística 0,620 [0,576-0,664]). Además, el bootstrap no reentrena el modelo.
- **Prueba exacta imposible para `core`:** con 3 + 3 pacientes hay solo 20 asignaciones posibles, así que ni acertando los 6 pacientes se baja de p = 0,05.
- **Selección posterior:** `core` se ejecutó después de ver fallar Xie completo, y la alineación y dentro de Xie son exploratorios (la alineación es transductiva). Los intervalos no corrigen nada de eso.

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

  Mejora en casi todas las filas, más con muchas dimensiones; en PCA-2 apenas. Apunta a que parte del fallo es un desplazamiento de medias entre estudios, pero con 7 pacientes y probando 12 filas, es una pista.

  **Variante sin usar las células del paciente evaluado** (`--loo`: el desplazamiento de cada paciente se calcula con los otros 6 de Xie; sigue usando células de Xie sin etiquetas, pero no las del paciente que se predice). Macro-F1 (pacientes de 7):

| Representación | Reg. logística | Random forest |
|---|---|---|
| PCA-2 | 0,566 (5/7) | 0,559 (5/7) |
| PCA-5 | 0,651 (6/7) | 0,626 (5/7) |
| PCA-10 | 0,653 (6/7) | 0,627 (5/7) |
| PCA-20 | 0,622 (4/7) | 0,601 (5/7) |
| PCA-50 | 0,743 (7/7) | 0,640 (6/7) |
| HVG-2000 | 0,755 (7/7) | 0,678 (7/7) |

  Los resultados son parecidos a la versión que usa a todos los pacientes (HVG-2000 con regresión logística 0,725 → 0,755), así que la mejora no depende de usar las células del propio paciente. Los intervalos de esta variante y de la periferia están en `results/bootstrap/bootstrap_ci.csv`; no están transcritos aquí (en la periferia, todos los F1 puntuales son ≤ 0,55).

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

### 4.11 Pseudobulks de conteos crudos (protocolo de los coautores, fase 1; `results/pseudobulk_counts`)
Configuración fijada de antemano: paquetes de 100 células, media de probabilidades por paciente, Xie `core` como validación externa principal, 5 semillas maestras. Los demás tamaños son sensibilidad. (Que el `core` sea el análisis principal viene de la guía de los coautores, pero ya habíamos visto que funcionaba a nivel de célula, así que no es una prueba ciega.)

**Auditoría de Xie** (`xie_audit.csv`; comparada con el artículo): sin incidencias en el mapeo muestra-paciente; profundidad muy desigual (mediana de UMIs por muestra de 1.262 a 6.428); la anotación endotelial publicada no está en los datos, pero el artículo da totales: **103.230 células tras el control de calidad, de ellas 57.324 endoteliales** (control de calidad: < 200 genes, > 10 % de genes mitocondriales y dobletes con DoubletFinder). Nuestra preparación carga **198.445 células** (solo filtro de 200 genes, sin filtro mitocondrial ni de dobletes) y selecciona **106.399 como endoteliales, casi el doble del artículo**. La selección por marcadores es por tanto mucho más laxa que la publicada y puede incluir células de baja calidad o no endoteliales; la fase 2 prueba filtros más estrictos (4.12). Pacientes de Xie: 3 LGG con mutación de IDH (edad media 49,3) y 4 GBM sin mutación de IDH (edad media 61,3); las edades individuales están en la Tabla S1A del artículo, que no tenemos. N036 tiene la puntuación mural más alta (0,66) y N045 la glial (0,47). No factibles con 500 células: GBM1 (264 células) en Wälchli y N027 (465) en Xie.

**Xie, modelo congelado (macro-F1 medio de 5 semillas)**

| Análisis | 50 | 100 | 250 | 500 | espécimen completo |
|---|---|---|---|---|---|
| `core`, HVG-2000 / PCA-2 / PCA-5 | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 / **0,33** / 1,00 |
| Xie completo, PCA-2 | 0,30 | 0,30 | 0,91 | 1,00 | 0,65 |
| Xie completo, PCA-5 | 0,30 | 0,64 | 0,77 | 0,86 | 0,86 |
| Xie completo, HVG-2000 | 0,53 | 0,71 | 0,83 | 0,86 | 0,86 |
| Periferia, PCA-2 | 0,30 | 0,30 | 0,39 | 0,67* | 0,53 |

*6 pacientes (N027 no factible). Con 100 células, PCA-10, PCA-20 y HVG-2000 dan 1,000 en `core`; todos los genes, 0,966.

- **Probabilidades por paciente en `core`** (100 células, semilla 0, PCA-5): GBM: P038 0,9996, P045 0,949, P046 0,99999. LGG: P027 0,00002, P037 0,0006, P043 0,00004. Separación muy limpia; la dispersión entre réplicas del mismo paciente es pequeña (P045, el más incierto, desviación 0,08).
- **Qué no se puede concluir:** 6 pacientes. El test exacto tiene 20 asignaciones, así que un acierto perfecto da como mucho p = 0,05; el IC exacto de 6/6 es [0,54-1,00]; el bootstrap de pacientes sale degenerado ([1,00-1,00]) y no es informativo. Es una dirección, no una prueba.
- **Por qué mirarlo con lupa:** (1) supera a la validación interna (0,65-0,81); (2) con el espécimen completo PCA-2 falla (0,33), así que no es indiferente a la representación; (3) el clasificador de estudio separa Wälchli de Xie con AUROC 1,000, o sea que hay un desplazamiento enorme, y aun así el eje de glioma sobrevive en `core`, lo que es a la vez una buena señal y algo que conviene entender; (4) no hemos comprobado si la **edad** separa a esos 6 pacientes de Xie (en Wälchli GBM y LGG se diferencian mucho por edad, 4.8).
- **Periferia:** con 100 células llama LGG a todos (macro-F1 0,30), pero el AUROC es 0,8-0,9: el modelo ordena a los pacientes, lo que falla es el umbral (desplazamiento de las probabilidades). Con 7 pacientes y el umbral recalibrado a posteriori no sería una validación; queda como observación.
- **Xie completo** mejora al aumentar el paquete (PCA-2: 0,30 → 0,91 → 1,00). No se interpreta como solución: es una comparación entre tamaños sobre 7 pacientes, sin corrección por haber probado varios.

**Validación interna (Wälchli, dejar un paciente fuera, macro-F1 medio de 5 semillas)**

| Representación | 50 | 100 | 250 | 500 | completo |
|---|---|---|---|---|---|
| PCA-2 | 0,75 | 0,80 | 0,78 | 0,71 | 0,64 |
| PCA-5 | 0,69 | 0,65 | 0,64 | 0,71 | 0,71 |
| HVG-2000 | 0,81 | 0,80 | 0,84 | 0,85 | 0,78 |

- A 100 células: PCA-2 0,800, PCA-5 0,654, PCA-10 0,767, PCA-20 0,811, HVG-2000 0,797, todos los genes 0,807. A nivel de célula: 0,65-0,74.
- **Permutación exacta** (3.003 asignaciones; máximo entre PCA-2, 5, 10 y 20; semilla 0, macro-F1 observado 0,857 / 0,708 / 0,854 / 0,854): p ajustado = **0,016** (PCA-2), 0,248 (PCA-5), **0,036** (PCA-10 y PCA-20). Con el espécimen completo ninguno baja de 0,05 (PCA-10: 0,081). HVG-2000 y todos los genes no tienen test.
- **Bootstrap de pacientes (100 células, semilla 0):** PCA-2 0,857 [0,63-1,00], PCA-5 0,708 [0,43-0,93], HVG-2000 0,854 [0,64-1,00]. Muy anchos. (La tabla de arriba usa la media de semillas, 0,800; la permutación y el bootstrap, la semilla 0.)
- **Lectura honesta:** el AUROC de PCA-2 cae con el tamaño del paquete (0,86; 0,71; 0,69; 0,46 con 100, 250, 500 y completo), mientras que el de HVG-2000 se mantiene en ≈ 0,9. La ventaja de PCA-2 es por tanto frágil; con pseudobulks la representación más estable es HVG-2000, que la permutación no cubre. No hay evidencia de que PCA-2 sea mejor que las demás.

**Estabilidad entre réplicas** (100 células): desviación mediana de la probabilidad de un paciente entre paquetes 0,002-0,025 (PCA-2 la más inestable, 0,025); entre el 57 % (PCA-2) y el 79 % de los pacientes tienen desviación < 0,05; en 79-93 % de los pacientes el 90 % de los paquetes coinciden en la clase. La estabilidad mejora con el tamaño (a 500 células, 100 % salvo PCA-2). El ICC está en `packet_stability_cohort.csv`.

**Controles (Wälchli, 14 pacientes, media de semillas)**

| Modelo | AUROC | Macro-F1 |
|---|---|---|
| Edad sola | 0,854 | 0,714 |
| Composición celular sola | 0,681 | 0,738 |
| Expresión (según representación) | 0,84-0,92 | 0,66-0,81 |
| Edad + expresión (PCA-2) | 0,861 | 0,713 |
| Composición + expresión (PCA-2) | 0,931 | 0,785 |

- **La edad sola rinde como la expresión**: con estos 14 pacientes no se puede decir que la expresión aporte más que la edad (confusión ya vista en 4.8).
- «Edad + expresión» sale casi igual que «expresión sola» **por construcción** (la edad es una columna entre miles o una entre pocas, y con el escalado apenas pesa); no es prueba de que la edad no aporte.
- **Clasificador de sitio** (núcleo frente a periferia dentro de Xie, 13 muestras): AUROC 0,63-0,78, es decir, el sitio se distingue débilmente; no explica por sí solo el fallo en la periferia.
- **Clasificador de estudio** (21 pacientes): AUROC 1,000 con todas las representaciones, balanced accuracy 0,976 con PCA-2 (a nivel de célula, 0,66).

**Comparación célula frente a pseudobulk (macro-F1)**

| Representación | interna, célula | interna, pseudobulk | Xie `core`, célula | Xie `core`, pseudobulk | Xie completo, célula | Xie completo, pseudobulk |
|---|---|---|---|---|---|---|
| PCA-2 | 0,739 | 0,800 | 0,704 | 1,000 | 0,545 | 0,300 |
| PCA-5 | 0,702 | 0,654 | 0,837 | 1,000 | 0,576 | 0,638 |
| PCA-10 | 0,674 | 0,767 | 0,797 | 1,000 | 0,590 | 0,708 |
| PCA-20 | 0,649 | 0,811 | 0,776 | 1,000 | 0,546 | 0,708 |
| HVG-2000 | 0,713 | 0,797 | 0,808 | 1,000 | 0,572 | 0,708 |
| Todos los genes | 0,707 | 0,807 | 0,824 | 0,966 | 0,536 | 0,300 |

**Resumen:** los pseudobulks mejoran la validación interna de forma modesta y, sobre todo, hacen que `core` de Xie se separe sin error con cinco de las seis representaciones y todos los tamaños. No resuelven el efecto de estudio ni la periferia. Siguen sin descartarse la edad y el desplazamiento de cohorte como explicaciones.

### 4.12 Fase 2: robustez del resultado `core` (`experiments.22_phase2`; `results/pseudobulk_phase2`)
Todo con paquetes de 100 células (salvo que se indique), 3 semillas, regresión logística (salvo random forest), mismo pipeline sin fugas que 4.11. Lo que se hizo y lo que salió:

**1. El `core` aguanta casi todas las variaciones (macro-F1 en Xie `core`, 6 pacientes; con 6 pacientes un solo error baja el valor a 0,83-0,89)**

| Variante | PCA-2 | PCA-5 | HVG-2000 |
|---|---|---|---|
| Referencia (fase 1) | 1,000 | 1,000 | 1,000 |
| Random forest | 0,943 | 1,000 | 1,000 |
| Paquetes disjuntos de 50 / de 100 células | 1,000 / 1,000 | 1,000 / 1,000 | 1,000 / **0,886** |
| Fracción del 10 / 25 / 50 % de las células del paciente | 1,000 / 0,829 / 0,829 | 1,000 en las tres | 1,000 en las tres |
| Igualados por UMI (~150.000) | 0,943 | 1,000 | **0,886** |
| Conteos adelgazados a igual profundidad | 1,000 | 1,000 | 1,000 |
| Solo genes comunes a ambos estudios | 1,000 | 1,000 | 1,000 |
| Xie con filtro mitocondrial < 10 % del artículo | 1,000 | 1,000 | 1,000 |
| Xie solo con células claramente endoteliales | 1,000 | 1,000 | 1,000 |
| Lo anterior + genes comunes + misma profundidad | 1,000 | 1,000 | 1,000 |
| Sin los 500 / 2.000 / 5.000 / 10.000 / 15.000 genes que más distinguen los estudios | 1,000 hasta 10.000; 0,886 con 15.000 | 1,000 en todos | 1,000 en todos |

- **PCA-5 es 1,000 en todas las variantes**; los fallos son un paciente en alguna semilla y alguna representación. Las variantes que bajan de 0,9 se repitieron con paquetes de 250 y 500: con UMI igualados vuelve a 1,000; con disjuntos de 100 y 250 células HVG-2000 queda en 0,829.
- **Filtrar Xie no cambia nada** (idénticos al milímetro): la laxitud de la selección endotelial frente al artículo (106.399 frente a 57.324 endoteliales, 4.11) no explica el resultado. Cuántas células sobreviven a cada filtro está en `xie_filter_counts.csv`.
- **Quitar 15.000 de los ≈ 22.700 genes comunes que más distinguen los estudios no rompe el `core`.** La señal no depende de los genes de estudio más extremos; tampoco está concentrada en pocos genes.
- La lectura: ninguna decisión técnica probada (modelo, forma del paquete, profundidad, genes, calidad de Xie) explica la separación perfecta. Lo que no descartan es una diferencia real entre las cohortes que coincide con GBM/LGG (edad, procesamiento), que ninguna de estas variantes toca.

**2. Dependencia del conjunto de entrenamiento**
- 100 subconjuntos de 10 pacientes de Wälchli (6 GBM + 4 LGG): separación sin error en el 91 % (PCA-5), 85 % (PCA-2) y 84 % (HVG-2000, PCA-10, PCA-20); con todos los genes, 39 %. La mediana del AUROC es 1,000 en todas las representaciones: ordena bien a los pacientes aunque el umbral a veces falle.
- Quitar un paciente de entrenamiento cada vez: PCA-2 y PCA-5 siguen en 1,000 siempre; HVG-2000 baja a 0,829 solo al quitar **LGG1** (el paciente sin metadatos clínicos, 4.8).

**3. La señal existe también dentro de Xie y en sentido inverso**
- **Dentro de Xie `core`** (dejando un paciente fuera, solo 5 pacientes de entrenamiento): macro-F1 y AUROC 1,000 con todas las representaciones. En periferia: AUROC 0,86-1,00 (HVG-2000: 1,000) con F1 0,57-0,91. Con las dos localizaciones juntas: AUROC 0,72-0,94. A nivel de célula esto no se veía (4.10).
- **Inversa** (entrenar con los 6 pacientes `core` de Xie, predecir los 14 de Wälchli): macro-F1 0,85 y AUROC 0,90 con todas las representaciones salvo todos los genes (0,78 y 0,85). El eje GBM/LGG se comparte en las dos direcciones. Con las dos localizaciones de Xie como entrenamiento: AUROC 0,74-0,92 y F1 0,43-0,80 (peor).
- Cautela: 6 pacientes en Xie `core` hacen que un acierto perfecto dentro de Xie (como el externo) tenga p mínimo 0,05; que sea 1,000 también dentro de Xie coincide con que ahí GBM y LGG difieran en edad y tengan IDH distinto (4.11).

**4. Edad (solo con medias publicadas)**
Un modelo solo de edad entrenado con Wälchli (LGG 40,5 años, GBM 62,5; rango 26-82) da P(GBM) = 0,47 a la edad media de los LGG de Xie (49,3) y 0,66 a la de los GBM (61,3). Las dos medias quedan a lados distintos de 0,5, así que **la edad por sí sola podría separar los grupos de Xie y la confusión con la edad no se descarta** (esperaba que no fuera así; no lo es). Con medias de 3 y 4 pacientes no se puede ir más allá: hacen falta las edades individuales (Tabla S1A del artículo). Dato a favor de que no es solo edad: el modelo de expresión da probabilidades extremas (≤ 0,001 frente a ≥ 0,95), muy lejos de las 0,47/0,66 de la edad, pero eso no demuestra que la edad no intervenga.

**5. Periferia: es un desplazamiento, no falta de señal**
- Las probabilidades de la periferia quedan **3,9-9,2 unidades de logit por debajo** de las del núcleo del mismo paciente, en el 100 % de los pacientes (todas más cerca de LGG).
- El AUROC en periferia es 0,82-0,92 (100 células) y no mejora con el tamaño. Si se centran las probabilidades sin usar etiquetas (restando su media), el macro-F1 pasa de 0,30 a 0,86 con HVG-2000, todos los genes y PCA-20; con el mejor umbral a posteriori llega a 0,82-0,86 con 100 y 250 células.
- Esto es un diagnóstico a posteriori y transductivo (usa la distribución de los propios datos de prueba), **no una validación**: indica que el orden se conserva y que el tejido periférico desplaza al modelo hacia LGG, compatible con la hipótesis de la sección 5, no la prueba.

**6. Resto de análisis**
- **Estados endoteliales** (solo Wälchli, paquetes de 50 células de un único estado): la señal aparece en varios estados, mejor en *Angiogenic capillary* (macro-F1 0,85-0,93, AUROC 0,90-0,95) y débil en *Large vein*, *Vein* y *Venule*. *Mitochondrial* sale 1,000 con 6 pacientes: es un estado de calidad dudosa y muy poco fiable. Equilibrado por estados (interno, 100 células): PCA-5 0,921, PCA-2 0,795, HVG-2000 0,764.
- **Convergencia de réplicas:** con 10 paquetes de 100 células la probabilidad media de un paciente ya cambia de clase en < 1,3 % de los remuestreos (internas) y 0 % en `core`, incluso con 1 paquete. Entre el 97 y el 99 % de la varianza está entre pacientes (ANOVA, no es un modelo mixto), así que más paquetes por paciente no añaden información; solo más pacientes.
- **Holm** entre las 7 permutaciones de la fase 1: PCA-2 (100 células) p = 0,047; las demás ≥ 0,136.
- **Xie completo y fracciones:** con paquetes definidos como fracción de las células, Xie completo llega a 0,857 (PCA-5 y HVG-2000). No se interpreta: la fracción da paquetes de tamaños muy distintos según la muestra (465 células frente a 25.000), y el tamaño del paquete ya se vio que cambia el resultado en Xie completo.

**Resumen de la fase 2:** el resultado `core` es robusto a las decisiones técnicas probadas y a qué pacientes de Wälchli se usan para entrenar, y la señal también aparece dentro de Xie y al revés. Sigue sin poderse separar de la edad y de diferencias reales entre cohortes (6 pacientes en Xie `core`, medias de edad solamente, estudio y diagnóstico confundidos), por lo que es un resultado prometedor y consistente, no una prueba.

---

## 5. Por qué podría pasar (hipótesis, NO verificadas)

- **PCA-2 sin pérdida frente a todos los genes:** con 14 pacientes, pocas dimensiones actúan como regularización; más variables dan más ocasión de ajustar ruido de paciente o de estudio. Coherente con que random forest empeore con más genes y con que el estudio se note más al subir de dimensión (4.6).
- **Por qué no se transfiere a Xie:** el modelo puede haber aprendido en parte un eje que mezcla diagnóstico con rasgos propios de la cohorte o del procesamiento de Wälchli. Xie se desplaza justo sobre ese eje hacia el lado LGG. No sabemos si el desplazamiento es técnico (protocolo, control de calidad, selección del endotelio por marcadores, aquí menos estricta) o biológico (otra cohorte, otro tejido).
- **Posible influencia del tejido periférico en Xie:** las muestras `N` son periferia del tumor y podrían parecerse más a LGG. El análisis `core` (4.7) mejora mucho el resultado, pero también quita a P036; no se ha separado qué de las dos cosas lo causa.
- **Confusión con la edad en Wälchli:** PC2, que sigue el glioma, correlaciona con la edad (rho -0,61), y edad y tipo van juntas en esa cohorte (4.8). No se pueden separar con estos datos.
- **Desplazamiento de medias entre estudios:** alinear las medias de Xie mejora el resultado (4.10 a), compatible con un efecto de estudio que actúa sobre las medias; sin validar.
- **Composición celular:** explica en parte el caso de LGG6 (4.3), pero no el resto ni el fallo en Xie.
- **Por qué `core` funciona con pseudobulks:** agregar 100 células elimina el ruido de célula, y el eje de glioma sobrevive aunque el desplazamiento de estudio sea total. Alternativas sin descartar: la edad de los pacientes de Xie, o un desplazamiento de cohorte que casualmente coincide con la dirección GBM/LGG en esos 6. Hace falta comprobar la edad en Xie.
- **Regresión de profundidad:** mejora a PCA-2 (macro-F1 0,663 → 0,739 con una semilla) pero no sabemos por qué.

---

## 6. Limitaciones

- **14 pacientes en entrenamiento y 7 en validación.** Diferencias de uno o dos puntos entre representaciones no son distinguibles del ruido. Con 7 pacientes el test exacto tiene 35 asignaciones posibles: p < 0,05 solo se alcanza acertando los 7 (mínimo p ≈ 0,029). La validación externa es una indicación de dirección, no una prueba.
- **Las semillas no miden la incertidumbre real**: con dejar-un-paciente-fuera el reparto no cambia entre semillas.
- **Bifurcaciones del análisis:** modelo, con/sin regresión de profundidad y tamaño de PCA son decisiones; el p ajustado solo cubre una de ellas. Conviene reportar todas.
- **Estudio y paciente están confundidos:** los pacientes de cada estudio son distintos, así que el diagnóstico de batch no separa efecto técnico de diferencias reales entre cohortes. Las muestras `N` y `T` de un mismo paciente de Xie no son independientes.
- **Selección del endotelio en Xie por marcadores** (decisión propia, no validada contra una anotación celular publicada). Control de calidad más laxo que el del artículo.
- **Edad en Xie:** solo disponemos de las medias publicadas (LGG 49,3; GBM 61,3), no de las edades individuales; el control de edad es por tanto muy grosero.
- **Pseudobulks:** el resultado `core` descansa en 6 pacientes (p mínimo exacto 0,05). La permutación y el bootstrap usan la semilla 0 mientras que las tablas promedian 5. Los pseudobulks de réplicas del mismo paciente no son independientes. Se probaron varios tamaños y representaciones sin corrección entre todas las configuraciones (Holm solo sobre las 7 permutaciones, 4.12).
- **Etiqueta confundida con el paciente:** `PATH`, `age` y `Patient` determinan `glioma_type` por sí solas; la separación es entre pacientes, no entre células.

---

## 7. Pendiente

Hecho: pseudobulks de conteos, fase 1 (4.11), análisis `core` (4.7), periferia (4.7), revisión de C (4.4), PC1/PC2 y edad (4.8), bootstrap (4.9), alineación de medias, su variante sin las células del paciente evaluado, y dentro de Xie (4.10).

1. **Metadatos clínicos de LGG1** (subtipo, localización, mutaciones): no están en el conjunto de datos (4.8). Pedirlos a la coautora o buscarlos en el artículo.
2. **Separar edad y tipo de glioma:** con estos 14 pacientes no se puede (4.8); haría falta una cohorte con edades solapadas entre GBM y LGG.
3. **Pseudobulks de conteos: fases 1 y 2 hechas (4.11, 4.12).** Pendiente inmediato: conseguir las **edades individuales de Xie** (Tabla S1A del artículo; no están en el PDF principal) y comprobar si la edad sola separa a los 6 pacientes `core`; y comprobar por qué LGG1 es el paciente que más influye (4.12).
   **De la guía, aún no implementado:** DESeq2/VST (necesita R), mediana de expresión y muestreos estratificados; componentes de varianza con modelo mixto (4.12 usa ANOVA); anotación endotelial publicada de Xie (no está en los datos descargados). `17_pseudobulk` (promedio de log-expresión) queda como secundario, sin ejecutar.
4. **Otro conjunto independiente, solo después de ver los pseudobulks:** Bejarano et al. (2025), *Single-cell atlas of endothelial and mural cells across primary and metastatic brain tumors*, Immunity 58, 1015-1032.e6 (propuesto por un coautor; no se ha revisado si trae GBM y LGG con etiqueta por paciente). Procesarlo cuesta tiempo (descarga, preparación, adaptación). Los pseudobulks no han reducido el efecto de estudio (AUROC 1,000), pero un tercer conjunto serviría justamente para saber si el resultado `core` se repite; primero una auditoría solo de metadatos (¿hay GBM y LGG con etiqueta por paciente y conteos?). Además, si es posible, anotación celular publicada de Xie para validar la selección de endotelio.

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

# pseudobulks de conteos crudos (protocolo de los coautores; 19 es reanudable)
.\.venv\Scripts\python.exe -m experiments.18_audit_xie
.\.venv\Scripts\python.exe -m experiments.19_pseudobulk_counts --config configs/full.yaml
.\.venv\Scripts\python.exe -m experiments.20_pseudobulk_controls --config configs/full.yaml
.\.venv\Scripts\python.exe -m experiments.21_pseudobulk_report --config configs/full.yaml

# fase 2 de los pseudobulks de conteos: robustez (reanudable; tarda horas)
.\.venv\Scripts\python.exe -m experiments.22_phase2 --config configs/full.yaml

# pseudobulks promediados (secundario; reanudable; --quick = prueba rápida)
.\.venv\Scripts\python.exe -m experiments.17_pseudobulk --config configs/full.yaml --xie data/xie/xie_endothelial.h5ad --out results/pseudobulk

# solo muestras periféricas de Xie (exploratorio)
.\.venv\Scripts\python.exe -m experiments.10_prepare_xie --folder "$([Environment]::GetFolderPath('Desktop'))\datos XIE" --reference-config configs/full.yaml --sites peripheral --out data/xie_periph
.\.venv\Scripts\python.exe -m experiments.08_external --bundle models/full_tuned/bundle.joblib --external data/xie_periph/xie_endothelial.h5ad --out results/external_xie_periph

# qué representan PC1/PC2, edad, bootstrap, y los exploratorios (4.8-4.10)
.\.venv\Scripts\python.exe -m experiments.12_pc_meaning --config configs/full.yaml --bundle models/full_tuned/bundle.joblib --out results/pc_meaning
.\.venv\Scripts\python.exe -m experiments.13_bootstrap --out results/bootstrap   # Wälchli, Xie, core, alineado y dentro de Xie (omite los que no existan)
.\.venv\Scripts\python.exe -m experiments.15_align_external --config configs/full.yaml --bundle models/full_tuned/bundle.joblib --out results/external_xie_aligned
.\.venv\Scripts\python.exe -m experiments.14_within_xie --config configs/full.yaml --out results/within_xie
.\.venv\Scripts\python.exe -m experiments.15_align_external --config configs/full.yaml --bundle models/full_tuned/bundle.joblib --out results/external_xie_aligned --loo
.\.venv\Scripts\python.exe -m experiments.16_age_check

# diagnóstico de estudio (batch): numérico + gráficas 09, 10, 11
.\.venv\Scripts\python.exe -m experiments.11_batch --config configs/full.yaml --xie data/xie/xie_endothelial.h5ad --bundle models/full_tuned/bundle.joblib --out results/batch
```

Si el estudio externo no trae `nCount_RNA` / `nFeature_RNA` ni conteos crudos, el paquete con regresión de profundidad no se puede aplicar y hay que usar un paquete entrenado sin ella (`configs/full_noreg.yaml`). Los genes ausentes se tratan como 0 y el script avisa si faltan muchos genes variables. Si las etiquetas externas se llaman distinto, usar `--label-map "Glioblastoma=GBM,..."`.

### Archivos de resultados (`results/<modo>/`, no versionados)
`01_cohort.png`, `02_pca_variance.png`, `03_confusion_*.png`, `04_components_vs_performance_*.png`, `05_per_patient_*.png`, `06_components_vs_performance_seeds_*.png`, `07_by_cluster_*.png`, `08_permutation_*.png`, `permutation_results_*.csv`, `results_*.csv`, `summary_*.md`, `summary.json`, `run_metadata.json`, `splits/`, `tuning/` (validación anidada). En `results/batch/`: `09_batch_model_space.png`, `10_batch_joint_pca.png`, `11_batch_domain_classifier.png`, `batch_domain_results.csv/.md`, `batch_domain_per_patient.csv`, `batch_pca_eta2.csv`, `batch_model_space_summary.csv`. Además: `results/pc_meaning/` (`12_pc_by_patient.png`, `patients_metadata_pcs.csv`, `pc_correlations.csv`), `results/bootstrap/` (`13_bootstrap_ci.png`, `bootstrap_ci.csv`), `results/external_xie_core/`, `results/external_xie_aligned/`, `results/within_xie/`, `results/pseudobulk/` (promediados: `17_pseudobulk.png`, `pseudobulk_*_summary.csv/.md`, `runs/`) y `results/pseudobulk_counts/` (conteos: `xie_audit.csv`, `pseudobulk_configuration_grid.csv`, `pseudobulk_seed_sensitivity.csv`, `internal_patient_predictions.csv`, `xie_core_predictions.csv`, `xie_periphery_predictions.csv`, `packet_stability_results.csv/_cohort.csv`, `pseudobulk_coverage_matrix.csv`, `permutation_results.csv`, `patient_bootstrap_results.csv`, `study_/site_classifier_results.csv`, `age_only_results.csv`, `composition_only_results.csv`, `comparison_cell_vs_pseudobulk.csv/.md`, figuras `21_*.png`, `run_metadata.json`).

**Descripciones de las gráficas:** cada `.png` tiene un `.txt` con el mismo nombre en la carpeta `descriptions/` de su resultado (p. ej. `results/full/descriptions/01_cohort.txt`). Explica qué expresa la gráfica y da sus cifras clave, calculadas de los propios datos. Se generan al crear la figura y se regeneran con `experiments.replot`.
