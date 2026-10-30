# bbb: ¿cuántos genes hacen falta para distinguir GBM de LGG?

Pipeline reproducible: matriz de expresión (células endoteliales x genes) → genes variables → regresión de profundidad → escalado → PCA (ROM) → regresión logística / random forest → predecir `glioma_type` (GBM vs LGG).

**Pregunta:** ¿cuánta dimensionalidad (genes) puede quitar la PCA sin perder la capacidad de predecir el tipo de glioma? Se compara `PCA-r` (r = 2, 5, 10, 20, 50) frente a `HVG-2000` (genes más variables) y frente a `All genes` (26.666), con regresión logística y random forest.

**Datos:** Wälchli et al., *Nature* 632, 603-613 (2024), `GSE256493`, objeto AnnData de 67.372 células x 26.666 genes: 14 pacientes (8 GBM, 6 LGG), 14 subtipos en `ECclusters`. El `.h5ad` NO se sube al repositorio (`.gitignore`). La ruta está en `configs/*.yaml` (`data.path`).

> **Estado (30-oct-2026):** análisis interno terminado. Falta la validación externa (Xie et al., *Neuron* 112(18), 2024), cuyo conjunto de datos aún no está localizado.

---

## 1. Resumen de resultados

| Hallazgo | Estado |
|---|---|
| Con regresión logística, 2 componentes de PCA predicen tan bien como 2.000 o 26.666 genes (macro-F1 0,740 frente a 0,716 y 0,713) | Sólido en la dirección; las diferencias entre representaciones no están contrastadas |
| Con random forest, más genes empeoran (todos los genes: 0,578; PCA-2: 0,680) | Sólido |
| La señal supera al azar (permutación exacta a nivel de paciente) | **Débil y dependiente de decisiones:** solo PCA-2 con regresión logística y regresión de profundidad (p ajustado = 0,033) |
| La señal es solo composición de tipos celulares | **No:** el acierto se mantiene dentro de casi todos los `ECclusters` (0,60-0,77) |
| Los resultados son estables entre semillas | Sí, pero esa estabilidad **no** mide la incertidumbre real (solo 14 pacientes) |
| Validación externa (Xie) | **Pendiente** |

Línea base: predecir siempre la clase mayoritaria acierta 8/14 pacientes = 0,571.

---

## 2. Qué se hizo, por orden

### 2.1 Comprobaciones de los datos (`src.validation`)
- Forma exacta 67.372 x 26.666: es el objeto de Wälchli, un único estudio. No hay columnas de estudio/muestra/lote en `obs`.
- Una sola etiqueta `glioma_type` por paciente.
- **Normalización:** la matriz ya es `log1p` de conteos normalizados a 10.000 por célula (suma de `expm1(X)` por célula: mediana 10.000, CV 0,0000; máximo de X = 8,75). Medido en una muestra de 500 células. No hace falta renormalizar.
- **Columnas que determinan la etiqueta por sí solas:** `PATH`, `age`, `Patient`. Nunca se usan como características (solo se usa `X`).
- No se hizo diagnóstico de lote: solo hay un estudio. Solo será necesario cuando se mezclen Wälchli y Xie.

### 2.2 Diseño del experimento
- **Unidad independiente: el paciente.** Las células de un mismo paciente se parecen mucho entre sí. Validación dejando un paciente fuera (14 rondas); IDs en `results/<modo>/splits/`.
- **Sin fugas:** genes variables, regresión de profundidad, escalado y PCA se ajustan en cada ronda solo con las células de los pacientes de entrenamiento (test automático).
- **Desequilibrio** (GBM 49.999 células frente a LGG 17.373): tope de 1.000 células por paciente en entrenamiento, `class_weight="balanced"`, y evaluación con todas las células del paciente de test. Las métricas pesan igual a cada paciente.
- **Regresión de profundidad** (`rom.regress_out`): quita por regresión lineal, gen a gen, el efecto de `log1p(nCount_RNA)` y `log1p(nFeature_RNA)`, ajustada solo con entrenamiento. Se decidió por ser práctica habitual, antes de ver los resultados del conjunto completo. Se repitió sin ella (`full_noreg`) como análisis de sensibilidad.
- **Control:** etiquetas permutadas entre pacientes. Sale por debajo del azar (0,38-0,47 en vez de 0,5): artefacto conocido de dejar-un-paciente-fuera con pocos pacientes, no un fallo.
- **Test de permutación exacto** a nivel de paciente: C(14,6) = 3.003 asignaciones, todas recorridas. `p_adjusted_max` corrige por mirar todas las representaciones a la vez (estadístico máximo).

### 2.3 Lo que se probó y se descartó
- **Diagnósticos de lote** (varianza de la PCA explicada por estudio, etc.): se añadieron y se eliminaron. Con un solo estudio no aportan nada.
- **Renormalizar la matriz:** innecesario, ya viene normalizada.
- **Etiqueta de localización:** descartada; el objetivo final es solo `glioma_type`.
- **Permutación con "Todos los genes":** no se hizo. Requiere matrices densas de 26.666 columnas para las 14 rondas y miles de ajustes; la memoria y el tiempo se disparan.
- **Hiperparámetros:** los análisis de las secciones 3.1-3.4 usan valores por defecto (regresión logística C = 1; random forest 200 árboles). El ajuste de hiperparámetros se hace aparte con validación anidada (sección 2.4); sus resultados se añadirán aquí cuando termine.

### 2.4 Ajuste de hiperparámetros con validación anidada (`experiments.09_tune`)
- **Rejilla** (por pareja representación-modelo): regresión logística con C ∈ {0,001; 0,01; 0,1; 1; 10}; random forest con `min_samples_leaf` ∈ {1, 20} (`max_depth` sin límite, 200 árboles). `All genes` queda con valores por defecto (demasiado costoso de ajustar).
- **Criterio:** macro-F1 ponderado por paciente, con dejar-un-paciente-fuera interno. Empates: gana la opción más regularizada.
- **Anidada:** para cada paciente de test, los hiperparámetros se eligen solo con los otros 13 pacientes. La estimación de rendimiento resultante (`results/<modo>/tuning/results_nested_glioma_type.md`) es la honrada.
- **Modelo final:** hiperparámetros elegidos con los 14 pacientes, guardado en `models/<modo>_tuned/bundle.joblib`. El rendimiento del LOPO interno de los 14 (`tuning_scores_all_patients.csv`) es optimista (es el máximo sobre la rejilla): no usarlo como estimación.
- No se ajustan la regresión de profundidad ni el tamaño de HVG: se fijan por diseño. Ajustar solo lo que tiene sentido ajustar con 14 pacientes reduce el riesgo de sobreajustar.
- Es posible que el ajuste no mejore, o que incluso empeore, respecto a los valores por defecto: la estimación anidada lo dirá.

---

## 3. Resultados

### 3.1 Rendimiento (10 semillas, media ± desviación; regresión de profundidad activa)

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

Con permutación de etiquetas (control), todas las filas quedan entre 0,38 y 0,47 de macro-F1.

### 3.2 ¿Supera el azar? Permutación exacta (regresión logística)

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

### 3.3 Qué pacientes fallan (PCA-2, regresión logística)
- **LGG1 y LGG6** se clasifican como GBM de forma sistemática (7 % y 7 % de sus células bien; con PCA-10, 9 % y 9 %). Es un fallo consistente, no ruido.
- **GBM6** es límite: bien con PCA-2 (58 % de células), mal con PCA-10 (36 %).
- El resto se clasifica con claridad: 12 de 14 pacientes con PCA-2, 11 de 14 con PCA-10.
- **Intervalos:** 12/14 corresponde a un intervalo de confianza del 95 % de 57 %-98 % (Clopper-Pearson). El límite inferior coincide con la línea base de 8/14 = 57 %. Con 11/14 el intervalo es 49 %-95 %.

### 3.4 ¿Es solo composición de tipos celulares?
- La composición de `ECclusters` difiere entre GBM y LGG: por ejemplo *Angiogenic capillary* 42 % en GBM frente a 27 % en LGG; *Stem-to-EC* 3,5 % frente a 0,2 %.
- **LGG6** tiene un 52 % de *Angiogenic capillary*, igual que los GBM típicos (48-55 %), lo que encaja con su error. **LGG1** tiene una composición casi idéntica a LGG2 y LGG4 (bien clasificados), así que la composición **no** lo explica.
- **GBM8** tiene solo un 2 % de *Angiogenic capillary* y aun así se acierta con el 82 % de sus células: el modelo usa algo más que esa proporción.
- Acierto dentro de cada clúster (PCA-2, balanced accuracy): *Artery* 0,77, *Vein* 0,77, *Proliferating cell* 0,76, *Large vein* 0,76, *Venule* 0,74, *Angiogenic capillary* 0,73, *Mitochondrial* 0,70, *Capillary* 0,69, *Large artery* 0,67, *EndoMT* 0,64, *Arteriole* 0,61, *Stem-to-EC* 0,60. Los dos clústeres proliferativos con un solo LGG (o ninguno) no se pueden evaluar.
- Estas cifras no pasaron por permutación y los 14 pacientes son los mismos en todos los clústeres: no son réplicas independientes.

---

## 4. Por qué podría funcionar (hipótesis, NO verificadas)

- **PCA-2 sin pérdida frente a todos los genes:** con 14 pacientes, pocas dimensiones actúan como regularización; más variables dan más ocasión de ajustar ruido de paciente. Coherente con que random forest empeore con más genes. No se ha comprobado qué representan PC1 y PC2.
- **Posible confusión con la edad:** `age` determina `glioma_type` en esta cohorte (los GBM son mayores). Si PC1 o PC2 correlacionan con la edad u otros rasgos del paciente, la señal podría no ser específica del tipo de glioma. No se ha comprobado.
- **Composición celular:** explica en parte el caso de LGG6 (ver 3.4), pero no el resto.
- **Efecto de la regresión de profundidad:** mejora a PCA-2 (macro-F1 0,663 → 0,739 con una semilla) pero no sabemos por qué; podría ser que la profundidad difiera entre pacientes y ensucie las primeras componentes.
- **Descenso no monótono con r** (2 > 5 > 10 > 20, luego 50 sube): no se interpreta como tendencia; con 14 pacientes un solo paciente mueve el acierto 0,071.

---

## 5. Limitaciones

- **14 pacientes.** Es el tamaño muestral real. Diferencias de uno o dos puntos entre representaciones no son distinguibles del ruido. No se ha contrastado formalmente PCA-2 frente a HVG-2000.
- **Las semillas no miden la incertidumbre real.** Con dejar-un-paciente-fuera el reparto no cambia entre semillas; la desviación mide el submuestreo de células y el azar del random forest.
- **Bifurcaciones del análisis.** Modelo, con/sin regresión de profundidad y tamaño de PCA son decisiones; el p ajustado solo cubre una de ellas. Conviene reportar todas.
- **Hiperparámetros.** Los resultados de la sección 3 usan valores por defecto. El ajuste se hace con validación anidada (sección 2.4) y la configuración final debe fijarse antes de aplicar el modelo a Xie; ajustar mirando Xie lo invalidaría.
- **Etiqueta confundida con el paciente:** `PATH`, `age` y `Patient` determinan `glioma_type` por sí solas; la separación es entre pacientes, no entre células.

---

## 6. Pendiente

1. **Localizar el conjunto de datos de Xie et al.** (*Neuron* 112(18), 2024). Pedir a la coautora el código de acceso (GEO, GSA-Human u otro) y comprobar si trae etiqueta de tipo de glioma.
2. **Validación externa** con el paquete entrenado (sección 7): genes en común, normalización, profundidad. Comprobación de lote **solo** si se mezclan ambos conjuntos.
3. **Metadatos clínicos de LGG1** (edad, subtipo, localización) en `obs`: único fallo aún sin explicación.
4. **Qué representan PC1 y PC2:** correlación con edad, composición de `ECclusters`, profundidad y paciente.
5. **Intervalos por remuestreo de pacientes** (bootstrap) para cuantificar la incertidumbre.
6. Permutación con "Todos los genes" usando pocas permutaciones y sin regresión de profundidad (opcional).

---

## 7. Uso (PowerShell, sin activar el entorno)

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe -m src.validation --run-tests                          # tests con datos sintéticos
.\.venv\Scripts\python.exe -m src.validation                                      # valida el dataset real
.\.venv\Scripts\python.exe -m experiments.04_final --config configs/full.yaml     # análisis completo (dejar-un-paciente-fuera)
.\.venv\Scripts\python.exe -m experiments.05_seeds --config configs/full.yaml --n-seeds 10
.\.venv\Scripts\python.exe -m experiments.06_permutation --config configs/full.yaml
.\.venv\Scripts\python.exe -m experiments.replot --config configs/full.yaml
```

Configuraciones: `demo` (comprueba que todo funciona; no sacar conclusiones), `mini`, `full`, `full_noreg` (sin regresión de profundidad), `full_hvgperm` (permutación también con HVG-2000; se genera a partir de `full`).

### Entrenamiento final y validación externa

```powershell
# recomendado: hiperparámetros con validación anidada + modelo final (reanudable, tarda horas)
.\.venv\Scripts\python.exe -m experiments.09_tune --config configs/full.yaml
.\.venv\Scripts\python.exe -m experiments.09_tune --config configs/full_noreg.yaml
```

Versión rápida, con valores por defecto y sin ajuste:

```powershell
# entrena con los 14 pacientes y guarda models/<modo>/bundle.joblib (fuera de GitHub)
.\.venv\Scripts\python.exe -m experiments.07_train_final --config configs/full.yaml
.\.venv\Scripts\python.exe -m experiments.07_train_final --config configs/full_noreg.yaml

# aplica el paquete a otro estudio SIN reentrenar (alinea genes por nombre, comprueba la normalización)
.\.venv\Scripts\python.exe -m experiments.08_external --bundle models/full_tuned/bundle.joblib --external RUTA\xie.h5ad --out results/external_xie --patient-col Patient --label-col glioma_type
```

Si el estudio externo no trae `nCount_RNA` / `nFeature_RNA` ni conteos crudos, el paquete con regresión de profundidad no se puede aplicar y hay que usar `models/full_noreg`. Los genes ausentes se tratan como 0 y el script avisa si faltan muchos genes variables. Si las etiquetas de Xie se llaman distinto, usar `--label-map "Glioblastoma=GBM,..."`.

### Archivos de resultados (`results/<modo>/`, no versionados)
`01_cohort.png`, `02_pca_variance.png`, `03_confusion_*.png`, `04_components_vs_performance_*.png`, `05_per_patient_*.png`, `07_by_cluster_*.png`, `08_permutation_*.png`, `permutation_results_*.csv`, `results_*.csv`, `summary_*.md`, `summary.json`, `run_metadata.json`, `splits/`.

Tiempos de referencia en el PC de Gabi: una ejecución completa dejando-un-paciente-fuera tarda unos 38 minutos; la permutación exacta de los PCA, unos 15 minutos.
