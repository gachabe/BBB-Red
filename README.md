# bbb: ¿cuántos genes hacen falta para distinguir GBM de LGG?

Pipeline reproducible: matriz de expresión (células endoteliales x genes) → genes variables → escalado → PCA (ROM) → regresión logística / random forest → predecir `glioma_type` (GBM vs LGG).

**Datos:** Wälchli et al., *Nature* 632, 603-613 (2024), objeto AnnData de 67.372 células x 26.666 genes (14 pacientes: 8 GBM, 6 LGG). El `.h5ad` NO se sube al repositorio (`.gitignore`). Ruta en `configs/*.yaml` (`data.path`).

## Uso (PowerShell, sin activar el entorno)

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe -m src.validation --run-tests      # tests con datos sintéticos
.\.venv\Scripts\python.exe -m src.validation                  # valida el dataset real (la 1ª vez crea la caché)
.\.venv\Scripts\python.exe -m experiments.04_final --config configs/demo.yaml
.\.venv\Scripts\python.exe -m experiments.04_final --config configs/mini.yaml
.\.venv\Scripts\python.exe -m experiments.04_final --config configs/full.yaml
.\.venv\Scripts\python.exe -m experiments.05_seeds --config configs/full.yaml --n-seeds 10
.\.venv\Scripts\python.exe -m experiments.replot --config configs/full.yaml
```

`demo` (pocos pacientes y células) solo comprueba que todo funciona: no saques conclusiones de él.

## Decisiones de diseño

- **Unidad independiente: el paciente.** Las células de un mismo paciente se parecen mucho entre sí; el tipo de glioma es del paciente. Validación **dejando un paciente fuera** (14 rondas): cada paciente es test una vez y se entrena con los demás. IDs en `results/<modo>/splits/`.
- **Características: solo `X` (genes).** Ninguna columna de `obs` es característica. `01_eda` informa de qué columnas determinarían la etiqueta por sí solas (`PATH`, `Patient`, `age`...).
- **Representaciones** (de menos a más dimensiones): `PCA-r` (r = 2, 5, 10, 20, 50) < `HVG-2000` (genes más variables) < `All genes` (26.666, matriz dispersa).
- **Sin fugas:** genes variables, escalado y PCA se ajustan en cada ronda solo con las células de los pacientes de entrenamiento (test automático). `All genes` solo se escala (sin centrar) para no destruir la dispersión.
- **Profundidad de secuenciación:** `rom.regress_out` quita por regresión lineal, gen a gen, el efecto de `log1p(nCount_RNA)` y `log1p(nFeature_RNA)` (después de elegir los genes variables y antes de escalar). Los coeficientes se ajustan solo con los pacientes de entrenamiento. Con esto activo, `All genes` pasa a ser una matriz densa (más memoria y tiempo). Para desactivarlo: `regress_out: null`.
- **Desequilibrio:** entrenamiento con tope de células por paciente (`ml.max_train_cells_per_patient`) y `class_weight="balanced"`. Se evalúa con todas las células del paciente de test. Las métricas pesan igual a cada paciente (Macro-F1, balanced accuracy) y se da también el acierto por paciente (voto mayoritario).
- **Control:** etiquetas permutadas entre pacientes (mismos pacientes por clase). Si todo está bien, su acierto es el del azar.
- **Un solo estudio:** `src.validation` comprueba que el archivo tiene la forma esperada (67.372 x 26.666, solo Wälchli) y que `obs` no trae columnas de estudio/muestra. Si algún día se mezcla otro dataset (p. ej. Xie), habrá que añadir una comprobación de lote entre estudios.
- **Por tipo de vaso:** acierto dentro de cada `ECclusters` (`07_by_cluster_*.png`), porque su composición difiere entre GBM y LGG.
- Mismo código para demo, mini y full: solo cambia el YAML.

## Test de permutación (`experiments.06_permutation`)

¿Supera el acierto al azar? La etiqueta es del paciente, así que se permuta a nivel de paciente: con 14 pacientes (6 LGG) existen 3.003 asignaciones distintas y se recorren todas (p-valor exacto). Solo regresión logística sobre los PCA (`permutation.representations`). `p_adjusted_max` corrige por mirar todos los tamaños de PCA a la vez (estadístico máximo). Se puede interrumpir con Ctrl+C y relanzar: reanuda desde `results/<modo>/permutation/perm_stats.csv`.

`configs/full_noreg.yaml` es el mismo `full` sin `regress_out` (análisis de sensibilidad), con resultados en `results/full_noreg/`.

## Resultados (`results/<modo>/`)

`01_cohort.png`, `02_pca_variance.png`, `03_confusion_*.png`, `04_components_vs_performance_*.png`, `05_per_patient_*.png`, `07_by_cluster_*.png`, `08_permutation_*.png`, `permutation_results_*.csv`, `results_*.csv`, `summary_*.md`, `summary.json`, `run_metadata.json`, `splits/`.

## Limitaciones

- El tamaño muestral real es 14 pacientes: diferencias de uno o dos puntos entre representaciones no son distinguibles del ruido.
- Con dejar-un-paciente-fuera el reparto no cambia entre semillas; la desviación de `05_seeds` mide el ruido del submuestreo y del random forest, no la incertidumbre por tener pocos pacientes.
- Validación externa con Xie et al. (*Neuron* 2024): pendiente, no incluida.
