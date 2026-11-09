# bbb: ¿cuántos genes hacen falta para distinguir GBM de LGG?

Pipeline reproducible: matriz de expresión (células endoteliales x genes) → genes variables → regresión de profundidad → escalado → PCA (ROM) → regresión logística / random forest → predecir `glioma_type` (GBM vs LGG).

**Pregunta:** ¿cuánta dimensionalidad (genes) puede quitar la PCA sin perder la capacidad de predecir el tipo de glioma? Se compara `PCA-r` (r = 2, 5, 10, 20, 50) frente a `HVG-2000` (genes más variables) y frente a `All genes` (26.666), con regresión logística y random forest. El modelo se entrena con un estudio (Wälchli) y se valida internamente (dejando un paciente fuera) y externamente (Xie).

> **Estado (9-nov-2026):** análisis a nivel de célula, ajuste de hiperparámetros, validación externa, diagnóstico de estudio, análisis `core`, bootstrap, significado de PC1/PC2, exploratorios y **pseudobulks de conteos crudos (fase 1 del protocolo de los coautores, sección 4.11)** terminados. Conclusión: señal modesta dentro de Wälchli; **no se reproduce en Xie completo**; **con pseudobulks y solo muestras `core` (6 pacientes) la separación es perfecta y robusta a las variantes técnicas probadas (fase 2)**, pero con 6 pacientes eso no se puede demostrar estadísticamente, la edad no se puede descartar y la periferia está desplazada; el efecto de estudio sigue siendo total (sección 1). **Novedad (9-nov, v39): el README tiene ahora un mapa del proyecto, la PCA explicada paso a paso y el historial de lo probado y descartado (secciones 0 y 3.8); nuevos análisis: genes que componen cada componente (4.14), solapamiento de pacientes con Bejarano (4.15), cuántos genes hacen falta (4.16) y transferencia entre cohortes (4.17).** **Novedad (9-nov): en una cohorte independiente de RNA-seq en masa (Bejarano et al., 18 pacientes con edades solapadas) el modelo congelado de Wälchli ordena perfectamente a los pacientes (AUROC 1,00; p exacto < 0,0001), también dentro de la zona de edades solapadas, donde la edad sola no discrimina (sección 7, punto 5).**

---

## 0. Mapa del proyecto y guía de lectura

Esta sección es para quien llega nuevo al repositorio (por ejemplo, para escribir el artículo): dónde está cada cosa, cómo se construye el modelo paso a paso, qué es una componente principal y en qué sección del README está cada resultado.

### 0.1 Cómo está montado el proyecto

```
bbb/
├── README.md              este documento (todo lo que se ha hecho, probado y descartado)
├── configs/               configuración y tablas de entrada pequeñas
│   ├── full.yaml, full_noreg.yaml, mini.yaml, demo.yaml   parámetros de cada ejecución
│   ├── xie_samples.csv, xie_ages.csv, xie_published_counts.csv   muestras, edades y recuentos publicados de Xie
│   └── bejarano_bulk_ages.csv     Tabla S1 de Bejarano: edad, sexo, IDH, qué pacientes tienen RNA-seq en masa
├── src/                   código reutilizable (sin resultados): carga, PCA, modelos, métricas, figuras
├── experiments/           un script por análisis, numerados en el orden en que se hicieron (0.2)
├── tests/                 pruebas con datos sintéticos (`python -m pytest -q`)
├── tablas/                tablas CSV/MD clave, copiadas de results/ para citarlas y revisarlas sin ejecutar nada
├── results/               salida de cada análisis: una carpeta por análisis, con sus PNG y sus descripciones (0.3)
├── data/                  datos de entrada (no se suben a GitHub: son grandes y no son nuestros)
└── models/                modelos entrenados (no se suben a GitHub)
```

`src/` en una línea por módulo: `data.py` (carga del objeto de Wälchli y validación), `features.py` (reparto por paciente, submuestreo, etiquetas permutadas), `rom.py` (genes variables, escalado y PCA a nivel de célula), `models.py` (regresión logística y random forest), `evaluation.py` (métricas, tablas y figuras de la parte a nivel de célula), `permutation.py` y `tuning.py` (permutación exacta y validación anidada), `batch.py` (efecto de estudio), `final_model.py` (modelo final, aplicación a otro estudio, alineación de genes), `xie.py` (preparación de Xie y marcadores celulares), `pseudobulk.py` (pseudobulks promediados), **`pbcount.py` (pseudobulks de conteos crudos: TMM, PCA, regresión logística, dejar un paciente fuera, métricas por paciente; es el núcleo del análisis actual)** y `figs.py` (guardar cada figura con su descripción y rellenar bloques del README).

### 0.2 Qué script produce qué

| Script | Qué hace | Carpeta de resultados | Sección |
|---|---|---|---|
| `01_eda` | Cohorte, composición por tipo de vaso, fuga de información, normalización | `results/<modo>/` | 3.1 |
| `02_rom`, `03_ml`, `04_final` | Genes variables + escalado + PCA, regresión logística y random forest dejando un paciente fuera, con control de etiquetas permutadas | `results/<modo>/` | 3.2, 4.1 |
| `05_seeds` | Repite lo anterior con varias semillas | `results/<modo>/` | 4.1 |
| `06_permutation` | Permutación exacta a nivel de paciente | `results/<modo>/` | 4.2 |
| `07_train_final`, `09_tune` | Modelo final con los 14 pacientes; hiperparámetros con validación anidada | `models/`, `results/<modo>/tuning/` | 3.3, 4.4 |
| `10_prepare_xie`, `08_external` | Prepara Xie y le aplica el modelo congelado de Wälchli | `data/xie*/`, `results/external_xie*/` | 3.4, 4.5, 4.7 |
| `11_batch` | ¿Se identifica el estudio de una célula nueva? | `results/batch/` | 3.5, 4.6 |
| `12_pc_meaning`, `16_age_check` | Qué representan PC1 y PC2; relación con la edad | `results/pc_meaning/`, `results/age_check/` | 4.8 |
| `13_bootstrap` | Intervalos por remuestreo de pacientes | `results/bootstrap/` | 4.9 |
| `14_within_xie`, `15_align_external` | Exploratorios: dentro de Xie y alineando medias | `results/within_xie/`, `results/external_xie_aligned/` | 4.10 |
| `17_pseudobulk` | Pseudobulks promediados (secundario) | `results/pseudobulk/` | 3.6, 4.13 |
| `18_audit_xie`, `19`, `20`, `21` | **Pseudobulks de conteos crudos (análisis principal)**: auditoría de Xie, validación interna y externa, controles, informe | `results/pseudobulk_counts/` | 3.7, 4.11 |
| `22_phase2` | Robustez del resultado `core`, edad, periferia, filtros de Xie | `results/pseudobulk_phase2/` | 4.12 |
| `23_bejarano_bulk` | Validación con la cohorte de RNA-seq en masa de Bejarano | `results/bejarano_bulk/` | 7 (punto 5) |
| `24_pc_genes` | **Qué genes componen cada componente principal** | `results/pc_genes/` | 4.14 |
| `25_overlap_check` | ¿Puede un paciente de Bejarano ser uno de Wälchli? | `results/overlap_check/` | 4.15 |
| `27_gene_budget` | **¿Cuántos genes hacen falta?** Rendimiento frente al nº de genes | `results/gene_budget/` | 4.16 |
| `26_transfer_matrix` | Entrenar con una cohorte y predecir las otras (incluye Bejarano → Wälchli y Xie) | `results/transfer_matrix/` | 4.17 |
| `28_cohort_overview` | Edades de las tres cohortes y diagrama del método | `results/cohort_overview/` | 0.3 |
| `replot` | Regenera figuras y descripciones sin reentrenar | -- | 8 |

### 0.3 Figuras y sus descripciones

**Convención:** cada análisis guarda sus gráficas (`.png`) en su carpeta de `results/` y, dentro, en `descriptions/`, un `.txt` con **el mismo nombre** que explica qué muestra la gráfica y da sus cifras clave calculadas de los propios datos (por ejemplo `results/pc_genes/24_pc_variance.png` y `results/pc_genes/descriptions/24_pc_variance.txt`). Quien escriba el artículo puede copiar la descripción como pie de figura. Los números de las figuras indican el script que las crea.

| Carpeta | Figura | Qué muestra |
|---|---|---|
| `results/cohort_overview/` | `28_pipeline_diagram` | Diagrama del método, de la matriz de expresión al paciente, marcando lo que se ajusta solo con entrenamiento |
| | `28_cohort_ages` | Edad de cada paciente en las tres cohortes por diagnóstico, con el solape de edades |
| | `28_cohort_composition` | Pacientes y células de la cohorte de entrenamiento |
| `results/<modo>/` | `01_cohort`, `02_pca_variance` | Composición de la cohorte; varianza explicada de la PCA a nivel de célula |
| | `03_confusion_*`, `05_per_patient_*`, `07_by_cluster_*` | Matrices de confusión, acierto por paciente y por tipo de vaso |
| | `04_components_vs_performance_*`, `06_..._seeds_*` | Rendimiento frente al nº de componentes (una semilla y varias) |
| | `08_permutation_*` | Distribución de la permutación exacta frente al valor observado |
| `results/batch/` | `09_batch_model_space`, `10_batch_joint_pca`, `11_batch_domain_classifier` | Efecto de estudio: Xie en el espacio del modelo, PCA conjunta e identificación del estudio |
| `results/pc_meaning/`, `results/age_check/` | `12_pc_by_patient`, `16_age_vs_pc2` | PC1/PC2 por paciente y relación de PC2 con la edad |
| `results/bootstrap/` | `13_bootstrap_ci` | Intervalos de confianza por remuestreo de pacientes |
| `results/pseudobulk/` | `17_pseudobulk` | Pseudobulks promediados (secundario) |
| `results/pseudobulk_counts/` | `21_cells_per_pseudobulk`, `21_stability_vs_packet_size` | Células por pseudobulk y estabilidad según el tamaño del paquete |
| | `21_patient_predictions`, `21_cell_vs_pseudobulk`, `21_core_vs_periphery` | Probabilidad por paciente; célula frente a pseudobulk; núcleo frente a periferia |
| | `21_expression_vs_composition_age`, `21_pseudobulk_pca` | Expresión frente a composición y edad; PCA de los pseudobulks |
| `results/pseudobulk_phase2/` | `22_robustness`, `22_training_subsample` | Robustez del `core` a las variantes técnicas; dependencia del conjunto de entrenamiento |
| `results/bejarano_bulk/` | `23_bejarano_probability_vs_age` | Probabilidad de GBM de cada paciente de Bejarano frente a su edad, con la zona de edades solapadas |
| | `23_bejarano_auroc_by_mode`, `23_bejarano_roc` | AUROC según cómo se trate la edad; curvas ROC frente a un modelo de solo edad |
| `results/pc_genes/` | `24_pc_variance` | Varianza, utilidad (AUROC) y estabilidad de cada componente |
| | `24_pc_scores_heatmap`, `24_pc_scatter`, `24_pc_vs_age` | Puntuación de cada paciente en cada componente; pacientes en el plano PC1-PC2-PC3; edad frente a puntuación |
| | `24_pc_top_genes` | Genes con más peso en PC1-PC4, coloreados por tipo de marcador celular |
| | `24_classifier_drivers`, `24_gene_overlap_heatmap`, `24_marker_share` | Genes que empujan cada modelo; solapamiento entre listas de genes; marcadores celulares por componente |
| `results/overlap_check/` | `25_overlap_check` | Edad y sexo de los pacientes de Wälchli y Bejarano, parejas compatibles y lo esperable por azar |
| `results/transfer_matrix/` | `26_transfer_matrix_auroc`, `26_transfer_matrix_macro_f1` | Entrenar con una cohorte y predecir las otras (orden frente a umbral) |
| | `26_reverse_patient_probabilities` | Predicción por paciente al entrenar solo con Bejarano |
| `results/gene_budget/` | `27_gene_budget_auroc`, `27_gene_budget_macro_f1` | Rendimiento frente al número de genes variables, en tres cohortes |

### 0.4 Cómo se construye el modelo, paso a paso

Es el modelo de los pseudobulks de conteos (4.11-4.17). Cada fila dice qué se hace, por qué, y con qué datos se ajusta (todo lo que depende de los datos se ajusta **solo con los pacientes de entrenamiento** y se aplica tal cual a los demás).

| Paso | Qué hace | Por qué | Se ajusta con |
|---|---|---|---|
| 1. Conteos | Se reconstruyen los conteos crudos de cada célula a partir de la matriz publicada: `expm1(X) × nCount_RNA / 10.000` | Los pseudobulks suman conteos, no logaritmos | -- (exacto, se comprueba) |
| 2. Paquetes | Se agrupan al azar 100 células **del mismo paciente** (en Xie, de la misma muestra) y se suman sus conteos: un pseudobulk por paquete | Una célula tiene mucho ruido; un paquete se parece a una muestra de RNA-seq en masa. Todos los paquetes de un paciente van siempre al mismo bloque de validación | -- |
| 3. Filtro de genes | Se quedan los genes con CPM > 1 en al menos el 25 % de los pseudobulks de entrenamiento | Quitar genes casi siempre a cero | Entrenamiento |
| 4. TMM + log2-CPM | Se corrige el tamaño de librería con TMM (frente a una referencia de entrenamiento) y se pasa a log2 de cuentas por millón | Hacer comparables pseudobulks de distinta profundidad | Entrenamiento |
| 5. Genes variables | Se eligen los 2.000 genes con más varianza | Quedarse con los genes que más cambian entre pseudobulks, que son los que pueden distinguir grupos | Entrenamiento |
| 6. Escalado | A cada gen se le resta la media y se divide por la desviación típica (recorte a ±10) | Que ningún gen domine la PCA solo por tener valores más grandes | Entrenamiento |
| 7. **PCA** | Se buscan las direcciones del espacio de genes en las que más varían los pseudobulks, ordenadas de mayor a menor varianza (PC1, PC2, ...). **PCA-r = las r primeras componentes de esa misma PCA** | Resumir 2.000 genes en unas pocas coordenadas; es la «reducción de dimensionalidad» de la pregunta | Entrenamiento |
| 8. Regresión logística | Aprende una combinación lineal de las coordenadas que separa GBM de LGG (penalización L2, C = 1, clases balanceadas, cada paciente pesa igual) | Clasificador simple y poco propenso a sobreajustar con 14 pacientes | Entrenamiento |
| 9. Probabilidad por paciente | Cada paquete recibe P(GBM); la probabilidad del paciente es la **media** de la de sus paquetes | Una predicción por paciente, que es la unidad independiente | -- |
| 10. Evaluación | AUROC (¿ordena bien a los pacientes?), macro-F1 con corte 0,5 (¿acierta la clase?), permutación exacta y bootstrap de pacientes | El AUROC mide el orden; el macro-F1 depende además de que el umbral esté bien colocado | -- |

Las tres representaciones comparadas: **PCA-r** (pasos 1-8 con r componentes), **HVG-2000** (se salta el paso 7: regresión logística directa sobre los 2.000 genes) y **All genes** (todos los genes filtrados).

### 0.5 Qué es una componente principal y cómo leerla

La PCA no elige genes: **crea ejes nuevos, cada uno mezcla muchos genes**. Una componente es una receta: «súmale a cada paciente un poco de cada gen, con estos pesos». Para un pseudobulk:

`puntuación en la componente k = Σ sobre los genes (carga del gen en k × valor escalado del gen)`

| Concepto | Qué es | Dónde verlo |
|---|---|---|
| **Carga (peso, *loading*)** | Cuánto contribuye un gen a la componente. Positiva: el gen sube cuando sube la puntuación; negativa: baja | `tablas/pc_gene_loadings_top.csv`, 4.14 |
| **Puntuación (*score*)** | Valor de un paquete o paciente en la componente: dónde cae en ese eje | `24_pc_scores_heatmap`, `24_pc_scatter` |
| **Varianza explicada** | Qué parte de la variación total entre pseudobulks recoge la componente. PC1 recoge más que PC2, y así sucesivamente | `24_pc_variance`, `tablas/pc_summary.csv` |
| **PCA-r** | Las primeras r componentes juntas. PCA-2 usa solo PC1 y PC2 | 0.4 |
| **Signo** | El signo de una componente es arbitrario; aquí se orienta para que una puntuación alta signifique «más GBM» | 4.14 |
| **Genes que empujan el modelo** | Combinación de los coeficientes de la regresión logística con las cargas: qué genes pesan realmente en la predicción de PCA-r | `24_classifier_drivers`, `tablas/classifier_gene_drivers.csv` |

**Cómo interpretar una componente (guía para la parte biológica):** (1) mirar si separa GBM de LGG o si sigue la edad (`24_pc_variance`, `24_pc_vs_age`); (2) leer los genes con más peso en cada sentido (`24_pc_top_genes`, tabla de 4.14); (3) mirar si hay marcadores de otros tipos celulares, lo que indicaría cambios de composición y no de la célula endotelial (`24_marker_share`); (4) comprobar que la lista es estable entre semillas (columna `estabilidad`); (5) pasar las listas por un análisis de enriquecimiento (rutas, tipos celulares). **Cautela:** las cargas describen cómo se construye la componente, no una causa; una componente puede mezclar diagnóstico, edad, profundidad y estudio (4.8 lo muestra para PC1 y PC2).

### 0.6 Glosario

| Término | Significado en este proyecto |
|---|---|
| GBM / LGG | Glioblastoma (grado 4, IDH nativo) / glioma de bajo grado (IDH mutado). Etiqueta `glioma_type` |
| Paciente | Unidad independiente: todo se reparte, se valida y se cuenta por paciente |
| Pseudobulk / paquete | Suma de conteos de ~100 células del mismo paciente; se comporta como una muestra de RNA-seq en masa |
| Dejar un paciente fuera (LOPO) | Se entrena con 13 pacientes y se predice el que falta; se repite con los 14 |
| Modelo congelado | Modelo entrenado con Wälchli y aplicado a otra cohorte sin reajustar nada |
| `core` / periferia | Muestras de Xie del núcleo del tumor (`T`) o de tejido periférico (`N`) |
| AUROC | Probabilidad de que un GBM reciba más probabilidad que un LGG (0,5 = azar, 1 = orden perfecto). No depende del umbral |
| Macro-F1 | Acierto medio de las dos clases con corte 0,5. Sí depende del umbral |
| Orden frente a umbral | Una cohorte puede conservar el orden (AUROC alto) y desplazar el umbral (macro-F1 bajo); es lo que pasa en la periferia de Xie y en Bejarano |
| Permutación exacta | Se recorren todas las asignaciones posibles de etiquetas a pacientes: p exacto sin aleatoriedad |
| Efecto de estudio (*batch*) | Diferencias entre cohortes por laboratorio, tecnología o procesado, no por diagnóstico |
| Confusión con la edad | Edad y diagnóstico van juntos, así que no se sabe cuál usa el modelo; Bejarano permite separarlas |
| TMM, log2-CPM | Normalización de tamaño de librería y escala logarítmica de cuentas por millón |
| HVG | Genes más variables (*highly variable genes*) |
| IDH | Gen cuya mutación define los gliomas de bajo grado (mutado) frente a los GBM (nativo) |

### 0.7 Dónde mirar según lo que se vaya a escribir

| Si hay que escribir sobre... | Mirar |
|---|---|
| Resumen y conclusión | Sección 1 y «Cómo contarlo» |
| Datos y cohortes | Sección 2, 28_cohort_ages |
| Métodos | 0.4 (paso a paso), 3.2, 3.7 y `28_pipeline_diagram` |
| Rendimiento interno y significación | 4.1, 4.2, 4.9, 4.11 |
| Validación externa y efecto de estudio | 4.5, 4.6, 4.7, 4.11, 4.12 |
| Edad como confusor | 4.12 (punto 4), Bejarano (sección 7, punto 5), 4.15 |
| Interpretación biológica | 4.14 (genes por componente) y `tablas/` |
| Cuántos genes hacen falta (pregunta central) | 4.16 y 4.1 |
| Qué se probó y se descartó | 3.8 |
| Limitaciones | Sección 6 |

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
| **Fase 2: robustez del `core` (4.12, 4.13)** | **Aguanta:** modelo, forma del paquete, profundidad por adelgazamiento, genes comunes, quedarse con el mismo número de células endoteliales que el artículo (57.318, `core` 1,000), C de la regresión, otra agregación (promedio de log-expresión: 0,95-0,99) y quitar 15.000 genes de estudio no lo rompen (PCA-5 1,000 casi siempre); **lo rompe la regresión de profundidad** en la agregación por promedio y PCA-2 con 500 genes variables; se mantiene con subconjuntos de entrenamiento (84-91 %). La señal aparece también dentro de Xie (AUROC 1,000) y en sentido inverso (AUROC 0,90). **No descartado:** edad (las medias publicadas de Xie quedan a ambos lados de 0,5 en un modelo solo de edad; dentro de cada diagnóstico la predicción correlaciona con la edad, 0,4-0,5 en GBM, sin significación) y diferencias entre cohortes; la periferia es un desplazamiento constante (AUROC 0,8-0,9), no ausencia de señal |
| **Edad en Xie (4.12, punto 4)** | **Confusión total:** LGG 28-48 años, GBM 56-70 (sin solape). Un modelo solo de edad da macro-F1 1,000 en `core`, periferia y conjunto completo, **igual o mejor que la expresión** (que en periferia y completo falla con umbral 0,5). La expresión no demuestra aportar más que la edad |
| **Validación con Bejarano (RNA-seq en masa, 18 pacientes, edades solapadas; sección 7, punto 5)** | **Funciona en orden:** AUROC 1,000 con todas las representaciones (p exacto < 0,0001; la edad sola da 0,77), y 1,000 también en la zona de edades solapadas (49-73 años, n = 13; la edad sola: 0,51). Con la pendiente de la edad quitada dentro de cada diagnóstico se mantiene (AUROC 0,93-0,97). **El umbral de 0,5 se desplaza** (macro-F1 0,72-0,83; con probabilidades centradas 0,83-0,94). Cautelas: tecnología distinta (RNA-seq en masa), 18 pacientes, y no se ha podido descartar solapamiento con pacientes de Wälchli |
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
Primera versión de la idea de los coautores: promediar paquetes de células del mismo paciente. Usa la media de la expresión log-normalizada (la regla 3 de la guía, marcada allí como solo exploratoria) y paquetes disjuntos (familia D). **Queda como análisis secundario; el principal es el protocolo de conteos de 3.7.** Ejecutado: resultados en 4.13.
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

### 3.8 Historial de lo probado, lo descartado y por qué

Tabla cronológica de las decisiones del proyecto. «Decisión» dice qué quedó como análisis principal, qué como secundario o exploratorio, y qué se abandonó.

| # | Qué se probó | Motivo | Qué salió | Decisión | Sección |
|---|---|---|---|---|---|
| 1 | PCA-r, HVG-2000 y todos los genes con regresión logística y random forest, a nivel de **célula**, dejando un paciente fuera | Pregunta original: cuánta dimensionalidad se puede quitar | Con regresión logística, PCA-2 predice como 2.000 o 26.666 genes (0,74 frente a 0,72 y 0,71); con random forest, más genes empeoran | Línea base | 4.1 |
| 2 | Regresión de profundidad (`nCount_RNA`, `nFeature_RNA`) a nivel de célula | Práctica habitual; fijada antes de ver resultados | Mejora a PCA-2; se repitió sin ella (`full_noreg`) como sensibilidad | Se mantiene; sensibilidad informada | 3.2, 5 |
| 3 | Permutación exacta a nivel de paciente | Saber si la señal supera al azar | Solo PCA-2 con regresión logística y regresión de profundidad (p ajustado 0,033) | Señal débil y dependiente de decisiones | 4.2 |
| 4 | Hiperparámetros con validación anidada | Comprobar que los valores por defecto no limitan | Diferencias de ±0,02, dentro del ruido | **No cambia nada**; se usan los valores por defecto | 4.4 |
| 5 | Validación externa en Xie (7 pacientes), a nivel de célula | Probar que la señal no es solo de Wälchli | **No se replica** (macro-F1 0,54-0,60) | Resultado negativo, se informa | 4.5 |
| 6 | Solo muestras `core` de Xie | Hipótesis fijada de antemano: la periferia puede parecerse a LGG | Mejora (0,67-0,84) | Secundario fijado de antemano | 4.7 |
| 7 | Solo muestras periféricas de Xie | Comprobar la hipótesis anterior | Azar (0,41-0,55) | Exploratorio | 4.7 |
| 8 | Corte de profundidad en Xie (≥ 1.500 genes por célula) | Ver si la baja profundidad explicaba el fallo | Empeoró | **Descartado**; se hizo tras ver resultados y no debe presentarse como corrección | 4.5 |
| 9 | Diagnóstico de efecto de estudio | ¿Se identifica el estudio de una célula nueva? | Casi sin error (AUC hasta 1,000); no lo explica la profundidad | Hallazgo: el efecto de estudio es muy fuerte | 4.6 |
| 10 | Alinear las medias por gen de Xie con las de Wälchli | Ver si el fallo es un desplazamiento de medias | Mejora en casi todas las filas, más con muchas dimensiones | Exploratorio y transductivo; solo una pista | 4.10 |
| 11 | Dejar un paciente fuera solo dentro de Xie | ¿Hay señal dentro de Xie a nivel de célula? | Cercano al azar | Exploratorio | 4.10 |
| 12 | Qué representan PC1 y PC2; edad | Entender qué aprende PCA-2 | PC1 sigue profundidad y composición; PC2 sigue el glioma pero también la edad (no separables) | Alerta sobre la edad | 4.8 |
| 13 | Bootstrap de pacientes | Cuantificar la incertidumbre real | Intervalos muy anchos | Se informa siempre junto a cada cifra | 4.9 |
| 14 | **Pseudobulks promediados** (media de log-expresión) | Propuesta de los coautores | `core` se reproduce; la regresión de profundidad lo rompe | Secundario | 3.6, 4.13 |
| 15 | **Pseudobulks de conteos crudos** (suma de conteos, TMM, log-CPM), paquetes de 50 a 500 células | Protocolo de los coautores | **`core` con macro-F1 1,000**; Xie completo y periferia no | **Análisis principal** | 3.7, 4.11 |
| 16 | Variantes técnicas de la fase 2: random forest, paquetes disjuntos, UMI igualados, conteos adelgazados, genes comunes, filtros de Xie, quitar genes de estudio, C de la regresión, 5.000 genes | Ver si el `core` depende de alguna decisión técnica | Aguanta todas; **PCA-2 es frágil** (500 genes variables: 0,333) y PCA-5 no | Resultado robusto a lo técnico | 4.12 |
| 17 | Normalización DESeq/VST en lugar de TMM | Inspirada en MLSeq | No cambia nada | Descartada como mejora; no es DESeq2 exacto | 4.12 |
| 18 | Filtro duro de endotelio en Xie (80 / 53,9 / 25 % de células más endoteliales) | Nuestra selección tenía casi el doble de células que el artículo | `core` sigue en 1,000 con el mismo nº de células que el artículo | Robustez confirmada | 4.12 |
| 19 | Entrenar al revés (Xie `core` → Wälchli) y señal dentro de Xie con pseudobulks | ¿La señal es compartida? | AUROC 0,90 al revés; 1,000 dentro de Xie `core` | Apoya señal compartida, con cautelas | 4.12 |
| 20 | Edades individuales de Xie y modelo de solo edad | Hipótesis de confusión con la edad | **La edad separa perfectamente los grupos de Xie** (28-48 frente a 56-70); el modelo de edad iguala o supera a la expresión | **Confusión confirmada**: la validación en Xie no demuestra más que la edad | 4.12 (punto 4) |
| 21 | Quitar la pendiente de la edad de los genes (versión simple) | Ver qué queda sin la edad | La señal interna casi desaparece (AUROC 0,54-0,73) | Descartada como modelo final: quita también parte del diagnóstico | 4.12 |
| 22 | Pendiente de la edad estimada **dentro de cada diagnóstico** | Corregir lo anterior | La señal se conserva (AUROC 0,71-0,91 interno; 1,00 en Xie) | Post hoc; indica que la señal no es solo edad | 4.12 |
| 23 | Paquetes pequeños (10, 25, 50 células) en Xie `core` | ¿Sirven conjuntos con pocas células por paciente? | Con 25-50 células el orden se conserva en todos los sorteos | Utilizable, sobre todo con PCA | 4.12 |
| 24 | Estados endoteliales por separado | ¿La señal está en un estado concreto? | Mejor en *Angiogenic capillary*; débil en venas | Descriptivo | 4.12 |
| 25 | Paquetes definidos como fracción de las células, en Xie completo | Probar si el tamaño explica el fallo en Xie completo | Resultado no interpretable (paquetes de tamaños muy distintos) | Descartado | 4.12 |
| 26 | **Cohorte de Bejarano** (RNA-seq en masa, 18 pacientes, edades solapadas) | Única cohorte que permite separar edad y diagnóstico | **AUROC 1,000**, también en la zona de edades solapadas (la edad sola: 0,51); el umbral se desplaza | Validación más fuerte hasta ahora; falta confirmar que no hay pacientes comunes | Sección 7 (punto 5) |
| 27 | Comprobación de solapamiento de pacientes Wälchli-Bejarano (clase, sexo, edad) | Los autores no lo dicen | Ver 4.15 | Se asume sin solapamiento hasta que lo confirmen los autores | 4.15 |
| 28 | Genes que componen cada componente principal | Interpretación biológica | Ver 4.14 | Explicabilidad para el grupo | 4.14 |
| 29 | Rendimiento frente al número de genes (curva de «cuántos genes hacen falta») | Responder a la pregunta central con las tres cohortes | Ver 4.16 | Pendiente de ejecutar | 4.16 |
| 30 | Matriz de transferencia entre cohortes (incluye Bejarano → Wälchli y Xie) | Ver si el eje GBM/LGG se comparte en todas direcciones | Ver 4.17 | Pendiente de ejecutar | 4.17 |

**Descartes menores:**
- **Renormalizar la matriz de Wälchli:** innecesario, ya viene normalizada.
- **Etiqueta de localización:** descartada; el objetivo final es solo `glioma_type`.
- **Permutación con «Todos los genes»:** no se hizo; requiere matrices densas de 26.666 columnas y miles de ajustes.
- **Diagnóstico de batch con un solo estudio:** se quitó (no aportaba nada) y se recuperó cuando hubo dos conjuntos (3.5).
- **Corrección de C en PCA-2:** no era un fallo, la puntuación no depende de C (4.8).

**Lo que no se puede hacer con los datos de Wälchli y Xie:** separar edad y diagnóstico (en Xie las edades no se solapan; en Wälchli solo hay un LGG dentro de la zona de edades compartida). Por eso Bejarano es la cohorte clave.

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

**Auditoría de Xie** (`xie_audit.csv`; comparada con el artículo): sin incidencias en el mapeo muestra-paciente; profundidad muy desigual (mediana de UMIs por muestra de 1.262 a 6.428); la anotación endotelial publicada no está en los datos, pero el artículo da totales: **103.230 células tras el control de calidad, de ellas 57.324 endoteliales** (control de calidad: < 200 genes, > 10 % de genes mitocondriales y dobletes con DoubletFinder). Nuestra preparación carga **198.445 células** (solo filtro de 200 genes, sin filtro mitocondrial ni de dobletes) y selecciona **106.399 como endoteliales, casi el doble del artículo**. La selección por marcadores es por tanto mucho más laxa que la publicada y puede incluir células de baja calidad o no endoteliales; la fase 2 prueba filtros más estrictos (4.12). Pacientes de Xie: 3 LGG con mutación de IDH y 4 GBM sin mutación de IDH. Edades individuales (Tabla S1A del artículo, `configs/xie_ages.csv`): LGG 28, 40 y 48 años (media 38,7); GBM 56, 59, 60 y 70 (media 61,3). **Comprobado contra el artículo:** el texto (Métodos) da edades medias de 49,3 (LGG) y 61,3 (GBM) y sexos 2 M/1 H y 3 M/1 H. La media GBM (61,25) y los sexos coinciden con la Tabla S1A, pero la media LGG de la tabla es 38,7, no 49,3. No sabemos cuál de las dos cifras es la errónea; usamos las edades individuales de la tabla. En ambos casos los LGG son más jóvenes que los GBM, pero con 49,3 la separación individual dependería de edades que no vemos, así que conviene preguntarlo a los autores si el resultado va a un artículo. N036 tiene la puntuación mural más alta (0,66) y N045 la glial (0,47). No factibles con 500 células: GBM1 (264 células) en Wälchli y N027 (465) en Xie.

**Xie: células publicadas frente a las nuestras** (Tablas S1B/S1C del artículo; `configs/xie_published_counts.csv`, `xie_audit_vs_published.csv`)
| Muestra | Células cargadas / publicadas | Endoteliales seleccionadas / publicadas | Razón endoteliales |
|---|---|---|---|
| N027 | 691 / 339 | 465 / 288 | 1,61 |
| T027 | 25.476 / 14.086 | 18.267 / 10.599 | 1,72 |
| N036 | 30.818 / 13.254 | 11.596 / 4.642 | 2,50 |
| N037 | 17.999 / 11.757 | 9.932 / 6.433 | 1,54 |
| T037 | 24.568 / 13.075 | 14.168 / 8.016 | 1,77 |
| N038 | 16.525 / 9.682 | 11.271 / 6.417 | 1,76 |
| T038 | 8.342 / 4.684 | 2.236 / 1.333 | 1,68 |
| N043 | 8.785 / 6.344 | 3.724 / 2.678 | 1,39 |
| T043 | 8.408 / 5.600 | 3.768 / 2.607 | 1,45 |
| N045 | 23.197 / 8.747 | 13.450 / 5.084 | 2,65 |
| T045 | 8.487 / 4.305 | 6.225 / 3.222 | 1,93 |
| N046 | 8.801 / 6.531 | 5.757 / 4.268 | 1,35 |
| T046 | 16.348 / 4.826 | 5.540 / 1.737 | 3,19 |

Cargamos entre 1,35 y 3,39 veces las células publicadas por muestra (los datos descargados parecen no tener el control de calidad del artículo), y la selección por marcadores da entre 1,35 y 3,19 veces las endoteliales publicadas. La variante `xie_pub` (sección 4.12) iguala el número por muestra al publicado: 57.324 células, `core` 1,000 con PCA-2, PCA-5 y HVG-2000.

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

*Segunda tanda (resultados abajo, apartado «Segunda tanda»):* Xie con solo el 25 / 53,9 / 80 % de células más endoteliales por muestra, 500 y 5.000 genes variables, C = 0,01 y 100 en la regresión, y la correlación entre predicción y edad dentro de cada diagnóstico en Wälchli.

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
| Xie con filtro mitocondrial < 10 % del artículo (retiene el 98 %) | 1,000 | 1,000 | 1,000 |
| Xie solo con células claramente endoteliales (retiene el 98 %) | 1,000 | 1,000 | 1,000 |
| Lo anterior + genes comunes + misma profundidad | 1,000 | 1,000 | 1,000 |
| Sin los 500 / 2.000 / 5.000 / 10.000 / 15.000 genes que más distinguen los estudios | 1,000 hasta 10.000; 0,886 con 15.000 | 1,000 en todos | 1,000 en todos |

- **PCA-5 es 1,000 en todas las variantes**; los fallos son un paciente en alguna semilla y alguna representación. Las variantes que bajan de 0,9 se repitieron con paquetes de 250 y 500: con UMI igualados vuelve a 1,000; con disjuntos de 100 y 250 células HVG-2000 queda en 0,829.
- **Filtrar Xie:** el filtro mitocondrial del artículo y la exigencia de ≥ 2 marcadores endoteliales con puntuación endotelial mayor que mural, inmune y glial solo eliminan el 2 % de las células (104.305 y 103.796 de 106.399 sobreviven), así que casi no cambian nada. El control duro está en la segunda tanda (`top25`, `top54`, `top80`).
- **Quitar 15.000 de los ≈ 22.700 genes comunes que más distinguen los estudios no rompe el `core`.** La señal no depende de los genes de estudio más extremos; tampoco está concentrada en pocos genes.
- La lectura: ninguna decisión técnica probada (modelo, forma del paquete, profundidad, genes) explica la separación perfecta. Lo que no descartan es una diferencia real entre las cohortes que coincide con GBM/LGG (edad, procesamiento), que ninguna de estas variantes toca.

**2. Dependencia del conjunto de entrenamiento**
- 100 subconjuntos de 10 pacientes de Wälchli (6 GBM + 4 LGG): separación sin error en el 91 % (PCA-5), 85 % (PCA-2) y 84 % (HVG-2000, PCA-10, PCA-20); con todos los genes, 39 %. La mediana del AUROC es 1,000 en todas las representaciones: ordena bien a los pacientes aunque el umbral a veces falle.
- Quitar un paciente de entrenamiento cada vez: PCA-2 y PCA-5 siguen en 1,000 siempre; HVG-2000 baja a 0,829 solo al quitar **LGG1** (el paciente sin metadatos clínicos, 4.8).

**3. La señal existe también dentro de Xie y en sentido inverso**
- **Dentro de Xie `core`** (dejando un paciente fuera, solo 5 pacientes de entrenamiento): macro-F1 y AUROC 1,000 con todas las representaciones. En periferia: AUROC 0,86-1,00 (HVG-2000: 1,000) con F1 0,57-0,91. Con las dos localizaciones juntas: AUROC 0,72-0,94. A nivel de célula esto no se veía (4.10).
- **Inversa** (entrenar con los 6 pacientes `core` de Xie, predecir los 14 de Wälchli): macro-F1 0,85 y AUROC 0,90 con todas las representaciones salvo todos los genes (0,78 y 0,85). El eje GBM/LGG se comparte en las dos direcciones. Con las dos localizaciones de Xie como entrenamiento: AUROC 0,74-0,92 y F1 0,43-0,80 (peor).
- Cautela: 6 pacientes en Xie `core` hacen que un acierto perfecto dentro de Xie (como el externo) tenga p mínimo 0,05; que sea 1,000 también dentro de Xie coincide con que ahí GBM y LGG difieran en edad y tengan IDH distinto (4.11).

**4. Edad: confusión confirmada con las edades individuales de Xie**
| Grupo | Pacientes (edad) |
|---|---|
| LGG (IDH mutado) | P027 (28), P037 (40), P043 (48) |
| GBM (IDH nativo) | P038 (56), P036 (59), P045 (60), P046 (70) |

Resultados en Xie (`xie_age_baseline.csv`; modelo congelado de Wälchli; el modelo de edad es una regresión logística solo de edad entrenada con Wälchli, umbral 0,5):

| Subconjunto | Edad sola (AUROC; p exacto) | Modelo de edad (macro-F1) | Expresión (AUROC; macro-F1 con HVG-2000) |
|---|---|---|---|
| `core` (6 pacientes) | 1,000; p = 0,050 | **1,000** | 1,000; 1,000 |
| Periferia (7) | 1,000; p = 0,029 | **1,000** | 0,917; 0,300 (p = 0,057) |
| Todo, por paciente (7) | 1,000; p = 0,029 | **1,000** | 1,000; 0,708 (todos los genes y PCA-2: 0,300) |

- **La edad separa los grupos de Xie sin solapamiento y el modelo de edad iguala al de expresión en `core` y lo supera en periferia y en el conjunto completo** (macro-F1 1,000 frente a 0,30-0,71). Con los datos de Xie y de Wälchli no se puede decir que la expresión aporte algo que la edad no aporte; en la validación externa completa, la edad predice mejor. **La cohorte de Bejarano (sección 7, punto 5), con edades solapadas, sí apunta a que la expresión aporta más que la edad.**
- En `core`, la probabilidad de GBM de la expresión correlaciona con la edad (Spearman 0,89-0,94), pero con el diagnóstico confundido eso no distingue nada.
- **Wälchli (14 pacientes), AUROC dejando un paciente fuera** (`walchli_age_adjusted.csv`): edad sola 0,833; expresión 0,812-0,917 (todos los genes 0,917, HVG-2000 0,854, PCA-2 0,812, PCA-5 0,812); edad + expresión 0,79-0,83 (sin mejora sobre la edad sola).
- **Zona de edades solapadas** (48-69 años: 5 GBM y **un solo LGG**): la edad rinde AUROC 0,0 y la expresión 0,4-0,8. Con un único LGG esto es anecdótico.
- **Edad dentro de cada diagnóstico (Wälchli):** la correlación entre edad y probabilidad de GBM es positiva en ambos grupos (GBM 0,38-0,52; LGG 0,09-0,26) pero no significativa con n = 6-8 (p = 0,19-0,87); con los 14 juntos es 0,64-0,72 (p ≤ 0,015). **Por tanto no se puede descartar que parte de la predicción siga a la edad ni afirmar lo contrario.**
- Lo único que apunta en contra de una lectura de edad pura: las probabilidades de expresión son extremas (≤ 0,001 frente a ≥ 0,95) y el tejido periférico de un mismo paciente (misma edad) se desplaza hacia LGG; esto último, sin embargo, empeora la expresión en periferia, no la explica la edad. Hace falta una cohorte con edades solapadas para separar ambos efectos.

- **Modelo con la edad quitada de los genes** (`age_adjust`, exploratorio; a cada gen se le quita su pendiente lineal con la edad, ajustada solo con los pacientes de entrenamiento; edades de Wälchli y de Xie individuales):

| Análisis | Representación | Sin corregir (macro-F1 / AUROC) | Edad quitada (macro-F1 / AUROC) |
|---|---|---|---|
| Interno (Wälchli, 14) | PCA-2 | 0,79 / 0,87 | 0,49 / 0,54 |
| Interno | PCA-5 | 0,66 / 0,84 | 0,64 / 0,73 |
| Interno | HVG-2000 | 0,78 / 0,91 | 0,69 / 0,65 |
| Interno | Todos los genes | 0,80 / 0,92 | 0,62 / 0,55 |
| Xie `core` (6) | PCA-2 | 1,00 / 1,00 | 0,83 / 1,00 |
| Xie `core` | PCA-5 | 1,00 / 1,00 | 0,33 / 0,89 |
| Xie `core` | HVG-2000 | 1,00 / 1,00 | 0,89 / 0,96 |
| Xie `core` | Todos los genes | 0,94 / 1,00 | 1,00 / 1,00 |

  **Lectura:** al quitar la tendencia con la edad, **la señal interna casi desaparece** (AUROC 0,54-0,73 frente a 0,84-0,92), coherente con que la expresión no aporta mucho más que la edad en Wälchli (n = 14). En Xie `core` el orden entre pacientes se mantiene (AUROC 0,89-1,00) pero el umbral deja de ser fiable (PCA-5: 0,33). **No es una prueba de que la señal vaya más allá de la edad:** la pendiente se estima donde edad y diagnóstico están confundidos (quita parte del diagnóstico), solo corrige tendencias lineales y con 6 pacientes en Xie el AUROC alto puede conservar restos de edad. En periferia y conjunto completo el modelo corregido empeora (macro-F1 0,30-0,84). Tabla: `age_adjusted_model.csv`.
- **Pendiente de la edad estimada dentro de cada diagnóstico** (`age_adjust_dx`: gen ~ edad + diagnóstico, solo se quita el término de la edad; así no se arrastra el efecto del diagnóstico). Exploratorio; es una de varias variantes probadas, no estaba fijada de antemano:

| Análisis | Representación | Sin corregir (AUROC) | Edad quitada dentro de cada diagnóstico (macro-F1 / AUROC) |
|---|---|---|---|
| Interno (Wälchli, 14) | PCA-2 | 0,87 | 0,71 / 0,71 |
| Interno | PCA-5 | 0,84 | 0,71 / 0,80 |
| Interno | HVG-2000 | 0,91 | 0,78 / 0,91 |
| Interno | Todos los genes | 0,92 | 0,78 / 0,90 |
| Xie `core` (6) | PCA-2, PCA-5, HVG-2000 y todos los genes | 1,00 | 1,00 / 1,00 (las cuatro) |
| Xie completo (7) | PCA-2, PCA-5, HVG-2000 | 1,00 | 1,00 / 1,00 (macro-F1 antes 0,30-0,71) |
| Xie periferia (7) | PCA-5 | 0,83 | 1,00 / 1,00 |

  **Lectura:** estimando la pendiente de la edad dentro de cada diagnóstico, **la señal interna se conserva casi entera** (AUROC 0,71-0,91; con HVG-2000 igual que sin corregir), a diferencia de la versión simple, que la destruye. En Xie se mantiene el `core` y mejoran el conjunto completo y la periferia, probablemente porque quitar la edad reduce parte del desplazamiento entre el núcleo y la periferia. Es un análisis post hoc y no se debe presentar como el modelo final; lo que sí indica es que **la señal de expresión no desaparece al quitar la edad cuando no se quita a la vez el diagnóstico**, y se confirma en Bejarano (punto 5 de la sección 7).

**5. Periferia: es un desplazamiento, no falta de señal**
- Las probabilidades de la periferia quedan **3,9-9,2 unidades de logit por debajo** de las del núcleo del mismo paciente, en el 100 % de los pacientes (todas más cerca de LGG).
- El AUROC en periferia es 0,82-0,92 (100 células) y no mejora con el tamaño. Si se centran las probabilidades sin usar etiquetas (restando su media), el macro-F1 pasa de 0,30 a 0,86 con HVG-2000, todos los genes y PCA-20; con el mejor umbral a posteriori llega a 0,82-0,86 con 100 y 250 células.
- Esto es un diagnóstico a posteriori y transductivo (usa la distribución de los propios datos de prueba), **no una validación**: indica que el orden se conserva y que el tejido periférico desplaza al modelo hacia LGG, compatible con la hipótesis de la sección 5, no la prueba.

**Pocas células por paciente: ¿sirven conjuntos externos con ~50 células endoteliales por paciente?** (`small_packets`, `small_packets_core.csv`)
Entrenamiento como siempre (paquetes de 100 células de Wälchli); evaluación en Xie `core` con **un solo paquete por muestra** de 10, 25 o 50 células, 30 sorteos cada uno:

| Células por paquete | Representación | macro-F1 medio (mínimo) | Sorteos perfectos | AUROC medio (mínimo) |
|---|---|---|---|---|
| 10 | PCA-2 | 0,84 (0,63) | 27 % | 1,00 (1,00) |
| 10 | PCA-5 | 0,86 (0,63) | 37 % | 1,00 (1,00) |
| 10 | HVG-2000 | 0,85 (0,63) | 20 % | 1,00 (0,89) |
| 25 | PCA-2 | 0,97 (0,83) | 83 % | 1,00 (1,00) |
| 25 | PCA-5 | 0,98 (0,83) | 87 % | 1,00 (1,00) |
| 25 | HVG-2000 | 0,91 (0,83) | 47 % | 1,00 (1,00) |
| 50 | PCA-2 | 0,99 (0,83) | 93 % | 1,00 (1,00) |
| 50 | PCA-5 | 0,98 (0,83) | 90 % | 1,00 (1,00) |
| 50 | HVG-2000 | 0,96 (0,83) | 77 % | 1,00 (1,00) |

Con 25-50 células por paquete **el orden entre pacientes se conserva en los 30 sorteos (AUROC 1,00)** y el macro-F1 es 0,91-0,99; con 10 células empieza a fallar el umbral (0,84-0,86). El mínimo de 0,83 corresponde a un paciente mal clasificado en un sorteo. Es decir, un conjunto externo con ~50 células por paciente es utilizable, sobre todo con PCA. Cautelas: aquí las células salen de Xie, que está enriquecido para endotelio (un conjunto sin enriquecer tendrá más células de otros tipos), y no resuelve la confusión con la edad; hay que conocer las edades de esos pacientes.

**6. Resto de análisis**
- **Estados endoteliales** (solo Wälchli, paquetes de 50 células de un único estado): la señal aparece en varios estados, mejor en *Angiogenic capillary* (macro-F1 0,85-0,93, AUROC 0,90-0,95) y débil en *Large vein*, *Vein* y *Venule*. *Mitochondrial* sale 1,000 con 6 pacientes: es un estado de calidad dudosa y muy poco fiable. Equilibrado por estados (interno, 100 células): PCA-5 0,921, PCA-2 0,795, HVG-2000 0,764.
- **Convergencia de réplicas:** con 10 paquetes de 100 células la probabilidad media de un paciente ya cambia de clase en < 1,3 % de los remuestreos (internas) y 0 % en `core`, incluso con 1 paquete. Entre el 97 y el 99 % de la varianza está entre pacientes (ANOVA, no es un modelo mixto), así que más paquetes por paciente no añaden información; solo más pacientes.
- **Holm** entre las 7 permutaciones de la fase 1: PCA-2 (100 células) p = 0,047; las demás ≥ 0,136.
- **Xie completo y fracciones:** con paquetes definidos como fracción de las células, Xie completo llega a 0,857 (PCA-5 y HVG-2000). No se interpreta: la fracción da paquetes de tamaños muy distintos según la muestra (465 células frente a 25.000), y el tamaño del paquete ya se vio que cambia el resultado en Xie completo.

**Segunda tanda (filtros duros de Xie, hiperparámetros, edad dentro de cada diagnóstico)**
- **Filtro duro de endotelio.** Nuestra selección tiene 106.399 células «endoteliales» frente a las 57.324 publicadas. Quedándonos en cada muestra con el X % de células con mayor puntuación endotelial (puntuación endotelial menos la mayor de las contaminantes):

| Células de Xie conservadas | Células | `core` PCA-2 | `core` PCA-5 | `core` HVG-2000 |
|---|---|---|---|---|
| 80 % más endotelial | 85.114 | 1,000 | 1,000 | 1,000 |
| **53,9 % (= recuento publicado)** | **57.318** | **1,000** | **1,000** | **1,000** |
| 25 % más endotelial | 26.597 | 1,000 | 1,000 | 0,943 |

  El resultado `core` no depende de las células menos claramente endoteliales: con el mismo número de células que el artículo sigue separando sin error. (Es el mismo número, no el mismo conjunto de células: no tenemos su anotación.) Combinado con genes comunes y misma profundidad: 1,000.
- **Normalización alternativa (`norm_deseq_vst`):** aproximación en Python de la normalización de DESeq2 (factores de tamaño por mediana de cocientes frente a la media geométrica de los pseudobulks de entrenamiento y transformación estabilizadora de varianza con la tendencia de dispersión ajustada en entrenamiento y congelada), en lugar de TMM + log-CPM. Propuesta a raíz de MLSeq (paquete de R de clasificación de RNA-seq que ofrece DESeq/VST y TMM/voom); a diferencia de su función `classify()`, que normaliza todo el conjunto de una vez, aquí la normalización se ajusta solo con el entrenamiento. **Resultado: `core` 1,000 con PCA-2, PCA-5 y HVG-2000; Xie completo 0,708 y periferia 0,300 (igual que TMM); validación interna HVG-2000 0,802 (TMM 0,782).** La normalización no cambia nada de lo anterior. No es idéntica a DESeq2.
- **Hiperparámetros:** C = 0,01 y C = 100 en la regresión: 1,000 en las tres representaciones. 5.000 genes variables: 1,000. **500 genes variables: PCA-2 cae a 0,333**, PCA-5 sigue en 1,000 (con 500 genes el HVG no entra en la tabla porque se llama HVG-500). PCA-2, con solo dos componentes, es la representación frágil en todos los controles.
- **Edad dentro de cada diagnóstico (Wälchli).** Correlación de Spearman entre la edad y la probabilidad de GBM que da el modelo en validación interna:

| Representación | GBM (n = 8) | LGG (n = 6) | todos (n = 14) |
|---|---|---|---|
| HVG-2000 | 0,52 (p = 0,19) | 0,20 (p = 0,70) | 0,69 (p = 0,006) |
| PCA-2 | 0,38 (p = 0,35) | 0,26 (p = 0,62) | 0,72 (p = 0,004) |
| PCA-5 | 0,46 (p = 0,26) | 0,09 (p = 0,87) | 0,64 (p = 0,015) |

  La correlación es positiva dentro de los dos diagnósticos, ninguna es significativa (con 8 y 6 pacientes no hay potencia) y las estimaciones puntuales no son pequeñas (hasta 0,5 en GBM). **No se puede decir ni que la expresión recoja edad ni que no la recoja**; con estos datos la edad sigue sin descartarse. Tabla por paciente en `walchli_patient_table.csv`: los dos GBM peor clasificados (GBM8 de 48 años y GBM7 de 50, probabilidad 0,28 y 0,51 con HVG-2000) son además los de **menor profundidad** (mediana de 1.364 y 1.114 UMI por célula frente a 3.100-9.000 en los demás) y de los más jóvenes; LGG1 (69 años) queda en 0,45; y LGG6 (31 años) se clasifica como GBM (0,92 con HVG-2000) pese a ser joven. Los errores se solapan con la edad y con la profundidad, no se pueden separar con 14 pacientes.

**Resumen de la fase 2:** el resultado `core` es robusto a las decisiones técnicas probadas y a qué pacientes de Wälchli se usan para entrenar, y la señal también aparece dentro de Xie y al revés. Sigue sin poderse separar de la edad y de diferencias reales entre cohortes (6 pacientes en Xie `core`, medias de edad solamente, estudio y diagnóstico confundidos), por lo que es un resultado prometedor y consistente, no una prueba.

### 4.13 Pseudobulks promediados (`experiments.17_pseudobulk`, secundario; `results/pseudobulk`)
Promedio de la expresión log-normalizada en paquetes **disjuntos** de 25, 50 y 100 células (5 particiones aleatorias), con y sin regresión de profundidad, regresión logística y random forest, 667 pseudobulks de Wälchli y 1.057 de Xie. Es una agregación distinta de la de conteos crudos (4.11) y sirve de comprobación independiente.
- **Xie `core`, sin regresión de profundidad, 100 células, regresión logística (macro-F1):** PCA-5 0,992; PCA-10 0,986; PCA-20 0,980; todos los genes 0,963; HVG-2000 0,953; PCA-2 0,898. Con random forest, HVG-2000 0,998 y todos los genes 0,998 (las PCA con random forest fallan: 0,66-0,74). La separación del `core` se reproduce con otra agregación, otro tipo de paquete y otro modelo. Con 25 y 50 células es parecida (0,86-0,99).
- **Con regresión de profundidad el `core` cae mucho:** HVG-2000 0,775; todos los genes 0,697; PCA-2 0,468; PCA-10 0,775; solo PCA-5 se mantiene (0,906). Con random forest también (HVG-2000 0,809). Es la única manipulación que rompe el resultado, y apunta a que parte de la señal está entrelazada con la profundidad o con la cantidad de genes detectados. Pero en el análisis de conteos, igualar la profundidad adelgazando los conteos (`thin_auto`, 4.12) lo deja en 1,000. Ambas cosas no son equivalentes (la regresión quita también la parte de expresión que varía con la profundidad), y no se han reconciliado: queda como punto abierto.
- **Xie completo y periferia:** como siempre, mal (todo: macro-F1 0,50-0,80, mejor PCA-5 0,80; periferia: 0,30-0,71).
- **Dentro de Wälchli (dejar un paciente fuera):** macro-F1 0,71-0,80 (la regresión logística con PCA-20 llega a 0,794; random forest con HVG-2000, 0,774).
- **Efecto de estudio:** el estudio se identifica con balanced accuracy de 0,89 (regresión logística) a 0,95 (random forest), 0,96 sin regresión de profundidad.

<!-- PC_GENES_START -->
### 4.14 Qué genes componen las componentes principales (para interpretar el modelo)

Generado por `experiments/24_pc_genes.py` (modelo de pseudobulks ajustado con los 14 pacientes de Wälchli, paquetes de 100 células, 2.000 genes más variables). **PCA-r usa las r primeras componentes de la misma PCA** (PCA-2 = PC1-PC2, PCA-5 = PC1-PC5, etc.). Cada componente está orientada para que una puntuación alta signifique «más GBM». El peso de un gen indica cuánto contribuye a la componente (sobre el gen escalado): **no es una causa biológica**, y una componente puede mezclar tipo de glioma, edad y estudio. Las tablas completas (50 genes por sentido y componente hasta PC20) están en `tablas/pc_gene_loadings_top.csv`.

**Resumen de las componentes** (AUROC: capacidad de la puntuación del paciente para separar GBM de LGG; edad: correlación de Spearman con la edad; estabilidad: solapamiento de los 30 genes principales entre semillas, 1 = idéntico):

| componente | varianza_% | auroc_gbm_lgg | spearman_edad | estabilidad |
|---|---|---|---|---|
| PC1 | 30.512 | 0.542 | 0.370 | 0.672 |
| PC2 | 17.416 | 0.979 | 0.609 | 0.765 |
| PC3 | 13.070 | 0.458 | 0.196 | 0.668 |
| PC4 | 11.794 | 0.417 | -0.145 | 0.627 |
| PC5 | 4.616 | 0.479 | 0.334 | 0.652 |
| PC6 | 2.769 | 0.750 | 0.189 | 0.717 |
| PC7 | 1.892 | 0.604 | 0.403 | 0.515 |
| PC8 | 1.431 | 0.521 | 0.187 | 0.354 |
| PC9 | 1.325 | 0.625 | 0.134 | 0.454 |
| PC10 | 1.194 | 0.646 | 0.011 | 0.653 |
| PC11 | 1.039 | 0.500 | -0.359 | 0.580 |
| PC12 | 0.772 | 0.479 | 0.086 | 0.402 |
| PC13 | 0.701 | 0.583 | 0.123 | 0.513 |
| PC14 | 0.235 | 0.479 | 0.121 | 0.351 |
| PC15 | 0.193 | 0.500 | -0.279 | 0.204 |
| PC16 | 0.187 | 0.521 | -0.048 | 0.041 |
| PC17 | 0.179 | 0.625 | 0.583 | 0.023 |
| PC18 | 0.176 | 0.562 | 0.185 | 0.029 |
| PC19 | 0.170 | 0.521 | 0.108 | 0.011 |
| PC20 | 0.166 | 0.604 | -0.108 | 0.029 |

**PC1** (30.5 % de varianza; AUROC GBM/LGG 0.54; edad +0.37)
- Hacia GBM: RPS17, C6orf48, METTL12, ATP5EP2, LINC00116, SLC25A6, HINT2, C10orf10, RARRES3, WRB, C5orf56, SUPT4H1
- Hacia LGG: ARMCX6, SPATA13, KLHL11, ZNF805, SEPTIN11, TENT5A, SEPTIN9, SEPTIN7, SPTLC1, ATP5MF, TUT7, CPLANE1

**PC2** (17.4 % de varianza; AUROC GBM/LGG 0.98; edad +0.61)
- Hacia GBM: PCDH17, HTRA1, EIF4EBP1, HRH1, MMP2, GABRD, KLHDC8A, THY1, CD276, PGM2L1, DUSP4, SCAMP5
- Hacia LGG: NET1, USP54, MT1M, ADAMTS8, SLCO1A2, AIF1L, USP53, ZNF385B, HLF, BTG2, PDCD4, ST6GAL1

**PC3** (13.1 % de varianza; AUROC GBM/LGG 0.46; edad +0.20)
- Hacia GBM: C14orf37, KIAA1147, AK1, C11orf84, C6orf106, SDCCAG3, PQLC1, AL136164.4, FAM129A, DIRC2, KIAA1551, MATR3-ENSG00000015479
- Hacia LGG: AC093323.1, ARRDC3, NFKBIZ, AP001528.2, CLU, HSPB1, MT-RNR1, MT-RNR2, HNRNPA1P48, BHLHE40, LGALS3, GLUL

**PC4** (11.8 % de varianza; AUROC GBM/LGG 0.42; edad -0.15)
- Hacia GBM: FZD3, ATP1B2, SCRN1, DENND2A, ACSS3, MT-ND6, KCNE4, LZTS1, AGT, SMOC2, KCNB1, PLEKHA4
- Hacia LGG: GSN, CALCRL, THBD, CXCL12, PIK3R3, ITPRIP, SGK1, ADAMTS1, FAM241A, LRRC32, DUSP5, RAMP3

**PC5** (4.6 % de varianza; AUROC GBM/LGG 0.48; edad +0.33)
- Hacia GBM: DPEP1, TNFRSF21, APLN, VASH1, SYT1, IGLON5, TMEM37, QPCT, NID2, MMP25, PGF, CA2
- Hacia LGG: PLAC9, ARHGAP20, ARHGAP10, NPY1R, INPP4B, TXNIP, RUNX1T1, ARID5B, INMT, SULF2, LITAF, PDE8B

**Genes que más empujan cada modelo** (coeficiente de la regresión logística llevado al espacio de genes; positivo = hacia GBM):

- **PCA-2** hacia GBM: EIF4EBP1, PCDH17, PTP4A3, HTRA1, HRH1, KLHDC8A, TIMP1, MMP2, SCAMP5, CD276, THY1, FOLH1, OAF, TMED3, GABRD
- **PCA-2** hacia LGG: USP54, ADAMTS8, NET1, MT1M, SYNE1, ZNF385B, ARRDC2, TACC1, BTG2, TIPARP, USP53, PDK4, IGF1R, RHBDL3, GADD45G
- **PCA-5** hacia GBM: EIF4EBP1, SEZ6L2, PTTG1, NEK6, TUBB2B, THY1, CPVL, HRH1, SCG5, TIMP1, KLHDC8A, DOK5, ETV1, FABP7, LYPD1
- **PCA-5** hacia LGG: CAVIN2, TSC22D1, ARRDC2, TACC1, BTG2, ST8SIA6, IFNGR1, GRASP, NET1, IRAK3, ADAMTS8, HERPUD1, PDCD4, USP54, PDK4

Para el análisis de enriquecimiento (rutas, tipos celulares) se pueden usar directamente las listas de `tablas/pc_gene_loadings_top.csv` y `tablas/classifier_gene_drivers.csv`.

<!-- PC_GENES_END -->

**Figuras de 4.14** (`results/pc_genes/`, cada una con su descripción en `descriptions/`): `24_pc_variance` (varianza, AUROC, edad y estabilidad por componente), `24_pc_scores_heatmap` (puntuación de cada paciente en PC1-PC10), `24_pc_top_genes` (genes con más peso en PC1-PC4), `24_pc_scatter` y `24_pc_vs_age` (pacientes en el plano de las componentes y frente a la edad), `24_classifier_drivers` (genes que empujan cada modelo), `24_gene_overlap_heatmap` (solapamiento entre listas) y `24_marker_share` (marcadores celulares por componente). Tablas: `tablas/pc_summary.csv`, `tablas/pc_gene_loadings_top.csv`, `tablas/classifier_gene_drivers.csv`.

### 4.15 ¿Hay pacientes comunes entre Wälchli y Bejarano?

**Supuesto de trabajo:** mientras los autores no confirmen lo contrario, **se asume que no hay pacientes comunes** entre los 14 de Wälchli y los 18 de Bejarano con RNA-seq en masa, y todos los resultados con Bejarano (sección 7, punto 5; 4.17) se leen con esa condición. Los identificadores no coinciden y el artículo no dice nada; son del mismo hospital, así que no se puede excluir. Si hubiera solapamiento, la validación en Bejarano dejaría de ser independiente para esos pacientes y habría que quitarlos de la validación (con la tabla de abajo se sabría cuáles).

<!-- OVERLAP_START -->
**Comprobación de solapamiento de pacientes con Wälchli** (`experiments/25_overlap_check.py`; `tablas/overlap_*.csv`; `results/overlap_check/25_overlap_check.png`)

Se comparan clase (GBM frente a IDH nativo, LGG frente a IDH mutado), sexo y edad (tolerancia de 1 año) entre los 14 pacientes de Wälchli y los 18 de Bejarano con RNA-seq en masa. **Parejas compatibles: 5; esperadas por azar: 5.0 (p = 1.00).** 9 de los 14 pacientes de Wälchli no tienen ninguna pareja compatible, así que no pueden estar repetidos.

No prueba que haya o no solapamiento (con 14 y 18 pacientes hay parejas compatibles por azar); señala a quién preguntar:

| paciente_walchli | paciente_bejarano | clase | sexo | edad_walchli | edad_bejarano | dif_edad |
|---|---|---|---|---|---|---|
| GBM1 | GBM123 | GBM | M | 71.000 | 70 | 1.000 |
| GBM6 | GBM149 | GBM | M | 60.000 | 59 | 1.000 |
| GBM8 | GBM159 | GBM | M | 48.000 | 49 | 1.000 |
| LGG1 | GBM144 | LGG | F | 69.000 | 70 | 1.000 |
| LGG2 | GBM141 | LGG | M | 46.000 | 47 | 1.000 |
<!-- OVERLAP_END -->

### 4.16 ¿Cuántos genes hacen falta? (pregunta central)

Se repite el modelo variando cuántos genes variables se le dan (10 a 2.000) y se mide el rendimiento por paciente en tres sitios: dentro de Wälchli, en Xie `core` y en Bejarano. Responde a la pregunta del proyecto con una curva en lugar de tres puntos (PCA-2, HVG-2000, todos los genes).

<!-- GENE_BUDGET_START -->
**Cuántos genes hacen falta** (`experiments/27_gene_budget.py`; `tablas/gene_budget_summary.csv`; figuras `results/gene_budget/27_gene_budget_*.png`). AUROC por paciente (media de las semillas) con n genes variables:

*Wälchli (dejar un paciente fuera)* (14 pacientes):

| n genes | HVG-n | PCA-2 | PCA-5 | PCA-10 |
|---|---|---|---|---|
| 10 | 0.340 | 0.500 | 0.340 | -- |
| 25 | 0.576 | 0.500 | 0.583 | 0.576 |
| 50 | 0.611 | 0.528 | 0.528 | 0.611 |
| 100 | 0.743 | 0.590 | 0.708 | 0.743 |
| 250 | 0.903 | 0.625 | 0.854 | 0.889 |
| 500 | 0.917 | 0.458 | 0.882 | 0.917 |
| 1000 | 0.917 | 0.674 | 0.875 | 0.931 |
| 2000 | 0.910 | 0.868 | 0.840 | 0.896 |

*Xie core (6 pacientes)* (6 pacientes):

| n genes | HVG-n | PCA-2 | PCA-5 | PCA-10 |
|---|---|---|---|---|
| 10 | 0.556 | 0.852 | 0.926 | -- |
| 25 | 0.556 | 0.556 | 0.667 | 0.593 |
| 50 | 0.963 | 0.630 | 0.926 | 0.963 |
| 100 | 1.000 | 0.259 | 1.000 | 1.000 |
| 250 | 1.000 | 0.148 | 1.000 | 1.000 |
| 500 | 1.000 | 0.407 | 1.000 | 1.000 |
| 1000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2000 | 1.000 | 1.000 | 1.000 | 1.000 |

*Bejarano (18, RNA-seq en masa)* (18 pacientes):

| n genes | HVG-n | PCA-2 | PCA-5 | PCA-10 |
|---|---|---|---|---|
| 10 | 0.567 | 0.429 | 0.433 | -- |
| 25 | 0.667 | 0.442 | 0.628 | 0.610 |
| 50 | 0.970 | 0.602 | 0.957 | 0.965 |
| 100 | 0.961 | 0.411 | 0.996 | 0.961 |
| 250 | 0.991 | 0.225 | 1.000 | 0.991 |
| 500 | 1.000 | 0.558 | 1.000 | 1.000 |
| 1000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2000 | 1.000 | 1.000 | 1.000 | 1.000 |

Lectura: ver la figura; es un barrido descriptivo (3 semillas de paquetes, 14 pacientes de entrenamiento), sin intervalos de confianza entre pacientes.
<!-- GENE_BUDGET_END -->

### 4.17 Transferencia entre cohortes (incluye entrenar con Bejarano)

Hasta ahora siempre se entrenaba con Wälchli. Aquí se prueban **todas las direcciones**: entrenar con Wälchli, con Xie `core` o con Bejarano y predecir las otras cohortes. Si el eje GBM/LGG es real y compartido, el AUROC debería mantenerse en todas las direcciones aunque el umbral se desplace; si lo que aprende el modelo es una diferencia de cohorte, solo funcionará dentro de la propia cohorte. Entrenar con Bejarano es especialmente informativo porque tiene más pacientes y edades solapadas.

<!-- TRANSFER_START -->
**Matriz de transferencia entre cohortes** (`experiments/26_transfer_matrix.py`; `tablas/transfer_matrix.csv`; figuras `results/transfer_matrix/26_*.png`). Se entrena con la cohorte de cada fila y se predice la de cada columna, sin reajustar; la diagonal es dejar un paciente fuera dentro de la cohorte.

| entrena | prueba | representacion | n_pacientes | auroc | p_exacto | macro_f1_corte_05 | macro_f1_centrado | tipo |
|---|---|---|---|---|---|---|---|---|
| Bejarano | Bejarano | HVG | 18 | 1.000 | 0.000 | 0.833 | 0.833 | dentro de la cohorte (dejar un paciente fuera) |
| Bejarano | Wälchli | HVG | 14 | 0.938 | 0.002 | 0.785 | 0.854 | entre cohortes |
| Bejarano | Xie core | HVG | 6 | 1.000 | 0.050 | 1.000 | 1.000 | entre cohortes |
| Bejarano | Xie periferia | HVG | 7 | 1.000 | 0.029 | 0.300 | 0.708 | entre cohortes |
| Wälchli | Bejarano | HVG | 18 | 1.000 | 0.000 | 0.833 | 0.888 | entre cohortes |
| Wälchli | Wälchli | HVG | 14 | 0.917 | 0.004 | 0.785 | 0.775 | dentro de la cohorte (dejar un paciente fuera) |
| Wälchli | Xie core | HVG | 6 | 1.000 | 0.050 | 1.000 | 1.000 | entre cohortes |
| Wälchli | Xie periferia | HVG | 7 | 0.917 | 0.057 | 0.300 | 0.857 | entre cohortes |
| Xie core | Bejarano | HVG | 18 | 1.000 | 0.000 | 1.000 | 0.833 | entre cohortes |
| Xie core | Wälchli | HVG | 14 | 0.896 | 0.006 | 0.854 | 0.785 | entre cohortes |
| Xie core | Xie core | HVG | 6 | 1.000 | 0.050 | 1.000 | 1.000 | dentro de la cohorte (dejar un paciente fuera) |
| Bejarano | Bejarano | PCA-5 | 18 | 0.935 | 0.001 | 0.833 | 0.833 | dentro de la cohorte (dejar un paciente fuera) |
| Bejarano | Wälchli | PCA-5 | 14 | 0.917 | 0.004 | 0.854 | 0.854 | entre cohortes |
| Bejarano | Xie core | PCA-5 | 6 | 1.000 | 0.050 | 1.000 | 1.000 | entre cohortes |
| Bejarano | Xie periferia | PCA-5 | 7 | 0.917 | 0.057 | 0.300 | 0.857 | entre cohortes |
| Wälchli | Bejarano | PCA-5 | 18 | 1.000 | 0.000 | 0.778 | 0.833 | entre cohortes |
| Wälchli | Wälchli | PCA-5 | 14 | 0.833 | 0.021 | 0.641 | 0.708 | dentro de la cohorte (dejar un paciente fuera) |
| Wälchli | Xie core | PCA-5 | 6 | 1.000 | 0.050 | 1.000 | 1.000 | entre cohortes |
| Wälchli | Xie periferia | PCA-5 | 7 | 0.833 | 0.114 | 0.300 | 0.571 | entre cohortes |
| Xie core | Bejarano | PCA-5 | 18 | 1.000 | 0.000 | 1.000 | 0.833 | entre cohortes |
| Xie core | Wälchli | PCA-5 | 14 | 0.896 | 0.006 | 0.854 | 0.785 | entre cohortes |
| Xie core | Xie core | PCA-5 | 6 | 1.000 | 0.050 | 1.000 | 1.000 | dentro de la cohorte (dejar un paciente fuera) |

Cautelas: Xie core tiene la edad confundida con el diagnóstico; Bejarano es RNA-seq en masa; 6-18 pacientes por cohorte (p exacto mínimo limitado por el nº de pacientes).
<!-- TRANSFER_END -->


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
- **Edad en Xie:** con las edades individuales (Tabla S1A) se ve que **la edad separa perfectamente LGG y GBM** (28-48 frente a 56-70), así que edad, IDH y tipo de glioma están confundidos en la validación externa; con 6-7 pacientes no se puede ajustar por edad (4.12, punto 4).
- **Pseudobulks:** el resultado `core` descansa en 6 pacientes (p mínimo exacto 0,05). La permutación y el bootstrap usan la semilla 0 mientras que las tablas promedian 5. Los pseudobulks de réplicas del mismo paciente no son independientes. Se probaron varios tamaños y representaciones sin corrección entre todas las configuraciones (Holm solo sobre las 7 permutaciones, 4.12).
- **Etiqueta confundida con el paciente:** `PATH`, `age` y `Patient` determinan `glioma_type` por sí solas; la separación es entre pacientes, no entre células.

---

## 7. Pendiente

Hecho: pseudobulks de conteos, fase 1 (4.11), análisis `core` (4.7), periferia (4.7), revisión de C (4.4), PC1/PC2 y edad (4.8), bootstrap (4.9), alineación de medias, su variante sin las células del paciente evaluado, y dentro de Xie (4.10).

1. **Metadatos clínicos de LGG1** (subtipo, localización, mutaciones): no están en el conjunto de datos (4.8). Pedirlos a la coautora o buscarlos en el artículo.
2. **Separar edad y tipo de glioma:** con estos 14 pacientes no se puede (4.8) y en Xie tampoco (4.12, punto 4); haría falta una cohorte con edades solapadas entre GBM y LGG (Bejarano lo es: punto 5).
3. **Pseudobulks de conteos: fases 1 y 2 hechas (4.11, 4.12).** Hecho: edades individuales de Xie (la edad sola separa a los pacientes y supera a la expresión en periferia y conjunto completo, 4.12 punto 4), variantes `xie_pub` y `norm_deseq_vst`. Pendiente inmediato: comprobar por qué LGG1 es el paciente que más influye (4.12).
   **De la guía:** normalización DESeq/VST ya implementada en Python y ejecutada (`norm_deseq_vst`; inspirada en el paquete de R MLSeq, que normaliza todo el conjunto a la vez, mientras que aquí se ajusta solo con el entrenamiento); aún no implementado: mediana de expresión y muestreos estratificados; componentes de varianza con modelo mixto (4.12 usa ANOVA); anotación endotelial publicada de Xie (no está en los datos descargados; `xie_pub` solo iguala el número de células). `17_pseudobulk` (promedio de log-expresión) ya está ejecutado (4.13); queda abierto reconciliar por qué la regresión de profundidad lo rompe y el adelgazamiento de conteos no.
4. **Hechos en la v30 (resultados en 4.12):** paquetes pequeños en Xie (`small_packets`: 25-50 células por paciente bastan para conservar el orden, AUROC 1,00) y modelo con la edad quitada de los genes (`age_adjust`: la señal interna casi desaparece, AUROC 0,54-0,73; en Xie `core` se conserva el orden pero no el umbral). Sin resolver la confusión con la edad.
   Hecho también en la v32: `age_adjust_dx` (pendiente de la edad dentro de cada diagnóstico; resultados en 4.12, punto 4).
   **Tablas:** la carpeta `tablas/` del repositorio guarda copias pequeñas de las tablas de resultados (CSV y resúmenes) citadas en este README; los resultados completos (`results/`) no se suben.
5. **Otro conjunto independiente: Bejarano et al. (2025)** (*Single-cell atlas of endothelial and mural cells across primary and metastatic brain tumors*, Immunity 58, 1015-1032.e6; DOI 10.1016/j.immuni.2025.02.022; fuente: texto de Resultados, tabla "Key resources" y "Data and code availability" del PDF).

| Parte del artículo | Qué es | ¿Sirve para validar? |
|---|---|---|
| **scRNA-seq** | Solo **4 pacientes** nuevos; integra datos publicados de glioma con IDH mutado de Wälchli et al. (accesos `GSE256493; GSE242044`, fila "Deposited data" de la tabla de recursos) | **No:** reutiliza nuestros dos conjuntos (al menos la parte de IDH mutado) y aporta muy pocos pacientes |
| **RNA-seq en masa de células endoteliales ordenadas por FACS (CD31+)** | **Cohorte nueva de 19 pacientes: 7 con glioma IDH mutado (todos grado II, salvo un oligodendroglioma grado III) y 12 con GBM IDH nativo** (Resultados, Tabla S1); tejido del hospital universitario de Lausana (CHUV) | **Posible tercera cohorte**, con 19 pacientes (permutación exacta: 50.388 asignaciones, p mínimo ≈ 2·10⁻⁵ frente al 0,05 de Xie `core`) |

Datos: portal del laboratorio (`joycelab.shinyapps.io/braintime`; "RNA-seq and scRNA-seq count data can be visualized and downloaded"), no hay acceso en GEO; los datos crudos solo bajo petición al autor correspondiente. Las edades y los sexos están en la Tabla S1; el artículo dice que la edad no difiere entre los grupos tumorales (sí entre tumor y no tumor, que son jóvenes) y que incluir edad y sexo como covariables no cambió la expresión diferencial, pero eso es expresión diferencial, no clasificación.

**Tabla S1 de Bejarano (revisada, hoja «A» «Seq»):** 23 pacientes con glioma, todos con datos «de este estudio»; **18 tienen RNA-seq en masa de células endoteliales** (7 con IDH mutado y 11 con GBM IDH nativo; un GBM más solo tiene células murales) y 4 tienen scRNA-seq (otros pacientes, sin RNA-seq en masa).

| Grupo (RNA-seq en masa de células endoteliales) | n | Edades (años) | Media |
|---|---|---|---|
| IDH mutado (astrocitoma/oligodendroglioma grado 2-3) | 7 | 36, 40, 47, 49, 51, 70, 73 | 52,3 |
| GBM IDH nativo (grado 4) | 11 | 49, 54, 59, 64, 64, 65, 66, 70, 73, 76, 86 | 66,0 |

- **La confusión con la edad es mucho menor que en Xie:** las edades se **solapan** (49-73 años: 4 pacientes con IDH mutado y 9 GBM). Un modelo solo de edad tendría AUROC 0,77 (Mann-Whitney unilateral p = 0,031), frente a 1,00 en Xie y 0,83 en Wälchli. Es justo la cohorte con edades solapadas que nos faltaba.
- Hay 8 mujeres y 10 hombres; los sexos están repartidos (IDH mutado 3 M/4 H, GBM 4 M/7 H).
- Los identificadores de paciente (`GBM116`, ...) no coinciden con los nuestros; no se puede descartar solapamiento con pacientes de Wälchli (mismo hospital y grupo) sin comparar edad, sexo y diagnóstico.

**Resultados** (`experiments/23_bejarano_bulk.py`; tablas en `tablas/bejarano_*.csv`). Conteos crudos y metadatos descargados del portal (pestaña «Bejarano et al., Immunity 2025», Download, «Human data»; 78 muestras CD31+ y PDGFRB+ de 18.874 genes; 18 CD31+ de glioma: 7 IDH mutado y 11 GBM, que coinciden con la Tabla S1; 16.810 de los 26.666 genes de Wälchli presentes). Se entrena con los 14 pacientes de Wälchli (paquetes de 100 células, 3 semillas) y se predice cada muestra en masa sin reajustar nada.

| Modelo | AUROC | p exacto | macro-F1 (corte 0,5) | sensibilidad / especificidad | macro-F1 centrado | AUROC en la zona de edades solapadas (49-73, n = 13) |
|---|---|---|---|---|---|---|
| Expresión, PCA-2 | 1,000 | < 0,0001 | 0,78 | 0,64 / 1,00 | 0,83 | 1,000 |
| Expresión, PCA-5 | 1,000 | < 0,0001 | 0,78 | 0,64 / 1,00 | 0,83 | 1,000 |
| Expresión, PCA-10 y PCA-20 | 1,000 | < 0,0001 | 0,83 | 0,73 / 1,00 | 0,89 | 1,000 |
| Expresión, HVG-2000 | 1,000 | < 0,0001 | 0,83 | 0,73 / 1,00 | 0,89 | 1,000 |
| Expresión, todos los genes | 1,000 | < 0,0001 | 0,72 | 0,55 / 1,00 | 0,94 | 1,000 |
| Edad quitada dentro de cada diagnóstico (HVG-2000, PCA-10/20) | 0,974 | < 0,0001 | 0,83 | 0,73 / 1,00 | 0,89 | 0,944 |
| Edad quitada de los genes (versión simple, HVG-2000) | 0,935 | 0,001 | 0,72 | 0,55 / 1,00 | 0,94 | 0,944 |
| **Edad sola (referencia)** | **0,773** | **0,030** | -- | -- | -- | **0,514** |
| Control: dejar un paciente fuera dentro de Bejarano (HVG-2000, PCA-10) | 1,000 | < 0,0001 | 0,83 | -- | -- | -- |

(El p exacto mínimo con 18 muestras es 1/31.824 ≈ 3·10⁻⁵.)

**Lectura:**
- **Es la prueba más fuerte hasta ahora de que la expresión aporta más que la edad.** En la zona de edades solapadas (4 pacientes con IDH mutado y 9 GBM de 49 a 73 años) la edad no discrimina (AUROC 0,51) y la expresión ordena a los 13 sin error (1,000); la probabilidad de que una ordenación al azar separe así 4 y 9 pacientes es 1/715 ≈ 0,14 %.
- **El modelo entrenado solo con scRNA-seq de Wälchli ordena perfectamente muestras en masa de otro estudio y otra tecnología**, con 18 pacientes y un p exacto de ~3·10⁻⁵ como mínimo, frente al 0,05 que impone Xie. El control interno de Bejarano (AUROC 0,83-1,00) confirma que la señal existe también dentro de esa cohorte.
- **El umbral sí se desplaza:** la especificidad es 1,00 y la sensibilidad 0,55-0,73 (el modelo llama IDH mutado a parte de los GBM, la misma tendencia que en la periferia de Xie). Con las probabilidades centradas, a posteriori, el macro-F1 sube a 0,83-0,94. El orden se conserva, la calibración no.
- **Con la edad quitada** (pendiente dentro de cada diagnóstico) el AUROC sigue en 0,93-0,97 y en la zona de edades solapadas 0,89-0,94: no hay indicios de que la señal sea solo edad.

**Cautelas que siguen vigentes:**
1. **Posible solapamiento de pacientes con Wälchli** (mismo hospital y grupo): no se puede descartar sin sexo, fecha y otros datos de Wälchli. Con las edades solo hay 8 coincidencias de ±1 año dentro de la misma clase, lo que es lo esperado por azar (media bajo la hipótesis nula 8,3; p = 0,62), así que no hay indicio de solapamiento, pero tampoco se excluye. Hay que preguntar a los autores.
2. **Tecnología distinta** (RNA-seq en masa de células ordenadas frente a scRNA-seq de 10x) y probablemente efecto de estudio: lo observado es el orden, no un umbral transferible.
3. **IDH mutado frente a GBM, no exactamente «LGG frente a GBM»:** hay astrocitomas y oligodendrogliomas de grado 2 y un oligodendroglioma de grado 3; el artículo no menciona gliomas de bajo grado con IDH nativo.
4. **18 pacientes**, de nuevo pocos; una cohorte más reforzaría.
5. La edad sigue sin estar totalmente equilibrada: por debajo de 49 años solo hay IDH mutado y por encima de 73 solo GBM; la edad sola da 0,77.

**Siguiente paso:** la transferencia inversa (entrenar con Bejarano, predecir Wälchli y Xie) y las demás direcciones están en 4.17 (`experiments/26_transfer_matrix.py`); la comprobación de pacientes comunes, en 4.15; y la curva de cuántos genes hacen falta, en 4.16. Figuras de este apartado: `results/bejarano_bulk/23_bejarano_probability_vs_age.png`, `23_bejarano_auroc_by_mode.png` y `23_bejarano_roc.png` (se regeneran con `--replot`). Si más adelante se quisiera un modelo con las tres cohortes dejando una fuera cada vez, la matriz de 4.17 es el punto de partida.


**Comprobaciones previas (la 1 y la 3 resueltas; la 2 sigue abierta):**
1. Si los **conteos de las muestras en masa son descargables** del portal (`joycelab.shinyapps.io/braintime`) por muestra.
2. Si **algún paciente coincide con los de Wälchli** (comparar edad, sexo y diagnóstico con nuestros 14; LGG1 tiene 69 años y no coincide con ninguno de los IDH mutado de esta tabla, 70 y 73). **Comprobación hecha en 4.15** (con sexo, además de edad y clase); se mantiene el supuesto de que no hay pacientes comunes hasta que lo confirmen los autores.
3. Diferencias de plataforma: RNA-seq en masa de células ordenadas (STAR + RSEM) frente a scRNA-seq 3' de 10x; el efecto de estudio ya es total entre Wälchli y Xie (4.11). Se puede esperar que se conserve el orden y falle el umbral; se vería con el AUROC.

**Pendiente tras la v39:**
- **Ejecutar y revisar** `24_pc_genes`, `25_overlap_check`, `27_gene_budget`, `26_transfer_matrix` y `28_cohort_overview` con los datos reales (sus secciones 4.14-4.17 se rellenan solas al ejecutarlos) y leer los resultados antes de citarlos: son análisis nuevos y no se han interpretado todavía.
- **Preguntar a los autores de Bejarano** si hay pacientes comunes con Wälchli (4.15) y si reutilizan datos de GBM de otros estudios.
- **Biología de las componentes:** pasar las listas de 4.14 por enriquecimiento de rutas y tipos celulares y contrastarlas con lo conocido del endotelio en glioma.
- **Edad de LGG1 y metadatos clínicos** (punto 1) y la discrepancia de la edad media de los LGG de Xie (49,3 en el texto frente a 38,7 en la tabla).

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

# pseudobulks promediados (secundario; ya ejecutado; reanudable; --quick = prueba rápida)
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

# validación con Bejarano (RNA-seq en masa; datos en data/bejarano/) y sus figuras; --replot solo regenera las figuras
.\.venv\Scripts\python.exe -m experiments.23_bejarano_bulk --config configs/full.yaml
.\.venv\Scripts\python.exe -m experiments.23_bejarano_bulk --config configs/full.yaml --replot

# explicabilidad, solapamiento, cuántos genes hacen falta, transferencia entre cohortes y visión de conjunto (rellenan 4.14-4.17 del README)
.\.venv\Scripts\python.exe -m experiments.24_pc_genes --config configs/full.yaml
.\.venv\Scripts\python.exe -m experiments.25_overlap_check --config configs/full.yaml
.\.venv\Scripts\python.exe -m experiments.27_gene_budget --config configs/full.yaml
.\.venv\Scripts\python.exe -m experiments.26_transfer_matrix --config configs/full.yaml
.\.venv\Scripts\python.exe -m experiments.28_cohort_overview --config configs/full.yaml

# diagnóstico de estudio (batch): numérico + gráficas 09, 10, 11
.\.venv\Scripts\python.exe -m experiments.11_batch --config configs/full.yaml --xie data/xie/xie_endothelial.h5ad --bundle models/full_tuned/bundle.joblib --out results/batch
```

Si el estudio externo no trae `nCount_RNA` / `nFeature_RNA` ni conteos crudos, el paquete con regresión de profundidad no se puede aplicar y hay que usar un paquete entrenado sin ella (`configs/full_noreg.yaml`). Los genes ausentes se tratan como 0 y el script avisa si faltan muchos genes variables. Si las etiquetas externas se llaman distinto, usar `--label-map "Glioblastoma=GBM,..."`.

### Archivos de resultados (`results/<modo>/`, no versionados)
`01_cohort.png`, `02_pca_variance.png`, `03_confusion_*.png`, `04_components_vs_performance_*.png`, `05_per_patient_*.png`, `06_components_vs_performance_seeds_*.png`, `07_by_cluster_*.png`, `08_permutation_*.png`, `permutation_results_*.csv`, `results_*.csv`, `summary_*.md`, `summary.json`, `run_metadata.json`, `splits/`, `tuning/` (validación anidada). En `results/batch/`: `09_batch_model_space.png`, `10_batch_joint_pca.png`, `11_batch_domain_classifier.png`, `batch_domain_results.csv/.md`, `batch_domain_per_patient.csv`, `batch_pca_eta2.csv`, `batch_model_space_summary.csv`. Además: `results/pc_meaning/` (`12_pc_by_patient.png`, `patients_metadata_pcs.csv`, `pc_correlations.csv`), `results/bootstrap/` (`13_bootstrap_ci.png`, `bootstrap_ci.csv`), `results/external_xie_core/`, `results/external_xie_aligned/`, `results/within_xie/`, `results/pseudobulk/` (promediados: `17_pseudobulk.png`, `pseudobulk_*_summary.csv/.md`, `runs/`) y `results/pseudobulk_counts/` (conteos: `xie_audit.csv`, `pseudobulk_configuration_grid.csv`, `pseudobulk_seed_sensitivity.csv`, `internal_patient_predictions.csv`, `xie_core_predictions.csv`, `xie_periphery_predictions.csv`, `packet_stability_results.csv/_cohort.csv`, `pseudobulk_coverage_matrix.csv`, `permutation_results.csv`, `patient_bootstrap_results.csv`, `study_/site_classifier_results.csv`, `age_only_results.csv`, `composition_only_results.csv`, `comparison_cell_vs_pseudobulk.csv/.md`, figuras `21_*.png`, `run_metadata.json`).

Carpetas nuevas (v39): `results/bejarano_bulk/` (`23_*.png`), `results/pc_genes/` (`24_*.png`, `pc_*.csv`, `classifier_gene_drivers.csv`), `results/overlap_check/` (`25_overlap_check.png`, `overlap_*.csv`), `results/transfer_matrix/` (`26_*.png`, `transfer_matrix.csv`), `results/gene_budget/` (`27_*.png`, `gene_budget_*.csv`) y `results/cohort_overview/` (`28_*.png`, `cohort_summary.csv`). El índice completo de figuras está en 0.3.

**Descripciones de las gráficas:** cada `.png` tiene un `.txt` con el mismo nombre en la carpeta `descriptions/` de su resultado (p. ej. `results/full/descriptions/01_cohort.txt`). Explica qué expresa la gráfica y da sus cifras clave, calculadas de los propios datos. Se generan al crear la figura y se regeneran con `experiments.replot`.
