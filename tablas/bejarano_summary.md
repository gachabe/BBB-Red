# Validación con la cohorte en masa de Bejarano et al. 2025

Muestras: 18 (11 GBM IDH nativo, 7 IDH mutado). Zona de edades solapadas: 49-73 años (n = 13).

| modo | representacion | n | auroc | p_exacto | macro_f1_corte_05 | sensibilidad | especificidad | macro_f1_centrado | p_medio_gbm | p_medio_idh_mutado | auroc_zona_solape | n_zona_solape | spearman_con_edad |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edad quitada de los genes | All genes | 18 | 0.974 | 0.000 | 0.280 | 0.000 | 1.000 | 0.778 | 0.018 | 0.000 | 1.000 | 13.000 | 0.251 |
| edad quitada de los genes | HVG-2000 | 18 | 0.935 | 0.001 | 0.721 | 0.545 | 1.000 | 0.943 | 0.501 | 0.001 | 0.944 | 13.000 | 0.178 |
| edad quitada de los genes | PCA-10 | 18 | 0.922 | 0.001 | 0.662 | 0.455 | 1.000 | 0.943 | 0.471 | 0.000 | 0.917 | 13.000 | 0.170 |
| edad quitada de los genes | PCA-2 | 18 | 0.935 | 0.001 | 0.888 | 0.818 | 1.000 | 0.833 | 0.605 | 0.414 | 0.889 | 13.000 | 0.385 |
| edad quitada de los genes | PCA-20 | 18 | 0.922 | 0.001 | 0.721 | 0.545 | 1.000 | 0.943 | 0.473 | 0.000 | 0.917 | 13.000 | 0.123 |
| edad quitada de los genes | PCA-5 | 18 | 0.766 | 0.035 | 0.721 | 0.545 | 1.000 | 0.667 | 0.472 | 0.202 | 0.750 | 13.000 | 0.239 |
| edad quitada dentro de cada diagnostico | All genes | 18 | 0.974 | 0.000 | 0.600 | 0.364 | 1.000 | 0.888 | 0.455 | 0.000 | 0.944 | 13.000 | 0.647 |
| edad quitada dentro de cada diagnostico | HVG-2000 | 18 | 0.974 | 0.000 | 0.833 | 0.727 | 1.000 | 0.888 | 0.673 | 0.005 | 0.944 | 13.000 | 0.608 |
| edad quitada dentro de cada diagnostico | PCA-10 | 18 | 0.974 | 0.000 | 0.833 | 0.727 | 1.000 | 0.888 | 0.670 | 0.004 | 0.944 | 13.000 | 0.598 |
| edad quitada dentro de cada diagnostico | PCA-2 | 18 | 0.948 | 0.000 | 0.888 | 0.818 | 1.000 | 0.888 | 0.764 | 0.053 | 0.889 | 13.000 | 0.660 |
| edad quitada dentro de cada diagnostico | PCA-20 | 18 | 0.974 | 0.000 | 0.833 | 0.727 | 1.000 | 0.888 | 0.671 | 0.004 | 0.944 | 13.000 | 0.608 |
| edad quitada dentro de cada diagnostico | PCA-5 | 18 | 0.935 | 0.001 | 0.662 | 0.455 | 1.000 | 0.829 | 0.473 | 0.006 | 0.889 | 13.000 | 0.629 |
| expresion | All genes | 18 | 1.000 | 0.000 | 0.721 | 0.545 | 1.000 | 0.943 | 0.596 | 0.000 | 1.000 | 13.000 | 0.428 |
| expresion | HVG-2000 | 18 | 1.000 | 0.000 | 0.833 | 0.727 | 1.000 | 0.888 | 0.673 | 0.000 | 1.000 | 13.000 | 0.431 |
| expresion | PCA-10 | 18 | 1.000 | 0.000 | 0.833 | 0.727 | 1.000 | 0.888 | 0.669 | 0.000 | 1.000 | 13.000 | 0.459 |
| expresion | PCA-2 | 18 | 1.000 | 0.000 | 0.778 | 0.636 | 1.000 | 0.833 | 0.647 | 0.000 | 1.000 | 13.000 | 0.493 |
| expresion | PCA-20 | 18 | 1.000 | 0.000 | 0.833 | 0.727 | 1.000 | 0.888 | 0.672 | 0.000 | 1.000 | 13.000 | 0.431 |
| expresion | PCA-5 | 18 | 1.000 | 0.000 | 0.778 | 0.636 | 1.000 | 0.833 | 0.621 | 0.000 | 1.000 | 13.000 | 0.486 |
| edad sola (referencia) | -- | 18 | 0.773 | 0.030 | -- | -- | -- | -- | -- | -- | 0.514 | 13.000 | -- |
| control: dejar un paciente fuera dentro de Bejarano | All genes | 18 | 0.987 | 0.000 | 0.833 | -- | -- | -- | -- | -- | -- | -- | -- |
| control: dejar un paciente fuera dentro de Bejarano | HVG-2000 | 18 | 1.000 | 0.000 | 0.833 | -- | -- | -- | -- | -- | -- | -- | -- |
| control: dejar un paciente fuera dentro de Bejarano | PCA-10 | 18 | 1.000 | 0.000 | 0.833 | -- | -- | -- | -- | -- | -- | -- | -- |
| control: dejar un paciente fuera dentro de Bejarano | PCA-2 | 18 | 0.831 | 0.010 | 0.775 | -- | -- | -- | -- | -- | -- | -- | -- |
| control: dejar un paciente fuera dentro de Bejarano | PCA-5 | 18 | 0.935 | 0.001 | 0.833 | -- | -- | -- | -- | -- | -- | -- | -- |


Comprobar a mano si algún paciente coincide con los de Wälchli (`walchli_patients_for_overlap_check.csv` frente a `bejarano_samples_used.csv`: edad, sexo y diagnóstico).
