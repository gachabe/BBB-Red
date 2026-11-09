# Fase 2: robustez del resultado `core` y análisis de la guía

## Veredicto automático (revisar con las tablas)

- Variantes que rompen el `core` (macro-F1 < 0,9 en HVG-2000 o PCA-5): disjoint_100, umi_150k
- Quitando los genes que más distinguen los estudios, el `core` aguanta hasta k = 15000 genes eliminados (HVG-2000 ≥ 0,9)
- Con subconjuntos de entrenamiento, HVG-2000 separa el `core` sin error en el 84 % de los casos
- Un modelo solo de edad daría P(GBM) = 0.47 a la edad media de los LGG de Xie y 0.66 a la de los GBM: la edad sí separa los grupos de Xie; la confusión con la edad no se descarta
- dentro_de_xie_core: AUROC medio 1.00 (HVG-2000: 1.00)
- dentro_de_xie_periferia: AUROC medio 0.89 (HVG-2000: 1.00)
- dentro_de_xie_todo: AUROC medio 0.86 (HVG-2000: 0.92)
- inverso_entrena_xie_core: AUROC medio 0.89 (HVG-2000: 0.90)
- inverso_entrena_xie_todo: AUROC medio 0.85 (HVG-2000: 0.83)

## robustness_table.csv

| variante | core PCA-2 | core PCA-5 | core HVG-2000 | todo HVG-2000 | periferia HVG-2000 | interno HVG-2000 |
|---|---|---|---|---|---|---|
| base_lr_100 | 1.000 | 1.000 | 1.000 | 0.708 | 0.300 | 0.782 |
| rf_100 | 0.943 | 1.000 | 1.000 | 0.857 | 0.300 | 0.753 |
| disjoint_50 | 1.000 | 1.000 | 1.000 | 0.592 | 0.300 | 0.775 |
| disjoint_100 | 1.000 | 1.000 | 0.886 | 0.533 | 0.300 | 0.769 |
| fraction_010 | 1.000 | 1.000 | 1.000 | 0.857 | 0.300 | 0.845 |
| fraction_025 | 0.829 | 1.000 | 1.000 | 0.857 | 0.533 | 0.808 |
| fraction_050 | 0.829 | 1.000 | 1.000 | 0.857 | 0.533 | 0.785 |
| umi_150k | 0.943 | 1.000 | 0.886 | 0.533 | 0.300 | 0.775 |
| thin_auto | 1.000 | 1.000 | 1.000 | 0.650 | 0.300 | 0.775 |
| common_genes | 1.000 | 1.000 | 1.000 | 0.708 | 0.300 | 0.828 |
| balanced_100 | -- | -- | -- | -- | -- | 0.764 |
| xie_qc_paper | 1.000 | 1.000 | 1.000 | 0.708 | 0.300 | 0.782 |
| xie_pure | 1.000 | 1.000 | 1.000 | 0.708 | 0.300 | 0.782 |
| xie_top54 | 1.000 | 1.000 | 1.000 | 0.708 | 0.300 | 0.782 |
| xie_pub | 1.000 | 1.000 | 1.000 | 0.708 | 0.300 | 0.782 |
| xie_top25 | 1.000 | 1.000 | 0.943 | 0.708 | 0.300 | 0.782 |
| xie_top80 | 1.000 | 1.000 | 1.000 | 0.708 | 0.300 | 0.782 |
| xie_top54_strictest | 1.000 | 1.000 | 1.000 | 0.533 | 0.300 | 0.775 |
| hvg_500 | 0.333 | 1.000 | -- | -- | -- | -- |
| hvg_5000 | 1.000 | 1.000 | -- | -- | -- | -- |
| norm_deseq_vst | 1.000 | 1.000 | 1.000 | 0.708 | 0.300 | 0.802 |
| lr_C0.01 | 1.000 | 1.000 | 1.000 | 0.708 | 0.300 | 0.782 |
| lr_C100 | 1.000 | 1.000 | 1.000 | 0.708 | 0.300 | 0.782 |
| xie_pure_strictest | 1.000 | 1.000 | 1.000 | 0.708 | 0.300 | 0.775 |
| dropstudy_500 | 1.000 | 1.000 | 1.000 | 0.708 | 0.300 | 0.808 |
| dropstudy_2000 | 1.000 | 1.000 | 1.000 | 0.708 | 0.300 | 0.831 |
| dropstudy_5000 | 1.000 | 1.000 | 1.000 | 0.708 | 0.300 | 0.854 |
| dropstudy_10000 | 1.000 | 1.000 | 1.000 | 0.708 | 0.300 | 0.854 |
| dropstudy_15000 | 0.886 | 1.000 | 1.000 | 0.857 | 0.533 | 0.878 |
| followup_disjoint_100_250 | 1.000 | 1.000 | 0.829 | 0.300 | 0.250 | 0.764 |
| followup_umi_150k_250 | 1.000 | 1.000 | 1.000 | 0.708 | 0.300 | 0.854 |
| followup_umi_150k_500 | 1.000 | 1.000 | 1.000 | 0.708 | 0.300 | 0.854 |


## xie_age_baseline.csv

| subset | model | representation | n | auroc | macro_f1 | exact_p_one_sided | spearman_with_age |
|---|---|---|---|---|---|---|---|
| core | edad sola (ranking) | -- | 6 | 1.000 | -- | 0.050 | -- |
| core | modelo de edad entrenado con Wälchli (umbral 0,5) | -- | 6 | 1.000 | 1.000 | -- | -- |
| core | expresión (modelo congelado de Wälchli) | All genes | 6 | 1.000 | 1.000 | 0.050 | 0.943 |
| core | expresión (modelo congelado de Wälchli) | HVG-2000 | 6 | 1.000 | 1.000 | 0.050 | 0.886 |
| core | expresión (modelo congelado de Wälchli) | PCA-10 | 6 | 1.000 | 1.000 | 0.050 | 0.886 |
| core | expresión (modelo congelado de Wälchli) | PCA-2 | 6 | 1.000 | 1.000 | 0.050 | 0.886 |
| core | expresión (modelo congelado de Wälchli) | PCA-20 | 6 | 1.000 | 1.000 | 0.050 | 0.886 |
| core | expresión (modelo congelado de Wälchli) | PCA-5 | 6 | 1.000 | 1.000 | 0.050 | 0.886 |
| periferia | edad sola (ranking) | -- | 7 | 1.000 | -- | 0.029 | -- |
| periferia | modelo de edad entrenado con Wälchli (umbral 0,5) | -- | 7 | 1.000 | 1.000 | -- | -- |
| periferia | expresión (modelo congelado de Wälchli) | All genes | 7 | 0.917 | 0.300 | 0.057 | 0.607 |
| periferia | expresión (modelo congelado de Wälchli) | HVG-2000 | 7 | 0.917 | 0.300 | 0.057 | 0.571 |
| periferia | expresión (modelo congelado de Wälchli) | PCA-10 | 7 | 0.917 | 0.300 | 0.057 | 0.571 |
| periferia | expresión (modelo congelado de Wälchli) | PCA-2 | 7 | 0.917 | 0.300 | 0.057 | 0.571 |
| periferia | expresión (modelo congelado de Wälchli) | PCA-20 | 7 | 0.917 | 0.300 | 0.057 | 0.571 |
| periferia | expresión (modelo congelado de Wälchli) | PCA-5 | 7 | 0.833 | 0.300 | 0.114 | 0.464 |
| todo (paciente) | edad sola (ranking) | -- | 7 | 1.000 | -- | 0.029 | -- |
| todo (paciente) | modelo de edad entrenado con Wälchli (umbral 0,5) | -- | 7 | 1.000 | 1.000 | -- | -- |
| todo (paciente) | expresión (modelo congelado de Wälchli) | All genes | 7 | 1.000 | 0.300 | 0.029 | 0.893 |
| todo (paciente) | expresión (modelo congelado de Wälchli) | HVG-2000 | 7 | 1.000 | 0.708 | 0.029 | 0.750 |
| todo (paciente) | expresión (modelo congelado de Wälchli) | PCA-10 | 7 | 1.000 | 0.708 | 0.029 | 0.750 |
| todo (paciente) | expresión (modelo congelado de Wälchli) | PCA-2 | 7 | 1.000 | 0.300 | 0.029 | 0.857 |
| todo (paciente) | expresión (modelo congelado de Wälchli) | PCA-20 | 7 | 1.000 | 0.708 | 0.029 | 0.750 |
| todo (paciente) | expresión (modelo congelado de Wälchli) | PCA-5 | 7 | 1.000 | 0.708 | 0.029 | 0.893 |


## walchli_age_adjusted.csv

| representation | zone | auroc_loo_edad | auroc_loo_expresión | auroc_loo_edad + expresión | auroc_zona_solape_edad | auroc_zona_solape_expresion |
|---|---|---|---|---|---|---|
| All genes | edades 48-69 (n=6, GBM 5, LGG 1) | 0.833 | 0.917 | 0.833 | 0.000 | 0.800 |
| HVG-2000 | edades 48-69 (n=6, GBM 5, LGG 1) | 0.833 | 0.854 | 0.833 | 0.000 | 0.800 |
| PCA-10 | edades 48-69 (n=6, GBM 5, LGG 1) | 0.833 | 0.854 | 0.813 | 0.000 | 0.600 |
| PCA-2 | edades 48-69 (n=6, GBM 5, LGG 1) | 0.833 | 0.812 | 0.833 | 0.000 | 0.600 |
| PCA-20 | edades 48-69 (n=6, GBM 5, LGG 1) | 0.833 | 0.854 | 0.833 | 0.000 | 0.800 |
| PCA-5 | edades 48-69 (n=6, GBM 5, LGG 1) | 0.833 | 0.812 | 0.792 | 0.000 | 0.400 |


## age_vs_prediction_within_class.csv

| representation | diagnosis | n | spearman_age_vs_prob | p_value |
|---|---|---|---|---|
| HVG-2000 | GBM | 8 | 0.515 | 0.192 |
| HVG-2000 | LGG | 6 | 0.200 | 0.704 |
| HVG-2000 | todos | 14 | 0.689 | 0.006 |
| PCA-2 | GBM | 8 | 0.383 | 0.349 |
| PCA-2 | LGG | 6 | 0.257 | 0.623 |
| PCA-2 | todos | 14 | 0.715 | 0.004 |
| PCA-5 | GBM | 8 | 0.455 | 0.257 |
| PCA-5 | LGG | 6 | 0.086 | 0.872 |
| PCA-5 | todos | 14 | 0.636 | 0.015 |


## xie_filter_totals.csv

| filter | n_cells_loaded | n_cells_kept | published_total_cells | published_endothelial |
|---|---|---|---|---|
| pub | 106399 | 57324 | 103230 | 57324 |
| pure | 106399 | 103796 | 103230 | 57324 |
| qc_paper | 106399 | 104305 | 103230 | 57324 |
| top25 | 106399 | 26597 | 103230 | 57324 |
| top54 | 106399 | 57318 | 103230 | 57324 |
| top80 | 106399 | 85114 | 103230 | 57324 |


## dropstudy_breakpoint.csv

| k_breaks | k_last_ok | refinement |
|---|---|---|
| -1 | 10000 | [15000] |


## small_packets_core.csv

| celulas_por_paquete | representacion | sorteos | macro_f1_medio | macro_f1_minimo | fraccion_sorteos_perfectos | auroc_medio | auroc_minimo |
|---|---|---|---|---|---|---|---|
| 10 | PCA-2 | 30 | 0.840 | 0.625 | 0.267 | 1.000 | 1.000 |
| 10 | PCA-5 | 30 | 0.864 | 0.625 | 0.367 | 1.000 | 1.000 |
| 10 | HVG-2000 | 30 | 0.849 | 0.625 | 0.200 | 0.996 | 0.889 |
| 10 | All genes | 30 | 0.739 | 0.333 | 0.133 | 0.959 | 0.667 |
| 25 | PCA-2 | 30 | 0.971 | 0.829 | 0.833 | 1.000 | 1.000 |
| 25 | PCA-5 | 30 | 0.977 | 0.829 | 0.867 | 1.000 | 1.000 |
| 25 | HVG-2000 | 30 | 0.909 | 0.829 | 0.467 | 1.000 | 1.000 |
| 25 | All genes | 30 | 0.983 | 0.829 | 0.900 | 1.000 | 1.000 |
| 50 | PCA-2 | 30 | 0.989 | 0.829 | 0.933 | 1.000 | 1.000 |
| 50 | PCA-5 | 30 | 0.983 | 0.829 | 0.900 | 1.000 | 1.000 |
| 50 | HVG-2000 | 30 | 0.960 | 0.829 | 0.767 | 1.000 | 1.000 |
| 50 | All genes | 30 | 0.949 | 0.829 | 0.700 | 1.000 | 1.000 |


## age_adjusted_model.csv

| modelo | analisis | representacion | macro_f1 | auroc |
|---|---|---|---|---|
| edad quitada de los genes | internal | PCA-2 | 0.485 | 0.542 |
| edad quitada de los genes | internal | PCA-5 | 0.639 | 0.729 |
| edad quitada de los genes | internal | HVG-2000 | 0.686 | 0.646 |
| edad quitada de los genes | internal | All genes | 0.618 | 0.549 |
| edad quitada de los genes | xie_core | PCA-2 | 0.829 | 1.000 |
| edad quitada de los genes | xie_core | PCA-5 | 0.333 | 0.889 |
| edad quitada de los genes | xie_core | HVG-2000 | 0.886 | 0.963 |
| edad quitada de los genes | xie_core | All genes | 1.000 | 1.000 |
| edad quitada de los genes | xie_all | PCA-2 | 0.844 | 1.000 |
| edad quitada de los genes | xie_all | PCA-5 | 0.364 | 0.750 |
| edad quitada de los genes | xie_all | HVG-2000 | 0.708 | 0.639 |
| edad quitada de los genes | xie_all | All genes | 0.300 | 0.889 |
| edad quitada de los genes | xie_periphery | PCA-2 | 0.571 | 0.722 |
| edad quitada de los genes | xie_periphery | PCA-5 | 0.436 | 0.639 |
| edad quitada de los genes | xie_periphery | HVG-2000 | 0.300 | 0.361 |
| edad quitada de los genes | xie_periphery | All genes | 0.300 | 0.694 |
| edad quitada (pendiente dentro de cada diagnóstico) | internal | PCA-2 | 0.708 | 0.708 |
| edad quitada (pendiente dentro de cada diagnóstico) | internal | PCA-5 | 0.708 | 0.799 |
| edad quitada (pendiente dentro de cada diagnóstico) | internal | HVG-2000 | 0.775 | 0.910 |
| edad quitada (pendiente dentro de cada diagnóstico) | internal | All genes | 0.775 | 0.896 |
| edad quitada (pendiente dentro de cada diagnóstico) | xie_core | PCA-2 | 1.000 | 1.000 |
| edad quitada (pendiente dentro de cada diagnóstico) | xie_core | PCA-5 | 1.000 | 1.000 |
| edad quitada (pendiente dentro de cada diagnóstico) | xie_core | HVG-2000 | 1.000 | 1.000 |
| edad quitada (pendiente dentro de cada diagnóstico) | xie_core | All genes | 1.000 | 1.000 |
| edad quitada (pendiente dentro de cada diagnóstico) | xie_all | PCA-2 | 1.000 | 1.000 |
| edad quitada (pendiente dentro de cada diagnóstico) | xie_all | PCA-5 | 1.000 | 1.000 |
| edad quitada (pendiente dentro de cada diagnóstico) | xie_all | HVG-2000 | 1.000 | 1.000 |
| edad quitada (pendiente dentro de cada diagnóstico) | xie_all | All genes | 0.592 | 1.000 |
| edad quitada (pendiente dentro de cada diagnóstico) | xie_periphery | PCA-2 | 0.708 | 1.000 |
| edad quitada (pendiente dentro de cada diagnóstico) | xie_periphery | PCA-5 | 1.000 | 1.000 |
| edad quitada (pendiente dentro de cada diagnóstico) | xie_periphery | HVG-2000 | 0.749 | 1.000 |
| edad quitada (pendiente dentro de cada diagnóstico) | xie_periphery | All genes | 0.300 | 1.000 |
| sin edad (referencia) | internal | PCA-2 | 0.785 | 0.868 |
| sin edad (referencia) | internal | PCA-5 | 0.663 | 0.840 |
| sin edad (referencia) | internal | HVG-2000 | 0.782 | 0.910 |
| sin edad (referencia) | internal | All genes | 0.802 | 0.924 |
| sin edad (referencia) | xie_core | PCA-2 | 1.000 | 1.000 |
| sin edad (referencia) | xie_core | PCA-5 | 1.000 | 1.000 |
| sin edad (referencia) | xie_core | HVG-2000 | 1.000 | 1.000 |
| sin edad (referencia) | xie_core | All genes | 0.943 | 1.000 |
| sin edad (referencia) | xie_all | PCA-2 | 0.300 | 1.000 |
| sin edad (referencia) | xie_all | PCA-5 | 0.650 | 1.000 |
| sin edad (referencia) | xie_all | HVG-2000 | 0.708 | 1.000 |
| sin edad (referencia) | xie_all | All genes | 0.300 | 1.000 |
| sin edad (referencia) | xie_periphery | PCA-2 | 0.300 | 0.889 |
| sin edad (referencia) | xie_periphery | PCA-5 | 0.300 | 0.833 |
| sin edad (referencia) | xie_periphery | HVG-2000 | 0.300 | 0.889 |
| sin edad (referencia) | xie_periphery | All genes | 0.300 | 0.917 |

