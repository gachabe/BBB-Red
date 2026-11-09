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

## Sensibilidad (macro-F1 medio entre semillas)

| variante | core PCA-2 | core PCA-5 | core HVG-2000 | todo PCA-2 | todo PCA-5 | todo HVG-2000 | periferia PCA-2 | periferia PCA-5 | periferia HVG-2000 | interno PCA-2 | interno PCA-5 | interno HVG-2000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_lr_100 | 1.000 | 1.000 | 1.000 | 0.300 | 0.650 | 0.708 | 0.300 | 0.300 | 0.300 | 0.785 | 0.663 | 0.782 |
| rf_100 | 0.943 | 1.000 | 1.000 | 0.592 | 0.708 | 0.857 | 0.300 | 0.300 | 0.300 | 0.610 | 0.708 | 0.753 |
| disjoint_50 | 1.000 | 1.000 | 1.000 | 0.300 | 0.300 | 0.592 | 0.300 | 0.300 | 0.300 | 0.708 | 0.641 | 0.775 |
| disjoint_100 | 1.000 | 1.000 | 0.886 | 0.300 | 0.300 | 0.533 | 0.300 | 0.300 | 0.300 | 0.769 | 0.690 | 0.769 |
| fraction_010 | 1.000 | 1.000 | 1.000 | 1.000 | 0.857 | 0.857 | 0.708 | 0.300 | 0.300 | 0.845 | 0.717 | 0.845 |
| fraction_025 | 0.829 | 1.000 | 1.000 | 0.844 | 0.857 | 0.857 | 0.617 | 0.456 | 0.533 | 0.641 | 0.690 | 0.808 |
| fraction_050 | 0.829 | 1.000 | 1.000 | 0.844 | 0.857 | 0.857 | 0.571 | 0.533 | 0.533 | 0.641 | 0.714 | 0.785 |
| umi_150k | 0.943 | 1.000 | 0.886 | 0.300 | 0.300 | 0.533 | 0.300 | 0.300 | 0.300 | 0.785 | 0.708 | 0.775 |
| thin_auto | 1.000 | 1.000 | 1.000 | 0.300 | 0.300 | 0.650 | 0.300 | 0.300 | 0.300 | 0.759 | 0.708 | 0.775 |
| common_genes | 1.000 | 1.000 | 1.000 | 0.300 | 1.000 | 0.708 | 0.300 | 0.533 | 0.300 | 0.785 | 0.686 | 0.828 |
| balanced_100 | -- | -- | -- | -- | -- | -- | -- | -- | -- | 0.795 | 0.921 | 0.764 |
| xie_qc_paper | 1.000 | 1.000 | 1.000 | 0.300 | 0.592 | 0.708 | 0.300 | 0.300 | 0.300 | 0.785 | 0.663 | 0.782 |
| xie_pure | 1.000 | 1.000 | 1.000 | 0.300 | 0.650 | 0.708 | 0.300 | 0.300 | 0.300 | 0.785 | 0.663 | 0.782 |
| xie_top54 | 1.000 | 1.000 | 1.000 | 0.300 | 0.300 | 0.708 | 0.300 | 0.300 | 0.300 | 0.785 | 0.663 | 0.782 |
| xie_pub | 1.000 | 1.000 | 1.000 | 0.300 | 0.300 | 0.708 | 0.300 | 0.300 | 0.300 | 0.785 | 0.663 | 0.782 |
| xie_top25 | 1.000 | 1.000 | 0.943 | 0.300 | 0.300 | 0.708 | 0.300 | 0.300 | 0.300 | 0.785 | 0.663 | 0.782 |
| xie_top80 | 1.000 | 1.000 | 1.000 | 0.300 | 0.300 | 0.708 | 0.300 | 0.300 | 0.300 | 0.785 | 0.663 | 0.782 |
| xie_top54_strictest | 1.000 | 1.000 | 1.000 | 0.300 | 0.300 | 0.533 | 0.300 | 0.300 | 0.300 | 0.785 | 0.686 | 0.775 |
| hvg_500 | 0.333 | 1.000 | -- | 0.300 | 0.300 | -- | 0.300 | 0.300 | -- | 0.417 | 0.714 | -- |
| hvg_5000 | 1.000 | 1.000 | -- | 0.300 | 0.533 | -- | 0.300 | 0.300 | -- | 0.734 | 0.686 | -- |
| norm_deseq_vst | 1.000 | 1.000 | 1.000 | 0.300 | 0.700 | 0.708 | 0.300 | 0.300 | 0.300 | 0.809 | 0.686 | 0.802 |
| lr_C0.01 | 1.000 | 1.000 | 1.000 | 0.300 | 0.300 | 0.708 | 0.300 | 0.300 | 0.300 | 0.761 | 0.641 | 0.782 |
| lr_C100 | 1.000 | 1.000 | 1.000 | 0.300 | 0.563 | 0.708 | 0.300 | 0.300 | 0.300 | 0.663 | 0.617 | 0.782 |
| xie_pure_strictest | 1.000 | 1.000 | 1.000 | 0.300 | 0.300 | 0.708 | 0.300 | 0.300 | 0.300 | 0.785 | 0.686 | 0.775 |
| dropstudy_500 | 1.000 | 1.000 | 1.000 | 0.300 | 0.300 | 0.708 | 0.300 | 0.300 | 0.300 | 0.785 | 0.663 | 0.808 |
| dropstudy_2000 | 1.000 | 1.000 | 1.000 | 0.300 | 0.300 | 0.708 | 0.300 | 0.300 | 0.300 | 0.833 | 0.640 | 0.831 |
| dropstudy_5000 | 1.000 | 1.000 | 1.000 | 0.300 | 0.300 | 0.708 | 0.300 | 0.300 | 0.300 | 0.785 | 0.663 | 0.854 |
| dropstudy_10000 | 1.000 | 1.000 | 1.000 | 0.700 | 0.592 | 0.708 | 0.456 | 0.300 | 0.300 | 0.690 | 0.686 | 0.854 |
| dropstudy_15000 | 0.886 | 1.000 | 1.000 | 1.000 | 0.708 | 0.857 | 0.708 | 0.300 | 0.533 | 0.665 | 0.731 | 0.878 |
| followup_disjoint_100_250 | 1.000 | 1.000 | 0.829 | 0.300 | 0.300 | 0.300 | 0.250 | 0.250 | 0.250 | 0.536 | 0.536 | 0.764 |
| followup_umi_150k_250 | 1.000 | 1.000 | 1.000 | 0.300 | 0.300 | 0.708 | 0.300 | 0.300 | 0.300 | 0.783 | 0.708 | 0.854 |
| followup_umi_150k_500 | 1.000 | 1.000 | 1.000 | 0.378 | 0.708 | 0.708 | 0.300 | 0.300 | 0.300 | 0.833 | 0.708 | 0.854 |


## Remuestreo del entrenamiento (100 subconjuntos de 6 GBM + 4 LGG)

| representation | f1_mediana | f1_p05 | prop_perfecto | auroc_mediana |
|---|---|---|---|---|
| All genes | 0.829 | 0.818 | 0.390 | 1.000 |
| HVG-2000 | 1.000 | 0.829 | 0.840 | 1.000 |
| PCA-10 | 1.000 | 0.829 | 0.840 | 1.000 |
| PCA-2 | 1.000 | 0.829 | 0.850 | 1.000 |
| PCA-20 | 1.000 | 0.829 | 0.840 | 1.000 |
| PCA-5 | 1.000 | 0.829 | 0.910 | 1.000 |


## Quitar un paciente de entrenamiento cada vez (macro-F1 en `core`)

| dropped | dropped_dx | representation | macro_f1 | auroc |
|---|---|---|---|---|
| GBM1 | GBM | PCA-2 | 1.000 | 1.000 |
| GBM1 | GBM | PCA-5 | 1.000 | 1.000 |
| GBM1 | GBM | HVG-2000 | 1.000 | 1.000 |
| GBM2 | GBM | PCA-2 | 1.000 | 1.000 |
| GBM2 | GBM | PCA-5 | 1.000 | 1.000 |
| GBM2 | GBM | HVG-2000 | 1.000 | 1.000 |
| GBM3 | GBM | PCA-2 | 1.000 | 1.000 |
| GBM3 | GBM | PCA-5 | 1.000 | 1.000 |
| GBM3 | GBM | HVG-2000 | 1.000 | 1.000 |
| GBM4 | GBM | PCA-2 | 1.000 | 1.000 |
| GBM4 | GBM | PCA-5 | 1.000 | 1.000 |
| GBM4 | GBM | HVG-2000 | 1.000 | 1.000 |
| GBM5 | GBM | PCA-2 | 1.000 | 1.000 |
| GBM5 | GBM | PCA-5 | 1.000 | 1.000 |
| GBM5 | GBM | HVG-2000 | 1.000 | 1.000 |
| GBM6 | GBM | PCA-2 | 1.000 | 1.000 |
| GBM6 | GBM | PCA-5 | 1.000 | 1.000 |
| GBM6 | GBM | HVG-2000 | 1.000 | 1.000 |
| GBM7 | GBM | PCA-2 | 1.000 | 1.000 |
| GBM7 | GBM | PCA-5 | 1.000 | 1.000 |
| GBM7 | GBM | HVG-2000 | 1.000 | 1.000 |
| GBM8 | GBM | PCA-2 | 1.000 | 1.000 |
| GBM8 | GBM | PCA-5 | 1.000 | 1.000 |
| GBM8 | GBM | HVG-2000 | 1.000 | 1.000 |
| LGG1 | LGG | PCA-2 | 1.000 | 1.000 |
| LGG1 | LGG | PCA-5 | 1.000 | 1.000 |
| LGG1 | LGG | HVG-2000 | 0.829 | 1.000 |
| LGG2 | LGG | PCA-2 | 1.000 | 1.000 |
| LGG2 | LGG | PCA-5 | 1.000 | 1.000 |
| LGG2 | LGG | HVG-2000 | 1.000 | 1.000 |
| LGG3 | LGG | PCA-2 | 1.000 | 1.000 |
| LGG3 | LGG | PCA-5 | 1.000 | 1.000 |
| LGG3 | LGG | HVG-2000 | 1.000 | 1.000 |
| LGG4 | LGG | PCA-2 | 1.000 | 1.000 |
| LGG4 | LGG | PCA-5 | 1.000 | 1.000 |
| LGG4 | LGG | HVG-2000 | 1.000 | 1.000 |
| LGG5 | LGG | PCA-2 | 1.000 | 1.000 |
| LGG5 | LGG | PCA-5 | 1.000 | 1.000 |
| LGG5 | LGG | HVG-2000 | 1.000 | 1.000 |
| LGG6 | LGG | PCA-2 | 1.000 | 1.000 |
| LGG6 | LGG | PCA-5 | 1.000 | 1.000 |
| LGG6 | LGG | HVG-2000 | 1.000 | 1.000 |


## Transferencia inversa y señal dentro de Xie

| analysis | representation | macro_f1 | auroc |
|---|---|---|---|
| dentro_de_xie_core | All genes | 1.000 | 1.000 |
| dentro_de_xie_core | HVG-2000 | 1.000 | 1.000 |
| dentro_de_xie_core | PCA-10 | 1.000 | 1.000 |
| dentro_de_xie_core | PCA-2 | 1.000 | 1.000 |
| dentro_de_xie_core | PCA-20 | 1.000 | 1.000 |
| dentro_de_xie_core | PCA-5 | 1.000 | 1.000 |
| dentro_de_xie_periferia | All genes | 0.808 | 0.861 |
| dentro_de_xie_periferia | HVG-2000 | 0.857 | 1.000 |
| dentro_de_xie_periferia | PCA-10 | 0.855 | 1.000 |
| dentro_de_xie_periferia | PCA-2 | 0.571 | 0.500 |
| dentro_de_xie_periferia | PCA-20 | 0.905 | 1.000 |
| dentro_de_xie_periferia | PCA-5 | 0.758 | 1.000 |
| dentro_de_xie_todo | All genes | 0.708 | 0.722 |
| dentro_de_xie_todo | HVG-2000 | 0.857 | 0.917 |
| dentro_de_xie_todo | PCA-10 | 0.857 | 0.889 |
| dentro_de_xie_todo | PCA-2 | 0.758 | 0.917 |
| dentro_de_xie_todo | PCA-20 | 0.857 | 0.944 |
| dentro_de_xie_todo | PCA-5 | 0.533 | 0.750 |
| inverso_entrena_xie_core | All genes | 0.775 | 0.847 |
| inverso_entrena_xie_core | HVG-2000 | 0.854 | 0.896 |
| inverso_entrena_xie_core | PCA-10 | 0.854 | 0.896 |
| inverso_entrena_xie_core | PCA-2 | 0.854 | 0.896 |
| inverso_entrena_xie_core | PCA-20 | 0.854 | 0.896 |
| inverso_entrena_xie_core | PCA-5 | 0.854 | 0.896 |
| inverso_entrena_xie_todo | All genes | 0.427 | 0.743 |
| inverso_entrena_xie_todo | HVG-2000 | 0.462 | 0.833 |
| inverso_entrena_xie_todo | PCA-10 | 0.428 | 0.833 |
| inverso_entrena_xie_todo | PCA-2 | 0.802 | 0.917 |
| inverso_entrena_xie_todo | PCA-20 | 0.462 | 0.833 |
| inverso_entrena_xie_todo | PCA-5 | 0.790 | 0.924 |


## Paquetes específicos de cada estado endotelial (interno)

| state | representation | macro_f1 | auroc | n |
|---|---|---|---|---|
| Angiogenic capillary | HVG-2000 | 0.925 | 0.938 | 14 |
| Angiogenic capillary | PCA-2 | 0.854 | 0.896 | 14 |
| Angiogenic capillary | PCA-5 | 0.925 | 0.948 | 14 |
| Arteriole | HVG-2000 | 0.748 | 0.944 | 12 |
| Arteriole | PCA-2 | 0.748 | 0.722 | 12 |
| Arteriole | PCA-5 | 0.748 | 0.944 | 12 |
| Artery | HVG-2000 | 0.775 | 0.850 | 9 |
| Artery | PCA-2 | 0.775 | 0.950 | 9 |
| Artery | PCA-5 | 0.775 | 0.850 | 9 |
| Capillary | HVG-2000 | 0.708 | 0.865 | 14 |
| Capillary | PCA-2 | 0.785 | 0.771 | 14 |
| Capillary | PCA-5 | 0.641 | 0.875 | 14 |
| EndoMT | HVG-2000 | 0.829 | 1.000 | 6 |
| EndoMT | PCA-2 | 0.667 | 0.938 | 6 |
| EndoMT | PCA-5 | 0.829 | 1.000 | 6 |
| Large vein | HVG-2000 | 0.417 | 0.600 | 7 |
| Large vein | PCA-2 | 0.844 | 0.800 | 7 |
| Large vein | PCA-5 | 0.417 | 0.600 | 7 |
| Mitochondrial | HVG-2000 | 1.000 | 1.000 | 6 |
| Mitochondrial | PCA-2 | 1.000 | 1.000 | 6 |
| Mitochondrial | PCA-5 | 1.000 | 1.000 | 6 |
| Vein | HVG-2000 | 0.621 | 0.650 | 7 |
| Vein | PCA-2 | 0.844 | 0.800 | 7 |
| Vein | PCA-5 | 0.621 | 0.650 | 7 |
| Venule | HVG-2000 | 0.718 | 0.900 | 11 |
| Venule | PCA-2 | 0.436 | 0.417 | 11 |
| Venule | PCA-5 | 0.676 | 0.867 | 11 |


## Periferia: umbral

| size | representation | n_patients | auroc | macro_f1_thr05 | macro_f1_centered_unlabeled | macro_f1_best_threshold_posthoc | mean_p_periphery |
|---|---|---|---|---|---|---|---|
| 100 | All genes | 7.000 | 0.917 | 0.300 | 0.857 | 0.857 | 0.004 |
| 100 | HVG-2000 | 7.000 | 0.900 | 0.300 | 0.857 | 0.857 | 0.038 |
| 100 | PCA-10 | 7.000 | 0.883 | 0.300 | 0.798 | 0.857 | 0.040 |
| 100 | PCA-2 | 7.000 | 0.900 | 0.300 | 0.768 | 0.855 | 0.001 |
| 100 | PCA-20 | 7.000 | 0.883 | 0.300 | 0.857 | 0.857 | 0.039 |
| 100 | PCA-5 | 7.000 | 0.817 | 0.300 | 0.571 | 0.817 | 0.026 |
| 250 | All genes | 7.000 | 0.917 | 0.300 | 0.857 | 0.857 | 0.003 |
| 250 | HVG-2000 | 7.000 | 0.867 | 0.300 | 0.708 | 0.857 | 0.059 |
| 250 | PCA-10 | 7.000 | 0.867 | 0.300 | 0.708 | 0.857 | 0.055 |
| 250 | PCA-2 | 7.000 | 0.800 | 0.393 | 0.571 | 0.790 | 0.116 |
| 250 | PCA-20 | 7.000 | 0.867 | 0.300 | 0.708 | 0.857 | 0.062 |
| 250 | PCA-5 | 7.000 | 0.800 | 0.300 | 0.571 | 0.790 | 0.010 |
| 500 | All genes | 6.000 | 0.900 | 0.250 | 0.829 | 0.863 | 0.007 |
| 500 | HVG-2000 | 6.000 | 0.875 | 0.250 | 0.829 | 0.829 | 0.058 |
| 500 | PCA-10 | 6.000 | 0.775 | 0.250 | 0.594 | 0.788 | 0.061 |
| 500 | PCA-2 | 6.000 | 0.625 | 0.667 | 0.486 | 0.667 | 0.323 |
| 500 | PCA-20 | 6.000 | 0.875 | 0.250 | 0.829 | 0.829 | 0.059 |
| 500 | PCA-5 | 6.000 | 0.675 | 0.250 | 0.486 | 0.711 | 0.087 |


## Periferia: desplazamiento de logit frente al núcleo

| size | representation | mean_logit_shift_per_minus_core | frac_patients_per_lower_than_core | n |
|---|---|---|---|---|
| 100 | All genes | -5.089 | 1.000 | 6.000 |
| 100 | HVG-2000 | -3.962 | 1.000 | 6.000 |
| 100 | PCA-10 | -4.046 | 1.000 | 6.000 |
| 100 | PCA-2 | -8.894 | 1.000 | 6.000 |
| 100 | PCA-20 | -3.954 | 1.000 | 6.000 |
| 100 | PCA-5 | -9.245 | 1.000 | 6.000 |
| 250 | All genes | -5.210 | 1.000 | 6.000 |
| 250 | HVG-2000 | -4.160 | 1.000 | 6.000 |
| 250 | PCA-10 | -4.288 | 1.000 | 6.000 |
| 250 | PCA-2 | -8.520 | 1.000 | 6.000 |
| 250 | PCA-20 | -4.166 | 1.000 | 6.000 |
| 250 | PCA-5 | -10.641 | 1.000 | 6.000 |
| 500 | All genes | -5.626 | 1.000 | 5.000 |
| 500 | HVG-2000 | -4.337 | 1.000 | 5.000 |
| 500 | PCA-10 | -4.427 | 0.960 | 5.000 |
| 500 | PCA-2 | -4.228 | 1.000 | 5.000 |
| 500 | PCA-20 | -4.337 | 1.000 | 5.000 |
| 500 | PCA-5 | -9.286 | 1.000 | 5.000 |


## Convergencia de réplicas

| analysis | representation | n_packets | median_sd_of_patient_mean | max_sd_of_patient_mean | mean_prob_class_flip_vs_full |
|---|---|---|---|---|---|
| internal | HVG-2000 | 1 | 0.006 | 0.141 | 0.053 |
| internal | HVG-2000 | 2 | 0.004 | 0.098 | 0.035 |
| internal | HVG-2000 | 3 | 0.003 | 0.077 | 0.027 |
| internal | HVG-2000 | 5 | 0.002 | 0.056 | 0.020 |
| internal | HVG-2000 | 10 | 0.001 | 0.033 | 0.013 |
| internal | HVG-2000 | 15 | 0.001 | 0.019 | 0.006 |
| internal | HVG-2000 | 20 | 0.000 | 0.000 | 0.000 |
| internal | PCA-5 | 1 | 0.002 | 0.201 | 0.025 |
| internal | PCA-5 | 2 | 0.001 | 0.139 | 0.015 |
| internal | PCA-5 | 3 | 0.001 | 0.110 | 0.013 |
| internal | PCA-5 | 5 | 0.001 | 0.079 | 0.009 |
| internal | PCA-5 | 10 | 0.000 | 0.046 | 0.004 |
| internal | PCA-5 | 15 | 0.000 | 0.027 | 0.001 |
| internal | PCA-5 | 20 | 0.000 | 0.000 | 0.000 |
| xie_core | HVG-2000 | 1 | 0.011 | 0.114 | 0.010 |
| xie_core | HVG-2000 | 2 | 0.007 | 0.079 | 0.002 |
| xie_core | HVG-2000 | 3 | 0.006 | 0.064 | 0.000 |
| xie_core | HVG-2000 | 5 | 0.004 | 0.047 | 0.000 |
| xie_core | HVG-2000 | 10 | 0.003 | 0.026 | 0.000 |
| xie_core | HVG-2000 | 15 | 0.001 | 0.015 | 0.000 |
| xie_core | HVG-2000 | 20 | 0.000 | 0.000 | 0.000 |
| xie_core | PCA-5 | 1 | 0.000 | 0.058 | 0.000 |
| xie_core | PCA-5 | 2 | 0.000 | 0.045 | 0.000 |
| xie_core | PCA-5 | 3 | 0.000 | 0.034 | 0.000 |
| xie_core | PCA-5 | 5 | 0.000 | 0.025 | 0.000 |
| xie_core | PCA-5 | 10 | 0.000 | 0.015 | 0.000 |
| xie_core | PCA-5 | 15 | 0.000 | 0.009 | 0.000 |
| xie_core | PCA-5 | 20 | 0.000 | 0.000 | 0.000 |


## Componentes de varianza (ANOVA)

| representation | var_between_patients | var_within_patient | share_between |
|---|---|---|---|
| All genes | 0.198 | 0.002 | 0.988 |
| HVG-2000 | 0.189 | 0.003 | 0.983 |
| PCA-10 | 0.192 | 0.003 | 0.983 |
| PCA-2 | 0.158 | 0.005 | 0.968 |
| PCA-20 | 0.190 | 0.003 | 0.983 |
| PCA-5 | 0.218 | 0.005 | 0.979 |


## Modelo solo de edad con las edades medias de Xie

| group | age | p_gbm_age_only | edad_min_walchli | edad_max_walchli |
|---|---|---|---|---|
| Wälchli LGG (media) | 40.500 | 0.334 | 26.000 | 82.000 |
| Wälchli GBM (media) | 62.500 | 0.674 | 26.000 | 82.000 |
| Xie LGG (media publicada) | 49.300 | 0.469 | 26.000 | 82.000 |
| Xie GBM (media publicada) | 61.300 | 0.657 | 26.000 | 82.000 |


## Edad sola frente a expresión en las validaciones externas de Xie

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


## Wälchli: edad, expresión y ambas (dejando un paciente fuera)

| representation | zone | auroc_loo_edad | auroc_loo_expresión | auroc_loo_edad + expresión | auroc_zona_solape_edad | auroc_zona_solape_expresion |
|---|---|---|---|---|---|---|
| All genes | edades 48-69 (n=6, GBM 5, LGG 1) | 0.833 | 0.917 | 0.833 | 0.000 | 0.800 |
| HVG-2000 | edades 48-69 (n=6, GBM 5, LGG 1) | 0.833 | 0.854 | 0.833 | 0.000 | 0.800 |
| PCA-10 | edades 48-69 (n=6, GBM 5, LGG 1) | 0.833 | 0.854 | 0.813 | 0.000 | 0.600 |
| PCA-2 | edades 48-69 (n=6, GBM 5, LGG 1) | 0.833 | 0.812 | 0.833 | 0.000 | 0.600 |
| PCA-20 | edades 48-69 (n=6, GBM 5, LGG 1) | 0.833 | 0.854 | 0.833 | 0.000 | 0.800 |
| PCA-5 | edades 48-69 (n=6, GBM 5, LGG 1) | 0.833 | 0.812 | 0.792 | 0.000 | 0.400 |


## Células de Xie: nuestras frente a las publicadas

| sample | n_cells_before_qc | total_cells_published | ratio_cargadas_vs_publicadas | n_endothelial_selected | endothelial_published | ratio_endoteliales_vs_publicadas |
|---|---|---|---|---|---|---|
| N027 | 691 | 339 | 2.038 | 465 | 288 | 1.615 |
| T027 | 25476 | 14086 | 1.809 | 18267 | 10599 | 1.723 |
| N036 | 30818 | 13254 | 2.325 | 11596 | 4642 | 2.498 |
| N037 | 17999 | 11757 | 1.531 | 9932 | 6433 | 1.544 |
| T037 | 24568 | 13075 | 1.879 | 14168 | 8016 | 1.767 |
| N038 | 16525 | 9682 | 1.707 | 11271 | 6417 | 1.756 |
| T038 | 8342 | 4684 | 1.781 | 2236 | 1333 | 1.677 |
| N043 | 8785 | 6344 | 1.385 | 3724 | 2678 | 1.391 |
| T043 | 8408 | 5600 | 1.501 | 3768 | 2607 | 1.445 |
| N045 | 23197 | 8747 | 2.652 | 13450 | 5084 | 2.646 |
| T045 | 8487 | 4305 | 1.971 | 6225 | 3222 | 1.932 |
| N046 | 8801 | 6531 | 1.348 | 5757 | 4268 | 1.349 |
| T046 | 16348 | 4826 | 3.387 | 5540 | 1737 | 3.189 |


## Correlación edad-predicción dentro de cada diagnóstico (Wälchli)

| representation | diagnosis | n | spearman_age_vs_prob | p_value |
|---|---|---|---|---|
| All genes | GBM | 8 | 0.503 | 0.204 |
| All genes | LGG | 6 | 0.371 | 0.468 |
| All genes | todos | 14 | 0.728 | 0.003 |
| HVG-2000 | GBM | 8 | 0.515 | 0.192 |
| HVG-2000 | LGG | 6 | 0.200 | 0.704 |
| HVG-2000 | todos | 14 | 0.689 | 0.006 |
| PCA-10 | GBM | 8 | 0.515 | 0.192 |
| PCA-10 | LGG | 6 | 0.200 | 0.704 |
| PCA-10 | todos | 14 | 0.658 | 0.011 |
| PCA-2 | GBM | 8 | 0.383 | 0.349 |
| PCA-2 | LGG | 6 | 0.257 | 0.623 |
| PCA-2 | todos | 14 | 0.715 | 0.004 |
| PCA-20 | GBM | 8 | 0.515 | 0.192 |
| PCA-20 | LGG | 6 | 0.200 | 0.704 |
| PCA-20 | todos | 14 | 0.689 | 0.006 |
| PCA-5 | GBM | 8 | 0.455 | 0.257 |
| PCA-5 | LGG | 6 | 0.086 | 0.872 |
| PCA-5 | todos | 14 | 0.636 | 0.015 |


## Pacientes de Wälchli

| patient | diagnosis | age | n_cells | median_umi | All genes | HVG-2000 | PCA-10 | PCA-2 | PCA-20 | PCA-5 |
|---|---|---|---|---|---|---|---|---|---|---|
| GBM1 | GBM | 71.000 | 264 | 3943.500 | 0.907 | 0.971 | 0.974 | 0.942 | 0.972 | 0.997 |
| GBM2 | GBM | 51.000 | 8316 | 6226.500 | 1.000 | 0.999 | 0.999 | 1.000 | 0.999 | 1.000 |
| GBM3 | GBM | 60.000 | 8107 | 4882.000 | 0.996 | 0.996 | 0.995 | 0.988 | 0.996 | 0.998 |
| GBM4 | GBM | 78.000 | 3990 | 8892.500 | 0.979 | 0.956 | 0.936 | 0.896 | 0.955 | 0.659 |
| GBM5 | GBM | 82.000 | 17580 | 8966.000 | 0.999 | 0.997 | 0.997 | 0.902 | 0.997 | 1.000 |
| GBM6 | GBM | 60.000 | 3503 | 8028.000 | 0.940 | 0.853 | 0.902 | 0.525 | 0.859 | 0.020 |
| GBM7 | GBM | 50.000 | 1249 | 1114.000 | 0.565 | 0.510 | 0.443 | 0.169 | 0.522 | 0.424 |
| GBM8 | GBM | 48.000 | 6990 | 1364.000 | 0.035 | 0.280 | 0.272 | 0.071 | 0.276 | 0.018 |
| LGG1 | LGG | 69.000 | 923 | 4286.000 | 0.516 | 0.452 | 0.505 | 0.473 | 0.452 | 0.799 |
| LGG2 | LGG | 46.000 | 2447 | 4992.000 | 0.002 | 0.007 | 0.005 | 0.056 | 0.006 | 0.000 |
| LGG3 | LGG | 26.000 | 1501 | 5281.000 | 0.000 | 0.000 | 0.000 | 0.060 | 0.000 | 0.000 |
| LGG4 | LGG | 32.000 | 4284 | 4632.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| LGG5 | LGG | 39.000 | 2866 | 3105.500 | 0.000 | 0.000 | 0.000 | 0.192 | 0.000 | 0.000 |
| LGG6 | LGG | 31.000 | 5352 | 6284.000 | 0.711 | 0.924 | 0.936 | 0.361 | 0.923 | 0.960 |


## Células de Xie que sobreviven a cada filtro (total)

| filter | n_cells_loaded | n_cells_kept | published_total_cells | published_endothelial |
|---|---|---|---|---|
| pub | 106399 | 57324 | 103230 | 57324 |
| pure | 106399 | 103796 | 103230 | 57324 |
| qc_paper | 106399 | 104305 | 103230 | 57324 |
| top25 | 106399 | 26597 | 103230 | 57324 |
| top54 | 106399 | 57318 | 103230 | 57324 |
| top80 | 106399 | 85114 | 103230 | 57324 |


## Holm sobre las permutaciones de la fase 1

| representation | macro_f1_observed | n_assignments | p_exact | p_adjusted_maxT | size | p_holm_across_all_tests | n_tests_in_family |
|---|---|---|---|---|---|---|---|
| PCA-2 | 0.857 | 3003 | 0.007 | 0.016 | 100 | 0.047 | 7 |
| PCA-5 | 0.708 | 3003 | 0.124 | 0.248 | 100 | 0.250 | 7 |
| PCA-10 | 0.854 | 3003 | 0.023 | 0.036 | 100 | 0.136 | 7 |
| PCA-20 | 0.854 | 3003 | 0.023 | 0.036 | 100 | 0.136 | 7 |
| PCA-2 | 0.641 | 3003 | 0.210 | 0.371 | whole | 0.250 | 7 |
| PCA-5 | 0.714 | 3003 | 0.083 | 0.171 | whole | 0.250 | 7 |
| PCA-10 | 0.785 | 3003 | 0.040 | 0.081 | whole | 0.159 | 7 |

