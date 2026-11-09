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
