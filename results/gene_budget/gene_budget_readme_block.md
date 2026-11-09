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
