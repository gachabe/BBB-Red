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
