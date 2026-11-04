"""Pseudobulks: se promedia la expresión (log-normalizada) de paquetes de células DEL MISMO paciente (y muestra).

Un pseudobulk no es un sujeto independiente: todos los de un paciente deben ir siempre al mismo pliegue (lo garantiza
`features.lopo_folds`, que reparte por paciente) y las métricas se resumen una sola vez por paciente.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.evaluation import classification_metrics


def make_pseudobulks(X, obs: pd.DataFrame, size: int, seed: int, group_cols=("Patient",), keep_cols=("Patient", "glioma_type"),
                     mean_cols=("nCount_RNA", "nFeature_RNA"), min_cells: int = 10):
    """Devuelve (Xp, meta). Dentro de cada grupo se barajan las células y se parten en n // size paquetes de tamaño casi igual
    (al menos `size` células cada uno; un grupo con menos de `size` células da un único paquete). Grupos con < min_cells se descartan."""
    assert size >= 1 and X.shape[0] == len(obs)
    obs = obs.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    keys = obs[list(group_cols)].astype(str).agg("|".join, axis=1).to_numpy()
    rows, cols, vals, metas, dropped = [], [], [], [], []
    k = 0
    for key in sorted(set(keys)):
        idx = np.flatnonzero(keys == key)
        if len(idx) < min_cells:
            dropped.append(key)
            continue
        rng.shuffle(idx)
        for chunk in np.array_split(idx, max(1, len(idx) // size)):
            rows.append(np.full(len(chunk), k))
            cols.append(chunk)
            vals.append(np.full(len(chunk), 1.0 / len(chunk), dtype=np.float32))
            sub = obs.iloc[chunk]
            m = {c: sub[c].iloc[0] for c in keep_cols if c in obs.columns}
            m.update({c: float(sub[c].mean()) for c in mean_cols if c in obs.columns})
            m.update(cell=f"PB{k:05d}", n_cells=len(chunk))
            metas.append(m)
            k += 1
    G = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(k, X.shape[0]), dtype=np.float32)
    Xp = (G @ X).tocsr().astype(np.float32)
    meta = pd.DataFrame(metas)
    meta["cell"] = meta.pop("cell")
    return Xp, meta, dropped


def patient_level(pp: pd.DataFrame, classes) -> dict:
    """Una sola predicción por paciente (la mayoritaria entre sus pseudobulks) y métricas sobre esas predicciones."""
    y = np.array([classes.index(v) for v in pp["label"]])
    p = np.array([classes.index(v) for v in pp["pred_majority"]])
    m = classification_metrics(y, p, classes)
    return dict(macro_f1_patient=m["macro_f1"], balanced_accuracy_patient=m["balanced_accuracy"])
