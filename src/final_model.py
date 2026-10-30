"""Modelo final: ROM + clasificadores ajustados con TODOS los pacientes, y su aplicación a un dataset externo.

El paquete (bundle) guarda representaciones, modelos y la lista de genes de Walchli. Al aplicarlo a otro estudio
no se reajusta nada: se alinean los genes por nombre y se usa exactamente lo aprendido.
"""
from __future__ import annotations

import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn

from src.data import GROUP
from src.evaluation import classification_metrics, per_patient_table
from src.features import capped_indices, task_classes
from src.models import make_model
from src.rom import covariates, fit_representations

TASK = "glioma_type"


def train_bundle(ds, cfg: dict) -> dict:
    meta = ds.meta
    cap = cfg.get("ml", {}).get("max_train_cells_per_patient")
    by = capped_indices(meta, cap, cfg["seed"])
    idx = np.sort(np.concatenate([by[p] for p in sorted(by)]))
    regress = cfg["rom"].get("regress_out")
    C = covariates(meta, regress)
    Xtr, Ctr = ds.X[idx], None if C is None else C[idx]
    classes = task_classes(meta, TASK)
    y = np.searchsorted(classes, meta[TASK].astype(str).to_numpy()[idx]).astype(np.int64)
    reps = fit_representations(Xtr, cfg["rom"], Ctr)
    models = {}
    for rep in reps:
        Z = rep.transform(Xtr, Ctr)
        for m in cfg["models"]:
            models[(rep.name, m)] = make_model(m, cfg["seed"], cfg.get("ml", {}).get("rf_trees", 200)).fit(Z, y)
            print(f"  entrenado {rep.name} / {m}", flush=True)
    return dict(
        genes=np.asarray(ds.genes).astype(str), classes=list(classes), reps=reps, models=models,
        regress_out=list(regress) if regress else None, seed=cfg["seed"], cap=cap,
        patients=sorted(by), n_train_cells=int(len(idx)), mode=cfg.get("mode"),
        versions=dict(sklearn=sklearn.__version__, python=platform.python_version()),
    )


def save_bundle(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)


def load_bundle(path) -> dict:
    return joblib.load(path)


def check_normalization(X) -> dict:
    """Clasifica la matriz: 'counts' (enteros), 'log_norm' (log1p con total fijo) u 'other'."""
    assert sp.issparse(X)
    assert X.data.size == 0 or X.data.min() >= 0, "valores negativos: la matriz parece escalada, no normalizada"
    n = min(500, X.shape[0])
    rows = np.random.default_rng(0).choice(X.shape[0], n, replace=False)
    sub = X[rows]
    if np.allclose(sub.data, np.round(sub.data)):
        return dict(kind="counts", detail="valores enteros (conteos crudos)")
    tot = np.asarray(np.expm1(sub).sum(axis=1)).ravel()
    cv = float(tot.std() / tot.mean())
    if cv < 1e-3:
        return dict(kind="log_norm", detail=f"log1p con total fijo {np.median(tot):.0f} por célula (CV {cv:.4f})")
    return dict(kind="other", detail=f"no son conteos ni log1p con total fijo (CV de expm1 = {cv:.3f})")


def normalize_counts(X, target=1e4):
    X = X.tocsr().astype(np.float32)
    tot = np.asarray(X.sum(axis=1)).ravel()
    assert (tot > 0).all(), "hay células sin conteos"
    Xn = sp.diags(target / tot).dot(X).tocsr()
    Xn.data = np.log1p(Xn.data)
    return Xn.astype(np.float32)


def align_genes(X, ext_genes, bundle_genes):
    """Devuelve X en el orden de genes del paquete (ceros donde falta el gen) y el vector de genes presentes."""
    ext = pd.Series(np.arange(len(ext_genes)), index=pd.Index([str(g).upper() for g in ext_genes]))
    ext = ext[~ext.index.duplicated(keep="first")]
    b_up = pd.Index([str(g).upper() for g in bundle_genes])
    pos = ext.reindex(b_up).to_numpy()                       # posición en X externo o NaN
    present = ~np.isnan(pos)
    src = pos[present].astype(int)
    coo = X[:, src].tocoo()
    new_col = np.flatnonzero(present)[coo.col]
    Xa = sp.csr_matrix((coo.data, (coo.row, new_col)), shape=(X.shape[0], len(bundle_genes)), dtype=np.float32)
    return Xa, present


def depth_covariates(bundle, obs, counts):
    """Covariables de profundidad: de obs si existen; si no, de los conteos crudos; si no, error claro."""
    cols = bundle["regress_out"]
    if not cols:
        return None
    if all(c in obs.columns for c in cols):
        return covariates(obs, cols)
    if counts is None:
        raise RuntimeError(
            f"El paquete usa regress_out={cols} pero el dataset externo no trae esas columnas ni conteos crudos. "
            "Opciones: aportar las columnas, o usar el paquete entrenado sin regress_out (configs/full_noreg.yaml).")
    d = {"nCount_RNA": np.asarray(counts.sum(axis=1)).ravel(), "nFeature_RNA": np.asarray((counts > 0).sum(axis=1)).ravel()}
    unknown = [c for c in cols if c not in d]
    assert not unknown, f"no sé calcular las covariables {unknown} desde los conteos"
    return covariates(pd.DataFrame({c: d[c] for c in cols}), cols)


def _labels(bundle, obs, patient_col, label_col, label_map=None):
    """Etiquetas externas como enteros, paciente de cada célula y pesos (cada paciente pesa igual)."""
    classes = bundle["classes"]
    labels = obs[label_col].astype(str)
    if label_map:
        labels = labels.map(lambda v: label_map.get(v, v))
    unknown = sorted(set(labels) - set(classes))
    assert not unknown, f"etiquetas externas {unknown} fuera de {classes}; usa label_map"
    y = np.searchsorted(classes, labels.to_numpy()).astype(np.int64)
    pats = obs[patient_col].astype(str).to_numpy()
    w = 1.0 / pd.Series(pats).groupby(pats).transform("size").to_numpy()
    return y, pats, w


def apply_all(bundle, X, obs, C, patient_col, label_col, label_map=None):
    y, pats, w = _labels(bundle, obs, patient_col, label_col, label_map)
    classes = bundle["classes"]
    rows, per_pat = [], []
    for rep in bundle["reps"]:
        Z = rep.transform(X, C)
        for (rn, mn), model in bundle["models"].items():
            if rn != rep.name:
                continue
            pred = model.predict(Z)
            m = classification_metrics(y, pred, classes, w)
            pp = per_patient_table(y, pred, pats, classes)
            m.update(representation=rn, model=mn, n_features=rep.n_features,
                     patient_acc=float((pp["label"] == pp["pred_majority"]).mean()), n_patients=len(pp), n_cells=len(y))
            rows.append(m)
            per_pat.append(pp.assign(representation=rn, model=mn))
    return pd.DataFrame(rows), pd.concat(per_pat, ignore_index=True)
