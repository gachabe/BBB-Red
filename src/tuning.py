"""Ajuste de hiperparámetros con validación ANIDADA (dejar un paciente fuera por fuera y por dentro).

Para cada paciente de test externo, los hiperparámetros se eligen solo con los otros 13 pacientes (LOPO interno) y
luego se evalúa en el paciente de test: la estimación no ve nunca al paciente que se evalúa. El modelo final usa
los hiperparámetros elegidos con los 14 pacientes; su rendimiento honrado es el de la validación anidada, no el
del LOPO interno final (que es optimista porque es el máximo sobre la rejilla).
Se ajusta cada pareja (representación, modelo) por separado. 'All genes' queda con valores por defecto.
"""
from __future__ import annotations

import platform
from dataclasses import dataclass

import numpy as np
import pandas as pd
import sklearn

from src.data import GROUP
from src.evaluation import classification_metrics
from src.features import capped_indices, task_classes
from src.models import make_model
from src.rom import covariates, fit_representations

TASK = "glioma_type"
DEFAULT_LR_C = [0.001, 0.01, 0.1, 1.0, 10.0]
DEFAULT_RF = [dict(max_depth=None, min_samples_leaf=1), dict(max_depth=None, min_samples_leaf=20)]


def hp_grid(model: str, tcfg: dict) -> list[dict]:
    if model == "logistic_regression":
        return [dict(C=float(c)) for c in tcfg.get("lr_C", DEFAULT_LR_C)]
    return [dict(d) for d in tcfg.get("rf_grid", DEFAULT_RF)]


def hp_key(hp: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(hp.items()))


@dataclass
class Ctx:
    X: object
    meta: pd.DataFrame
    C: object
    y: np.ndarray
    w: np.ndarray
    classes: list
    full: dict
    capped: dict
    cfg: dict


def make_ctx(ds, cfg: dict) -> Ctx:
    meta = ds.meta
    classes = task_classes(meta, TASK)
    y = np.searchsorted(classes, meta[TASK].astype(str).to_numpy()).astype(np.int64)
    w = 1.0 / meta.groupby(GROUP)["cell"].transform("size").to_numpy()
    full = {p: np.asarray(i) for p, i in meta.groupby(GROUP, sort=True).indices.items()}
    capped = capped_indices(meta, cfg.get("ml", {}).get("max_train_cells_per_patient"), cfg["seed"])
    return Ctx(ds.X, meta, covariates(meta, cfg["rom"].get("regress_out")), y, w, list(classes), full, capped, cfg)


def _rows(ctx: Ctx, patients, q=None):
    return np.concatenate([ctx.capped[p] for p in patients if p != q])


def inner_scores(ctx: Ctx, patients: list[str], log=print) -> pd.DataFrame:
    """LOPO sobre 'patients': macro-F1 (cada paciente pesa igual) de cada (representación, modelo, hiperparámetros)."""
    cfg, tcfg = ctx.cfg, ctx.cfg.get("tuning", {})
    rom = {**cfg["rom"], "include_all_genes": False}
    n_trees, seed = cfg.get("ml", {}).get("rf_trees", 200), cfg["seed"]
    preds: dict = {}
    for q in patients:
        tr, te = _rows(ctx, patients, q), ctx.full[q]
        Ctr = None if ctx.C is None else ctx.C[tr]
        Cte = None if ctx.C is None else ctx.C[te]
        Xtr, Xte, ytr = ctx.X[tr], ctx.X[te], ctx.y[tr]
        for rep in fit_representations(Xtr, rom, Ctr):
            Ztr, Zte = rep.transform(Xtr, Ctr), rep.transform(Xte, Cte)
            for m in cfg["models"]:
                for hp in hp_grid(m, tcfg):
                    mod = make_model(m, seed, n_trees, **hp).fit(Ztr, ytr)
                    preds.setdefault((rep.name, m, hp_key(hp)), {})[q] = mod.predict(Zte).astype(np.int8)
        log(f"    ronda interna {q} hecha")
    idx = np.concatenate([ctx.full[q] for q in patients])
    rows = []
    for (rn, m, k), d in preds.items():
        met = classification_metrics(ctx.y[idx], np.concatenate([d[q] for q in patients]), ctx.classes, ctx.w[idx])
        rows.append(dict(representation=rn, model=m, hp=k, macro_f1=met["macro_f1"], balanced_accuracy=met["balanced_accuracy"]))
    return pd.DataFrame(rows)


def select_best(scores: pd.DataFrame, cfg: dict) -> dict:
    """{(representación, modelo): hiperparámetros}. Empates: gana el primero de la rejilla (el más regularizado)."""
    tcfg, chosen = cfg.get("tuning", {}), {}
    for (rn, m), g in scores.groupby(["representation", "model"], sort=False):
        best = g.loc[g["macro_f1"].idxmax(), "hp"]
        chosen[(rn, m)] = next(hp for hp in hp_grid(m, tcfg) if hp_key(hp) == best)
    return chosen


def fit_final(ctx: Ctx, patients: list[str], chosen: dict):
    """Ajusta ROM y modelos con 'patients' (con tope de células); parejas sin elegir usan los valores por defecto."""
    cfg = ctx.cfg
    tr = np.sort(_rows(ctx, patients))
    Ctr = None if ctx.C is None else ctx.C[tr]
    Xtr = ctx.X[tr]
    reps = fit_representations(Xtr, cfg["rom"], Ctr)
    models = {}
    for rep in reps:
        Z = rep.transform(Xtr, Ctr)
        for m in cfg["models"]:
            models[(rep.name, m)] = make_model(m, cfg["seed"], cfg.get("ml", {}).get("rf_trees", 200),
                                               **chosen.get((rep.name, m), {})).fit(Z, ctx.y[tr])
    return reps, models, tr


def predict_patient(ctx: Ctx, reps, models, q: str) -> dict:
    te = ctx.full[q]
    Cte = None if ctx.C is None else ctx.C[te]
    Xte = ctx.X[te]
    out = {}
    for rep in reps:
        Z = rep.transform(Xte, Cte)
        for (rn, m), mod in models.items():
            if rn == rep.name:
                out[(rn, m)] = mod.predict(Z).astype(np.int8)
    return out


def make_bundle(ctx: Ctx, reps, models, chosen, n_train_cells: int) -> dict:
    cfg = ctx.cfg
    reg = cfg["rom"].get("regress_out")
    return dict(
        genes=np.asarray(ctx.meta.attrs.get("genes", [])), classes=ctx.classes, reps=reps, models=models,
        regress_out=list(reg) if reg else None, seed=cfg["seed"], cap=cfg.get("ml", {}).get("max_train_cells_per_patient"),
        patients=sorted(ctx.full), n_train_cells=int(n_train_cells), mode=cfg.get("mode"),
        hparams={f"{r} / {m}": hp for (r, m), hp in chosen.items()},
        versions=dict(sklearn=sklearn.__version__, python=platform.python_version()),
    )
