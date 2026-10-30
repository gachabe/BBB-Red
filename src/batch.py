"""Diagnóstico de estudio (batch) entre Walchli y Xie.

Pregunta: ¿se puede adivinar de qué estudio es una célula (de un paciente NUEVO) solo por su expresión?
Si sí, hay un efecto de estudio que puede explicar por qué un modelo entrenado en uno no funciona en el otro.
Todo con el paciente como unidad: dejar un paciente fuera (21 rondas: 14 de Walchli + 7 de Xie).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data import GROUP
from src.evaluation import classification_metrics
from src.features import capped_indices
from src.models import make_model
from src.rom import covariates, fit_representations

STUDIES = ["Walchli", "Xie"]


def common_genes(Xa, ga, Xb, gb):
    """Recorta ambas matrices a los genes en común (por nombre, sin distinguir mayúsculas)."""
    def first(g):
        s = pd.Series(np.arange(len(g)), index=pd.Index([str(x).upper() for x in g]))
        return s[~s.index.duplicated()]
    sa, sb = first(ga), first(gb)
    common = sa.index.intersection(sb.index)
    common = sa.index[sa.index.isin(common)]
    return Xa[:, sa[common].to_numpy()], Xb[:, sb[common].to_numpy()], np.asarray(common)


def eta_squared(values, groups) -> float:
    v = np.asarray(values, dtype=float)
    g = pd.Series(v).groupby(np.asarray(groups)).agg(["count", "mean"])
    total = ((v - v.mean()) ** 2).sum()
    return float((g["count"] * (g["mean"] - v.mean()) ** 2).sum() / total) if total > 0 else float("nan")


def domain_cv(X, meta, rom_cfg, model_names, cap, seed, n_trees, regress, log=print):
    """LOPO para predecir el estudio. Devuelve (métricas, probabilidad media de 'Xie' por paciente)."""
    rom = {**rom_cfg, "include_all_genes": False, "regress_out": rom_cfg.get("regress_out") if regress else None}
    C = covariates(meta, rom["regress_out"])
    y = (meta["study"] == STUDIES[1]).to_numpy().astype(np.int64)
    w = 1.0 / meta.groupby(GROUP)["cell"].transform("size").to_numpy()
    full = {p: np.asarray(i) for p, i in meta.groupby(GROUP, sort=True).indices.items()}
    capped = capped_indices(meta, cap, seed)
    proba: dict = {}
    for q in sorted(full):
        tr = np.sort(np.concatenate([capped[p] for p in capped if p != q]))
        te = full[q]
        Ctr = None if C is None else C[tr]
        Cte = None if C is None else C[te]
        for rep in fit_representations(X[tr], rom, Ctr):
            Ztr, Zte = rep.transform(X[tr], Ctr), rep.transform(X[te], Cte)
            for m in model_names:
                mod = make_model(m, seed, n_trees).fit(Ztr, y[tr])
                proba.setdefault((rep.name, rep.n_features, m), {})[q] = mod.predict_proba(Zte)[:, 1]
        log(f"    {'con' if regress else 'sin'} regresión de profundidad: paciente {q} fuera, hecho")
    pats = sorted(full)
    idx = np.concatenate([full[q] for q in pats])
    rows, per_pat = [], []
    for (rn, nf, m), d in proba.items():
        pr = np.concatenate([d[q] for q in pats])
        met = classification_metrics(y[idx], (pr > 0.5).astype(int), STUDIES, w[idx])
        pm = pd.Series({q: float(d[q].mean()) for q in pats})
        truth = pd.Series({q: int(y[full[q][0]]) for q in pats})
        rows.append(dict(representation=rn, n_features=nf, model=m, regress_depth=regress,
                         balanced_accuracy=met["balanced_accuracy"], auc=float(roc_auc_score(y[idx], pr, sample_weight=w[idx])),
                         patient_acc=float(((pm > 0.5).astype(int) == truth).mean())))
        per_pat.append(pd.DataFrame(dict(Patient=pm.index, study=[STUDIES[t] for t in truth], mean_prob_xie=pm.to_numpy(),
                                         representation=rn, model=m, regress_depth=regress)))
    return pd.DataFrame(rows), pd.concat(per_pat, ignore_index=True)


def depth_only_cv(meta, seed) -> dict:
    """Control: predecir el estudio solo con la profundidad (log nCount, log nFeature)."""
    D = covariates(meta, ["nCount_RNA", "nFeature_RNA"])
    y = (meta["study"] == STUDIES[1]).to_numpy().astype(np.int64)
    w = 1.0 / meta.groupby(GROUP)["cell"].transform("size").to_numpy()
    full = {p: np.asarray(i) for p, i in meta.groupby(GROUP, sort=True).indices.items()}
    pr = np.zeros(len(y))
    for q, te in full.items():
        tr = np.concatenate([i for p, i in full.items() if p != q])
        sc = StandardScaler().fit(D[tr])
        pr[te] = LogisticRegression(class_weight="balanced", random_state=seed).fit(sc.transform(D[tr]), y[tr]).predict_proba(sc.transform(D[te]))[:, 1]
    met = classification_metrics(y, (pr > 0.5).astype(int), STUDIES, w)
    pm = pd.Series(pr).groupby(meta[GROUP].to_numpy()).mean()
    truth = pd.Series(y).groupby(meta[GROUP].to_numpy()).first()
    return dict(balanced_accuracy=met["balanced_accuracy"], auc=float(roc_auc_score(y, pr, sample_weight=w)),
                patient_acc=float(((pm > 0.5).astype(int) == truth).mean()))


def joint_pca(X, meta, rom_cfg, cap, seed, n_plot=4000):
    """PCA ajustada con ambos estudios (solo visualización y varianza explicada). Devuelve coordenadas, y tabla eta²."""
    rom = {**rom_cfg, "components": [2, 5], "include_all_genes": False}
    C = covariates(meta, rom.get("regress_out"))
    capped = capped_indices(meta, cap, seed)
    sub = np.sort(np.concatenate(list(capped.values())))
    reps = {r.name: r for r in fit_representations(X[sub], rom, None if C is None else C[sub])}
    rep = reps["PCA-5"]
    Zs = rep.transform(X[sub], None if C is None else C[sub])
    ms = meta.iloc[sub]
    eta = pd.DataFrame([dict(component=f"PC{k + 1}", eta2_study=eta_squared(Zs[:, k], ms["study"]),
                             eta2_glioma_type=eta_squared(Zs[:, k], ms["glioma_type"]), eta2_patient=eta_squared(Zs[:, k], ms[GROUP]))
                        for k in range(Zs.shape[1])])
    rng = np.random.default_rng(seed)
    pick = np.sort(np.concatenate([rng.choice(np.flatnonzero((meta["study"] == s).to_numpy()),
                                              min(n_plot, int((meta["study"] == s).sum())), replace=False) for s in STUDIES]))
    Zp = rep.transform(X[pick], None if C is None else C[pick])
    return Zp, meta.iloc[pick].reset_index(drop=True), eta, float(rep.explained_variance)


def model_space(bundle, Xw, mw, Xx, gx, mx, n=20000, seed=0):
    """Proyecta Walchli y Xie al espacio PCA-2 del paquete entrenado (el que ve el modelo)."""
    from src.final_model import align_genes
    rep = next(r for r in bundle["reps"] if r.name == "PCA-2")
    lr = bundle["models"][("PCA-2", "logistic_regression")]
    reg = bundle["regress_out"]
    Xxa, _ = align_genes(Xx, gx, bundle["genes"])
    rng = np.random.default_rng(seed)
    out = []
    for X, m, s in ((Xw, mw, STUDIES[0]), (Xxa, mx, STUDIES[1])):
        idx = np.sort(rng.choice(X.shape[0], min(n, X.shape[0]), replace=False))
        C = covariates(m.iloc[idx], reg)
        Z = rep.transform(X[idx], C)
        out.append(pd.DataFrame(dict(PC1=Z[:, 0], PC2=Z[:, 1], study=s, glioma_type=m["glioma_type"].to_numpy()[idx],
                                     Patient=m[GROUP].to_numpy()[idx])))
    df = pd.concat(out, ignore_index=True)
    w, b = lr.coef_[0], float(lr.intercept_[0])
    df["side"] = np.where(df[["PC1", "PC2"]].to_numpy() @ w + b > 0, bundle["classes"][1], bundle["classes"][0])
    return df, w, b, bundle["classes"]
