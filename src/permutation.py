"""Test de permutación a nivel de paciente (regresión logística sobre representaciones ya calculadas).

La etiqueta (GBM/LGG) es del paciente: permutar = reasignar qué pacientes son LGG, conservando cuántos hay
de cada clase. Con 14 pacientes (6 LGG) solo existen C(14,6) = 3003 asignaciones distintas: se pueden recorrer todas.
Cada asignación se evalúa con la misma validación (dejar un paciente fuera) que el análisis real.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from src.models import make_model


def enumerate_assignments(n_patients: int, k: int, n, seed: int, real: tuple):
    """Lista de tuplas con los índices de pacientes de clase 1. Devuelve (lista, exacto)."""
    total = math.comb(n_patients, k)
    if n in ("all", None) or int(n) >= total:
        return list(itertools.combinations(range(n_patients), k)), True
    rng = np.random.default_rng(seed)
    seen, combos = {tuple(sorted(real))}, []
    while len(combos) < int(n):
        c = tuple(sorted(rng.choice(n_patients, size=k, replace=False).tolist()))
        if c not in seen:
            seen.add(c)
            combos.append(c)
    return combos, False


def stats_from_fraction(frac, a) -> tuple[float, float, float]:
    """frac[p] = fracción de células del paciente p predichas como clase 1; a[p] = etiqueta del paciente.
    Cada paciente pesa 1. Devuelve (macro_f1, balanced_accuracy, patient_acc), igual que el análisis real."""
    cm = np.zeros((2, 2))
    for k in (0, 1):
        f = frac[a == k]
        cm[k, 1], cm[k, 0] = f.sum(), (1 - f).sum()
    tp, support, predicted = np.diag(cm), cm.sum(axis=1), cm.sum(axis=0)
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros(2), where=denom > 0)
    recall = np.divide(tp, support, out=np.zeros(2), where=support > 0)
    pred_majority = (frac > 0.5).astype(int)  # empate -> clase 0, como la moda en el análisis real
    return float(f1.mean()), float(recall[support > 0].mean()), float((pred_majority == a).mean())


def fractions(a, folds_data, rep_names, model: str, seed: int):
    """Una pasada de validación (dejar un paciente fuera) con la asignación `a`."""
    frac = np.zeros((len(rep_names), len(a)))
    for p, pidx_tr, mats in folds_data:
        ytr = a[pidx_tr]
        assert len(np.unique(ytr)) == 2, "una asignación dejó a entrenamiento sin una clase"
        for j, rn in enumerate(rep_names):
            Xtr, Xte = mats[rn]
            frac[j, p] = make_model(model, seed).fit(Xtr, ytr).predict(Xte).mean()
    return frac


def chunk_stats(assign_chunk, folds_data, rep_names, model: str, seed: int) -> np.ndarray:
    """(len(chunk), n_reps, 3): macro_f1, balanced_accuracy, patient_acc."""
    out = []
    for a in assign_chunk:
        fr = fractions(a, folds_data, rep_names, model, seed)
        out.append([stats_from_fraction(fr[j], a) for j in range(len(rep_names))])
    return np.array(out)


def p_values(real: np.ndarray, null: np.ndarray, exact: bool):
    """real: (R,), null: (n, R) estadístico por asignación (ya incluye la real si exact).
    Devuelve (p sin ajustar (R,), p del estadístico máximo (R,)). Monte Carlo: (1+#>=)/(1+n)."""
    eps = 1e-12
    mx = null.max(axis=1)
    if exact:
        p = (null >= real - eps).sum(axis=0) / len(null)
        p_max = np.array([(mx >= r - eps).sum() / len(null) for r in real])
    else:
        p = (1 + (null >= real - eps).sum(axis=0)) / (1 + len(null))
        p_max = np.array([(1 + (mx >= r - eps).sum()) / (1 + len(null)) for r in real])
    return p, p_max
