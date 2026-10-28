"""Test de permutación (regresión logística sobre PCA): ¿la señal GBM/LGG supera al azar? Con p-valor ajustado.

Se puede interrumpir (Ctrl+C) y volver a lanzar: reanuda desde results/<modo>/permutation/perm_stats.csv.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from src.data import GROUP, load_config
from src.evaluation import fig_permutation, to_markdown
from src.features import prepare, task_classes, task_is_patient_constant
from src.permutation import chunk_stats, enumerate_assignments, p_values
from src.rom import covariates, fit_representations

TASK = "glioma_type"


def _folds_data(ds, folds, names, cfg):
    pats = sorted(ds.meta[GROUP].unique())
    pindex = {p: i for i, p in enumerate(pats)}
    pat_idx = ds.meta[GROUP].map(pindex).to_numpy()
    rom = {**cfg["rom"], "include_all_genes": "All genes" in names}
    C = covariates(ds.meta, cfg["rom"].get("regress_out"))
    data = []
    for f in tqdm(folds, desc="Representaciones por ronda"):
        Ctr = None if C is None else C[f.train_idx]
        Cte = None if C is None else C[f.test_idx]
        reps = {r.name: r for r in fit_representations(ds.X[f.train_idx], rom, Ctr)}
        missing = [n for n in names if n not in reps]
        assert not missing, f"permutation.representations desconocidas: {missing}; disponibles: {list(reps)}"
        Xtr_raw, Xte_raw = ds.X[f.train_idx], ds.X[f.test_idx]
        mats = {n: (reps[n].transform(Xtr_raw, Ctr), reps[n].transform(Xte_raw, Cte)) for n in names}
        data.append((pindex[f.held_out], pat_idx[f.train_idx].astype(np.int16), mats))
    return pats, data


def main(cfg: dict) -> pd.DataFrame:
    out = Path(cfg.get("results_dir", f"results/{cfg['mode']}"))
    pdir = out / "permutation"
    pdir.mkdir(parents=True, exist_ok=True)
    ds, folds = prepare(cfg, out)
    pc = cfg.get("permutation", {})
    names = list(pc.get("representations", ["PCA-2", "PCA-5", "PCA-10", "PCA-20", "PCA-50"]))
    model = pc.get("model", "logistic_regression")
    chunk = int(pc.get("chunk", 10))
    n_jobs = int(pc.get("n_jobs", -1))
    seed = cfg["seed"]

    assert task_is_patient_constant(ds.meta, TASK), "la etiqueta debe ser constante por paciente"
    classes = task_classes(ds.meta, TASK)
    pats, data = _folds_data(ds, folds, names, cfg)
    lab = ds.meta.groupby(GROUP)[TASK].first().astype(str)
    real_a = np.array([classes.index(lab[p]) for p in pats])
    k = int(real_a.sum())
    combos, exact = enumerate_assignments(len(pats), k, pc.get("n", "all"), seed, tuple(np.flatnonzero(real_a)))
    vecs = []
    for c in combos:
        a = np.zeros(len(pats), dtype=np.int64)
        a[list(c)] = 1
        vecs.append(a)
    print(f"\n{len(pats)} pacientes ({k} de clase '{classes[1]}'): {len(vecs)} asignaciones "
          f"({'exactas, todas' if exact else 'Monte Carlo'}); representaciones: {names}")

    csv = pdir / "perm_stats.csv"
    done = set(pd.read_csv(csv)["id"]) if csv.exists() else set()
    pending = [i for i in range(len(vecs)) if i not in done]
    print(f"Hechas: {len(done)}; pendientes: {len(pending)}")
    nj = os.cpu_count() if n_jobs == -1 else n_jobs
    step = max(1, chunk) * max(1, nj or 1)
    for s in tqdm(range(0, len(pending), step), desc="Permutaciones (lotes)"):
        ids = pending[s: s + step]
        parts = [ids[i: i + chunk] for i in range(0, len(ids), chunk)]
        res = Parallel(n_jobs=n_jobs)(
            delayed(chunk_stats)([vecs[i] for i in part], data, names, model, seed) for part in parts)
        rows = []
        for part, arr in zip(parts, res):
            for i, mat in zip(part, arr):
                for j, rn in enumerate(names):
                    rows.append(dict(id=i, representation=rn, macro_f1=mat[j, 0], balanced_accuracy=mat[j, 1],
                                     patient_acc=mat[j, 2]))
        pd.DataFrame(rows).to_csv(csv, mode="a", header=not csv.exists(), index=False)

    st = pd.read_csv(csv).drop_duplicates(["id", "representation"])
    assert st["id"].nunique() == len(vecs), "faltan asignaciones; vuelve a lanzar para reanudar"
    mf = st.pivot(index="id", columns="representation", values="macro_f1")[names].to_numpy()
    pa = st.pivot(index="id", columns="representation", values="patient_acc")[names].to_numpy()
    real = chunk_stats([real_a], data, names, model, seed)[0]          # (R, 3)
    p_mf, pmax_mf = p_values(real[:, 0], mf, exact)
    p_pa, _ = p_values(real[:, 2], pa, exact)
    table = pd.DataFrame({
        "representation": names, "real_macro_f1": real[:, 0], "real_patient_acc": real[:, 2],
        "null_mean_macro_f1": mf.mean(axis=0), "p_unadjusted": p_mf, "p_adjusted_max": pmax_mf,
        "p_patient_acc_unadjusted": p_pa, "n_assignments": len(vecs), "exact": exact,
    })
    table.to_csv(out / f"permutation_results_{TASK}.csv", index=False)
    (out / f"permutation_results_{TASK}.md").write_text(
        f"### {TASK} — {model} — permutación a nivel de paciente ({'exacta' if exact else 'Monte Carlo'}, "
        f"{len(vecs)} asignaciones)\n\np_adjusted_max corrige por mirar todas las representaciones a la vez "
        "(estadístico máximo).\n\n" + to_markdown(table) + "\n", encoding="utf-8")
    fig_permutation(mf.max(axis=1), dict(zip(names, real[:, 0])), TASK, out)
    print("\n" + table.round(4).to_string(index=False))
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(load_config(ap.parse_args().config))
