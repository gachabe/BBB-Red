"""ML: LR y RF sobre PCA-r / HVG / todos los genes, dejando un paciente fuera; con control de etiquetas permutadas."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data import CELLTYPE, GROUP, load_config
from src.evaluation import (
    classification_metrics,
    cluster_breakdown,
    fig_by_cluster,
    fig_components_vs_performance,
    fig_confusions,
    fig_per_patient,
    per_patient_table,
    save_tables,
    weighted_confusion,
)
from src.features import permute_labels, task_classes, task_is_patient_constant
from src.models import make_model
from src.rom import covariates


def run(ds, fits, cfg: dict, outdir: Path, seed=None, save: bool = True) -> dict[str, pd.DataFrame]:
    seed = cfg["seed"] if seed is None else seed
    tasks = list(cfg.get("tasks", ["glioma_type"]))
    models = cfg["models"]
    mlc = cfg.get("ml", {})
    n_trees, do_control, do_cluster = mlc.get("rf_trees", 200), mlc.get("control", True), mlc.get("by_cluster", True)
    meta = ds.meta
    N = len(meta)
    C = covariates(meta, cfg["rom"].get("regress_out"))
    pats = meta[GROUP].to_numpy()
    clusters = meta[CELLTYPE].to_numpy()
    w = 1.0 / meta.groupby(GROUP)["cell"].transform("size").to_numpy()  # cada paciente pesa igual

    plan = {}
    for t in tasks:
        classes = task_classes(meta, t)
        variants = {"real": meta[t].astype(str).to_numpy()}
        if do_control:
            variants["permuted"] = permute_labels(meta, t, seed + 1000)
        plan[t] = (classes, {v: np.searchsorted(classes, y).astype(np.int64) for v, y in variants.items()})

    rep_names = [r.name for r in fits[0].reps]
    pred = {(t, v, m, r): np.full(N, -1, np.int8) for t in tasks for v in plan[t][1] for m in models for r in rep_names}

    for ff in tqdm(fits, desc="ML (paciente de test)"):
        tr, te = ff.fold.train_idx, ff.fold.test_idx
        assert set(tr).isdisjoint(te)  # antes de entrenar
        Xtr_raw, Xte_raw = ds.X[tr], ds.X[te]
        for rep in ff.reps:
            Xtr = rep.transform(Xtr_raw, None if C is None else C[tr])
            Xte = rep.transform(Xte_raw, None if C is None else C[te])
            for t in tasks:
                for v, codes in plan[t][1].items():
                    for m in models:
                        clf = make_model(m, seed, n_trees).fit(Xtr, codes[tr])
                        pred[(t, v, m, rep.name)][te] = clf.predict(Xte)

    n_feat = {r.name: r.n_features for r in fits[0].reps}
    expl = {n: float(np.mean([next(r.explained_variance for r in ff.reps if r.name == n) for ff in fits])) for n in rep_names}

    out = {}
    for t in tasks:
        classes, codes = plan[t]
        const = task_is_patient_constant(meta, t)
        rows, pp_rows, bc_rows, cms = [], [], [], {v: {} for v in codes}
        for v, y in codes.items():
            for m in models:
                for rn in rep_names:
                    p = pred[(t, v, m, rn)].astype(np.int64)
                    assert (p >= 0).all()  # cada célula se predice una vez, con un modelo que no vio a su paciente
                    mw, mc = classification_metrics(y, p, classes, w), classification_metrics(y, p, classes)
                    pp = per_patient_table(y, p, pats, classes)
                    rows.append(
                        dict(
                            task=t, variant=v, model=m, representation=rn, n_features=n_feat[rn], **mw,
                            cell_macro_f1=mc["macro_f1"], cell_balanced_accuracy=mc["balanced_accuracy"],
                            patient_acc=float((pp["label"] == pp["pred_majority"]).mean()) if const else np.nan,
                            explained_variance=expl[rn], n_patients=len(pp), n_test_cells=N,
                        )
                    )
                    pp_rows.append(pp.assign(task=t, variant=v, model=m, representation=rn))
                    cms[v][f"{m}|{rn}"] = weighted_confusion(y, p, classes, w).tolist()
                    if do_cluster and v == "real":
                        bc_rows.append(cluster_breakdown(y, p, pats, clusters, classes).assign(
                            task=t, variant=v, model=m, representation=rn))
        res = pd.DataFrame(rows)
        out[t] = res
        if save:
            o = Path(outdir)
            save_tables(res, t, o)
            pp_all = pd.concat(pp_rows, ignore_index=True)
            pp_all.to_csv(o / f"results_per_patient_{t}.csv", index=False)
            (o / f"confusion_matrices_{t}.json").write_text(json.dumps({"classes": classes, **cms}), encoding="utf-8")
            fig_confusions({tuple(k.split("|")): np.array(c) for k, c in cms["real"].items()}, classes, t, o)
            fig_components_vs_performance(res, t, o)
            fig_per_patient(pp_all, t, o)
            if bc_rows:
                bc = pd.concat(bc_rows, ignore_index=True)
                bc.to_csv(o / f"results_by_cluster_{t}.csv", index=False)
                fig_by_cluster(bc, t, o)
    return out


if __name__ == "__main__":
    from src.features import prepare

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    out = Path(cfg.get("results_dir", f"results/{cfg['mode']}"))
    out.mkdir(parents=True, exist_ok=True)
    ds, folds = prepare(cfg, out)
    fits = importlib.import_module("experiments.02_rom").run(ds, folds, cfg, out)
    run(ds, fits, cfg, out)
