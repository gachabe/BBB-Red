"""¿Hay señal GBM/LGG DENTRO de Xie? Mismo análisis que en Wälchli (dejar un paciente fuera) pero solo con los 7 pacientes de Xie."""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.data import GROUP, load_config
from src.evaluation import classification_metrics, fig_components_vs_performance, per_patient_table, to_markdown
from src.features import lopo_folds, task_classes
from src.models import make_model
from src.rom import covariates, fit_representations

TASK = "glioma_type"


def main(cfg: dict, xie_path, out, models=("logistic_regression", "random_forest")) -> pd.DataFrame:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(xie_path)
    X = a.X.astype(np.float32).tocsr() if sp.issparse(a.X) else sp.csr_matrix(np.asarray(a.X, dtype=np.float32))
    meta = a.obs[[GROUP, TASK, "nCount_RNA", "nFeature_RNA"]].reset_index(drop=True)
    meta["cell"] = np.arange(len(meta)).astype(str)
    classes = task_classes(meta, TASK)
    y = np.searchsorted(classes, meta[TASK].astype(str).to_numpy())
    w = 1.0 / meta.groupby(GROUP)["cell"].transform("size").to_numpy()
    rom = {**cfg["rom"], "include_all_genes": False}
    cap, seed = cfg.get("ml", {}).get("max_train_cells_per_patient"), cfg["seed"]
    C = covariates(meta, rom.get("regress_out"))
    preds: dict = {}
    for f in lopo_folds(meta, cap, seed):
        Ctr = None if C is None else C[f.train_idx]
        Cte = None if C is None else C[f.test_idx]
        for rep in fit_representations(X[f.train_idx], rom, Ctr):
            Ztr, Zte = rep.transform(X[f.train_idx], Ctr), rep.transform(X[f.test_idx], Cte)
            for m in models:
                mod = make_model(m, seed, cfg.get("ml", {}).get("rf_trees", 200)).fit(Ztr, y[f.train_idx])
                preds.setdefault((rep.name, rep.n_features, m), []).append((f.test_idx, mod.predict(Zte)))
        print(f"  paciente {f.held_out} fuera, hecho", flush=True)
    rows, pps = [], []
    for (rn, nf, m), parts in preds.items():
        idx = np.concatenate([p[0] for p in parts])
        pred = np.concatenate([p[1] for p in parts])
        met = classification_metrics(y[idx], pred, classes, w[idx])
        pp = per_patient_table(y[idx], pred, meta[GROUP].to_numpy()[idx], classes)
        rows.append(dict(model=m, variant="real", representation=rn, n_features=nf, macro_f1=met["macro_f1"],
                         balanced_accuracy=met["balanced_accuracy"], patient_acc=float((pp["label"] == pp["pred_majority"]).mean())))
        pps.append(pp.assign(representation=rn, model=m))
    res = pd.DataFrame(rows)
    res.to_csv(out / "results_within_xie.csv", index=False)
    pd.concat(pps).to_csv(out / "results_per_patient_within_xie.csv", index=False)
    (out / "results_within_xie.md").write_text(to_markdown(res.round(3)) + "\n", encoding="utf-8")
    fig_components_vs_performance(res, "glioma_type_within_xie", out)
    print("\n" + res.round(3).to_string(index=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--xie", default="data/xie/xie_endothelial.h5ad")
    ap.add_argument("--out", default="results/within_xie")
    a = ap.parse_args()
    main(load_config(a.config), a.xie, a.out)
