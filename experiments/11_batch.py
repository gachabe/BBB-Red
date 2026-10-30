"""Diagnóstico de estudio (batch) entre Walchli y Xie: numérico + gráficas + descripciones.

Salida en --out (p. ej. results/batch): batch_domain_results.csv/.md, batch_domain_per_patient.csv, batch_pca_eta2.csv,
batch_model_space_summary.csv, 09/10/11_*.png y descriptions/*.txt.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.batch import STUDIES, common_genes, depth_only_cv, domain_cv, joint_pca, model_space
from src.data import GROUP, build_dataset, load_config
from src.evaluation import fig_batch_domain, fig_batch_joint, fig_batch_model_space, to_markdown
from src.final_model import load_bundle

KEEP = [GROUP, "glioma_type", "nCount_RNA", "nFeature_RNA"]


def main(cfg: dict, xie_path, out, bundle_path=None, models=("logistic_regression", "random_forest")) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ds = build_dataset(cfg)
    a = ad.read_h5ad(xie_path)
    Xx = a.X.astype(np.float32).tocsr() if sp.issparse(a.X) else sp.csr_matrix(np.asarray(a.X, dtype=np.float32))
    ox, gx = a.obs.copy(), np.asarray(a.var_names.astype(str))
    for c in KEEP:
        assert c in ox.columns, f"Xie no tiene la columna {c}"
    mw = ds.meta[KEEP].assign(study=STUDIES[0])
    mx = ox[KEEP].assign(study=STUDIES[1]).reset_index(drop=True)
    assert not set(mw[GROUP]) & set(mx[GROUP]), "hay pacientes con el mismo nombre en los dos estudios"
    Xw, Xxc, genes = common_genes(ds.X, ds.genes, Xx, gx)
    print(f"Walchli {Xw.shape[0]:,} células, Xie {Xxc.shape[0]:,} células, {len(genes):,} genes en común")
    meta = pd.concat([mw, mx], ignore_index=True)
    meta["cell"] = np.arange(len(meta)).astype(str)
    X = sp.vstack([Xw, Xxc]).tocsr()
    cap = cfg.get("ml", {}).get("max_train_cells_per_patient")
    seed, n_trees = cfg["seed"], cfg.get("ml", {}).get("rf_trees", 200)

    res, pp = [], []
    for regress in (True, False):
        r, p = domain_cv(X, meta, cfg["rom"], list(models), cap, seed, n_trees, regress)
        res.append(r)
        pp.append(p)
    res, pp = pd.concat(res, ignore_index=True), pd.concat(pp, ignore_index=True)
    depth = depth_only_cv(meta, seed)
    res.to_csv(out / "batch_domain_results.csv", index=False)
    pp.to_csv(out / "batch_domain_per_patient.csv", index=False)
    (out / "batch_domain_results.md").write_text(
        to_markdown(res) + f"\n\nSolo profundidad: {depth}\n", encoding="utf-8")
    fig_batch_domain(res, depth, out)

    Zp, mp, eta, var = joint_pca(X, meta, cfg["rom"], cap, seed)
    eta.to_csv(out / "batch_pca_eta2.csv", index=False)
    fig_batch_joint(pd.DataFrame(dict(PC1=Zp[:, 0], PC2=Zp[:, 1], study=mp["study"], glioma_type=mp["glioma_type"])), eta, var, out)

    summary = {}
    if bundle_path:
        bundle = load_bundle(bundle_path)
        df, w, b, classes = model_space(bundle, ds.X, ds.meta, Xx, gx, ox.reset_index(drop=True))
        df.groupby(["study", "glioma_type", "side"]).size().rename("cells").reset_index().to_csv(out / "batch_model_space_summary.csv", index=False)
        sample = df.groupby("study", group_keys=False).sample(n=min(4000, int(df.groupby("study").size().min())), random_state=0)
        fig_batch_model_space(sample.assign(side=sample["side"]), (w, b), classes, out)
        summary["model_space_cells"] = len(df)
    best = res.sort_values("balanced_accuracy", ascending=False).head(3)
    print("\n" + res.round(3).to_string(index=False))
    print(f"\nSolo profundidad: {depth}")
    print("\nMayor identificación del estudio:\n" + best.round(3).to_string(index=False))
    return dict(results=res, depth=depth, eta=eta, **summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--xie", required=True)
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--out", default="results/batch")
    ap.add_argument("--models", nargs="*", default=["logistic_regression", "random_forest"])
    a = ap.parse_args()
    main(load_config(a.config), a.xie, a.out, a.bundle, a.models)
