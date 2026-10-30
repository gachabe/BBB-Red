"""EXPLORATORIO: aplica el paquete a Xie tras alinear las medias por gen (patrón por paciente: media de Xie -> media de Walchli).

Usa las células de Xie SIN etiquetas para calcular las medias (transductivo); no es validación externa limpia. Omite 'All genes'.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.data import GROUP, build_dataset, load_config
from src.evaluation import to_markdown
from src.final_model import align_genes, apply_all, load_bundle
from src.permutation import stats_from_fraction
from src.rom import covariates


def patient_balanced_mean(X, meta) -> np.ndarray:
    """Media por gen donde cada paciente pesa igual."""
    means = [np.asarray(X[np.asarray(i)].mean(axis=0)).ravel() for _, i in meta.groupby(GROUP, sort=True).indices.items()]
    return np.mean(means, axis=0)


def loo_table(b, Xa, present, obs, ds, C) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Alineación SIN usar las células del paciente evaluado: el desplazamiento de cada paciente sale de los otros pacientes de Xie."""
    base = patient_balanced_mean(ds.X, ds.meta)
    pats = obs[GROUP].astype(str).to_numpy()
    pps = []
    for p in sorted(set(pats)):
        te = np.flatnonzero(pats == p)
        tr = np.flatnonzero(pats != p)
        shift = (patient_balanced_mean(Xa[tr], obs.iloc[tr].reset_index(drop=True)) - base).astype(np.float32)
        shift[~present] = 0.0
        _, pp = apply_all(b, Xa[te], obs.iloc[te].reset_index(drop=True), None if C is None else C[te], GROUP, "glioma_type", shift=shift)
        pps.append(pp)
        print(f"  paciente {p} evaluado con desplazamiento de los otros, hecho", flush=True)
    pp = pd.concat(pps, ignore_index=True)
    rows = []
    for (rep, model), g in pp.groupby(["representation", "model"], sort=False):
        a_ = (g["label"].to_numpy() == "LGG").astype(int)
        acc = g["accuracy"].to_numpy(dtype=float)
        f1, ba, pa = stats_from_fraction(np.where(a_ == 1, acc, 1 - acc), a_)
        rows.append(dict(representation=rep, model=model, macro_f1_loo=f1, balanced_accuracy_loo=ba, patient_acc_loo=pa))
    return pd.DataFrame(rows), pp


def main(cfg: dict, bundle_path, xie_path, out, unaligned_csv=None, loo: bool = False) -> pd.DataFrame:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    b = load_bundle(bundle_path)
    ds = build_dataset(cfg)
    a = ad.read_h5ad(xie_path)
    Xx = a.X.astype(np.float32).tocsr() if sp.issparse(a.X) else sp.csr_matrix(np.asarray(a.X, dtype=np.float32))
    obs = a.obs.reset_index(drop=True)
    Xa, present = align_genes(Xx, np.asarray(a.var_names.astype(str)), b["genes"])
    shift = (patient_balanced_mean(Xa, obs) - patient_balanced_mean(ds.X, ds.meta)).astype(np.float32)
    shift[~present] = 0.0
    print(f"desplazamiento medio |shift| sobre los genes en común: {np.abs(shift[present]).mean():.4f}")
    C = covariates(obs, b["regress_out"])
    if loo:
        res, pp = loo_table(b, Xa, present, obs, ds, C)
        pp.to_csv(out / "results_per_patient_external_aligned_loo.csv", index=False)
        res.to_csv(out / "results_external_aligned_loo.csv", index=False)
        (out / "results_external_aligned_loo.md").write_text(to_markdown(res.round(3)) + "\n", encoding="utf-8")
        print("\n" + res.round(3).to_string(index=False))
        return res
    res, pp = apply_all(b, Xa, obs, C, GROUP, "glioma_type", shift=shift)
    res = res.rename(columns={"macro_f1": "macro_f1_aligned", "balanced_accuracy": "balanced_accuracy_aligned", "patient_acc": "patient_acc_aligned"})
    if unaligned_csv and Path(unaligned_csv).exists():
        u = pd.read_csv(unaligned_csv)[["representation", "model", "macro_f1", "balanced_accuracy", "patient_acc"]]
        res = res.merge(u.rename(columns={"macro_f1": "macro_f1_unaligned", "balanced_accuracy": "balanced_accuracy_unaligned", "patient_acc": "patient_acc_unaligned"}),
                        on=["representation", "model"], how="left")
    res.to_csv(out / "results_external_aligned.csv", index=False)
    pp.to_csv(out / "results_per_patient_external_aligned.csv", index=False)
    cols = [c for c in res.columns if c.startswith(("representation", "model", "macro_f1", "balanced_accuracy", "patient_acc"))]
    (out / "results_external_aligned.md").write_text(to_markdown(res[cols].round(3)) + "\n", encoding="utf-8")
    print("\n" + res[cols].round(3).to_string(index=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--xie", default="data/xie/xie_endothelial.h5ad")
    ap.add_argument("--out", default="results/external_xie_aligned")
    ap.add_argument("--unaligned", default="results/external_xie/results_external.csv")
    ap.add_argument("--loo", action="store_true", help="el desplazamiento de cada paciente se calcula con los OTROS pacientes de Xie (no usa sus células)")
    a = ap.parse_args()
    main(load_config(a.config), a.bundle, a.xie, a.out, a.unaligned, a.loo)
