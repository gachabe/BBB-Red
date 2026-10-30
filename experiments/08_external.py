"""Aplica un paquete entrenado con Walchli a un dataset externo (p. ej. Xie et al.) SIN reajustar nada."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

from src.evaluation import to_markdown
from src.final_model import (align_genes, apply_all, check_normalization, depth_covariates, load_bundle,
                             normalize_counts)


def main(bundle_path, external, out, patient_col="Patient", label_col="glioma_type", label_map=None) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(bundle_path)
    a = ad.read_h5ad(external)
    X = a.X.astype(np.float32).tocsr() if sp.issparse(a.X) else sp.csr_matrix(np.asarray(a.X, dtype=np.float32))
    obs, genes = a.obs.copy(), np.asarray(a.var_names.astype(str))
    assert patient_col in obs.columns and label_col in obs.columns, f"faltan columnas; hay: {list(obs.columns)}"

    norm = check_normalization(X)
    print(f"[normalización] {norm['kind']}: {norm['detail']}")
    counts = X if norm["kind"] == "counts" else None
    if norm["kind"] == "counts":
        X = normalize_counts(X)
        print("  -> normalizado a 10.000 por célula + log1p (como Walchli)")
    elif norm["kind"] == "other":
        print("  AVISO: normalización distinta de la de Walchli; los resultados pueden no ser comparables")

    Xa, present = align_genes(X, genes, bundle["genes"])
    hvg = next(r for r in bundle["reps"] if r.kind == "hvg")
    hvg_present = float(present[hvg.gene_idx].mean())
    print(f"[genes] en común: {present.sum()}/{len(present)} ({present.mean():.1%}); "
          f"de los {len(hvg.gene_idx)} genes variables: {hvg_present:.1%}")
    if hvg_present < 0.9:
        print("  AVISO: faltan muchos genes variables; los genes ausentes se tratan como 0")

    C = depth_covariates(bundle, obs, counts)
    res, per_pat = apply_all(bundle, Xa, obs, C, patient_col, label_col, label_map)
    res.to_csv(out / "results_external.csv", index=False)
    per_pat.to_csv(out / "results_per_patient_external.csv", index=False)
    cols = ["representation", "model", "n_features", "macro_f1", "balanced_accuracy", "patient_acc", "n_patients", "n_cells"]
    (out / "results_external.md").write_text(to_markdown(res[cols]) + "\n", encoding="utf-8")
    (out / "external_report.json").write_text(json.dumps(dict(
        external=str(external), normalization=norm, genes_in_common=int(present.sum()), genes_bundle=int(len(present)),
        hvg_present_fraction=hvg_present, patients=int(obs[patient_col].nunique()), cells=int(X.shape[0]),
        regress_out=bundle["regress_out"]), indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + res[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--external", required=True, help="ruta al .h5ad del estudio externo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--patient-col", default="Patient")
    ap.add_argument("--label-col", default="glioma_type")
    ap.add_argument("--label-map", default=None, help="p. ej. Glioblastoma=GBM,Lower-grade=LGG")
    a = ap.parse_args()
    lm = dict(kv.split("=", 1) for kv in a.label_map.split(",")) if a.label_map else None
    main(a.bundle, a.external, a.out, a.patient_col, a.label_col, lm)
