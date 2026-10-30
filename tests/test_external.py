import importlib

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.final_model import align_genes, check_normalization, load_bundle, normalize_counts, save_bundle, train_bundle


def test_train_apply_and_gene_alignment(cfg, dataset_and_folds, tmp_path):
    ds, _ = dataset_and_folds
    bundle = train_bundle(ds, cfg)
    save_bundle(bundle, tmp_path / "b.joblib")
    b = load_bundle(tmp_path / "b.joblib")
    assert set(r.name for r in b["reps"]) >= {"PCA-2", "PCA-5", "HVG-60", "All genes"}
    # externo = mismo dato con los genes barajados y renombrados a minúsculas: debe dar lo mismo que el directo
    rng = np.random.default_rng(1)
    perm = rng.permutation(ds.X.shape[1])
    ext = ad.AnnData(X=sp.csr_matrix(ds.X[:, perm]), obs=ds.meta.set_index("cell"),
                     var=pd.DataFrame(index=[str(g).lower() for g in ds.genes[perm]]))
    p = tmp_path / "ext.h5ad"
    ext.write_h5ad(p)
    mod = importlib.import_module("experiments.08_external")
    mod.main(tmp_path / "b.joblib", p, tmp_path / "out")
    ext_res = pd.read_csv(tmp_path / "out" / "results_external.csv")
    from src.final_model import apply_all
    from src.rom import covariates
    direct, _ = apply_all(b, ds.X, ds.meta, covariates(ds.meta, b["regress_out"]), "Patient", "glioma_type")
    m = ext_res.merge(direct, on=["representation", "model"], suffixes=("_e", "_d"))
    assert len(m) == len(direct) > 0
    assert np.allclose(m["macro_f1_e"], m["macro_f1_d"], atol=1e-5)
    # con estos datos sintéticos el modelo entrenado ve señal (acierto por encima del azar en alguna fila)
    assert ext_res["balanced_accuracy"].max() > 0.6


def test_align_genes_missing_are_zero():
    X = sp.csr_matrix(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    Xa, present = align_genes(X, ["B", "A", "Z"], ["a", "b", "c", "z"])
    assert present.tolist() == [True, True, False, True]
    assert Xa.toarray().tolist() == [[2.0, 1.0, 0.0, 3.0]]


def test_normalization_detection():
    counts = sp.csr_matrix(np.array([[3, 0, 1], [0, 2, 2]], dtype=np.float32))
    assert check_normalization(counts)["kind"] == "counts"
    assert check_normalization(normalize_counts(counts))["kind"] == "log_norm"
    assert check_normalization(sp.csr_matrix(np.array([[0.5, 0.2], [0.1, 0.9]], dtype=np.float32)))["kind"] == "other"
