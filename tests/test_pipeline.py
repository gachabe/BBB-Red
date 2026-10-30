import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.rom import covariates, fit_representations


def test_no_preprocessing_leakage(dataset_and_folds, cfg):
    """HVG, escalado y PCA solo dependen de train: cambiar las células de test no los altera."""
    ds, folds = dataset_and_folds
    f = folds[0]
    Xtr = ds.X[f.train_idx]
    rom = {**cfg["rom"], "regress_out": None}
    reps = fit_representations(Xtr, rom)
    X_all_mean = np.asarray(ds.X.mean(axis=0)).ravel()
    hvg = next(r for r in reps if r.kind == "hvg")
    assert np.allclose(hvg.scaler.mean_, np.asarray(Xtr[:, hvg.gene_idx].mean(axis=0)).ravel(), atol=1e-5)
    assert not np.allclose(hvg.scaler.mean_, X_all_mean[hvg.gene_idx], atol=1e-5)  # no usa a todos
    ds2 = ds.X.copy()
    ds2[f.test_idx] = ds2[f.test_idx] * 1000.0   # alterar SOLO el paciente de test
    reps2 = fit_representations(ds2[f.train_idx], rom)
    assert np.array_equal(reps[0].gene_idx, reps2[0].gene_idx)
    assert np.allclose(reps[0].pca.components_, reps2[0].pca.components_)
    # y el modelo del paciente de test no cambia sus transformaciones de train
    assert np.allclose(reps[0].transform(Xtr), reps2[0].transform(ds2[f.train_idx]))


def test_representations_dims_and_sparse_all_genes(dataset_and_folds, cfg):
    ds, folds = dataset_and_folds
    f = folds[1]
    rom = {**cfg["rom"], "regress_out": None}
    reps = fit_representations(ds.X[f.train_idx], rom)
    assert [r.name for r in reps] == ["PCA-2", "PCA-5", "HVG-60", "All genes"]
    Xte = ds.X[f.test_idx]
    for r in reps:
        Z = r.transform(Xte)
        assert Z.shape == (len(f.test_idx), r.n_features)
    all_g = reps[-1].transform(Xte)
    import scipy.sparse as sp
    assert sp.issparse(all_g) and all_g.shape[1] == ds.X.shape[1]     # no se densifica
    pca5, pca2 = reps[1], reps[0]
    assert np.allclose(pca5.transform(Xte)[:, :2], pca2.transform(Xte))   # PCA-2 = primeras 2 de PCA-5
    assert pca5.explained_variance > pca2.explained_variance > 0


def test_end_to_end(cfg):
    final = importlib.import_module("experiments.04_final")
    summary = final.main(cfg)
    out = Path(cfg["results_dir"])
    for f in ("01_cohort.png", "02_pca_variance.png", "03_confusion_glioma_type.png",
              "04_components_vs_performance_glioma_type.png", "05_per_patient_glioma_type.png",
              "07_by_cluster_glioma_type.png", "summary.json", "run_metadata.json", "results_glioma_type.csv",
              "summary_glioma_type.md", "cohort_table.csv", "splits/folds.csv"):
        assert (out / f).exists(), f
    meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    assert len(meta["folds_held_out"]) == 9 and "eda" in meta
    res = pd.DataFrame(summary["results"]["glioma_type"])
    assert len(res) == 2 * 2 * 4                                   # variantes x modelos x representaciones
    assert np.isfinite(res["macro_f1"]).all()
    real = res[res["variant"] == "real"]
    perm = res[res["variant"] == "permuted"]
    assert real["macro_f1"].max() > 0.8                            # hay señal en el sintético
    assert perm["macro_f1"].mean() < real["macro_f1"].mean()       # el control rinde peor


def test_seeds(cfg):
    aggs = importlib.import_module("experiments.05_seeds").main(cfg, n_seeds=2)
    out = Path(cfg["results_dir"])
    assert (out / "results_seeds_summary_glioma_type.md").exists()
    assert (out / "06_components_vs_performance_seeds_glioma_type.png").exists()
    assert len(aggs["glioma_type"]) == 2 * 2 * 4 and np.isfinite(aggs["glioma_type"]["macro_f1_mean"]).all()


def test_replot(cfg):
    importlib.import_module("experiments.replot").main(cfg)
    assert (Path(cfg["results_dir"]) / "03_confusion_glioma_type.png").exists()


def test_regress_out_fit_only_on_train_and_removes_depth(dataset_and_folds, cfg):
    ds, folds = dataset_and_folds
    f = folds[2]
    C = covariates(ds.meta, cfg["rom"]["regress_out"])
    reps = fit_representations(ds.X[f.train_idx], cfg["rom"], C[f.train_idx])
    hvg = next(r for r in reps if r.kind == "hvg")
    # (1) en entrenamiento el residuo no correlaciona con la profundidad
    R = hvg.regress.residual(ds.X[f.train_idx][:, hvg.gene_idx], C[f.train_idx])
    for k in range(C.shape[1]):
        cc = [abs(np.corrcoef(R[:, g], C[f.train_idx][:, k])[0, 1]) for g in range(R.shape[1]) if R[:, g].std() > 0]
        assert max(cc) < 1e-3
    # (2) los coeficientes no dependen del paciente de test: alterar X y C del test no los cambia
    X2, C2 = ds.X.copy(), C.copy()
    X2[f.test_idx] = X2[f.test_idx] * 1000.0
    C2[f.test_idx] = C2[f.test_idx] + 5.0
    reps2 = fit_representations(X2[f.train_idx], cfg["rom"], C2[f.train_idx])
    assert np.allclose(hvg.regress.B, next(r for r in reps2 if r.kind == "hvg").regress.B)
    assert np.allclose(reps[0].pca.components_, reps2[0].pca.components_)
    # (3) las representaciones con regresión salen con las dimensiones esperadas y densas
    Xte = ds.X[f.test_idx]
    for r in reps:
        Z = r.transform(Xte, C[f.test_idx])
        assert Z.shape == (len(f.test_idx), r.n_features) and not hasattr(Z, "tocsr") and np.isfinite(Z).all()


def test_regress_out_needs_covariates(dataset_and_folds, cfg):
    import pytest
    ds, folds = dataset_and_folds
    with pytest.raises(AssertionError):
        fit_representations(ds.X[folds[0].train_idx], cfg["rom"], None)


def test_every_figure_has_a_description(cfg):
    out = Path(cfg["results_dir"])
    importlib.import_module("experiments.replot").main(cfg)
    pngs = sorted(out.glob("*.png"))
    assert pngs
    for png in pngs:
        txt = out / "descriptions" / (png.stem + ".txt")
        assert txt.exists() and len(txt.read_text(encoding="utf-8")) > 80, f"falta descripción de {png.name}"
