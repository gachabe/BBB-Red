import importlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.final_model import save_bundle, train_bundle
from tests.synthetic import make_synthetic


def _xie_like(tmp_path, shift: float):
    p = make_synthetic(tmp_path / "xie", n_gbm=4, n_lgg=3, seed=5)
    a = ad.read_h5ad(p)
    rng = np.random.default_rng(1)
    cols = rng.choice(a.n_vars, 120, replace=False)
    scale = np.ones(a.n_vars, dtype=np.float32)
    scale[cols] = shift                                                   # efecto de estudio: cambia 120 genes
    a.X = sp.csr_matrix(sp.csc_matrix(a.X).dot(sp.diags(scale)))
    a.obs["Patient"] = ["X" + str(x) for x in a.obs["Patient"]]
    a.obs["nCount_RNA"] = a.obs["nCount_RNA"] * 0.5
    a.write_h5ad(p)
    return p


def test_batch_detects_study_effect_and_writes_outputs(cfg, dataset_and_folds, tmp_path):
    ds, _ = dataset_and_folds
    bundle = train_bundle(ds, cfg)
    save_bundle(bundle, tmp_path / "b.joblib")
    xie = _xie_like(tmp_path, shift=2.5)
    out = tmp_path / "batch"
    r = importlib.import_module("experiments.11_batch").main(cfg, xie, out, tmp_path / "b.joblib", ("logistic_regression",))
    res = r["results"]
    assert res["balanced_accuracy"].max() > 0.8            # el estudio se identifica con facilidad
    for f in ("09_batch_model_space.png", "10_batch_joint_pca.png", "11_batch_domain_classifier.png"):
        assert (out / f).exists()
        txt = out / "descriptions" / (Path(f).stem + ".txt")
        assert txt.exists() and len(txt.read_text(encoding="utf-8")) > 100
    assert (out / "batch_domain_results.csv").exists() and (out / "batch_pca_eta2.csv").exists()
    assert set(r["eta"]["component"]) == {"PC1", "PC2", "PC3", "PC4", "PC5"}


def test_batch_no_effect_gives_chance_level(cfg, tmp_path):
    from src.batch import domain_cv
    rng = np.random.default_rng(0)
    n_pat, n_cell, G = 8, 60, 40
    X = sp.csr_matrix(rng.random((n_pat * n_cell, G)).astype(np.float32))      # sin diferencias entre estudios
    meta = pd.DataFrame(dict(Patient=np.repeat([f"P{i}" for i in range(n_pat)], n_cell),
                             study=np.repeat(["Walchli"] * 4 + ["Xie"] * 4, n_cell), glioma_type="GBM",
                             nCount_RNA=100.0, nFeature_RNA=50.0, cell=np.arange(n_pat * n_cell).astype(str)))
    rom = {"hvg": 20, "components": [2, 5], "scale_clip": 10.0, "regress_out": None}
    res, _ = domain_cv(X, meta, rom, ["logistic_regression"], 50, 0, 20, regress=False, log=lambda *_: None)
    assert res["auc"].max() < 0.75
