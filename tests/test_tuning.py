import importlib

import numpy as np

from src.final_model import apply_all, load_bundle
from src.rom import covariates
from src.tuning import hp_grid, hp_key, inner_scores, make_ctx, select_best

TUNING = {"lr_C": [0.1, 1.0], "rf_grid": [{"max_depth": 3, "min_samples_leaf": 5}, {"max_depth": None, "min_samples_leaf": 1}]}


def test_inner_scores_and_selection_only_use_given_patients(cfg, dataset_and_folds):
    ds, _ = dataset_and_folds
    c = {**cfg, "tuning": TUNING}
    ctx = make_ctx(ds, c)
    allp = sorted(ctx.full)
    pats = [p for p in allp if p.startswith("GBM")][:3] + [p for p in allp if p.startswith("LGG")][:3]
    sc = inner_scores(ctx, pats, log=lambda *_: None)
    # 4 representaciones sin 'All genes' (PCA-2, PCA-5, HVG) = 3; cada una con rejilla LR (2) + RF (2)
    assert len(sc) == 3 * (len(TUNING["lr_C"]) + len(TUNING["rf_grid"]))
    chosen = select_best(sc, c)
    for (rn, m), hp in chosen.items():
        assert hp_key(hp) in [hp_key(h) for h in hp_grid(m, TUNING)]


def test_nested_pipeline_end_to_end(cfg, tmp_path):
    c = {**cfg, "tuning": TUNING, "results_dir": str(tmp_path / "r"), "models_dir": str(tmp_path / "m")}
    mod = importlib.import_module("experiments.09_tune")
    nested = mod.main(c)
    assert {"PCA-2", "PCA-5", "HVG-60", "All genes"} <= set(nested["representation"])
    assert nested["macro_f1"].between(0, 1).all()
    assert (tmp_path / "r" / "tuning" / "results_nested_glioma_type.md").exists()
    b = load_bundle(tmp_path / "m" / "bundle.joblib")
    assert any("PCA-2" in k for k in b["hparams"])
    from src.data import build_dataset
    ds = build_dataset(c)
    res, _ = apply_all(b, ds.X, ds.meta, covariates(ds.meta, b["regress_out"]), "Patient", "glioma_type")
    assert len(res) == len(b["models"]) and np.isfinite(res["macro_f1"]).all()
    # reanudar no recalcula: segunda llamada usa los .pkl
    assert mod.main(c).shape == nested.shape
