import importlib

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.final_model import apply_all, load_bundle, save_bundle, train_bundle
from src.rom import covariates
from tests.test_batch import _xie_like


def _bundle(cfg, dataset_and_folds, tmp_path):
    ds, _ = dataset_and_folds
    save_bundle(train_bundle(ds, cfg), tmp_path / "b.joblib")
    return ds, tmp_path / "b.joblib"


def test_pc_meaning_runs_and_describes(cfg, dataset_and_folds, tmp_path):
    ds, b = _bundle(cfg, dataset_and_folds, tmp_path)
    t = importlib.import_module("experiments.12_pc_meaning").main(cfg, b, tmp_path / "pc")
    assert {"PC1", "PC2", "side", "n_cells"} <= set(t.columns) and len(t) == ds.patients.shape[0]
    assert (tmp_path / "pc" / "descriptions" / "12_pc_by_patient.txt").exists()
    assert (tmp_path / "pc" / "pc_correlations.csv").exists()


def test_bootstrap_ci_contains_point_and_is_wider_with_fewer_patients(tmp_path):
    from experiments import __name__ as _  # noqa: F401
    mod = importlib.import_module("experiments.13_bootstrap")
    rng = np.random.default_rng(0)
    def pp(n0, n1):
        rows = [dict(Patient=f"G{i}", label="GBM", accuracy=float(rng.uniform(0.6, 1)), representation="PCA-2", model="logistic_regression", variant="real") for i in range(n0)]
        rows += [dict(Patient=f"L{i}", label="LGG", accuracy=float(rng.uniform(0.6, 1)), representation="PCA-2", model="logistic_regression", variant="real") for i in range(n1)]
        return pd.DataFrame(rows)
    big, small = pp(8, 6), pp(4, 3)
    cb = mod.bootstrap_ci(big, ["GBM", "LGG"], B=500)["macro_f1"]
    cs = mod.bootstrap_ci(small, ["GBM", "LGG"], B=500)["macro_f1"]
    assert cb[1] <= cb[0] <= cb[2] + 1e-9 and cs[1] <= cs[0] <= cs[2] + 1e-9
    big.to_csv(tmp_path / "w.csv", index=False)
    small.drop(columns="variant").to_csv(tmp_path / "x.csv", index=False)
    mod.main({"Walchli": tmp_path / "w.csv", "Xie core": tmp_path / "x.csv", "No existe": tmp_path / "nada.csv"}, tmp_path / "o", B=200)
    assert (tmp_path / "o" / "descriptions" / "13_bootstrap_ci.txt").exists()
    assert (tmp_path / "o" / "bootstrap_ci.csv").exists()


def test_within_xie_runs(cfg, tmp_path):
    xie = _xie_like(tmp_path, shift=1.0)
    c = {**cfg, "rom": {**cfg["rom"], "components": [2, 5]}}
    res = importlib.import_module("experiments.14_within_xie").main(c, xie, tmp_path / "w", ("logistic_regression",))
    assert {"PCA-2", "PCA-5"} <= set(res["representation"]) and res["macro_f1"].between(0, 1).all()
    assert (tmp_path / "w" / "descriptions").exists()


def test_alignment_zero_shift_equals_unaligned_and_script_runs(cfg, dataset_and_folds, tmp_path):
    ds, b = _bundle(cfg, dataset_and_folds, tmp_path)
    bundle = load_bundle(b)
    C = covariates(ds.meta, bundle["regress_out"])
    base, _ = apply_all(bundle, ds.X, ds.meta, C, "Patient", "glioma_type")
    zero, _ = apply_all(bundle, ds.X, ds.meta, C, "Patient", "glioma_type", shift=np.zeros(ds.X.shape[1], dtype=np.float32))
    base = base[base["representation"] != "All genes"].reset_index(drop=True)
    assert np.allclose(base["macro_f1"], zero["macro_f1"], atol=1e-6)
    xie = _xie_like(tmp_path, shift=2.0)
    res = importlib.import_module("experiments.15_align_external").main(cfg, b, xie, tmp_path / "al")
    assert "macro_f1_aligned" in res.columns and "All genes" not in set(res["representation"])


def test_age_check_runs(tmp_path):
    rng = np.random.default_rng(1)
    t = pd.DataFrame({"Patient": [f"P{i}" for i in range(14)], "glioma_type": ["GBM"] * 8 + ["LGG"] * 6,
                      "age_num": np.r_[rng.uniform(48, 80, 8), rng.uniform(25, 50, 6)], "PC2": np.r_[rng.normal(-4, 3, 8), rng.normal(7, 3, 6)]})
    t.to_csv(tmp_path / "t.csv", index=False)
    res = importlib.import_module("experiments.16_age_check").main(tmp_path / "t.csv", tmp_path / "o")
    assert res["valor"].notna().all() and (tmp_path / "o" / "descriptions" / "16_age_vs_pc2.txt").exists()


def test_alignment_loo_runs(cfg, dataset_and_folds, tmp_path):
    ds, b = _bundle(cfg, dataset_and_folds, tmp_path)
    xie = _xie_like(tmp_path, shift=2.0)
    res = importlib.import_module("experiments.15_align_external").main(cfg, b, xie, tmp_path / "loo", loo=True)
    assert {"macro_f1_loo", "patient_acc_loo"} <= set(res.columns) and res["macro_f1_loo"].between(0, 1).all()
