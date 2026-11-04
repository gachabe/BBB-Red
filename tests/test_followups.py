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


def test_pseudobulks_same_patient_and_mean(cfg, dataset_and_folds):
    from src.pseudobulk import make_pseudobulks
    ds, _ = dataset_and_folds
    Xp, meta, dropped = make_pseudobulks(ds.X, ds.meta, size=20, seed=0)
    assert not dropped and Xp.shape[0] == len(meta) and Xp.shape[1] == ds.X.shape[1]
    for p, g in meta.groupby("Patient"):
        n = int((ds.meta["Patient"] == p).sum())
        assert g["n_cells"].sum() == n and (g["n_cells"] >= min(20, n)).all()          # cada célula en un solo paquete
        assert np.allclose(np.asarray(Xp[g.index.to_numpy()].multiply(g["n_cells"].to_numpy()[:, None]).sum(axis=0)).ravel(),
                           np.asarray(ds.X[(ds.meta["Patient"] == p).to_numpy()].sum(axis=0)).ravel(), atol=1e-3)
    assert meta.groupby("Patient")["glioma_type"].nunique().max() == 1


def test_pseudobulk_experiment_runs_and_resumes(cfg, tmp_path):
    xie = _xie_like(tmp_path, shift=2.0)
    a = ad.read_h5ad(xie)
    a.obs["sample"] = [("T" if i % 2 else "N") + p for i, p in enumerate(a.obs["Patient"])]
    a.obs["site"] = ["core" if s.startswith("T") else "peripheral" for s in a.obs["sample"]]
    a.write_h5ad(xie)
    c = {**cfg, "rom": {**cfg["rom"], "components": [2, 5]}, "ml": {**cfg["ml"], "max_train_cells_per_patient": 40}}
    mod = importlib.import_module("experiments.17_pseudobulk")
    r = mod.main(c, xie, tmp_path / "pb", sizes=(10,), n_seeds=2, regress="both", primary_size=10, models=["logistic_regression"], include_all=False)
    w = r["all"]["within"]
    assert set(w["regress_depth"]) == {True, False} and w["macro_f1"].between(0, 1).all() and {"macro_f1_patient", "patient_acc"} <= set(w.columns)
    assert set(r["all"]["external"]["subset"]) == {"todo", "core", "periferia"}
    assert r["all"]["batch"]["auc"].between(0, 1).all()
    assert (tmp_path / "pb" / "descriptions" / "17_pseudobulk.txt").exists() and (tmp_path / "pb" / "results_per_patient_pb_within.csv").exists()
    before = sorted((tmp_path / "pb" / "runs").iterdir())
    mod.main(c, xie, tmp_path / "pb", sizes=(10,), n_seeds=2, regress="both", primary_size=10, models=["logistic_regression"], include_all=False)
    assert sorted((tmp_path / "pb" / "runs").iterdir()) == before and len(before) == 4


def _synthetic_xie(tmp_path):
    from tests.synthetic import make_synthetic
    xie = make_synthetic(tmp_path / "xie_c", n_gbm=4, n_lgg=3, seed=5)
    a = ad.read_h5ad(xie)
    a.obs["Patient"] = ["X" + str(x) for x in a.obs["Patient"]]
    a.obs["sample"] = [("T" if i % 2 else "N") + p for i, p in enumerate(a.obs["Patient"])]
    a.obs["site"] = ["core" if s.startswith("T") else "peripheral" for s in a.obs["sample"]]
    a.write_h5ad(xie)
    return xie


def test_counts_tmm_and_packets(cfg, dataset_and_folds):
    from src.pbcount import TMM, make_packets, reconstruct_counts, _tmm_factor
    ds, _ = dataset_and_folds
    C, dev = reconstruct_counts(ds.X, ds.meta["nCount_RNA"])
    assert dev < 0.01 and np.allclose(np.asarray(C.sum(axis=1)).ravel(), ds.meta["nCount_RNA"].to_numpy())
    Xp, meta, inf = make_packets(C, ds.meta, 20, 3, 0, "Patient", dict(patient="Patient", diagnosis="glioma_type"), min_cells=10)
    assert (meta["n_cells"] == 20).all() and meta.groupby("patient").size().eq(3).all() and not inf
    Xq, _, _ = make_packets(C, ds.meta, 20, 3, 0, "Patient", dict(patient="Patient"), min_cells=10)
    assert (Xp != Xq).nnz == 0                                                    # misma semilla, mismos paquetes
    Xw, mw, _ = make_packets(C, ds.meta, "whole", 1, 0, "Patient", dict(patient="Patient"), min_cells=10)
    assert np.allclose(np.asarray(Xw.sum(axis=1)).ravel(), mw["lib_size"])
    D = np.asarray(Xp.todense(), dtype=float)
    t = TMM().fit(D)
    assert np.isfinite(t.transform(D)).all()
    assert abs(_tmm_factor(D[0][t.keep], D[0][t.keep], D[0][t.keep].sum(), D[0][t.keep].sum()) - 1) < 1e-9


def test_pbcount_lopo_no_leak_and_permutation(cfg, dataset_and_folds):
    from src.pbcount import lopo, make_packets, reconstruct_counts, to_patient, patient_metrics, exact_permutation
    ds, _ = dataset_and_folds
    C, _ = reconstruct_counts(ds.X, ds.meta["nCount_RNA"])
    Xp, m, _ = make_packets(C, ds.meta, 20, 3, 0, "Patient", dict(patient="Patient", diagnosis="glioma_type"), min_cells=10)
    y = (m["diagnosis"] == "GBM").astype(int).to_numpy()
    rom = dict(components=[2, 5], hvg=60, scale_clip=10.0, min_frac=0.25)
    df, cache = lopo(Xp, m, y, rom, keep_cache=True)
    assert all(f["held"] not in set(f["tr_group"]) for f in cache)                  # los paquetes de un paciente no se reparten
    assert df.groupby("group")["packet"].apply(lambda s: s.nunique()).eq(3).all()
    pat = to_patient(df)
    assert len(pat[pat["representation"] == "PCA-2"]) == 9 and patient_metrics(pat["y"], pat["p_mean"])["macro_f1"] >= 0
    yb = {p: int(v) for p, v in m.groupby("patient")["diagnosis"].first().eq("GBM").items()}
    perm = exact_permutation(cache, yb, ["PCA-2", "PCA-5"], n_jobs=1)
    assert (perm["n_assignments"] == 126).all() and perm["p_exact"].between(0, 1).all() and (perm["p_adjusted_maxT"] >= perm["p_exact"] - 1e-9).all()


def test_audit_xie_runs(tmp_path):
    import json
    xie = _synthetic_xie(tmp_path)
    a = ad.read_h5ad(xie)
    smp = a.obs.drop_duplicates("sample")[["sample", "Patient", "site", "glioma_type"]].rename(columns={"Patient": "patient"})
    smp.insert(0, "gsm", ["G" + str(i) for i in range(len(smp))])
    smp.to_csv(tmp_path / "s.csv", index=False)
    per = [dict(sample=s, celulas_crudas=500, tras_QC=480, endoteliales=int((a.obs["sample"] == s).sum())) for s in smp["sample"]]
    (tmp_path / "r.json").write_text(json.dumps(dict(por_muestra=per)), encoding="utf-8")
    t = importlib.import_module("experiments.18_audit_xie").main(xie, tmp_path / "s.csv", tmp_path / "r.json", tmp_path / "aud")
    assert len(t) == len(smp) and (tmp_path / "aud" / "xie_audit.csv").exists()


def test_pseudobulk_counts_experiment(cfg, tmp_path):
    xie = _synthetic_xie(tmp_path)
    c = {**cfg, "rom": {**cfg["rom"], "hvg": 60}}
    mod = importlib.import_module("experiments.19_pseudobulk_counts")
    r = mod.main(c, xie, tmp_path / "pb", sizes=(20, 40), R=3, n_seeds=2, primary_size=20, components=(2, 5), min_whole=10, n_jobs=1, do_perm=True, n_boot=50)
    g = r["grid"]
    assert {"internal", "xie_core", "xie_periphery", "xie_all"} <= set(g["analysis"]) and g["macro_f1"].between(0, 1).all()
    for f in ("pseudobulk_configuration_grid.csv", "pseudobulk_seed_sensitivity.csv", "internal_patient_predictions.csv", "xie_core_predictions.csv",
              "xie_periphery_predictions.csv", "packet_stability_results.csv", "pseudobulk_coverage_matrix.csv", "permutation_results.csv",
              "patient_bootstrap_results.csv", "run_metadata.json"):
        assert (tmp_path / "pb" / f).exists(), f
    before = sorted((tmp_path / "pb" / "runs").iterdir())
    mod.main(c, xie, tmp_path / "pb", sizes=(20, 40), R=3, n_seeds=2, primary_size=20, components=(2, 5), min_whole=10, n_jobs=1, do_perm=False, n_boot=20)
    assert sorted((tmp_path / "pb" / "runs").iterdir()) == before                       # reanudable


def test_controls_and_report(cfg, tmp_path):
    xie = _synthetic_xie(tmp_path)
    c = {**cfg, "rom": {**cfg["rom"], "hvg": 60}}
    m19 = importlib.import_module("experiments.19_pseudobulk_counts")
    res = tmp_path / "pb"
    m19.main(c, xie, res, sizes=(20,), R=3, n_seeds=2, primary_size=20, components=(2, 5), min_whole=10, n_jobs=1, do_perm=True, n_boot=30)
    r = importlib.import_module("experiments.20_pseudobulk_controls").main(c, xie, res, size=20, R=3, seeds=1, components=(2, 5), min_whole=10)
    assert {"study", "site", "age", "comp"} == set(r) and r["study"]["auroc"].between(0, 1).all() and {"edad sola", "expresión sola"} <= set(r["age"]["model"])
    pd.DataFrame(dict(representation=["PCA-2", "PCA-5"] * 2, model="logistic_regression", variant="real", Patient=["A", "A", "B", "B"],
                      label=["GBM", "GBM", "LGG", "LGG"], accuracy=[0.9, 0.8, 0.7, 0.6])).to_csv(tmp_path / "ci.csv", index=False)
    pd.DataFrame(dict(representation=["PCA-2", "PCA-5"], model="logistic_regression", macro_f1=[0.6, 0.7])).to_csv(tmp_path / "ce.csv", index=False)
    tab = importlib.import_module("experiments.21_pseudobulk_report").main(res, tmp_path / "ci.csv", tmp_path / "ce.csv", tmp_path / "ce.csv", 20, c, xie)
    assert {"cell_level_internal_macro_f1", "pseudobulk_xie_core_macro_f1", "permutation_p_value", "study_auc"} <= set(tab.columns)
    for f in ("21_stability_vs_packet_size", "21_patient_predictions", "21_cell_vs_pseudobulk", "21_core_vs_periphery", "21_pseudobulk_pca"):
        assert (res / (f + ".png")).exists() and (res / "descriptions" / (f + ".txt")).exists(), f
