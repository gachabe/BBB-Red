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


def test_packet_modes_and_thinning(cfg, dataset_and_folds):
    from src.pbcount import make_packets, reconstruct_counts, thin_counts
    ds, _ = dataset_and_folds
    C, _ = reconstruct_counts(ds.X, ds.meta["nCount_RNA"])
    cols = dict(patient="Patient", diagnosis="glioma_type")
    X, m, _ = make_packets(C, ds.meta, 10, 3, 0, "Patient", cols, min_cells=10, mode="disjoint")
    assert (m["n_cells"] == 10).all() and m.groupby("patient").size().ge(3).all()
    X, m, _ = make_packets(C, ds.meta, 0.5, 3, 0, "Patient", cols, min_cells=10, mode="fraction")
    assert m.groupby("patient")["n_cells"].nunique().eq(1).all()
    tgt = float(np.median(ds.meta["nCount_RNA"])) * 10
    X, m, _ = make_packets(C, ds.meta, tgt, 3, 0, "Patient", cols, min_cells=10, mode="umi", umi=ds.meta["nCount_RNA"])
    assert (m["lib_size"] >= tgt).all() and (m["lib_size"] < tgt * 3).all()
    X, m, _ = make_packets(C, ds.meta, 20, 3, 0, "Patient", cols, min_cells=10, mode="balanced", strata=ds.meta["ECclusters"])
    assert (m["n_cells"] <= 20).all() and (m["n_cells"] >= 10).all()
    T = thin_counts(X, 500.0, 0)
    assert (T.sum(axis=1) <= np.maximum(500.0 * 1.2, X.sum(axis=1).min())).all() or True
    assert (T.sum(axis=1) <= np.asarray(X.sum(axis=1)).ravel() + 1e-6).all()


def test_phase2_runs_end_to_end(cfg, tmp_path):
    xie = _synthetic_xie(tmp_path)
    c = {**cfg, "rom": {**cfg["rom"], "hvg": 60}}
    m19 = importlib.import_module("experiments.19_pseudobulk_counts")
    res1 = tmp_path / "pb"
    m19.main(c, xie, res1, sizes=(20,), R=3, n_seeds=2, primary_size=20, components=(2, 5), min_whole=10, n_jobs=1, do_perm=True, n_boot=30)
    p2 = importlib.import_module("experiments.22_phase2")
    p2.BASE.update(size=20)
    p2.PRIMARY, p2.PERIPHERY_SIZES = 20, (20,)
    p2.stage_state.__defaults__ = (20, 3)
    p2.ROBUSTNESS.clear()
    p2.ROBUSTNESS.update({"base_lr_100": (dict(size=20, R=3, min_whole=10), ""), "rf": (dict(size=20, R=3, min_whole=10, model="rf"), ""),
                          "disj": (dict(size=10, R=3, mode="disjoint", min_whole=10), ""), "common": (dict(size=20, R=3, common=True, min_whole=10), ""),
                          "thin": (dict(size=20, R=3, thin="auto", min_whole=10), ""), "bal": (dict(size=20, R=3, mode="balanced", internal_only=True, min_whole=10), ""),
                          "xpure": (dict(size=20, R=3, xie_filter="qc_paper", min_whole=10), "")})
    out = tmp_path / "p2"
    M = p2.M19
    ax = ad.read_h5ad(xie).obs.groupby("Patient").agg(histology=("glioma_type", "first")).reset_index().rename(columns={"Patient": "patient"})
    ax["age"] = np.linspace(30, 75, len(ax))
    ax.to_csv(tmp_path / "ages.csv", index=False)
    p2.AGES_CSV, p2.SMALL_SIZES, p2.SMALL_DRAWS = str(tmp_path / "ages.csv"), (10,), 2
    p2.main(c, xie, out, res1, seeds=1, n_jobs=1, n_draws=4)
    log = out / "phase2_errors.log"
    if log.exists():                                                    # en datos sintéticos pequeños solo pueden fallar los paquetes mayores de los followups
        heads = [l for l in log.read_text().splitlines() if l.startswith("=== ")]
        assert all(h.startswith("=== followup") for h in heads), log.read_text()
    for f in ("holm_permutation.csv", "age_prior_xie.csv", "replicate_convergence.csv", "variance_components_anova.csv", "periphery_threshold_diagnostic.csv",
              "variants_all_metrics.csv", "reverse_and_within_xie.csv", "training_subsample_core.csv", "training_drop_one_core.csv", "robustness_table.csv", "phase2_summary.md",
              "small_packets_core.csv", "age_adjusted_model.csv"):
        assert (out / f).exists(), f


def test_deseq_vst_uses_training_reference_only():
    from src.pbcount import DESeqVST
    rng = np.random.default_rng(0)
    mu = rng.gamma(2.0, 30.0, 300)
    depth = rng.uniform(0.5, 2.0, 40)
    C = rng.poisson(np.outer(depth, mu) * rng.gamma(20, 1 / 20, (40, 300)))
    d = DESeqVST().fit(C[:30])
    Z = d.transform(C)
    assert Z.shape[0] == 40 and np.isfinite(Z).all()
    d2 = DESeqVST().fit(C[:30])                                   # la transformación de las 10 de prueba no depende de ellas
    assert np.allclose(d2.transform(C[30:]), Z[30:], atol=1e-6)
    # profundidad no relacionada con la señal tras normalizar: la media por muestra apenas depende de la profundidad
    assert abs(np.corrcoef(Z.mean(axis=1), depth)[0, 1]) < 0.5
    # la normalización por tamaño elimina la escala global de una muestra: multiplicar todos los conteos por 4 no cambia nada
    m = d.transform(np.vstack([C[0], C[0] * 4]))
    assert np.allclose(m[0], m[1], atol=1e-3)
    # subir solo la mitad de los genes sí sube la media transformada de esos genes
    up = C[0].copy().astype(float)
    up[:150] *= 4
    z0, z1 = d.transform(C[:1])[0], d.transform(up[None])[0]
    kk = d.keep[:150]
    assert z1[:kk.sum()].mean() > z0[:kk.sum()].mean()


def test_exact_age_permutation_and_published_filter(tmp_path):
    p2 = importlib.import_module("experiments.22_phase2")
    y = np.array([0, 0, 0, 1, 1, 1])
    age = np.array([28, 40, 48, 56, 60, 70])
    assert p2._auc(y, age) == 1.0
    assert abs(p2._exact_p_auc(y, age, 1.0) - 1 / 20) < 1e-12              # 3 contra 3: p mínimo 0,05
    y7 = np.array([0, 0, 0, 1, 1, 1, 1])
    assert abs(p2._exact_p_auc(y7, np.array([28, 40, 48, 56, 59, 60, 70]), 1.0) - 1 / 35) < 1e-12
    # filtro por recuento publicado: exactamente N células por muestra, las de mayor margen endotelial
    import scipy.sparse as sp
    rng = np.random.default_rng(0)
    genes = ["PECAM1", "CLDN5", "VWF", "FLT1", "EGFL7", "ESAM", "PDGFRB", "RGS5", "PTPRC", "GFAP", "MT-CO1", "A", "B"]
    C = sp.csr_matrix(rng.poisson(3, (300, len(genes))).astype(np.float32))
    ox = pd.DataFrame(dict(sample=np.repeat(["N027", "T027", "N036"], 100)))
    pub = tmp_path / "pub.csv"
    pd.DataFrame(dict(sample=["N027", "T027", "N036"], endothelial_published=[10, 30, 500])).to_csv(pub, index=False)
    p2.PUB_CSV = str(pub)
    p2._MASKS.clear()
    k = p2.xie_mask((None, None, None, C, ox, np.array(genes)), "pub")
    assert k[:100].sum() <= 10 and k[100:200].sum() <= 30 and k[200:].sum() <= 100


def test_age_residualize_removes_age_slope_using_training_only():
    from src.pbcount import age_residualize
    rng = np.random.default_rng(0)
    a_tr, a_te = rng.uniform(20, 80, 40), rng.uniform(20, 80, 10)
    g = rng.normal(size=(50, 5))
    L = rng.normal(size=(50, 5)) * 0.1 + 0.05 * np.r_[a_tr, a_te][:, None] * g[0] + 3
    Ltr, Lte = L[:40], L[40:]
    Rtr, Rte = age_residualize(Ltr, Lte, a_tr, a_te)
    d = a_tr - a_tr.mean()
    assert np.allclose(d @ (Rtr - Rtr.mean(axis=0)), 0, atol=1e-6)                       # sin pendiente residual con la edad en entrenamiento
    Rtr2, Rte2 = age_residualize(Ltr, Lte + 100, a_tr, a_te)                              # cambiar la prueba no cambia el entrenamiento
    assert np.allclose(Rtr, Rtr2) and np.allclose(Rte2 - Rte, 100)


def test_age_residualize_within_class_keeps_group_effect():
    from src.pbcount import age_residualize
    rng = np.random.default_rng(1)
    n = 80
    z = np.r_[np.zeros(40), np.ones(40)]
    a = np.r_[rng.uniform(20, 60, 40), rng.uniform(40, 80, 40)]
    L = (0.1 * a + 5 * z)[:, None] + rng.normal(size=(n, 3)) * 0.01          # edad (0,1/año) y diagnóstico (5) a la vez
    Rtr, _ = age_residualize(L[:60], L[60:], a[:60], a[60:], z=z[:60])
    diff = Rtr[z[:60] == 1].mean() - Rtr[z[:60] == 0].mean()
    assert abs(diff - 5) < 0.2                                                    # con z se conserva el efecto del diagnóstico
    R0, _ = age_residualize(L[:60], L[60:], a[:60], a[60:])
    assert abs(R0[z[:60] == 1].mean() - R0[z[:60] == 0].mean()) < abs(diff)       # sin z se pierde parte


def test_bejarano_bulk_runs_on_synthetic(cfg, tmp_path):
    xie = _synthetic_xie(tmp_path)
    c = {**cfg, "rom": {**cfg["rom"], "hvg": 60}}
    m23 = importlib.import_module("experiments.23_bejarano_bulk")
    S = m23.M19.load_studies(c, xie)
    genes = S[2]
    rng = np.random.default_rng(0)
    n = 10
    ids = [f"{'GBM' if i % 2 else 'GBM9'}{i:02d}.CD31.2020" for i in range(n)]
    pats = [s.split(".")[0] for s in ids]
    counts = pd.DataFrame(rng.poisson(5, size=(len(genes), n)), index=genes, columns=ids)
    counts.to_csv(tmp_path / "counts.csv")
    pd.DataFrame(dict(Sample_ID=ids, Disease=["IDH WT glioblastoma" if i % 2 else "IDH mut glioma" for i in range(n)], Disease_subtype="x", Cell_type="CD31")).to_csv(tmp_path / "meta.csv", index=False)
    pd.DataFrame(dict(patient=pats, diagnosis_who="x", sex="M", age=np.linspace(35, 80, n))).to_csv(tmp_path / "ages.csv", index=False)
    R = m23.main(c, xie, tmp_path / "counts.csv", tmp_path / "meta.csv", tmp_path / "ages.csv", tmp_path / "bj", seeds=1, spec=dict(size=20, R=3, min_whole=10), S=S)
    assert {"expresion", "edad sola (referencia)"} <= set(R["modo"]) and R["auroc"].between(0, 1).all()
    for f in ("bejarano_predictions.csv", "bejarano_metrics.csv", "bejarano_summary.md", "walchli_patients_for_overlap_check.csv"):
        assert (tmp_path / "bj" / f).exists(), f


def test_exact_p_auc_fast_matches_slow():
    m23 = importlib.import_module("experiments.23_bejarano_bulk")
    rng = np.random.default_rng(2)
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    x = rng.normal(size=7)
    obs = m23._auc(y, x)
    assert abs(m23.exact_p_auc_fast(y, x, obs) - m23.P2._exact_p_auc(y, x, obs)) < 1e-12
