"""Pseudobulks de conteos crudos (guía de los coautores): validación interna (dejar un paciente fuera) y aplicación a Xie.

Conteos crudos reconstruidos de la matriz log-normalizada (conteos = expm1(X) * nCount_RNA / 1e4, comprobado entero).
Paquetes de 50, 100, 250 y 500 células (R réplicas, sin reemplazo dentro del paquete, pueden solaparse) y el espécimen completo.
Un pseudobulk de Wälchli = un paciente; en Xie = una muestra (núcleo y periferia separados, nunca mezclados).
Pipeline ajustado solo con pacientes de entrenamiento: filtro CPM, TMM, log-CPM, escalado, HVG/PCA y regresión logística (L2, balanceada, C=1).
Una probabilidad (de GBM) por paciente = media de las de sus paquetes. Todas las métricas, la permutación y el bootstrap son por paciente.
Análisis principal fijado de antemano: paquetes de 100 células, media de probabilidades, Xie núcleo; se informan todas las representaciones.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import warnings
from pathlib import Path

import anndata as ad
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.data import build_dataset, load_config
from src.evaluation import to_markdown
from src.final_model import align_genes
from src.pbcount import (age_from_obs, bootstrap_patients, exact_permutation, icc1, lopo, make_packets, patient_metrics, predict_block,
                         reconstruct_counts, to_patient)

warnings.filterwarnings("ignore")
COLS_W = dict(patient="Patient", diagnosis="glioma_type", sample="sample_id", site="site_id", age="age_num")
COLS_X = dict(patient="Patient", diagnosis="glioma_type", sample="sample", site="site")
AGGS = {"p_mean": "mean", "p_median": "median", "p_trim": "trimmed"}


def load_studies(cfg, xie_path):
    ds = build_dataset(cfg)
    ow = ds.meta.copy()
    ow["age_num"] = age_from_obs(ow)
    ow["sample_id"], ow["site_id"] = ow["Patient"], "tumour"
    Cw, dev_w = reconstruct_counts(ds.X, ow["nCount_RNA"])
    a = ad.read_h5ad(xie_path)
    Xe = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    ox = a.obs.reset_index(drop=True)
    Cx, dev_x = reconstruct_counts(Xe, ox["nCount_RNA"])
    print(f"conteos reconstruidos: Walchli {Cw.shape} (desviación máx. {dev_w:.4f}), Xie {Cx.shape} (desviación máx. {dev_x:.4f})", flush=True)
    return Cw, ow, np.asarray(ds.genes).astype(str), Cx, ox, np.asarray(a.var_names.astype(str))


def one_run(S, rom, size, seed, R, min_whole):
    Cw, ow, gw, Cx, ox, gx = S
    Xw, mw, infw = make_packets(Cw, ow, size, R, seed, "Patient", COLS_W, "ECclusters", min_whole)
    y = (mw["diagnosis"] == "GBM").astype(int).to_numpy()
    internal = lopo(Xw, mw, y, rom).assign(seed=seed, size=size)
    internal["patient"] = internal["group"]
    Xx, mx, infx = make_packets(Cx, ox, size, R, seed, "sample", COLS_X, None, min_whole)
    Xa, _ = align_genes(Xx, gx, gw)
    pr = predict_block(np.asarray(Xw.todense(), dtype=np.float64), y, mw["patient"].to_numpy(), np.asarray(Xa.todense(), dtype=np.float64), rom)
    ext = pd.concat([pd.DataFrame(dict(packet=np.arange(len(mx)), patient=mx["patient"], sample=mx["sample"], site=mx["site"], y=(mx["diagnosis"] == "GBM").astype(int),
                                        representation=n, p=p)) for n, p in pr.items()], ignore_index=True).assign(seed=seed, size=size)
    return dict(internal=internal, external=ext, meta_w=mw, meta_x=mx, infeasible_w=infw, infeasible_x=infx)


def specimen_levels(ext: pd.DataFrame) -> dict:
    """Nivel muestra (media de sus paquetes) y paciente para núcleo, periferia y todo (media de las muestras del paciente)."""
    k = ["seed", "size", "representation", "patient", "sample", "site"]
    sp_ = ext.groupby(k, sort=True).agg(p_mean=("p", "mean"), p_sd=("p", "std"), n_packets=("p", "size"), y=("y", "first")).reset_index()
    out = {"core": sp_[sp_["site"] == "core"].copy(), "periphery": sp_[sp_["site"] == "peripheral"].copy()}
    allp = sp_.groupby(["seed", "size", "representation", "patient"], sort=True).agg(p_mean=("p_mean", "mean"), y=("y", "first"), n_specimens=("sample", "size")).reset_index()
    out["all"] = allp
    out["specimens"] = sp_
    return out


def metric_rows(df, keys, pcol, analysis, patient_set):
    rows = []
    for kv, g in df.groupby(list(keys), sort=True):
        kv = kv if isinstance(kv, tuple) else (kv,)
        rows.append(dict(zip(keys, kv), analysis=analysis, patient_set=patient_set, aggregation=pcol, **patient_metrics(g["y"], g[pcol])))
    return rows


def grid_tables(allr):
    """Métricas por paciente (una probabilidad por paciente) para cada configuración, análisis y regla de agregación."""
    internal = pd.concat([r["internal"] for r in allr], ignore_index=True)
    pat = to_patient(internal, keys=("seed", "size", "group", "representation")).rename(columns={"group": "patient"})
    sizes_all = set(pat["size"].astype(str))
    common = {s: set.intersection(*[set(pat[(pat["seed"] == s) & (pat["size"].astype(str) == z)]["patient"]) for z in sizes_all]) for s in pat["seed"].unique()}
    rows = []
    for pcol in AGGS:
        rows += metric_rows(pat, ["seed", "size", "representation"], pcol, "internal", "all_eligible")
        cc = pat[[p in common[s] for p, s in zip(pat["patient"], pat["seed"])]]
        rows += metric_rows(cc, ["seed", "size", "representation"], pcol, "internal", "complete_case")
    ext = pd.concat([r["external"] for r in allr], ignore_index=True)
    lv = specimen_levels(ext)
    for name in ("core", "periphery", "all"):
        d = lv[name]
        rows += metric_rows(d, ["seed", "size", "representation"], "p_mean", f"xie_{name}", "all_eligible")
        sizes_x = set(d["size"].astype(str))
        cc = d[[p in set.intersection(*[set(d[(d["seed"] == s) & (d["size"].astype(str) == z)]["patient"]) for z in sizes_x]) for p, s in zip(d["patient"], d["seed"])]]
        rows += metric_rows(cc, ["seed", "size", "representation"], "p_mean", f"xie_{name}", "complete_case")
    return pd.DataFrame(rows), pat, lv, internal


def stability(pat, lv, internal):
    """Variabilidad entre réplicas por paciente (semilla 0) y a nivel de cohorte: ICC, cambio entre semillas, cambios de clase frente al espécimen completo."""
    p0 = pat[pat["seed"] == 0].copy()
    whole = p0[p0["size"].astype(str) == "whole"].set_index(["representation", "patient"])["p_mean"]
    p0["whole_p"] = [whole.get((r, p), np.nan) for r, p in zip(p0["representation"], p0["patient"])]
    p0["agrees_with_whole"] = ((p0["p_mean"] >= 0.5) == (p0["whole_p"] >= 0.5)).where(p0["whole_p"].notna())
    per_patient = p0[["size", "representation", "patient", "y", "n_packets", "p_mean", "p_sd", "p_median", "p_iqr", "p_q05", "p_q95", "frac_packets_gbm", "whole_p", "agrees_with_whole"]].copy()
    per_patient["p_range"] = p0["p_max"] - p0["p_min"]
    per_patient["agree_class_frac"] = np.where(per_patient["p_mean"] >= 0.5, per_patient["frac_packets_gbm"], 1 - per_patient["frac_packets_gbm"])
    rows = []
    i0 = internal[internal["seed"] == 0]
    for (size, rep), g in per_patient.groupby(["size", "representation"]):
        if str(size) == "whole":
            continue
        ps = pat[(pat["size"] == size) & (pat["representation"] == rep)]
        wide = ps.pivot(index="patient", columns="seed", values="p_mean")
        gg = i0[(i0["size"] == size) & (i0["representation"] == rep)]
        rows.append(dict(size=size, representation=rep, n_patients=len(g), median_patient_sd=g["p_sd"].median(),
                         frac_patients_sd_below_0_05=float((g["p_sd"] < 0.05).mean()),
                         frac_patients_90pct_packets_agree=float((g["agree_class_frac"] >= 0.9).mean()),
                         icc_packets_within_patient=icc1(gg["p"], gg["group"]),
                         max_change_across_seeds=float((wide.max(axis=1) - wide.min(axis=1)).max()) if wide.shape[1] > 1 else np.nan,
                         switch_rate_vs_whole=float(1 - g["agrees_with_whole"].dropna().mean()) if g["agrees_with_whole"].notna().any() else np.nan))
    return per_patient, pd.DataFrame(rows)


def main(cfg: dict, xie_path, out, sizes=(50, 100, 250, 500), R=20, n_seeds=5, primary_size=100, components=(2, 5, 10, 20), min_whole=100,
         n_jobs=-1, do_perm=True, n_boot=5000, force=False) -> dict:
    out = Path(out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    rom = dict(components=list(components), hvg=cfg["rom"].get("hvg", 2000), scale_clip=cfg["rom"].get("scale_clip", 10.0), min_cpm=1.0, min_frac=0.25)
    S = load_studies(cfg, xie_path)
    runs = []
    todo = [(s, z) for z in list(sizes) + ["whole"] for s in (range(n_seeds) if z != "whole" else [0])]
    todo.sort(key=lambda t: (t[1] != primary_size, str(t[1]), t[0]))
    for seed, size in todo:
        f = out / "runs" / f"seed{seed}_size{size}.joblib"
        if f.exists() and not force:
            runs.append(joblib.load(f))
            continue
        r = one_run(S, rom, size, seed, R, min_whole)
        joblib.dump(r, f)
        runs.append(r)
        print(f"  hecho: tamaño {size}, semilla {seed} (no factibles Walchli: {len(r['infeasible_w'])}, Xie: {len(r['infeasible_x'])})", flush=True)
    grid, pat, lv, internal = grid_tables(runs)
    grid.to_csv(out / "pseudobulk_configuration_grid.csv", index=False)
    (out / "pseudobulk_configuration_grid.md").write_text(to_markdown(grid[(grid["aggregation"] == "p_mean") & (grid["patient_set"] == "all_eligible") & (grid["seed"] == 0)].round(3)) + "\n", encoding="utf-8")
    sens = grid[(grid["aggregation"] == "p_mean") & (grid["patient_set"] == "all_eligible")].groupby(["analysis", "size", "representation"]).agg(
        n_seeds=("seed", "nunique"), macro_f1_mean=("macro_f1", "mean"), macro_f1_min=("macro_f1", "min"), macro_f1_max=("macro_f1", "max"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"), auroc_mean=("auroc", "mean"), auroc_min=("auroc", "min"), auroc_max=("auroc", "max"),
        n_patients=("n_patients", "min")).reset_index()
    sens.to_csv(out / "pseudobulk_seed_sensitivity.csv", index=False)
    pat.rename(columns={"p_mean": "p_gbm_mean", "p_median": "p_gbm_median", "p_trim": "p_gbm_trimmed"}).to_csv(out / "internal_patient_predictions.csv", index=False)
    spec = lv["specimens"]
    spec[spec["site"] == "core"].to_csv(out / "xie_core_predictions.csv", index=False)
    spec[spec["site"] == "peripheral"].to_csv(out / "xie_periphery_predictions.csv", index=False)
    lv["all"].to_csv(out / "xie_all_patient_level_predictions.csv", index=False)
    per_patient, cohort = stability(pat, lv, internal)
    per_patient.to_csv(out / "packet_stability_results.csv", index=False)
    cohort.to_csv(out / "packet_stability_cohort.csv", index=False)
    meta = pd.concat([r["meta_w"].assign(study="Walchli") for r in runs if r["meta_w"]["seed"].iloc[0] == 0] +
                     [r["meta_x"].assign(study="Xie") for r in runs if r["meta_x"]["seed"].iloc[0] == 0], ignore_index=True)
    meta.to_csv(out / "pseudobulk_metadata.csv", index=False)
    cov = []
    for r in runs:
        if r["meta_w"]["seed"].iloc[0] != 0:
            continue
        for st, m, inf in (("Walchli", r["meta_w"], r["infeasible_w"]), ("Xie", r["meta_x"], r["infeasible_x"])):
            sz = m["size"].iloc[0]
            for u, n in m.groupby("unit").size().items():
                cov.append(dict(study=st, unit=u, size=sz, feasible=True, n_packets=int(n)))
            cov += [dict(study=st, unit=u, size=sz, feasible=False, n_packets=0, n_cells_available=n) for u, n in inf]
    cov = pd.DataFrame(cov)
    cov.to_csv(out / "pseudobulk_coverage_matrix.csv", index=False)

    perm_rows, boot_rows = [], []
    if do_perm:
        for size in (primary_size, "whole"):
            Xw, mw, _ = make_packets(S[0], S[1], size, R, 0, "Patient", COLS_W, "ECclusters", min_whole)
            y = (mw["diagnosis"] == "GBM").astype(int).to_numpy()
            _, cache = lopo(Xw, mw, y, rom, keep_cache=True)
            yb = {p: int(v) for p, v in mw.groupby("patient")["diagnosis"].first().eq("GBM").items()}
            reps = [n for n in cache[0]["Ztr"] if n.startswith("PCA")]
            perm_rows.append(exact_permutation(cache, yb, reps, n_jobs=n_jobs).assign(size=size))
            print(f"  permutación exacta hecha (tamaño {size})", flush=True)
        pd.concat(perm_rows).to_csv(out / "permutation_results.csv", index=False)
    for analysis, d in (("internal", pat[(pat["seed"] == 0)]), ("xie_core", lv["core"][lv["core"]["seed"] == 0])):
        for (size, rep), g in d.groupby(["size", "representation"]):
            if str(size) not in (str(primary_size), "whole"):
                continue
            b = bootstrap_patients(g["y"], g["p_mean"], n_boot)
            boot_rows.append(dict(analysis=analysis, size=size, representation=rep, n_patients=len(g),
                                  **{f"{k}{s}": v[i] for k, v in b.items() for s, i in (("", 0), ("_lo", 1), ("_hi", 2))}))
    pd.DataFrame(boot_rows).to_csv(out / "patient_bootstrap_results.csv", index=False)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        commit = ""
    (out / "run_metadata.json").write_text(json.dumps(dict(
        git_commit=commit, seeds=list(range(n_seeds)), sizes=list(sizes) + ["whole"], replicates=R, primary=dict(size=primary_size, aggregation="mean", external="core"),
        components=list(components), hvg=rom["hvg"], min_cells_whole=min_whole, gene_filter="CPM>1 en >=25% de los pseudobulks de entrenamiento",
        normalization="TMM (referencia de entrenamiento) + log2-CPM (prior 2)", model="LogisticRegression(L2, class_weight=balanced, liblinear, C=1), peso 1/R por paciente",
        endothelial_definition="selección por marcadores (published: no disponible)", patients_walchli=sorted(S[1]["Patient"].unique()), patients_xie=sorted(S[4]["Patient"].unique()),
        specimens_xie=sorted(S[4]["sample"].unique())), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados en {out}")
    return dict(grid=grid, patient=pat, levels=lv, stability=(per_patient, cohort), coverage=cov)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--xie", default="data/xie/xie_endothelial.h5ad")
    ap.add_argument("--out", default="results/pseudobulk_counts")
    ap.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 250, 500])
    ap.add_argument("--replicates", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--primary-size", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--no-permutation", action="store_true")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(load_config(a.config), a.xie, a.out, tuple(a.sizes), a.replicates, a.seeds, a.primary_size, n_jobs=a.n_jobs, do_perm=not a.no_permutation,
         n_boot=a.n_boot, force=a.force)
