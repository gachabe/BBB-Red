"""Validación externa con la cohorte de RNA-seq en masa de células endoteliales (CD31+) de Bejarano et al. 2025 (Immunity).

18 pacientes con glioma (7 IDH mutado, 11 GBM IDH nativo), edades solapadas (Tabla S1 del artículo, configs/bejarano_bulk_ages.csv).
Datos: portal https://joycelab.shinyapps.io/braintime (pestaña «Bejarano et al., Immunity 2025», Download, Human data): conteos crudos y metadatos.

Se entrena con los 14 pacientes de Wälchli (paquetes de 100 células, como siempre) y se predice cada muestra en masa, sin reajustar nada.
Se informa, por representación y por modo (expresión, con la pendiente de la edad quitada, y con la pendiente dentro de cada diagnóstico):
AUROC, p exacto (todas las asignaciones de etiquetas), macro-F1 con corte 0,5 y con las probabilidades centradas, probabilidad media por clase,
AUROC dentro de la zona de edades solapadas, y la referencia de la edad sola. Control: dejar un paciente fuera dentro de la propia cohorte.
Aviso: RNA-seq en masa frente a scRNA-seq de 10x; se espera que se conserve el orden y falle el umbral.
"""
from __future__ import annotations

import argparse
import importlib
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata

from src.data import load_config
from src.evaluation import to_markdown
from src.final_model import align_genes
from src.pbcount import lopo, patient_metrics, predict_block, to_patient

P2 = importlib.import_module("experiments.22_phase2")
M19 = P2.M19
MODES = {"expresion": "expr", "edad quitada de los genes": "age_adj", "edad quitada dentro de cada diagnostico": "age_adj_dx"}
POS = "IDH WT glioblastoma"
NEG = "IDH mut glioma"


def load_bulk(counts_csv, meta_csv, ages_csv):
    d = pd.read_csv(counts_csv, index_col=0)                                  # genes x muestras
    m = pd.read_csv(meta_csv)
    m = m[(m["Cell_type"] == "CD31") & m["Disease"].isin([POS, NEG])].copy()
    m["patient"] = m["Sample_ID"].str.split(".").str[0]
    ages = pd.read_csv(ages_csv)
    ages["patient"] = ages["patient"].astype(str).str.strip()
    m = m.merge(ages[["patient", "age", "sex", "diagnosis_who"]], on="patient", how="left")
    miss = m[m["age"].isna()]["patient"].tolist()
    assert not miss, f"faltan edades de: {miss}"
    missing = [s for s in m["Sample_ID"] if s not in d.columns]
    assert not missing, f"muestras sin conteos: {missing}"
    X = sp.csr_matrix(np.rint(d[m["Sample_ID"]].to_numpy().T).astype(np.float32))
    m["y"] = (m["Disease"] == POS).astype(int)
    return X, np.asarray(d.index.astype(str)), m.reset_index(drop=True)


def exact_p_auc_fast(y, x, obs):
    """p exacto unilateral del AUROC: fracción de reparticiones de las etiquetas (mismo nº de positivos) con AUROC >= observado."""
    y, x = np.asarray(y), np.asarray(x, dtype=float)
    n, k = len(y), int(y.sum())
    r = rankdata(x)
    combos = np.array(list(combinations(range(n), k)))
    auc = (r[combos].sum(axis=1) - k * (k + 1) / 2) / (k * (n - k))
    return float((auc >= obs - 1e-12).mean())


def _auc(y, x):
    return P2._auc(y, x)


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main(cfg, xie_path, counts, meta, ages, out, seeds=3, spec=None, S=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    S = S or M19.load_studies(cfg, xie_path)
    rom = dict(components=[2, 5, 10, 20], hvg=cfg["rom"].get("hvg", 2000), scale_clip=cfg["rom"].get("scale_clip", 10.0), min_cpm=1.0, min_frac=0.25)
    Xb, gb, mb = load_bulk(counts, meta, ages)
    Xa, present = align_genes(Xb, gb, S[2])
    A = np.asarray(Xa.todense(), dtype=np.float64)
    print(f"muestras: {len(mb)} ({int(mb['y'].sum())} GBM, {int((1 - mb['y']).sum())} IDH mutado); genes de Wälchli presentes en la cohorte: {int(present.sum())} de {len(present)}", flush=True)
    spec = dict(spec or P2.BASE)
    age_b = mb["age"].to_numpy(dtype=float)[:, None]
    rows = []
    for seed in range(seeds):
        b = P2.build(S, spec, seed, need_ext=False)
        W, mw, y = b["W"], b["mw"], b["y"]
        age_w = pd.to_numeric(mw["age"], errors="coerce").to_numpy(dtype=float)[:, None]
        for mode_name, mode in MODES.items():
            pr = predict_block(W, y, mw["patient"].to_numpy(), A, rom, mode=mode, Etr=age_w, Ete=age_b)
            for rep, p in pr.items():
                rows.append(pd.DataFrame(dict(sample=mb["Sample_ID"], patient=mb["patient"], y=mb["y"], age=mb["age"], modo=mode_name, representacion=rep, seed=seed, p=p)))
        print(f"semilla {seed} hecha", flush=True)
    pred = pd.concat(rows, ignore_index=True)
    pred.to_csv(out / "bejarano_predictions.csv", index=False)
    avg = pred.groupby(["modo", "representacion", "sample", "patient", "y", "age"], as_index=False)["p"].mean()
    lo, hi = mb.groupby("y")["age"].min().max(), mb.groupby("y")["age"].max().min()          # zona de edades presente en las dos clases
    zone = mb["age"].between(lo, hi).to_numpy()
    res = []
    for (mode_name, rep), g in avg.groupby(["modo", "representacion"], sort=False):
        g = g.set_index("sample").loc[mb["Sample_ID"]]
        y, p, age = g["y"].to_numpy(), g["p"].to_numpy(), g["age"].to_numpy()
        pm = patient_metrics(y, p)
        lg = _logit(p)
        cen = patient_metrics(y, (lg - lg.mean() > 0).astype(float))
        res.append(dict(modo=mode_name, representacion=rep, n=len(y), auroc=pm["auroc"], p_exacto=exact_p_auc_fast(y, p, pm["auroc"]), macro_f1_corte_05=pm["macro_f1"],
                        sensibilidad=pm["sensitivity"], especificidad=pm["specificity"], macro_f1_centrado=cen["macro_f1"], p_medio_gbm=float(p[y == 1].mean()),
                        p_medio_idh_mutado=float(p[y == 0].mean()), auroc_zona_solape=_auc(y[zone], p[zone]) if len(set(y[zone])) == 2 else np.nan,
                        n_zona_solape=int(zone.sum()), spearman_con_edad=float(pd.Series(p).corr(pd.Series(age), method="spearman"))))
    yb, ageb = mb["y"].to_numpy(), mb["age"].to_numpy()
    ao = _auc(yb, ageb)
    res.append(dict(modo="edad sola (referencia)", representacion="--", n=len(yb), auroc=ao, p_exacto=exact_p_auc_fast(yb, ageb, ao),
                    auroc_zona_solape=_auc(yb[zone], ageb[zone]) if len(set(yb[zone])) == 2 else np.nan, n_zona_solape=int(zone.sum())))
    # control: dejar un paciente fuera dentro de la propia cohorte (cada muestra es un paciente)
    meta_b = pd.DataFrame(dict(patient=mb["patient"], diagnosis=mb["Disease"]))
    df = lopo(Xb.toarray().astype(np.float64), meta_b, mb["y"].to_numpy(), rom)
    for rep, g in to_patient(df, keys=("group", "representation")).groupby("representation"):
        pm = patient_metrics(g["y"], g["p_mean"])
        res.append(dict(modo="control: dejar un paciente fuera dentro de Bejarano", representacion=rep, n=len(g), auroc=pm["auroc"], p_exacto=exact_p_auc_fast(g["y"].to_numpy(), g["p_mean"].to_numpy(), pm["auroc"]),
                        macro_f1_corte_05=pm["macro_f1"]))
    R = pd.DataFrame(res)
    R.to_csv(out / "bejarano_metrics.csv", index=False)
    wp = S[1].groupby("Patient").agg(edad=("age_num", "first"), diagnostico=("glioma_type", "first")).reset_index()
    wp.to_csv(out / "walchli_patients_for_overlap_check.csv", index=False)
    mb[["Sample_ID", "patient", "Disease", "age", "sex", "diagnosis_who"]].to_csv(out / "bejarano_samples_used.csv", index=False)
    lines = ["# Validación con la cohorte en masa de Bejarano et al. 2025\n", f"Muestras: {len(mb)} ({int(mb['y'].sum())} GBM IDH nativo, {int((1 - mb['y']).sum())} IDH mutado). Zona de edades solapadas: {lo:.0f}-{hi:.0f} años (n = {int(zone.sum())}).\n",
             to_markdown(R.round(3)), "\n",
             "Comprobar a mano si algún paciente coincide con los de Wälchli (`walchli_patients_for_overlap_check.csv` frente a `bejarano_samples_used.csv`: edad, sexo y diagnóstico).\n"]
    (out / "bejarano_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--xie", default="data/xie/xie_endothelial.h5ad")
    ap.add_argument("--counts", default="data/bejarano/BrainTime_vasculature_human.counts.csv")
    ap.add_argument("--metadata", default="data/bejarano/BrainTime_vasculature_human.metadata.csv")
    ap.add_argument("--ages", default="configs/bejarano_bulk_ages.csv")
    ap.add_argument("--out", default="results/bejarano_bulk")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    main(load_config(a.config), a.xie, a.counts, a.metadata, a.ages, a.out, a.seeds)
