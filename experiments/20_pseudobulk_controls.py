"""Controles con pseudobulks de conteos (guía, sección 9): estudio, sitio anatómico, edad y composición celular.

Mismo pipeline que 19 (TMM, log-CPM, PCA, regresión logística; todo ajustado con el entrenamiento, dejando un paciente fuera).
  - Clasificador de estudio: ¿se predice Wälchli frente a Xie con pseudobulks? (21 pacientes; AUROC alto = el efecto de estudio sigue ahí)
  - Clasificador de sitio dentro de Xie: ¿se predice núcleo frente a periferia? (por muestra)
  - Solo edad / edad + expresión, y solo composición / composición + expresión (solo Wälchli: Xie no trae subclusters ni edad).
"""
from __future__ import annotations

import argparse
import importlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.batch import common_genes
from src.data import load_config
from src.evaluation import to_markdown
from src.pbcount import lopo, make_packets, patient_metrics, to_patient

warnings.filterwarnings("ignore")
M19 = importlib.import_module("experiments.19_pseudobulk_counts")


def _rows(pat, keys, extra):
    out = []
    for kv, g in pat.groupby(list(keys), sort=True):
        kv = kv if isinstance(kv, tuple) else (kv,)
        out.append(dict(zip(keys, kv), **extra, **patient_metrics(g["y"], g["p_mean"])))
    return out


def main(cfg, xie_path, out, size=100, R=20, seeds=3, components=(2, 5, 10, 20), min_whole=100) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rom = dict(components=list(components), hvg=cfg["rom"].get("hvg", 2000), scale_clip=cfg["rom"].get("scale_clip", 10.0), min_cpm=1.0, min_frac=0.25)
    Cw, ow, gw, Cx, ox, gx = M19.load_studies(cfg, xie_path)
    study, site, age, comp = [], [], [], []
    for seed in range(seeds):
        Xw, mw, _ = make_packets(Cw, ow, size, R, seed, "Patient", M19.COLS_W, "ECclusters", min_whole)
        Xx, mx, _ = make_packets(Cx, ox, size, R, seed, "sample", M19.COLS_X, None, min_whole)
        # estudio
        Xa, Xb, _ = common_genes(Xw, gw, Xx, gx)
        X = sp.vstack([Xa, Xb]).tocsr()
        m = pd.concat([mw[["patient"]].assign(study=0), mx[["patient"]].assign(study=1)], ignore_index=True)
        d = lopo(X, m, m["study"].to_numpy(), rom)
        p = to_patient(d, keys=("group", "representation"))
        study += _rows(p, ["representation"], dict(seed=seed, size=size, n_walchli=int((p["y"] == 0).sum() // p["representation"].nunique()),
                                                   n_xie=int((p["y"] == 1).sum() // p["representation"].nunique())))
        # sitio dentro de Xie (se mantiene todo el paciente fuera; resultado por muestra)
        ys = (mx["site"] == "peripheral").astype(int).to_numpy()
        d = lopo(Xx, mx, ys, rom)
        d["sample"] = mx["sample"].to_numpy()[d["packet"]]
        ps = to_patient(d.assign(group=d["sample"]), keys=("group", "representation"))
        site += _rows(ps, ["representation"], dict(seed=seed, size=size, n_specimens=int(ps["group"].nunique())))
        # edad y composición (solo Wälchli)
        y = (mw["diagnosis"] == "GBM").astype(int).to_numpy()
        E_age = mw[["age"]].to_numpy(float)
        E_comp = mw[[c for c in mw.columns if c.startswith("comp_")]].to_numpy(float)
        for name, mode, E in (("expresión sola", "expr", None), ("edad sola", "extra", E_age), ("edad + expresión", "both", E_age),
                              ("composición sola", "extra", E_comp), ("composición + expresión", "both", E_comp)):
            d = lopo(Xw, mw, y, rom, mode=mode, extra=E)
            pp = to_patient(d, keys=("group", "representation"))
            rows = _rows(pp, ["representation"], dict(seed=seed, size=size, model=name))
            (age if "edad" in name or name == "expresión sola" else []).extend(rows)
            (comp if "composición" in name or name == "expresión sola" else []).extend(rows)
        print(f"  controles hechos, semilla {seed}", flush=True)
    res = dict(study=pd.DataFrame(study), site=pd.DataFrame(site), age=pd.DataFrame(age), comp=pd.DataFrame(comp))
    names = dict(study="study_classifier_results.csv", site="site_classifier_results.csv", age="age_only_results.csv", comp="composition_only_results.csv")
    for k, f in names.items():
        res[k].to_csv(out / f, index=False)
        cols = [c for c in ("model", "representation") if c in res[k].columns]
        s = res[k].groupby(cols).agg(auroc=("auroc", "mean"), balanced_accuracy=("balanced_accuracy", "mean"), macro_f1=("macro_f1", "mean")).reset_index()
        (out / f.replace(".csv", ".md")).write_text(to_markdown(s.round(3)) + "\n", encoding="utf-8")
        print(f"\n{f} (media entre semillas):\n{s.round(3).to_string(index=False)}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--xie", default="data/xie/xie_endothelial.h5ad")
    ap.add_argument("--out", default="results/pseudobulk_counts")
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    main(load_config(a.config), a.xie, a.out, a.size, 20, a.seeds)
