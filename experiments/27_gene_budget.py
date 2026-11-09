"""¿Cuántos genes hacen falta? Curva de rendimiento frente al nº de genes variables (pregunta central del proyecto, a nivel de paciente).

Para cada n en `ns` (genes más variables elegidos SOLO con el entrenamiento) se ajusta el modelo de pseudobulks (paquetes de 100 células, TMM + log-CPM,
escalado, PCA, regresión logística) y se mide a nivel de paciente:
  - validación interna (dejar un paciente de Wälchli fuera);
  - validación externa en Xie `core` (modelo congelado, 6 pacientes);
  - validación externa en Bejarano (RNA-seq en masa, 18 pacientes), si se pasan los ficheros.
Representaciones: HVG-n (los n genes escalados tal cual) y PCA-2, PCA-5 y PCA-10 calculadas sobre esos n genes (solo si r < n).
Se repite con varias semillas de paquetes y se informa media y desviación entre semillas. Es un barrido descriptivo, no un contraste.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_config
from src.evaluation import to_markdown
from src.figs import inject_block, plt, save_fig
from src.final_model import align_genes
from src.pbcount import lopo, patient_metrics, predict_block, to_patient

P2 = importlib.import_module("experiments.22_phase2")
M19 = P2.M19
M23 = importlib.import_module("experiments.23_bejarano_bulk")
COMP = (2, 5, 10)
COHORTS = {"interna": "Wälchli (dejar un paciente fuera)", "xie_core": "Xie core (6 pacientes)", "bejarano": "Bejarano (18, RNA-seq en masa)"}


def _label(rep):
    return "HVG-n" if rep.startswith("HVG-") else rep


def _metrics(df_p, rep_col="rep"):
    out = []
    for rep, g in df_p.groupby(rep_col):
        pm = patient_metrics(g["y"], g["p"])
        out.append(dict(rep=rep, auroc=pm["auroc"], macro_f1=pm["macro_f1"], n_pacientes=len(g)))
    return out


def run(S, cfg, ns, seeds, spec, bej=None):
    base = dict(components=list(COMP), scale_clip=cfg["rom"].get("scale_clip", 10.0), min_cpm=1.0, min_frac=0.25)
    Ab = yb = None
    if bej is not None:
        Xb, gb, mb = M23.load_bulk(*bej)
        Xa, _ = align_genes(Xb, gb, S[2])
        Ab, yb, pb = np.asarray(Xa.todense(), dtype=np.float64), mb["y"].to_numpy(), mb["patient"].to_numpy()
    rows = []
    for seed in range(seeds):
        b = P2.build(S, dict(spec), seed, need_ext=True)
        W, mw, y, A, mx = b["W"], b["mw"], b["y"], b["A"], b["mx"]
        pat = mw["patient"].to_numpy()
        core = (mx["site"] == "core").to_numpy()
        yx = (mx["diagnosis"] == "GBM").astype(int).to_numpy()
        for n in ns:
            rom = dict(base, hvg=int(n))
            df = lopo(W, mw, y, rom)
            pt = to_patient(df, keys=("group", "representation")).rename(columns={"p_mean": "p", "representation": "rep"})
            for r in _metrics(pt):
                rows.append(dict(seed=seed, n_genes=n, cohorte="interna", **r))
            pr = predict_block(W, y, pat, A, rom)
            for rep, p in pr.items():
                t = pd.DataFrame(dict(patient=mx["patient"].to_numpy()[core], y=yx[core], p=p[core])).groupby("patient").agg(y=("y", "first"), p=("p", "mean"))
                pm = patient_metrics(t["y"], t["p"])
                rows.append(dict(seed=seed, n_genes=n, cohorte="xie_core", rep=rep, auroc=pm["auroc"], macro_f1=pm["macro_f1"], n_pacientes=len(t)))
            if Ab is not None:
                for rep, p in predict_block(W, y, pat, Ab, rom).items():
                    pm = patient_metrics(yb, p)
                    rows.append(dict(seed=seed, n_genes=n, cohorte="bejarano", rep=rep, auroc=pm["auroc"], macro_f1=pm["macro_f1"], n_pacientes=len(yb)))
            print(f"semilla {seed}, n = {n} hecho", flush=True)
    R = pd.DataFrame(rows)
    R["representacion"] = R["rep"].map(_label)
    return R


def summarize(R):
    g = R.groupby(["cohorte", "representacion", "n_genes"])
    return g.agg(auroc=("auroc", "mean"), auroc_sd=("auroc", "std"), macro_f1=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"), n_pacientes=("n_pacientes", "first"), n_semillas=("seed", "nunique")).reset_index()


def figures(out, T):
    cohs = [c for c in COHORTS if c in set(T["cohorte"])]
    cols = {"HVG-n": "#c0392b", "PCA-2": "#2471a3", "PCA-5": "#27ae60", "PCA-10": "#8e44ad"}
    for metric, name, ylab in (("auroc", "27_gene_budget_auroc", "AUROC por paciente"), ("macro_f1", "27_gene_budget_macro_f1", "macro-F1 por paciente (umbral 0,5)")):
        fig, ax = plt.subplots(1, len(cohs), figsize=(5.6 * len(cohs), 4.4), sharey=True)
        ax = np.atleast_1d(ax)
        for a, c in zip(ax, cohs):
            for rep, col in cols.items():
                t = T[(T["cohorte"] == c) & (T["representacion"] == rep)].sort_values("n_genes")
                if t.empty:
                    continue
                a.plot(t["n_genes"], t[metric], marker="o", color=col, label=rep)
                sd = t[metric + "_sd"].fillna(0)
                a.fill_between(t["n_genes"], t[metric] - sd, t[metric] + sd, color=col, alpha=0.15)
            a.axhline(0.5, color="k", linestyle=":", linewidth=1)
            a.set_xscale("log")
            a.set_xticks(sorted(T["n_genes"].unique()))
            a.set_xticklabels([str(int(v)) for v in sorted(T["n_genes"].unique())], rotation=45, fontsize=8)
            a.minorticks_off()
            a.set_xlabel("genes variables disponibles (n)")
            a.set_title(COHORTS[c], fontsize=10)
            a.grid(alpha=0.3)
        ax[0].set_ylabel(ylab)
        ax[0].legend(frameon=False, fontsize=8)
        mx_ = T[T["representacion"] == "HVG-n"]
        lines = [f"{ylab} frente al número de genes variables n con el que se construye el modelo (eje logarítmico); bandas = ± 1 desviación entre semillas de paquetes.",
                 "HVG-n: regresión logística sobre los n genes; PCA-r: PCA de esos n genes y las primeras r componentes. La línea de puntos es el azar (0,5 en AUROC).",
                 "Responde a la pregunta central del proyecto: cuántos genes (o componentes) hacen falta, medido de tres formas: dentro de Wälchli, en Xie core y en Bejarano."]
        for c in cohs:
            t = mx_[mx_["cohorte"] == c].sort_values("n_genes")
            if len(t):
                lines.append(f"{COHORTS[c]}, HVG-n: {metric} = " + ", ".join(f"{int(n)} genes: {v:.2f}" for n, v in zip(t["n_genes"], t[metric])))
        save_fig(fig, out, name, lines)


def block(T, ns):
    md = ["**Cuántos genes hacen falta** (`experiments/27_gene_budget.py`; `tablas/gene_budget_summary.csv`; figuras `results/gene_budget/27_gene_budget_*.png`). AUROC por paciente (media de las semillas) con n genes variables:", ""]
    for c in COHORTS:
        t = T[T["cohorte"] == c]
        if t.empty:
            continue
        pv = t.pivot(index="n_genes", columns="representacion", values="auroc").reindex(columns=[x for x in ("HVG-n", "PCA-2", "PCA-5", "PCA-10") if x in set(t["representacion"])])
        pv.insert(0, "n genes", pv.index)
        md += [f"*{COHORTS[c]}* ({int(t['n_pacientes'].iloc[0])} pacientes):", "", to_markdown(pv.reset_index(drop=True).round(3)), ""]
    md += ["Lectura: ver la figura; es un barrido descriptivo (3 semillas de paquetes, 14 pacientes de entrenamiento), sin intervalos de confianza entre pacientes.", ""]
    return "\n".join(md)


def main(cfg, xie_path, out, readme=None, ns=(10, 25, 50, 100, 250, 500, 1000, 2000), seeds=3, spec=None, S=None, bej=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    S = S or M19.load_studies(cfg, xie_path)
    R = run(S, cfg, ns, seeds, dict(spec or P2.BASE), bej)
    R.to_csv(out / "gene_budget_results.csv", index=False)
    T = summarize(R)
    T.to_csv(out / "gene_budget_summary.csv", index=False)
    figures(out, T)
    (out / "gene_budget_readme_block.md").write_text(block(T, ns), encoding="utf-8")
    if readme:
        inject_block(readme, "GENE_BUDGET", block(T, ns))
    return R, T


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--xie", default="data/xie/xie_endothelial.h5ad")
    ap.add_argument("--counts", default="data/bejarano/BrainTime_vasculature_human.counts.csv")
    ap.add_argument("--metadata", default="data/bejarano/BrainTime_vasculature_human.metadata.csv")
    ap.add_argument("--ages", default="configs/bejarano_bulk_ages.csv")
    ap.add_argument("--out", default="results/gene_budget")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    bej = (a.counts, a.metadata, a.ages) if Path(a.counts).exists() else None
    main(load_config(a.config), a.xie, a.out, a.readme, seeds=a.seeds, bej=bej)
