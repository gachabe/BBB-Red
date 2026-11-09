"""Visión de conjunto de las tres cohortes y del método: edades por cohorte y diagnóstico, y diagrama del flujo de trabajo.

Figuras (28_*.png, con descripción): 
  28_cohort_ages: edad de cada paciente en Wälchli, Xie y Bejarano por diagnóstico, con el solape de edades de cada cohorte (la confusión edad-diagnóstico de un vistazo);
  28_cohort_composition: nº de pacientes, de células o muestras y de pseudobulks por cohorte (barras);
  28_pipeline_diagram: de la matriz de expresión al paciente, paso a paso, y qué se ajusta solo con entrenamiento.
Tabla: cohort_summary.csv.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_config
from src.figs import COL, plt, save_fig

M19 = importlib.import_module("experiments.19_pseudobulk_counts")
M25 = importlib.import_module("experiments.25_overlap_check")


def cohort_tables(ow, xie_ages, bej_ages):
    w = ow.groupby("Patient").agg(edad=("age_num", "first"), clase=("glioma_type", "first"), n_celulas=("Patient", "size")).reset_index().rename(columns={"Patient": "paciente"})
    x = pd.read_csv(xie_ages)
    x = pd.DataFrame(dict(paciente=x["patient"], edad=x["age"].astype(float), clase=np.where(x["idh_status"].astype(str).str.lower().str.startswith("wild"), "GBM", "LGG")))
    b = M25.bejarano_table(bej_ages)
    return {"Wälchli (scRNA-seq)": w, "Xie (scRNA-seq)": x, "Bejarano (RNA-seq en masa)": b}


def fig_ages(out, T):
    fig, ax = plt.subplots(figsize=(10, 5))
    rng = np.random.default_rng(0)
    lines = ["Edad de cada paciente en las tres cohortes, por diagnóstico (rojo = GBM / IDH nativo, azul = LGG / IDH mutado). La zona sombreada es el intervalo de edades que comparten las dos clases dentro de la cohorte."]
    rows = []
    for i, (name, t) in enumerate(T.items()):
        for j, c in enumerate(("LGG", "GBM")):
            x0 = i * 3 + j
            g = t[t["clase"] == c]
            ax.scatter(x0 + rng.uniform(-0.18, 0.18, len(g)), g["edad"], color=COL[c], edgecolor="k", s=55, zorder=3)
            ax.hlines(g["edad"].mean(), x0 - 0.3, x0 + 0.3, color="k", linewidth=2)
        lo, hi = t.groupby("clase")["edad"].min().max(), t.groupby("clase")["edad"].max().min()
        solape = lo <= hi
        if solape:
            ax.fill_between([i * 3 - 0.45, i * 3 + 1.45], lo, hi, color="#f1c40f", alpha=0.25, zorder=1)
        n_in = int(t["edad"].between(lo, hi).sum()) if solape else 0
        lines.append(f"{name}: LGG {t[t['clase'] == 'LGG']['edad'].min():.0f}-{t[t['clase'] == 'LGG']['edad'].max():.0f} años (n = {int((t['clase'] == 'LGG').sum())}), GBM {t[t['clase'] == 'GBM']['edad'].min():.0f}-{t[t['clase'] == 'GBM']['edad'].max():.0f} (n = {int((t['clase'] == 'GBM').sum())}); "
                     + (f"zona de edades solapadas {lo:.0f}-{hi:.0f} con {n_in} pacientes." if solape else "sin edades solapadas: la edad separa perfectamente los dos grupos."))
        rows.append(dict(cohorte=name, n_pacientes=len(t), n_lgg=int((t["clase"] == "LGG").sum()), n_gbm=int((t["clase"] == "GBM").sum()), edad_min_lgg=t[t["clase"] == "LGG"]["edad"].min(), edad_max_lgg=t[t["clase"] == "LGG"]["edad"].max(),
                         edad_min_gbm=t[t["clase"] == "GBM"]["edad"].min(), edad_max_gbm=t[t["clase"] == "GBM"]["edad"].max(), solape_edades=bool(solape), n_en_solape=n_in))
    ax.set_xticks([k for i in range(len(T)) for k in (i * 3, i * 3 + 1)])
    ax.set_xticklabels([f"{n.split(' (')[0]}\n{c}" for n in T for c in ("LGG", "GBM")], fontsize=8)
    ax.set_ylabel("edad (años)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Edad por cohorte y diagnóstico (raya negra = media)")
    lines.append("Por qué importa: si las edades no se solapan (Xie), un modelo que solo usa la edad separa los grupos igual de bien que la expresión; Bejarano es la cohorte que permite distinguir ambas cosas.")
    save_fig(fig, out, "28_cohort_ages", lines)
    return pd.DataFrame(rows)


def fig_composition(out, ow, S=None):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    w = ow.groupby(["glioma_type"]).agg(pac=("Patient", "nunique"), cel=("Patient", "size"))
    ax[0].bar(["GBM", "LGG"], [w.loc["GBM", "pac"], w.loc["LGG", "pac"]], color=[COL["GBM"], COL["LGG"]], edgecolor="k")
    ax[0].set_ylabel("pacientes")
    ax[0].set_title("Wälchli: pacientes por diagnóstico")
    per = ow.groupby(["Patient", "glioma_type"]).size().reset_index(name="n").sort_values("n")
    ax[1].barh(per["Patient"], per["n"], color=[COL[c] for c in per["glioma_type"]], edgecolor="k", linewidth=0.3)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("células endoteliales (escala logarítmica)")
    ax[1].set_title("Wälchli: células por paciente")
    ax[1].tick_params(axis="y", labelsize=7)
    save_fig(fig, out, "28_cohort_composition", [
        "Composición de la cohorte de entrenamiento (Wälchli). Izquierda: pacientes por diagnóstico (%d GBM y %d LGG). Derecha: células endoteliales por paciente (escala logarítmica)." % (w.loc["GBM", "pac"], w.loc["LGG", "pac"]),
        f"El paciente con más células tiene {int(per['n'].max()):,} y el que menos {int(per['n'].min()):,}: por eso cada paciente pesa igual en el entrenamiento y las células se agrupan en paquetes de 100.",
        "Las diferencias de tamaño entre pacientes son la razón de usar pseudobulks por paciente en vez de células sueltas."])


def fig_pipeline(out):
    steps = [("Matriz de expresión\n(células x genes)", "#ecf0f1"), ("Paquetes de 100 células\nde un mismo paciente\n(suma de conteos)", "#d6eaf8"), ("Filtro de genes y TMM\n+ log2-CPM", "#d5f5e3"),
             ("2.000 genes más\nvariables, escalado", "#d5f5e3"), ("PCA\n(PC1 ... PC20)", "#fdebd0"), ("Regresión logística\nsobre PCA-r o genes", "#fadbd8"), ("P(GBM) del paquete\n→ media por paciente", "#e8daef")]
    fig, ax = plt.subplots(figsize=(15, 3.6))
    ax.axis("off")
    for i, (t, c) in enumerate(steps):
        x = i * 2.05
        ax.add_patch(plt.Rectangle((x, 0.3), 1.8, 1.4, facecolor=c, edgecolor="k", linewidth=1))
        ax.text(x + 0.9, 1.0, t, ha="center", va="center", fontsize=8.5)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 2.03, 1.0), xytext=(x + 1.82, 1.0), arrowprops=dict(arrowstyle="->"))
    ax.add_patch(plt.Rectangle((2 * 2.05 - 0.1, 0.05), 3 * 2.05 + 0.1 - 0.15, 1.9, fill=False, edgecolor="#c0392b", linestyle="--", linewidth=1.5))
    ax.text(2 * 2.05 + 3.0, 2.08, "se ajusta SOLO con los pacientes de entrenamiento (sin fugas)", color="#c0392b", fontsize=9, ha="center")
    ax.text(0, -0.15, "Validación: dejar un paciente fuera (Wälchli) · modelo congelado aplicado sin reajustar a Xie y Bejarano · todos los paquetes de un paciente siempre en el mismo bloque.", fontsize=9)
    ax.set_xlim(-0.2, 14.7)
    ax.set_ylim(-0.4, 2.4)
    save_fig(fig, out, "28_pipeline_diagram", [
        "Diagrama del flujo de trabajo, de la matriz de expresión a una probabilidad por paciente.",
        "Los cuadros rodeados en rojo (filtro de genes, TMM, genes variables, escalado y PCA) se calculan con los pacientes de entrenamiento y se aplican tal cual a los de prueba; es lo que evita las fugas de información.",
        "PCA-r = primeras r componentes de la misma PCA; HVG-2000 = regresión logística directamente sobre los 2.000 genes, sin PCA; All genes = todos los genes filtrados.",
        "La predicción final es una por paciente: la media de las probabilidades de sus paquetes."])


def main(cfg, xie_path, xie_ages, bej_ages, out, S=None, ow=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ow = ow if ow is not None else (S or M19.load_studies(cfg, xie_path))[1]
    T = cohort_tables(ow, xie_ages, bej_ages)
    summ = fig_ages(out, T)
    summ.to_csv(out / "cohort_summary.csv", index=False)
    fig_composition(out, ow)
    fig_pipeline(out)
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--xie", default="data/xie/xie_endothelial.h5ad")
    ap.add_argument("--xie-ages", default="configs/xie_ages.csv")
    ap.add_argument("--bej-ages", default="configs/bejarano_bulk_ages.csv")
    ap.add_argument("--out", default="results/cohort_overview")
    a = ap.parse_args()
    main(load_config(a.config), a.xie, a.xie_ages, a.bej_ages, a.out)
