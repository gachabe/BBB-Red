"""Qué genes componen las componentes principales (explicabilidad para el grupo).

Ajusta el modelo de pseudobulks sobre los 14 pacientes de Wälchli (paquetes de 100 células, TMM + log-CPM, 2.000 genes más variables,
escalado y PCA, igual que en los análisis) y describe:
  1. para cada componente (PC1-PC20; PCA-r usa las r primeras): los genes con más peso (cargas) en cada sentido, el porcentaje de varianza,
     el AUROC de su puntuación para separar GBM de LGG (por paciente), su correlación con la edad y la estabilidad de sus genes entre semillas;
  2. para cada modelo (PCA-2, PCA-5, PCA-10, PCA-20 y HVG-2000): los genes que más empujan la predicción hacia GBM o hacia LGG
     (coeficiente de la regresión logística llevado al espacio de genes: suma de coeficiente x carga).
Convención de signo: cada componente se orienta para que una puntuación alta signifique «más GBM» (media de pacientes GBM > media de pacientes LGG).
Salida: tablas CSV y un bloque de texto que se inserta en el README entre <!-- PC_GENES_START --> y <!-- PC_GENES_END -->.
Las cargas describen cómo se construye la componente, no una causa biológica; una componente puede mezclar tipo de glioma, edad y estudio.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data import load_config
from src.figs import COL, MARK_COL, inject_block, plt, save_fig
from src.pbcount import TMM, DenseReps, fit_lr, packet_weights
from src.xie import ENDO, GLIA, IMMUNE, MURAL

P2 = importlib.import_module("experiments.22_phase2")
M19 = P2.M19
MARK = {**{g: "endotelial" for g in ENDO}, **{g: "mural" for g in MURAL}, **{g: "inmune" for g in IMMUNE}, **{g: "glía" for g in GLIA}}
START, END = "<!-- PC_GENES_START -->", "<!-- PC_GENES_END -->"


def fit_once(S, rom, spec, seed):
    b = P2.build(S, spec, seed, need_ext=False)
    W, mw, y = b["W"], b["mw"], b["y"]
    genes = np.asarray(S[2]).astype(str)
    tmm = TMM(rom["min_cpm"], rom["min_frac"]).fit(W)
    L = tmm.transform(W)
    gk = genes[tmm.keep]
    reps = DenseReps(rom["components"], rom["hvg"], rom["scale_clip"]).fit(L)
    Sc = reps._scale(L)[:, reps.hvg]
    P = reps.pca.transform(Sc)                                            # puntuaciones de los paquetes (PC x paquetes)
    C = reps.pca.components_.copy()                                       # cargas (PC x genes HVG)
    pat = mw["patient"].to_numpy()
    pm = pd.DataFrame(P).assign(patient=pat).groupby("patient").mean()   # puntuación media por paciente
    yp = pd.Series(y, index=pat).groupby(level=0).first().reindex(pm.index).to_numpy()
    flip = np.where(pm.to_numpy()[yp == 1].mean(axis=0) >= pm.to_numpy()[yp == 0].mean(axis=0), 1.0, -1.0)
    P, C, pm = P * flip, C * flip[:, None], pm * flip
    age = pd.to_numeric(mw.drop_duplicates("patient").set_index("patient")["age"], errors="coerce").reindex(pm.index).to_numpy()
    ws = packet_weights(pat)
    Z = reps.transform(L)
    coefs = {}
    for n, z in Z.items():
        if n.startswith("PCA-"):
            coefs[n] = fit_lr(z * flip[: z.shape[1]], y, ws).coef_.ravel()
        elif n.startswith("HVG-"):
            coefs[n] = fit_lr(z, y, ws).coef_.ravel()
    return dict(genes=gk[reps.hvg], C=C, var=reps.pca.explained_variance_ratio_.copy(), pm=pm, ypat=yp, age=age, coefs=coefs)


def top_table(names, vec, n):
    o = np.argsort(vec)
    rows = []
    for side, idx in (("GBM", o[::-1][:n]), ("LGG", o[:n])):
        for r, j in enumerate(idx, 1):
            rows.append(dict(sentido=side, orden=r, gen=names[j], peso=float(vec[j]), marcador=MARK.get(names[j], "")))
    return pd.DataFrame(rows)


def _cls(g):
    return MARK.get(g, "otros")


def make_figures(out, runs, summary, top_pc, drivers, comps=(2, 5, 10, 20)):
    """Ocho figuras (24_*.png) con su descripción en descriptions/. Todo se calcula de los resultados de `main`."""
    r0 = runs[0]
    pm, ypat, age, genes, C = r0["pm"], r0["ypat"], r0["age"], r0["genes"], r0["C"]
    dx = np.where(ypat == 1, "GBM", "LGG")
    npc = len(summary)
    K = np.arange(1, npc + 1)
    # 1. varianza, utilidad y estabilidad por componente
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.4), gridspec_kw=dict(wspace=0.5))
    ax[0].bar(K, summary["varianza_%"], color="#34495e")
    ax2 = ax[0].twinx()
    ax2.plot(K, summary["varianza_%"].cumsum(), color="#e67e22", marker="o", markersize=3)
    ax2.set_ylabel("varianza acumulada (%)", color="#e67e22")
    for c in comps:
        if c <= npc:
            ax[0].axvline(c + 0.5, color="k", linestyle=":", linewidth=1)
            ax[0].text(c + 0.55, ax[0].get_ylim()[1] * 0.92, f"PCA-{c}", fontsize=8)
    ax[0].set_xlabel("componente")
    ax[0].set_ylabel("varianza explicada (%)")
    ax[0].set_title("Cuánta variación recoge cada componente")
    ax[1].bar(K - 0.2, summary["auroc_gbm_lgg"] - 0.5, width=0.4, bottom=0.5, color="#c0392b", label="AUROC GBM/LGG")
    ax[1].bar(K + 0.2, summary["spearman_edad"].abs(), width=0.4, color="#7f8c8d", label="|Spearman con la edad|")
    ax[1].axhline(0.5, color="k", linewidth=0.8)
    ax[1].set_xlabel("componente")
    ax[1].set_ylim(0, 1.02)
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title("Separa GBM/LGG frente a sigue la edad")
    ax[2].bar(K, summary["estabilidad"], color="#16a085")
    ax[2].set_ylim(0, 1)
    ax[2].set_xlabel("componente")
    ax[2].set_ylabel("solapamiento de los 30 genes")
    ax[2].set_title("Estabilidad entre semillas (1 = idénticos)")
    best = summary.sort_values("auroc_gbm_lgg", ascending=False).iloc[0]
    save_fig(fig, out, "24_pc_variance", [
        "Qué recoge cada componente principal (PC1-PC%d) de la PCA de los pseudobulks de Wälchli." % npc,
        "Izquierda: varianza explicada (barras) y acumulada (línea); las líneas de puntos marcan hasta dónde llega PCA-2, 5, 10 y 20 (PCA-r = primeras r componentes).",
        "Centro: AUROC de la puntuación media por paciente para separar GBM de LGG (rojo; 0,5 = azar) y valor absoluto de su correlación con la edad (gris).",
        "Derecha: estabilidad de los 30 genes de mayor |carga| al repetir el sorteo de paquetes (índice de Jaccard medio entre semillas).",
        f"PC1 explica {summary['varianza_%'].iloc[0]:.1f} % de la varianza; las {min(comps[-1], npc)} primeras suman {summary['varianza_%'].iloc[:min(comps[-1], npc)].sum():.1f} %.",
        f"La componente que mejor separa GBM de LGG es {best['componente']} (AUROC {best['auroc_gbm_lgg']:.2f}, correlación con la edad {best['spearman_edad']:+.2f})."])

    # 2. puntuaciones de cada paciente (mapa de calor)
    nh = min(10, npc)
    order = np.lexsort((np.nan_to_num(age, nan=0), -ypat))
    Z = pm.to_numpy()[order][:, :nh]
    Zs = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-9)
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(order) + 2))
    im = ax.imshow(Zs, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(nh))
    ax.set_xticklabels([f"PC{k + 1}" for k in range(nh)])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{pm.index[i]} ({dx[i]}, {'?' if np.isnan(age[i]) else int(age[i])} a)" for i in order], fontsize=8)
    for t, i in zip(ax.get_yticklabels(), order):
        t.set_color(COL[dx[i]])
    fig.colorbar(im, ax=ax, label="puntuación (z): rojo = más GBM")
    ax.set_title("Puntuación media de cada paciente en cada componente")
    save_fig(fig, out, "24_pc_scores_heatmap", [
        "Puntuación de cada paciente (media de sus paquetes) en PC1-PC%d, estandarizada por componente; filas ordenadas GBM primero y por edad." % nh,
        "Rojo = puntuación alta = el perfil se parece más al de un GBM (el signo de cada componente está orientado así). Una componente útil muestra un bloque rojo arriba (GBM) y azul abajo (LGG).",
        "Si la franja sigue la edad (de arriba abajo dentro de cada grupo) en vez del grupo, la componente recoge edad y no diagnóstico.",
        f"Pacientes: {int((ypat == 1).sum())} GBM y {int((ypat == 0).sum())} LGG."])

    # 3. genes con más peso en PC1-PC4
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for k, a in enumerate(axes.ravel()):
        if k >= npc:
            a.axis("off")
            continue
        t = top_pc[top_pc["pc"] == f"PC{k + 1}"]
        sel = pd.concat([t[t["sentido"] == "GBM"].head(12), t[t["sentido"] == "LGG"].head(12)]).sort_values("peso")
        a.barh(sel["gen"], sel["peso"], color=[MARK_COL[_cls(g)] for g in sel["gen"]], edgecolor="k", linewidth=0.3)
        a.axvline(0, color="k", linewidth=0.8)
        a.set_title(f"PC{k + 1}: derecha = más GBM, izquierda = más LGG", fontsize=10)
        a.tick_params(axis="y", labelsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in MARK_COL.values()]
    fig.legend(handles, MARK_COL.keys(), loc="lower center", ncol=5, frameon=False, title="tipo de gen marcador (listas de src/xie.py)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_fig(fig, out, "24_pc_top_genes", [
        "Los 12 genes con más peso (carga) hacia GBM y los 12 hacia LGG en PC1-PC4.",
        "El peso es la contribución del gen (escalado) a la componente; el color indica si es un marcador conocido (endotelial, mural, inmune o glía) o no.",
        "Es una descripción de cómo se construye la componente, no una prueba de que esos genes causen el tumor. Los genes inmunes, murales o gliales indican contaminación o cambios de composición celular.",
        "Tablas completas (50 genes por sentido hasta PC20) en pc_gene_loadings_top.csv."])

    # 4. pacientes en el plano de las componentes
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    pairs = [(0, 1), (0, 2), (1, 2)]
    for a, (i, j) in zip(ax, pairs):
        if max(i, j) >= npc:
            a.axis("off")
            continue
        for d in ("GBM", "LGG"):
            m = dx == d
            sz = 30 + 2.2 * np.nan_to_num(age[m], nan=40)
            a.scatter(pm[i].to_numpy()[m], pm[j].to_numpy()[m], s=sz, color=COL[d], alpha=0.8, edgecolor="k", linewidth=0.5, label=d)
        for n_, x_, y_ in zip(pm.index, pm[i], pm[j]):
            a.annotate(str(n_), (x_, y_), fontsize=7, xytext=(3, 3), textcoords="offset points")
        a.set_xlabel(f"PC{i + 1}")
        a.set_ylabel(f"PC{j + 1}")
        a.grid(alpha=0.3)
    ax[0].legend(frameon=False, title="tamaño = edad")
    fig.suptitle("Pacientes en el plano de las primeras componentes", y=1.02)
    save_fig(fig, out, "24_pc_scatter", [
        "Cada punto es un paciente de Wälchli (media de sus paquetes); color = diagnóstico, tamaño = edad.",
        "Paneles: PC1 frente a PC2, PC1 frente a PC3 y PC2 frente a PC3. Es lo que ve PCA-2 (primer panel) y PCA-5 (los tres, más PC4 y PC5).",
        "Un pacientes de un color rodeado de otro color es un paciente mal clasificado por esa proyección (ver sección 4.3 del README)."])

    # 5. edad frente a puntuación
    nk = min(4, npc)
    fig, ax = plt.subplots(1, nk, figsize=(4 * nk, 4))
    ax = np.atleast_1d(ax)
    for k in range(nk):
        ok = ~np.isnan(age)
        for d in ("GBM", "LGG"):
            m = (dx == d) & ok
            ax[k].scatter(age[m], pm[k].to_numpy()[m], color=COL[d], edgecolor="k", linewidth=0.5, s=50, label=d)
        ax[k].set_xlabel("edad (años)")
        ax[k].set_ylabel(f"puntuación PC{k + 1}")
        ax[k].set_title(f"PC{k + 1}: Spearman {summary['spearman_edad'].iloc[k]:+.2f}, AUROC {summary['auroc_gbm_lgg'].iloc[k]:.2f}", fontsize=9)
        ax[k].grid(alpha=0.3)
    ax[0].legend(frameon=False)
    save_fig(fig, out, "24_pc_vs_age", [
        "Puntuación de cada paciente en PC1-PC%d frente a su edad (color = diagnóstico)." % nk,
        "Sirve para ver si una componente separa por diagnóstico (dos nubes de colores a distinta altura a cualquier edad) o por edad (los puntos siguen una diagonal).",
        "En Wälchli los GBM son mayores que los LGG (salvo LGG1, 69 años), así que edad y diagnóstico están mezclados; ver sección 4.12 punto 4 del README."])

    # 6. genes que más empujan cada modelo
    mods = [m for m in ("PCA-2", "PCA-5", "PCA-10", "PCA-20") if m in set(drivers["modelo"])] + [m for m in drivers["modelo"].unique() if m.startswith("HVG-")][:1]
    fig, axes = plt.subplots(1, len(mods), figsize=(4.2 * len(mods), 6.2))
    axes = np.atleast_1d(axes)
    for a, m in zip(axes, mods):
        t = drivers[drivers["modelo"] == m]
        sel = pd.concat([t[t["sentido"] == "GBM"].head(12), t[t["sentido"] == "LGG"].head(12)]).sort_values("peso")
        a.barh(sel["gen"], sel["peso"], color=[MARK_COL[_cls(g)] for g in sel["gen"]], edgecolor="k", linewidth=0.3)
        a.axvline(0, color="k", linewidth=0.8)
        a.set_title(m, fontsize=10)
        a.tick_params(axis="y", labelsize=7)
    fig.suptitle("Genes que más empujan la predicción de cada modelo (derecha = hacia GBM)", y=1.0)
    save_fig(fig, out, "24_classifier_drivers", [
        "Genes que más empujan la predicción de cada modelo: para PCA-r, coeficiente de la regresión logística llevado al espacio de genes (suma de coeficiente x carga de las r componentes); para HVG, el coeficiente de cada gen.",
        "Positivo = empuja hacia GBM, negativo = hacia LGG. Mismos colores que en 24_pc_top_genes.",
        "Si PCA-2 y HVG comparten genes, el modelo de pocas dimensiones usa la misma señal que el de miles de genes; si no, usa otra (ver 24_gene_overlap_heatmap)."])

    # 7. solapamiento entre listas de genes
    sets = {}
    for k in range(min(10, npc)):
        gk = dict(zip(genes, np.abs(C[k])))
        sets[f"PC{k + 1}"] = set(sorted(gk, key=gk.get, reverse=True)[:50])
    for m in mods:
        t = drivers[drivers["modelo"] == m]
        sets[m] = set(pd.concat([t[t["sentido"] == "GBM"].head(25), t[t["sentido"] == "LGG"].head(25)])["gen"])
    names = list(sets)
    J = np.array([[len(sets[a] & sets[b]) / len(sets[a] | sets[b]) for b in names] for a in names])
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(J, cmap="viridis", vmin=0, vmax=max(0.3, np.nanmax(J - np.eye(len(names)))))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    fig.colorbar(im, label="índice de Jaccard")
    ax.set_title("Solapamiento entre las listas de genes principales")
    save_fig(fig, out, "24_gene_overlap_heatmap", [
        "Solapamiento (índice de Jaccard) entre listas de ~50 genes: los de mayor |carga| de PC1-PC%d y los de mayor peso de cada modelo." % min(10, npc),
        "Valores altos fuera de la diagonal indican que dos componentes (o un modelo y una componente) se apoyan en los mismos genes; las componentes ortogonales suelen compartir pocos.",
        f"Solapamiento medio entre componentes distintas: {np.mean(J[:min(10, npc), :min(10, npc)][~np.eye(min(10, npc), dtype=bool)]):.3f}."])

    # 8. qué tipos de genes marcadores aparecen
    rows = []
    for k in range(npc):
        t = top_pc[top_pc["pc"] == f"PC{k + 1}"]
        c = pd.Series([_cls(g) for g in t["gen"]]).value_counts(normalize=True)
        rows.append(c.reindex(list(MARK_COL), fill_value=0))
    D = pd.DataFrame(rows, index=[f"PC{k + 1}" for k in range(npc)])
    fig, ax = plt.subplots(figsize=(11, 4))
    bottom = np.zeros(npc)
    for c in MARK_COL:
        ax.bar(D.index, D[c], bottom=bottom, color=MARK_COL[c], label=c, edgecolor="w", linewidth=0.4)
        bottom += D[c].to_numpy()
    ax.set_ylabel("fracción de los 100 genes principales")
    ax.set_title("Marcadores celulares entre los genes con más peso de cada componente")
    ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save_fig(fig, out, "24_marker_share", [
        "Fracción de los 100 genes de mayor |carga| de cada componente (50 por sentido) que son marcadores endoteliales, murales, inmunes o gliales (listas cortas de src/xie.py; 'otros' = ninguno).",
        "Sirve para ver si una componente recoge cambios de composición celular (muchos marcadores de otro tipo celular) o cambios dentro de la célula endotelial.",
        "Las listas son cortas (6, 6, 4 y 4 genes), así que la fracción de marcadores es pequeña por construcción; lo informativo es qué componentes los contienen."])
    return D


def to_md(df):
    from src.evaluation import to_markdown
    return to_markdown(df)


def build_readme_block(summary, top_pc, drivers, npc_md=5, n_md=12, ndr_md=15):
    L = ["### 4.14 Qué genes componen las componentes principales (para interpretar el modelo)", "",
         "Generado por `experiments/24_pc_genes.py` (modelo de pseudobulks ajustado con los 14 pacientes de Wälchli, paquetes de 100 células, 2.000 genes más variables). "
         "**PCA-r usa las r primeras componentes de la misma PCA** (PCA-2 = PC1-PC2, PCA-5 = PC1-PC5, etc.). Cada componente está orientada para que una puntuación alta signifique «más GBM». "
         "El peso de un gen indica cuánto contribuye a la componente (sobre el gen escalado): **no es una causa biológica**, y una componente puede mezclar tipo de glioma, edad y estudio. "
         "Las tablas completas (50 genes por sentido y componente hasta PC20) están en `tablas/pc_gene_loadings_top.csv`.", "",
         "**Resumen de las componentes** (AUROC: capacidad de la puntuación del paciente para separar GBM de LGG; edad: correlación de Spearman con la edad; estabilidad: solapamiento de los 30 genes principales entre semillas, 1 = idéntico):", "",
         to_md(summary.round(3)), ""]
    for k in range(1, npc_md + 1):
        t = top_pc[top_pc["pc"] == f"PC{k}"]
        g = t[t["sentido"] == "GBM"].head(n_md)["gen"].tolist()
        l = t[t["sentido"] == "LGG"].head(n_md)["gen"].tolist()
        r = summary[summary["componente"] == f"PC{k}"].iloc[0]
        L += [f"**PC{k}** ({r['varianza_%']:.1f} % de varianza; AUROC GBM/LGG {r['auroc_gbm_lgg']:.2f}; edad {r['spearman_edad']:+.2f})",
              f"- Hacia GBM: {', '.join(g)}", f"- Hacia LGG: {', '.join(l)}", ""]
    L += ["**Genes que más empujan cada modelo** (coeficiente de la regresión logística llevado al espacio de genes; positivo = hacia GBM):", ""]
    for rep in ("PCA-2", "PCA-5"):
        t = drivers[drivers["modelo"] == rep]
        g = t[t["sentido"] == "GBM"].head(ndr_md)["gen"].tolist()
        l = t[t["sentido"] == "LGG"].head(ndr_md)["gen"].tolist()
        L += [f"- **{rep}** hacia GBM: {', '.join(g)}", f"- **{rep}** hacia LGG: {', '.join(l)}"]
    L += ["", "Para el análisis de enriquecimiento (rutas, tipos celulares) se pueden usar directamente las listas de `tablas/pc_gene_loadings_top.csv` y `tablas/classifier_gene_drivers.csv`.", ""]
    return "\n".join(L)


def main(cfg, xie_path, out, readme=None, seeds=3, spec=None, S=None, n_top=50, n_pc=20):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    S = S or M19.load_studies(cfg, xie_path)
    rom = dict(components=[2, 5, 10, 20], hvg=cfg["rom"].get("hvg", 2000), scale_clip=cfg["rom"].get("scale_clip", 10.0), min_cpm=1.0, min_frac=0.25)
    spec = dict(spec or P2.BASE)
    runs = [fit_once(S, rom, spec, s) for s in range(seeds)]
    r0 = runs[0]
    npc = min(n_pc, r0["C"].shape[0])
    top_rows, sum_rows = [], []
    ok = ~np.isnan(r0["age"])
    for k in range(npc):
        t = top_table(r0["genes"], r0["C"][k], n_top)
        t.insert(0, "pc", f"PC{k + 1}")
        top_rows.append(t)
        sets = []
        for r in runs:                                   # estabilidad: 30 genes de mayor |carga| con otro sorteo de paquetes
            gk = dict(zip(r["genes"], np.abs(r["C"][k])))
            sets.append(set(sorted(gk, key=gk.get, reverse=True)[:30]))
        jac = float(np.mean([len(x & z) / len(x | z) for i, x in enumerate(sets) for z in sets[i + 1:]])) if len(sets) > 1 else np.nan
        sc = r0["pm"][k].to_numpy()
        sum_rows.append(dict(componente=f"PC{k + 1}", **{"varianza_%": 100 * float(r0["var"][k])}, auroc_gbm_lgg=P2._auc(r0["ypat"], sc),
                             spearman_edad=float(spearmanr(sc[ok], r0["age"][ok])[0]) if ok.sum() > 3 else np.nan, estabilidad=jac))
    top_pc = pd.concat(top_rows, ignore_index=True)
    summary = pd.DataFrame(sum_rows)
    drv = []
    for name, coef in r0["coefs"].items():
        if name.startswith("PCA-"):
            r = int(name.split("-")[1])
            w = coef @ r0["C"][:r]                       # peso de cada gen en la predicción del modelo PCA-r
        else:
            w = coef                                      # HVG-2000: el coeficiente ya es por gen
        t = top_table(r0["genes"], w, n_top)
        t.insert(0, "modelo", name)
        drv.append(t)
    drivers = pd.concat(drv, ignore_index=True)
    top_pc.to_csv(out / "pc_gene_loadings_top.csv", index=False)
    summary.to_csv(out / "pc_summary.csv", index=False)
    drivers.to_csv(out / "classifier_gene_drivers.csv", index=False)
    make_figures(out, runs, summary, top_pc, drivers)
    block = build_readme_block(summary, top_pc, drivers)
    (out / "pc_genes_readme_block.md").write_text(block, encoding="utf-8")
    if readme:
        rp = Path(readme)
        txt = rp.read_text(encoding="utf-8")
        assert START in txt and END in txt, "el README no tiene los marcadores PC_GENES_START / PC_GENES_END"
        a, e = txt.index(START) + len(START), txt.index(END)
        rp.write_text(txt[:a] + "\n" + block + "\n" + txt[e:], encoding="utf-8")
    return summary, top_pc, drivers


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--xie", default="data/xie/xie_endothelial.h5ad")
    ap.add_argument("--out", default="results/pc_genes")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    main(load_config(a.config), a.xie, a.out, a.readme, a.seeds)
