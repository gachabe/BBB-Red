"""Tabla comparativa célula frente a pseudobulk (guía, sección 10) y figuras con su descripción."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import describe_figure, to_markdown
from src.permutation import stats_from_fraction

REPS = ["PCA-2", "PCA-5", "PCA-10", "PCA-20", "HVG-2000", "All genes"]


def cell_level_internal(path):
    d = pd.read_csv(path)
    d = d[(d["model"] == "logistic_regression")] if "model" in d.columns else d
    d = d[d["variant"] == "real"] if "variant" in d.columns else d
    out = {}
    for rep, g in d.groupby("representation"):
        a = (g["label"].to_numpy() == "LGG").astype(int)
        acc = g["accuracy"].to_numpy(float)
        out[rep] = stats_from_fraction(np.where(a == 1, acc, 1 - acc), a)[0]
    return out


def cell_level_ext(path):
    if not Path(path).exists():
        return {}
    d = pd.read_csv(path)
    return d[d["model"] == "logistic_regression"].set_index("representation")["macro_f1"].to_dict()


def comparison(res: Path, cell_internal, cell_core, cell_full, primary_size):
    sens = pd.read_csv(res / "pseudobulk_seed_sensitivity.csv")
    sens["size"] = sens["size"].astype(str)
    ci, cc, cf = cell_level_internal(cell_internal), cell_level_ext(cell_core), cell_level_ext(cell_full)
    perm = pd.read_csv(res / "permutation_results.csv") if (res / "permutation_results.csv").exists() else pd.DataFrame()
    stu = pd.read_csv(res / "study_classifier_results.csv") if (res / "study_classifier_results.csv").exists() else pd.DataFrame()
    rows = []
    for rep in REPS:
        def pb(analysis, col="macro_f1_mean"):
            g = sens[(sens["analysis"] == analysis) & (sens["size"] == str(primary_size)) & (sens["representation"] == rep)]
            return float(g[col].iloc[0]) if len(g) else np.nan
        p = perm[(perm["representation"] == rep) & (perm["size"].astype(str) == str(primary_size))] if len(perm) else perm
        rows.append(dict(representation=rep, cell_level_internal_macro_f1=ci.get(rep, np.nan), pseudobulk_internal_macro_f1=pb("internal"),
                         cell_level_xie_core_macro_f1=cc.get(rep, np.nan), pseudobulk_xie_core_macro_f1=pb("xie_core"),
                         cell_level_xie_full_macro_f1=cf.get(rep, np.nan), pseudobulk_xie_full_macro_f1=pb("xie_all"),
                         permutation_p_value=float(p["p_adjusted_maxT"].iloc[0]) if len(p) else np.nan,
                         study_auc=float(stu[stu["representation"] == rep]["auroc"].mean()) if len(stu) else np.nan))
    return pd.DataFrame(rows), sens


def main(res, cell_internal, cell_core, cell_full, primary_size=100, cfg=None, xie_path=None) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    res = Path(res)
    tab, sens = comparison(res, cell_internal, cell_core, cell_full, primary_size)
    tab.to_csv(res / "comparison_cell_vs_pseudobulk.csv", index=False)
    (res / "comparison_cell_vs_pseudobulk.md").write_text(to_markdown(tab.round(3)) + "\n", encoding="utf-8")
    print(tab.round(3).to_string(index=False))
    P = str(primary_size)

    # 1. células por pseudobulk (espécimen completo)
    meta = pd.read_csv(res / "pseudobulk_metadata.csv")
    wh = meta[meta["size"].astype(str) == "whole"]
    if len(wh):
        fig, ax = plt.subplots(figsize=(10, 4))
        lab = wh["study"].str[0] + ":" + wh["unit"].astype(str)
        ax.bar(range(len(wh)), wh["n_cells"], color=["#0072B2" if s == "Walchli" else "#E69F00" for s in wh["study"]])
        ax.axhline(100, color="k", ls=":")
        ax.set_xticks(range(len(wh)), lab, rotation=60, fontsize=7)
        ax.set(ylabel="Endothelial cells in the whole-specimen pseudobulk", yscale="log")
        fig.tight_layout(); fig.savefig(res / "21_cells_per_pseudobulk.png", dpi=130); plt.close(fig)
        describe_figure(res, "21_cells_per_pseudobulk.png", ["21_cells_per_pseudobulk.png",
            "Qué expresa: nº de células endoteliales de cada espécimen (azul Wälchli = paciente, naranja Xie = muestra), en escala logarítmica; la línea es el mínimo de 100 células.", "",
            f"Mínimo {int(wh['n_cells'].min())} células ({wh.loc[wh['n_cells'].idxmin(), 'unit']}), máximo {int(wh['n_cells'].max())}; {int((wh['n_cells'] < 500).sum())} especímenes con menos de 500 células (no admiten paquetes de 500)."])

    # 5. estabilidad frente al tamaño
    st = pd.read_csv(res / "packet_stability_cohort.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for rep, g in st.groupby("representation"):
        g = g.sort_values("size")
        ax[0].plot(g["size"].astype(str), g["median_patient_sd"], marker="o", label=rep)
        ax[1].plot(g["size"].astype(str), g["icc_packets_within_patient"], marker="o", label=rep)
    ax[0].axhline(0.05, color="gray", ls=":")
    ax[0].set(xlabel="Packet size (cells)", ylabel="Median per-patient SD of P(GBM) across replicates")
    ax[1].set(xlabel="Packet size (cells)", ylabel="ICC of packet probabilities within patients")
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(res / "21_stability_vs_packet_size.png", dpi=130); plt.close(fig)
    lines = ["21_stability_vs_packet_size.png",
             "Qué expresa: izquierda, desviación típica mediana (entre pacientes) de la probabilidad de GBM de sus paquetes réplica, según el tamaño del paquete; la línea punteada es el umbral 0,05 de la guía. "
             "Derecha, correlación intraclase de las probabilidades de los paquetes dentro de cada paciente (cerca de 1 = los paquetes de un paciente dicen lo mismo). Semilla 0, validación interna.", ""]
    for rep, g in st.groupby("representation"):
        g = g.sort_values("size")
        lines.append(f"{rep}: SD mediana " + ", ".join(f"{s}: {v:.3f}" for s, v in zip(g["size"], g["median_patient_sd"])) + "; ICC " + ", ".join(f"{s}: {v:.2f}" for s, v in zip(g["size"], g["icc_packets_within_patient"])))
    describe_figure(res, "21_stability_vs_packet_size.png", lines)

    # 4/6. probabilidades por paciente y variabilidad entre réplicas
    ip = pd.read_csv(res / "internal_patient_predictions.csv")
    ip = ip[(ip["seed"] == 0) & (ip["size"].astype(str) == P)]
    xc = pd.read_csv(res / "xie_core_predictions.csv")
    xc = xc[(xc["seed"] == 0) & (xc["size"].astype(str) == P)]
    reps = [r for r in ("PCA-2", "PCA-5", "PCA-10", "PCA-20") if r in set(ip["representation"])]
    fig, ax = plt.subplots(2, len(reps), figsize=(3.6 * len(reps), 7), squeeze=False, sharey=True)
    lines = ["21_patient_predictions.png",
             f"Qué expresa: probabilidad de GBM por paciente (media de sus paquetes de {P} células; barra = ±1 desviación típica entre paquetes). Arriba, validación interna en Wälchli; abajo, Xie solo núcleo con el modelo congelado. Naranja = GBM, verde = LGG; línea punteada = umbral 0,5.", ""]
    for j, rep in enumerate(reps):
        for i, (d, ycol, pcol, scol, ttl) in enumerate(((ip, "y", "p_gbm_mean", "p_sd", "Walchli LOPO"), (xc, "y", "p_mean", "p_sd", "Xie core"))):
            g = d[d["representation"] == rep].sort_values(ycol)
            pc = "p_gbm_mean" if "p_gbm_mean" in g.columns else "p_mean"
            ax[i][j].errorbar(range(len(g)), g[pc], yerr=g[scol].fillna(0), fmt="none", ecolor="gray", capsize=2)
            ax[i][j].scatter(range(len(g)), g[pc], c=["#D55E00" if v == 1 else "#009E73" for v in g[ycol]], zorder=3)
            ax[i][j].set_xticks(range(len(g)), g["patient"] if "patient" in g.columns else g["group"], rotation=70, fontsize=6)
            ax[i][j].axhline(0.5, color="gray", ls=":"); ax[i][j].set_title(f"{ttl}, {rep}", fontsize=8)
            wrong = g[((g[pc] >= 0.5) != (g[ycol] == 1))]
            lines.append(f"{ttl}, {rep}: " + ("todos bien clasificados" if wrong.empty else "mal clasificados: " + ", ".join(f"{(r.patient if 'patient' in g.columns else r.group)} (P={getattr(r, pc):.2f})" for r in wrong.itertuples())))
    ax[0][0].set_ylabel("P(GBM)"); ax[1][0].set_ylabel("P(GBM)")
    fig.tight_layout(); fig.savefig(res / "21_patient_predictions.png", dpi=130); plt.close(fig)
    describe_figure(res, "21_patient_predictions.png", lines)

    # 7. célula frente a pseudobulk
    fig, ax = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    lines = ["21_cell_vs_pseudobulk.png",
             "Qué expresa: macro-F1 (por paciente) del análisis a nivel de célula (ya existente) frente al de pseudobulks de conteos, por representación y para tres evaluaciones: validación interna en Wälchli, Xie solo núcleo y Xie completo. Línea punteada: 0,5.", ""]
    for a_, (c, p_, ttl) in zip(ax, (("cell_level_internal_macro_f1", "pseudobulk_internal_macro_f1", "Internal (Walchli)"),
                                      ("cell_level_xie_core_macro_f1", "pseudobulk_xie_core_macro_f1", "Xie core"),
                                      ("cell_level_xie_full_macro_f1", "pseudobulk_xie_full_macro_f1", "Xie full"))):
        x = np.arange(len(tab))
        a_.bar(x - 0.2, tab[c], 0.4, label="cell level"); a_.bar(x + 0.2, tab[p_], 0.4, label=f"pseudobulk ({P} cells)")
        a_.set_xticks(x, tab["representation"], rotation=35, fontsize=7); a_.axhline(0.5, color="gray", ls=":"); a_.set_title(ttl)
        lines.append(f"{ttl}: " + "; ".join(f"{r.representation} célula {getattr(r, c):.2f} / pseudobulk {getattr(r, p_):.2f}" for r in tab.itertuples()))
    ax[0].legend(fontsize=7); ax[0].set_ylabel("Macro-F1")
    fig.tight_layout(); fig.savefig(res / "21_cell_vs_pseudobulk.png", dpi=130); plt.close(fig)
    describe_figure(res, "21_cell_vs_pseudobulk.png", lines)

    # 8. núcleo frente a periferia
    fig, ax = plt.subplots(figsize=(8, 4))
    lines = ["21_core_vs_periphery.png",
             f"Qué expresa: AUROC por paciente en Xie con el modelo congelado (paquetes de {P} células), solo núcleo, solo periferia y todo (media de las muestras del paciente). Línea punteada: 0,5 (azar). Los conjuntos de pacientes difieren (núcleo 6, periferia 7).", ""]
    for k, an in enumerate(("xie_core", "xie_periphery", "xie_all")):
        g = sens[(sens["analysis"] == an) & (sens["size"] == P)].set_index("representation").reindex(REPS)
        ax.bar(np.arange(len(REPS)) + (k - 1) * 0.27, g["auroc_mean"], 0.27, label=an)
        lines.append(f"{an}: " + ", ".join(f"{r} {v:.2f}" for r, v in g["auroc_mean"].items() if pd.notna(v)))
    ax.axhline(0.5, color="gray", ls=":"); ax.set_xticks(range(len(REPS)), REPS, rotation=35, fontsize=8); ax.set_ylabel("AUROC"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(res / "21_core_vs_periphery.png", dpi=130); plt.close(fig)
    describe_figure(res, "21_core_vs_periphery.png", lines)

    # 9. expresión sola frente a composición sola
    cf, af = res / "composition_only_results.csv", res / "age_only_results.csv"
    if cf.exists() and af.exists():
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        lines = ["21_expression_vs_composition_age.png",
                 "Qué expresa: AUROC por paciente (validación interna en Wälchli, media entre semillas) de expresión sola, composición celular sola, edad sola y combinaciones. Sirve para ver si la expresión aporta algo más que la composición o la edad. Línea punteada: 0,5.", ""]
        for a_, f, ttl in ((ax[0], cf, "Composition"), (ax[1], af, "Age")):
            d = pd.read_csv(f).groupby(["model", "representation"])["auroc"].mean().reset_index()
            for k, (m, g) in enumerate(d.groupby("model")):
                g = g.set_index("representation").reindex(REPS + ["extra"])
                g = g.dropna()
                a_.plot(range(len(g)), g["auroc"], marker="o", label=m)
                lines.append(f"{ttl}: {m}: " + ", ".join(f"{r} {v:.2f}" for r, v in g["auroc"].items()))
            a_.axhline(0.5, color="gray", ls=":"); a_.set_title(ttl); a_.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(res / "21_expression_vs_composition_age.png", dpi=130); plt.close(fig)
        describe_figure(res, "21_expression_vs_composition_age.png", lines)

    # 2/3. PCA de los pseudobulks (ajustada con Wälchli; Xie proyectado)
    if cfg is not None and xie_path is not None:
        from src.final_model import align_genes
        from src.pbcount import DenseReps, TMM, make_packets
        M19 = importlib.import_module("experiments.19_pseudobulk_counts")
        Cw, ow, gw, Cx, ox, gx = M19.load_studies(cfg, xie_path)
        Xw, mw, _ = make_packets(Cw, ow, primary_size, 20, 0, "Patient", M19.COLS_W, "ECclusters", 100)
        Xx, mx, _ = make_packets(Cx, ox, primary_size, 20, 0, "sample", M19.COLS_X, None, 100)
        Xa, _ = align_genes(Xx, gx, gw)
        Dw, Dx = np.asarray(Xw.todense(), float), np.asarray(Xa.todense(), float)
        t = TMM().fit(Dw)
        r = DenseReps([2], cfg["rom"].get("hvg", 2000), cfg["rom"].get("scale_clip", 10.0)).fit(t.transform(Dw))
        Zw, Zx = r.transform(t.transform(Dw))["PCA-2"], r.transform(t.transform(Dx))["PCA-2"]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
        for dx, c in (("GBM", "#D55E00"), ("LGG", "#009E73")):
            for Z, m, mk in ((Zw, mw, "o"), (Zx, mx, "^")):
                s = (m["diagnosis"] == dx).to_numpy()
                ax[0].scatter(Z[s, 0], Z[s, 1], c=c, marker=mk, s=14, alpha=0.7, label=f"{dx} ({'Walchli' if mk == 'o' else 'Xie'})")
        ax[0].legend(fontsize=7)
        ax[1].scatter(Zw[:, 0], Zw[:, 1], c="#0072B2", s=14, alpha=0.5, label="Walchli")
        for site, c in (("core", "#E69F00"), ("peripheral", "#CC79A7")):
            s = (mx["site"] == site).to_numpy()
            ax[1].scatter(Zx[s, 0], Zx[s, 1], c=c, marker="^", s=18, alpha=0.8, label=f"Xie {site}")
        ax[1].legend(fontsize=7)
        for a_ in ax:
            a_.set(xlabel="PC1", ylabel="PC2")
        fig.tight_layout(); fig.savefig(res / "21_pseudobulk_pca.png", dpi=130); plt.close(fig)
        mw_c = Zw[:, 0].mean(); mx_c = Zx[:, 0].mean()
        describe_figure(res, "21_pseudobulk_pca.png", ["21_pseudobulk_pca.png",
            f"Qué expresa: PC1 y PC2 de los pseudobulks de {primary_size} células tras TMM y log-CPM, con la PCA ajustada solo con Wälchli y Xie proyectado. Izquierda, coloreado por diagnóstico (círculo Wälchli, triángulo Xie); derecha, por estudio y sitio (núcleo/periferia).", "",
            f"PC1 media: Wälchli {mw_c:.2f}, Xie {mx_c:.2f}; PC2 media: Wälchli {Zw[:, 1].mean():.2f}, Xie {Zx[:, 1].mean():.2f}. Si los triángulos se separan de los círculos, el efecto de estudio persiste con pseudobulks."])
    return tab


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--xie", default="data/xie/xie_endothelial.h5ad")
    ap.add_argument("--res", default="results/pseudobulk_counts")
    ap.add_argument("--cell-internal", default="results/full/results_per_patient_glioma_type.csv")
    ap.add_argument("--cell-core", default="results/external_xie_core/results_external.csv")
    ap.add_argument("--cell-full", default="results/external_xie/results_external.csv")
    ap.add_argument("--primary-size", type=int, default=100)
    ap.add_argument("--no-pca", action="store_true")
    a = ap.parse_args()
    from src.data import load_config
    main(a.res, a.cell_internal, a.cell_core, a.cell_full, a.primary_size, None if a.no_pca else load_config(a.config), None if a.no_pca else a.xie)
