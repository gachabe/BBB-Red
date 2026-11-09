"""Matriz de transferencia entre cohortes: entrenar con una y predecir las demás (incluye la transferencia inversa Bejarano -> Wälchli y Xie).

Cohortes: Wälchli (14 pacientes, scRNA-seq, paquetes de 100 células), Xie `core` (6 pacientes; la periferia solo como prueba, porque son los mismos pacientes),
Bejarano (18 pacientes, RNA-seq en masa de células endoteliales ordenadas). Mismo modelo en todos los casos (TMM + log-CPM, 2.000 genes variables, PCA, regresión
logística balanceada), ajustado solo con la cohorte de entrenamiento y aplicado sin reajustar. A nivel de paciente: AUROC, p exacto (todas las asignaciones de
etiquetas), macro-F1 con corte 0,5 y con las probabilidades centradas. La diagonal es dejar un paciente fuera dentro de la propia cohorte.
Avisos: Xie core tiene la edad totalmente confundida con el diagnóstico (28-48 frente a 56-70), así que entrenar con él aprende en parte edad; Bejarano es RNA-seq
en masa (tecnología distinta); los genes ausentes de Bejarano se rellenan con ceros.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_config
from src.evaluation import to_markdown
from src.figs import COL, inject_block, plt, save_fig
from src.final_model import align_genes
from src.pbcount import lopo, patient_metrics, predict_block, to_patient

P2 = importlib.import_module("experiments.22_phase2")
M19, M23 = P2.M19, importlib.import_module("experiments.23_bejarano_bulk")
TRAIN = ["Wälchli", "Xie core", "Bejarano"]
TEST = ["Wälchli", "Xie core", "Xie periferia", "Bejarano"]
REPS = ["PCA-2", "PCA-5", "PCA-10", "PCA-20", "HVG"]            # HVG = los n genes más variables (n = hvg de la configuración, 2.000)


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _pat(patient, y, p):
    return pd.DataFrame(dict(patient=np.asarray(patient), y=np.asarray(y), p=np.asarray(p))).groupby("patient").agg(y=("y", "first"), p=("p", "mean")).reset_index()


def run(S, cfg, spec, seeds, bej):
    rom = dict(components=[2, 5, 10, 20], hvg=cfg["rom"].get("hvg", 2000), scale_clip=cfg["rom"].get("scale_clip", 10.0), min_cpm=1.0, min_frac=0.25)
    Xb, gb, mb = M23.load_bulk(*bej)
    Ab = np.asarray(align_genes(Xb, gb, S[2])[0].todense(), dtype=np.float64)
    yb, pb = mb["y"].to_numpy(), mb["patient"].to_numpy()
    recs = []

    def add(train, test, seed, pr, pats, y):
        for rep, p in pr.items():
            rep = "HVG" if rep.startswith("HVG-") else rep
            if rep in REPS:
                t = _pat(pats, y, p)
                recs.append(t.assign(train=train, test=test, rep=rep, seed=seed))

    for seed in range(seeds):
        b = P2.build(S, dict(spec), seed, need_ext=True)
        W, mw, y, A, mx = b["W"], b["mw"], b["y"], b["A"], b["mx"]
        pw = mw["patient"].to_numpy()
        core = (mx["site"] == "core").to_numpy()
        yx, px = (mx["diagnosis"] == "GBM").astype(int).to_numpy(), mx["patient"].to_numpy()
        Ac, Ap = A[core], A[~core]
        # entrenar con Wälchli
        pr = predict_block(W, y, pw, A, rom)
        add("Wälchli", "Xie core", seed, {k: v[core] for k, v in pr.items()}, px[core], yx[core])
        add("Wälchli", "Xie periferia", seed, {k: v[~core] for k, v in pr.items()}, px[~core], yx[~core])
        add("Wälchli", "Bejarano", seed, predict_block(W, y, pw, Ab, rom), pb, yb)
        df = to_patient(lopo(W, mw, y, rom), keys=("group", "representation"))
        for rep, g in df.groupby(df["representation"].map(lambda r: "HVG" if r.startswith("HVG-") else r)):
            if rep in REPS:
                recs.append(pd.DataFrame(dict(patient=g["group"], y=g["y"], p=g["p_mean"], train="Wälchli", test="Wälchli", rep=rep, seed=seed)))
        # entrenar con Bejarano (transferencia inversa)
        add("Bejarano", "Wälchli", seed, predict_block(Ab, yb, pb, W, rom), pw, y)
        pr = predict_block(Ab, yb, pb, A, rom)
        add("Bejarano", "Xie core", seed, {k: v[core] for k, v in pr.items()}, px[core], yx[core])
        add("Bejarano", "Xie periferia", seed, {k: v[~core] for k, v in pr.items()}, px[~core], yx[~core])
        df = to_patient(lopo(Ab, pd.DataFrame(dict(patient=pb, diagnosis=mb["Disease"])), yb, rom), keys=("group", "representation"))
        for rep, g in df.groupby(df["representation"].map(lambda r: "HVG" if r.startswith("HVG-") else r)):
            if rep in REPS:
                recs.append(pd.DataFrame(dict(patient=g["group"], y=g["y"], p=g["p_mean"], train="Bejarano", test="Bejarano", rep=rep, seed=seed)))
        # entrenar con Xie core
        add("Xie core", "Wälchli", seed, predict_block(Ac, yx[core], px[core], W, rom), pw, y)
        add("Xie core", "Bejarano", seed, predict_block(Ac, yx[core], px[core], Ab, rom), pb, yb)
        df = to_patient(lopo(Ac, pd.DataFrame(dict(patient=px[core], diagnosis=mx["diagnosis"].to_numpy()[core])), yx[core], rom), keys=("group", "representation"))
        for rep, g in df.groupby(df["representation"].map(lambda r: "HVG" if r.startswith("HVG-") else r)):
            if rep in REPS:
                recs.append(pd.DataFrame(dict(patient=g["group"], y=g["y"], p=g["p_mean"], train="Xie core", test="Xie core", rep=rep, seed=seed)))
        print(f"semilla {seed} hecha", flush=True)
    return pd.concat(recs, ignore_index=True)


def metrics(P):
    avg = P.groupby(["train", "test", "rep", "patient"], as_index=False).agg(y=("y", "first"), p=("p", "mean"))
    rows = []
    for (tr, te, rep), g in avg.groupby(["train", "test", "rep"]):
        y, p = g["y"].to_numpy(), g["p"].to_numpy()
        pm = patient_metrics(y, p)
        lg = _logit(p)
        cen = patient_metrics(y, (lg - lg.mean() > 0).astype(float))
        rows.append(dict(entrena=tr, prueba=te, representacion=rep, n_pacientes=len(y), n_gbm=int(y.sum()), auroc=pm["auroc"],
                         p_exacto=M23.exact_p_auc_fast(y, p, pm["auroc"]) if len(set(y)) == 2 else np.nan, macro_f1_corte_05=pm["macro_f1"], macro_f1_centrado=cen["macro_f1"],
                         sensibilidad=pm["sensitivity"], especificidad=pm["specificity"], tipo="dentro de la cohorte (dejar un paciente fuera)" if tr == te else "entre cohortes"))
    return pd.DataFrame(rows), avg


def figures(out, M, avg):
    for key, name, ttl in (("auroc", "26_transfer_matrix_auroc", "AUROC por paciente"), ("macro_f1_corte_05", "26_transfer_matrix_macro_f1", "macro-F1 por paciente (corte 0,5)")):
        fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3))
        for a, rep in zip(ax, ("PCA-5", "HVG")):
            mat = np.full((len(TRAIN), len(TEST)), np.nan)
            txt = {}
            for i, tr in enumerate(TRAIN):
                for j, te in enumerate(TEST):
                    r = M[(M["entrena"] == tr) & (M["prueba"] == te) & (M["representacion"] == rep)]
                    if len(r):
                        mat[i, j] = r[key].iloc[0]
                        txt[(i, j)] = f"{mat[i, j]:.2f}" + (f"\np={r['p_exacto'].iloc[0]:.3f}" if key == "auroc" and not np.isnan(r["p_exacto"].iloc[0]) else "") + f"\n(n={int(r['n_pacientes'].iloc[0])})"
            im = a.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
            for (i, j), t in txt.items():
                a.text(j, i, t, ha="center", va="center", fontsize=8, color="w" if (mat[i, j] > 0.8 or mat[i, j] < 0.2) else "k")
                if TRAIN[i] == TEST[j]:
                    a.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="k", linewidth=2))
            a.set_xticks(range(len(TEST)))
            a.set_xticklabels(TEST, fontsize=9)
            a.set_yticks(range(len(TRAIN)))
            a.set_yticklabels(TRAIN)
            a.set_xlabel("cohorte de prueba")
            a.set_ylabel("cohorte de entrenamiento")
            a.set_title(rep if rep != "HVG" else "HVG (genes más variables, sin PCA)", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8, label=ttl)
        d = M[(M["representacion"] == "PCA-5")]
        lines = [f"{ttl} al entrenar con la cohorte de cada fila y predecir la de cada columna (sin reajustar). Recuadro negro = dentro de la propia cohorte, dejando un paciente fuera.",
                 "Filas: Wälchli (scRNA-seq), Xie core (scRNA-seq, edad confundida) y Bejarano (RNA-seq en masa). Columnas: las mismas más la periferia de Xie, que solo se usa como prueba.",
                 "Un modelo que solo aprende diferencias de estudio fallaría fuera de su cohorte; uno que aprende señal biológica compartida debería mantener el AUROC (el orden) aunque el macro-F1 (el umbral) se desplace.",
                 "Con 6-18 pacientes por cohorte los valores son muy inciertos; no hay intervalos de confianza."]
        for _, r in d.iterrows():
            lines.append(f"PCA-5, entrena {r['entrena']} -> prueba {r['prueba']}: AUROC {r['auroc']:.2f}, macro-F1 {r['macro_f1_corte_05']:.2f}")
        save_fig(fig, out, name, lines)
    # probabilidades por paciente al entrenar con Bejarano
    sub = avg[(avg["train"] == "Bejarano") & (avg["rep"] == "PCA-5") & (avg["test"] != "Bejarano")]
    tests = [t for t in TEST if t in set(sub["test"])]
    fig, ax = plt.subplots(1, len(tests), figsize=(4.6 * len(tests), 4.2), sharey=True)
    ax = np.atleast_1d(ax)
    rng = np.random.default_rng(0)
    for a, t in zip(ax, tests):
        g = sub[sub["test"] == t]
        for d_, yv in (("LGG", 0), ("GBM", 1)):
            gg = g[g["y"] == yv]
            a.scatter(yv + rng.uniform(-0.15, 0.15, len(gg)), gg["p"], color=COL[d_], edgecolor="k", s=60, zorder=3)
            for _, r in gg.iterrows():
                a.annotate(str(r["patient"]), (yv + 0.06, r["p"]), fontsize=6)
        a.axhline(0.5, color="k", linestyle=":")
        a.set_xticks([0, 1])
        a.set_xticklabels(["LGG", "GBM"])
        a.set_title(f"prueba: {t}", fontsize=10)
        a.set_ylim(-0.03, 1.03)
        a.grid(alpha=0.3)
    ax[0].set_ylabel("P(GBM) del modelo entrenado con Bejarano (PCA-5)")
    save_fig(fig, out, "26_reverse_patient_probabilities", [
        "Transferencia inversa: modelo PCA-5 entrenado solo con las 18 muestras en masa de Bejarano y aplicado a cada paciente de Wälchli y Xie (media de sus paquetes y semillas).",
        "Color = diagnóstico real; línea de puntos = umbral 0,5. Si GBM y LGG forman dos nubes separadas el modelo conserva el orden aunque el umbral esté desplazado.",
        "Se entrena con RNA-seq en masa y se predice scRNA-seq agregado en pseudobulks: diferencia de tecnología esperable."])


def block(M):
    sel = M[M["representacion"].isin(["PCA-5", "HVG"])].copy()
    t = sel[["entrena", "prueba", "representacion", "n_pacientes", "auroc", "p_exacto", "macro_f1_corte_05", "macro_f1_centrado", "tipo"]].sort_values(["representacion", "entrena", "prueba"])
    return "\n".join(["**Matriz de transferencia entre cohortes** (`experiments/26_transfer_matrix.py`; `tablas/transfer_matrix.csv`; figuras `results/transfer_matrix/26_*.png`). "
                      "Se entrena con la cohorte de cada fila y se predice la de cada columna, sin reajustar; la diagonal es dejar un paciente fuera dentro de la cohorte.", "",
                      to_markdown(t.round(3).reset_index(drop=True)), "",
                      "Cautelas: Xie core tiene la edad confundida con el diagnóstico; Bejarano es RNA-seq en masa; 6-18 pacientes por cohorte (p exacto mínimo limitado por el nº de pacientes).", ""])


def main(cfg, xie_path, bej, out, readme=None, seeds=3, spec=None, S=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    S = S or M19.load_studies(cfg, xie_path)
    P = run(S, cfg, dict(spec or P2.BASE), seeds, bej)
    P.to_csv(out / "transfer_patient_predictions.csv", index=False)
    M, avg = metrics(P)
    M.to_csv(out / "transfer_matrix.csv", index=False)
    figures(out, M, avg)
    if readme:
        inject_block(readme, "TRANSFER", block(M))
    (out / "transfer_readme_block.md").write_text(block(M), encoding="utf-8")
    return M


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--xie", default="data/xie/xie_endothelial.h5ad")
    ap.add_argument("--counts", default="data/bejarano/BrainTime_vasculature_human.counts.csv")
    ap.add_argument("--metadata", default="data/bejarano/BrainTime_vasculature_human.metadata.csv")
    ap.add_argument("--ages", default="configs/bejarano_bulk_ages.csv")
    ap.add_argument("--out", default="results/transfer_matrix")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    main(load_config(a.config), a.xie, (a.counts, a.metadata, a.ages), a.out, a.readme, a.seeds)
