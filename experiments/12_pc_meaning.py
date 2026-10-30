"""¿Qué representan PC1 y PC2 del modelo? Por paciente: metadatos, posición en PCA-2 y correlaciones. También deja a la vista
los metadatos de LGG1 y LGG6 (los pacientes que el modelo clasifica mal)."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data import CELLTYPE, GROUP, build_dataset, load_config
from src.evaluation import describe_figure, to_markdown
from src.features import capped_indices
from src.final_model import load_bundle
from src.rom import covariates


def _num(v):
    m = re.search(r"\d+(\.\d+)?", str(v))
    return float(m.group()) if m else np.nan


def patient_table(meta: pd.DataFrame, Z: np.ndarray, idx: np.ndarray, classes) -> pd.DataFrame:
    g = meta.groupby(GROUP, sort=True)
    t = g.size().rename("n_cells").to_frame()
    t["glioma_type"] = g["glioma_type"].first()
    for c in meta.columns:
        if c in (GROUP, "cell", "glioma_type", CELLTYPE, "nCount_RNA", "nFeature_RNA") or meta[c].dtype.kind in "fi":
            continue
        if (g[c].nunique() == 1).all() and 1 <= meta[c].nunique() <= 30:
            t[c] = g[c].first()
    if "age" in meta.columns:
        t["age_num"] = t.index.map(lambda p: _num(g["age"].first()[p]))
    t["median_nCount_RNA"] = g["nCount_RNA"].median()
    t["median_nFeature_RNA"] = g["nFeature_RNA"].median()
    if CELLTYPE in meta.columns:
        comp = pd.crosstab(meta[GROUP], meta[CELLTYPE], normalize="index")
        t = t.join(comp.add_prefix("frac:"))
    pcs = pd.DataFrame(Z[:, :2], columns=["PC1", "PC2"]).assign(Patient=meta[GROUP].to_numpy()[idx]).groupby("Patient").mean()
    return t.join(pcs).reset_index()


def correlations(t: pd.DataFrame) -> pd.DataFrame:
    cand = ["age_num", "median_nCount_RNA", "median_nFeature_RNA", "n_cells"]
    cand += sorted([c for c in t.columns if c.startswith("frac:")], key=lambda c: -t[c].mean())[:6]
    t = t.assign(is_LGG=(t["glioma_type"] == sorted(t["glioma_type"].unique())[-1]).astype(float))
    cand.append("is_LGG")
    rows = []
    for v in cand:
        if v not in t.columns or t[v].nunique() < 3 and v != "is_LGG":
            continue
        for pc in ("PC1", "PC2"):
            ok = t[[v, pc]].dropna()
            if len(ok) >= 4 and ok[v].nunique() > 1:
                r, p = spearmanr(ok[v], ok[pc])
                rows.append(dict(variable=v, component=pc, spearman_rho=float(r), p_unadjusted=float(p), n_patients=len(ok)))
    return pd.DataFrame(rows)


def main(cfg: dict, bundle_path, out, n_per_patient: int = 3000) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ds = build_dataset(cfg)
    b = load_bundle(bundle_path)
    rep = next(r for r in b["reps"] if r.name == "PCA-2")
    lr = b["models"][("PCA-2", "logistic_regression")]
    idx = np.sort(np.concatenate(list(capped_indices(ds.meta, n_per_patient, cfg["seed"]).values())))
    Z = rep.transform(ds.X[idx], covariates(ds.meta.iloc[idx], b["regress_out"]))
    t = patient_table(ds.meta, Z, idx, b["classes"])
    w, bias = lr.coef_[0], float(lr.intercept_[0])
    t["side"] = np.where(t[["PC1", "PC2"]].to_numpy() @ w + bias > 0, b["classes"][1], b["classes"][0])
    t["side_matches_label"] = t["side"] == t["glioma_type"]
    cor = correlations(t)
    t.to_csv(out / "patients_metadata_pcs.csv", index=False)
    cor.to_csv(out / "pc_correlations.csv", index=False)
    (out / "pc_correlations.md").write_text(to_markdown(cor.round(3)) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {"GBM": "#D55E00", "LGG": "#009E73"}
    for gt, g in t.groupby("glioma_type"):
        ax[0].scatter(g["PC1"], g["PC2"], s=40 + 160 * g["n_cells"] / t["n_cells"].max(), color=colors.get(gt, "gray"), alpha=0.8, label=gt)
    for r in t.itertuples():
        ax[0].annotate(r.Patient, (r.PC1, r.PC2), fontsize=7, xytext=(3, 3), textcoords="offset points")
    xs, yl = np.array(ax[0].get_xlim()), ax[0].get_ylim()
    if abs(w[1]) > 1e-9:
        ax[0].plot(xs, -(w[0] * xs + bias) / w[1], "k--", lw=1, label="decision boundary")
    ax[0].set(xlim=tuple(xs), ylim=yl, xlabel="Mean PC1 per patient", ylabel="Mean PC2 per patient")
    ax[0].legend(fontsize=8)
    xcol = "age_num" if "age_num" in t.columns and t["age_num"].notna().sum() > 3 else "median_nCount_RNA"
    for gt, g in t.groupby("glioma_type"):
        ax[1].scatter(g[xcol], g["PC1"], s=40, color=colors.get(gt, "gray"), label=gt)
    ax[1].set(xlabel=xcol, ylabel="Mean PC1 per patient")
    ax[1].legend(fontsize=8)
    fig.suptitle("Wälchli patients in the PCA-2 space of the trained model", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "12_pc_by_patient.png", dpi=130)
    plt.close(fig)

    lines = ["12_pc_by_patient.png",
             "Qué expresa: cada punto es un paciente de Wälchli (tamaño = nº de células), en la media de PC1 y PC2 del modelo PCA-2; la línea es la "
             "frontera de la regresión logística final. Derecha: PC1 media frente a " + xcol + ". Sirve para ver si los ejes del modelo "
             "siguen al tipo de glioma o a otra variable del paciente.", ""]
    for r in t.itertuples():
        lines.append(f"{r.Patient} ({r.glioma_type}): PC1 {r.PC1:.2f}, PC2 {r.PC2:.2f}, lado {r.side}" + ("" if r.side_matches_label else "  <-- del lado contrario a su etiqueta"))
    if len(cor):
        lines.append("")
        lines.append("Correlaciones de Spearman entre pacientes (sin ajustar por comparaciones múltiples; pocos pacientes):")
        for r in cor.reindex(cor["spearman_rho"].abs().sort_values(ascending=False).index).head(8).itertuples():
            lines.append(f"  {r.component} con {r.variable}: rho {r.spearman_rho:+.2f} (p {r.p_unadjusted:.3f}, n {r.n_patients})")
    describe_figure(out, "12_pc_by_patient.png", lines)
    print("\n".join(lines))
    print("\nMetadatos de los pacientes mal clasificados y del resto:\n" + t.drop(columns=[c for c in t.columns if c.startswith("frac:")]).round(2).to_string(index=False))
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", default="results/pc_meaning")
    a = ap.parse_args()
    main(load_config(a.config), a.bundle, a.out)
