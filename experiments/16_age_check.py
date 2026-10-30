"""¿Explica la edad lo que sigue PC2? Con los 14 pacientes de Walchli (salida de 12_pc_meaning). Exploratorio, potencia muy baja."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.evaluation import describe_figure, to_markdown


def _lopo(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    pred = np.zeros(len(y), dtype=int)
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = LogisticRegression(max_iter=1000).fit((X[tr] - mu) / sd, y[tr])
        pred[i] = m.predict(((X[i:i + 1] - mu) / sd))[0]
    return pred


def partial_spearman(x, y, z) -> float:
    """Correlación de rangos entre x e y quitando linealmente (en rangos) z."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def res(a):
        k = np.polyfit(rz, a, 1)
        return a - np.polyval(k, rz)
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def analyze(t: pd.DataFrame) -> pd.DataFrame:
    t = t.dropna(subset=["age_num", "PC2"]).reset_index(drop=True)
    y = (t["glioma_type"] == "LGG").astype(int).to_numpy()
    age, pc2 = t["age_num"].to_numpy(float), t["PC2"].to_numpy(float)
    rows = []
    for name, X in (("solo edad", age[:, None]), ("solo PC2", pc2[:, None]), ("PC2 + edad", np.c_[pc2, age])):
        pred = _lopo(X, y)
        wrong = ",".join(t.loc[pred != y, "Patient"])
        rows.append(dict(analisis=f"Dejar un paciente fuera, {name}", valor=float((pred == y).mean()), detalle=f"pacientes mal clasificados: {wrong or 'ninguno'}"))
    rows.append(dict(analisis="AUC de la edad para LGG (invertida)", valor=float(roc_auc_score(y, -age)), detalle=""))
    rows.append(dict(analisis="AUC de PC2 para LGG", valor=float(roc_auc_score(y, pc2)), detalle=""))
    rows.append(dict(analisis="Spearman PC2 con edad (todos)", valor=float(spearmanr(pc2, age)[0]), detalle=f"n={len(t)}"))
    for c in ("GBM", "LGG"):
        m = (t["glioma_type"] == c).to_numpy()
        if m.sum() >= 4:
            rows.append(dict(analisis=f"Spearman PC2 con edad solo en {c}", valor=float(spearmanr(pc2[m], age[m])[0]), detalle=f"n={int(m.sum())}"))
    rows.append(dict(analisis="Spearman parcial PC2 con LGG quitando la edad", valor=partial_spearman(pc2, y, age), detalle="rangos; indicativo"))
    return pd.DataFrame(rows)


def main(pcs_csv, out) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(pcs_csv)
    res = analyze(t)
    res.to_csv(out / "age_check.csv", index=False)
    (out / "age_check.md").write_text(to_markdown(res.round(3)) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6, 4.6))
    for gt, c in (("GBM", "#D55E00"), ("LGG", "#009E73")):
        g = t[t["glioma_type"] == gt]
        ax.scatter(g["age_num"], g["PC2"], color=c, label=gt, s=45)
        for r in g.itertuples():
            ax.annotate(r.Patient, (r.age_num, r.PC2), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set(xlabel="Age", ylabel="Mean PC2 per patient")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "16_age_vs_pc2.png", dpi=130)
    plt.close(fig)
    lines = ["16_age_vs_pc2.png",
             "Qué expresa: PC2 media por paciente frente a su edad, coloreado por tipo de glioma (14 pacientes de Wälchli). Si la edad lo explicara "
             "todo, los puntos de cada color no se solaparían en edad; sirve para ver si PC2 separa los tipos más allá de la edad.", ""]
    lines += [f"{r.analisis}: {r.valor:.3f}" + (f" ({r.detalle})" if r.detalle else "") for r in res.itertuples()]
    describe_figure(out, "16_age_vs_pc2.png", lines)
    print("\n".join(lines))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcs", default="results/pc_meaning/patients_metadata_pcs.csv")
    ap.add_argument("--out", default="results/age_check")
    a = ap.parse_args()
    main(a.pcs, a.out)
