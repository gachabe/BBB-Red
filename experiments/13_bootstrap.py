"""Intervalos de confianza por remuestreo de PACIENTES (bootstrap estratificado por clase) para Wälchli y para Xie."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import describe_figure, to_markdown
from src.permutation import stats_from_fraction

METRICS = ("macro_f1", "balanced_accuracy", "patient_acc")


def bootstrap_ci(pp: pd.DataFrame, classes, B: int = 5000, seed: int = 0) -> dict:
    """pp: Patient, label, accuracy (fracción de células bien clasificadas) de UNA pareja representación-modelo."""
    a = (pp["label"].to_numpy() == classes[1]).astype(int)
    acc = pp["accuracy"].to_numpy(dtype=float)
    frac = np.where(a == 1, acc, 1 - acc)                      # fracción de células predichas como clase 1
    point = stats_from_fraction(frac, a)
    rng = np.random.default_rng(seed)
    i0, i1 = np.flatnonzero(a == 0), np.flatnonzero(a == 1)
    boots = np.empty((B, 3))
    for b in range(B):
        idx = np.concatenate([rng.choice(i0, len(i0)), rng.choice(i1, len(i1))])
        boots[b] = stats_from_fraction(frac[idx], a[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return {m: (point[k], lo[k], hi[k]) for k, m in enumerate(METRICS)}


def table(per_patient: pd.DataFrame, classes, study: str, B: int, seed: int) -> pd.DataFrame:
    d = per_patient[per_patient["variant"] == "real"] if "variant" in per_patient.columns else per_patient
    rows = []
    for (rep, model), g in d.groupby(["representation", "model"], sort=False):
        r = bootstrap_ci(g, classes, B, seed)
        row = dict(study=study, representation=rep, model=model, n_patients=len(g))
        for m in METRICS:
            row[m], row[m + "_lo"], row[m + "_hi"] = r[m]
        rows.append(row)
    return pd.DataFrame(rows)


DEFAULT_STUDIES = {
    "Walchli": "results/full/results_per_patient_glioma_type.csv",
    "Xie": "results/external_xie/results_per_patient_external.csv",
    "Xie core": "results/external_xie_core/results_per_patient_external.csv",
    "Xie alineado (exploratorio)": "results/external_xie_aligned/results_per_patient_external_aligned.csv",
    "Dentro de Xie (exploratorio)": "results/within_xie/results_per_patient_within_xie.csv",
    "Xie alineado sin sus células (exploratorio)": "results/external_xie_aligned/results_per_patient_external_aligned_loo.csv",
    "Xie solo periferia (exploratorio)": "results/external_xie_periph/results_per_patient_external.csv",
}


def main(studies: dict, out, B: int = 5000, seed: int = 0) -> pd.DataFrame:
    """studies: {nombre: csv por paciente}. Se omiten los que no existen."""
    import matplotlib.pyplot as plt
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    classes = ["GBM", "LGG"]
    parts = []
    for name, path in studies.items():
        if Path(path).exists():
            parts.append(table(pd.read_csv(path), classes, name, B, seed))
        else:
            print(f"(omitido, no existe) {name}: {path}")
    res = pd.concat(parts, ignore_index=True)
    res.to_csv(out / "bootstrap_ci.csv", index=False)
    (out / "bootstrap_ci.md").write_text(to_markdown(res.round(3)) + "\n", encoding="utf-8")

    names = list(dict.fromkeys(res["study"]))
    reps = list(dict.fromkeys(res["representation"]))
    ncol = min(3, len(names))
    nrow = -(-len(names) // ncol)
    fig, ax = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.4 * nrow), sharey=True, squeeze=False)
    for a_ in ax.ravel()[len(names):]:
        a_.axis("off")
    for a_, s_ in zip(ax.ravel(), names):
        g_s = res[res["study"] == s_]
        for k, (model, g) in enumerate(g_s.groupby("model", sort=False)):
            g = g.set_index("representation").reindex(reps).dropna(subset=["macro_f1"])
            x = np.array([reps.index(r) for r in g.index]) + (k - 0.5) * 0.18
            a_.errorbar(x, g["macro_f1"], yerr=[g["macro_f1"] - g["macro_f1_lo"], g["macro_f1_hi"] - g["macro_f1"]], fmt="o", capsize=3, label=model)
        a_.axhline(0.5, color="gray", ls=":")
        a_.set_xticks(range(len(reps)), reps, fontsize=7, rotation=35)
        a_.set(title=f"{s_} ({int(g_s['n_patients'].iloc[0])} patients)", ylim=(0, 1.02))
        a_.set_ylabel("Macro-F1 (95% bootstrap CI)")
        a_.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "13_bootstrap_ci.png", dpi=130)
    plt.close(fig)

    lines = ["13_bootstrap_ci.png",
             "Qué expresa: Macro-F1 con su intervalo de confianza al 95 % obtenido remuestreando PACIENTES (con reemplazo, manteniendo cuántos hay "
             "de cada clase), por representación y modelo, un panel por análisis. La línea punteada es 0,5 (azar). No reentrena el modelo, así que "
             "los intervalos reales serían algo más anchos. Muestra cuánta incertidumbre hay al tener tan pocos pacientes.", ""]
    for s_ in names:
        g = res[res["study"] == s_]
        lines.append(f"{s_} ({int(g['n_patients'].iloc[0])} pacientes):")
        for r in g.itertuples():
            lines.append(f"  {r.representation} | {r.model}: F1 {r.macro_f1:.3f} [{r.macro_f1_lo:.3f}, {r.macro_f1_hi:.3f}]; "
                         f"acierto por paciente {r.patient_acc:.2f} [{r.patient_acc_lo:.2f}, {r.patient_acc_hi:.2f}]")
        lines.append(f"  Filas con el límite inferior del intervalo por encima de 0,5: {int((g['macro_f1_lo'] > 0.5).sum())} de {len(g)}.")
    describe_figure(out, "13_bootstrap_ci.png", lines)
    print("\n".join(lines))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", action="append", default=[], metavar="NOMBRE=CSV",
                    help="análisis a añadir o sustituir (repetible); por defecto Walchli, Xie, Xie core, alineado y dentro de Xie")
    ap.add_argument("--out", default="results/bootstrap")
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args()
    st = dict(DEFAULT_STUDIES)
    for item in a.study:
        k, v = item.split("=", 1)
        st[k] = v
    main(st, a.out, a.n_boot)
