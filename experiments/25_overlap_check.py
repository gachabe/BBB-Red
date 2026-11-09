"""¿Podría algún paciente de Bejarano et al. ser también un paciente de Wälchli?

Los autores no lo dicen y los identificadores no coinciden, así que se comparan los datos que sí tenemos de los dos lados: clase (GBM frente a
IDH nativo, LGG frente a IDH mutado), sexo y edad. Una pareja (paciente de Wälchli, paciente de Bejarano) es «compatible» si es de la misma clase,
del mismo sexo y con edades que difieren como mucho `tol` años. Esto NO demuestra que sean la misma persona (con 14 y 18 pacientes habrá parejas
compatibles por azar): sirve para (1) saber qué pacientes habría que preguntar a los autores y (2) comprobar si hay MÁS parejas compatibles de las
esperables por azar (permutando las edades y los sexos de Bejarano dentro de cada clase). Si un paciente de Wälchli no tiene ninguna pareja compatible,
no puede estar repetido.
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

M19 = importlib.import_module("experiments.19_pseudobulk_counts")


def _sex(v):
    """Male/Female/M/F/H(ombre)/W(oman)... -> 'M' o 'F'."""
    t = str(v).strip().upper()
    return "F" if t.startswith(("F", "W", "MUJER")) else ("M" if t.startswith(("M", "H")) else "?")


def walchli_table(ow):
    t = ow.groupby("Patient").agg(edad=("age_num", "first"), sexo=("sex", "first"), clase=("glioma_type", "first")).reset_index().rename(columns={"Patient": "paciente"})
    t["sexo"] = t["sexo"].map(_sex)
    return t


def bejarano_table(ages_csv, only_bulk=True):
    b = pd.read_csv(ages_csv)
    if only_bulk and "bulk_ec" in b:
        b = b[b["bulk_ec"].astype(str).str.lower().eq("yes")]
    b = b.rename(columns={"patient": "paciente", "age": "edad", "sex": "sexo"})
    b["clase"] = np.where(b["idh"].astype(str).str.lower().str.startswith("wild"), "GBM", "LGG")
    b["sexo"] = b["sexo"].map(_sex)
    return b[["paciente", "edad", "sexo", "clase"]].reset_index(drop=True)


def compatible(w, b, tol):
    rows = []
    for _, r in w.iterrows():
        for _, q in b.iterrows():
            if r["clase"] == q["clase"] and r["sexo"] == q["sexo"] and abs(r["edad"] - q["edad"]) <= tol:
                rows.append(dict(paciente_walchli=r["paciente"], paciente_bejarano=q["paciente"], clase=r["clase"], sexo=r["sexo"], edad_walchli=r["edad"], edad_bejarano=q["edad"], dif_edad=abs(r["edad"] - q["edad"])))
    return pd.DataFrame(rows, columns=["paciente_walchli", "paciente_bejarano", "clase", "sexo", "edad_walchli", "edad_bejarano", "dif_edad"])


def null_count(w, b, tol, n_perm, seed=0):
    """Nº de parejas compatibles al permutar (edad, sexo) de Bejarano dentro de cada clase: lo esperable por azar."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_perm, dtype=int)
    for i in range(n_perm):
        bb = b.copy()
        for c in bb["clase"].unique():
            m = (bb["clase"] == c).to_numpy()
            k = rng.permutation(m.sum())
            bb.loc[m, "edad"] = bb.loc[m, "edad"].to_numpy()[k]
            bb.loc[m, "sexo"] = bb.loc[m, "sexo"].to_numpy()[k]
        out[i] = len(compatible(w, bb, tol))
    return out


def main(cfg, xie_path, ages_csv, out, readme=None, tol=1.0, n_perm=2000, S=None, ow=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if ow is None:
        ow = (S or M19.load_studies(cfg, xie_path))[1]
    w, b = walchli_table(ow), bejarano_table(ages_csv)
    pairs = compatible(w, b, tol)
    obs = len(pairs)
    nul = null_count(w, b, tol, n_perm)
    p = float((nul >= obs).mean())
    per = w.assign(n_parejas=w["paciente"].map(pairs["paciente_walchli"].value_counts()).fillna(0).astype(int))
    pairs.to_csv(out / "overlap_compatible_pairs.csv", index=False)
    per.to_csv(out / "overlap_per_walchli_patient.csv", index=False)
    pd.DataFrame(dict(parejas_compatibles_observadas=[obs], media_azar=[nul.mean()], p_azar_ge_observado=[p], tolerancia_edad=[tol], n_permutaciones=[n_perm],
                      n_walchli=[len(w)], n_bejarano=[len(b)])).to_csv(out / "overlap_summary.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw=dict(width_ratios=[1.5, 1]))
    a = ax[0]
    xs = {("Wälchli", "GBM"): 0, ("Bejarano", "GBM"): 1, ("Wälchli", "LGG"): 3, ("Bejarano", "LGG"): 4}
    rng = np.random.default_rng(1)
    pos = {}
    for coh, t in (("Wälchli", w), ("Bejarano", b)):
        for _, r in t.iterrows():
            x = xs[(coh, r["clase"])] + rng.uniform(-0.2, 0.2)
            pos[(coh, r["paciente"])] = (x, r["edad"])
            a.scatter(x, r["edad"], s=55, marker="o" if r["sexo"] == "M" else "^", color=COL[r["clase"]], edgecolor="k", linewidth=0.5, zorder=3)
    for _, r in pairs.iterrows():
        (x1, y1), (x2, y2) = pos[("Wälchli", r["paciente_walchli"])], pos[("Bejarano", r["paciente_bejarano"])]
        a.plot([x1, x2], [y1, y2], color="k", alpha=0.35, linewidth=1, zorder=2)
    a.set_xticks(list(xs.values()))
    a.set_xticklabels(["Wälchli\nGBM", "Bejarano\nGBM IDH nativo", "Wälchli\nLGG", "Bejarano\nIDH mutado"])
    a.set_ylabel("edad (años)")
    a.set_title(f"Edad y sexo por paciente (círculo = hombre, triángulo = mujer)\nlíneas: parejas compatibles (misma clase y sexo, ≤ {tol:g} año)")
    a.grid(axis="y", alpha=0.3)
    h = ax[1]
    h.hist(nul, bins=np.arange(nul.min() - 0.5, max(nul.max(), obs) + 1.5, 1), color="#95a5a6", edgecolor="w")
    h.axvline(obs, color="#c0392b", linewidth=2, label=f"observado: {obs}")
    h.set_xlabel("nº de parejas compatibles")
    h.set_ylabel("permutaciones")
    h.set_title(f"Lo esperable por azar (media {nul.mean():.1f}); p = {p:.2f}")
    h.legend(frameon=False)
    save_fig(fig, out, "25_overlap_check", [
        "Posible solapamiento de pacientes entre Wälchli (14) y Bejarano (18 con RNA-seq en masa).",
        "Izquierda: edad de cada paciente por cohorte y clase (círculo = hombre, triángulo = mujer); las líneas unen parejas de la misma clase y sexo con edades que difieren ≤ "
        f"{tol:g} año. Derecha: nº de parejas compatibles que salen al permutar edades y sexos de Bejarano dentro de cada clase ({n_perm} permutaciones).",
        f"Parejas compatibles observadas: {obs}; esperadas por azar: {nul.mean():.1f}; P(azar >= observado) = {p:.3f}.",
        f"Pacientes de Wälchli sin ninguna pareja compatible (no pueden estar repetidos): {int((per['n_parejas'] == 0).sum())} de {len(per)}.",
        "Esto no prueba que sean la misma persona ni que no lo sean: solo acota a quién preguntar a los autores."])
    if readme:
        md = ["**Comprobación de solapamiento de pacientes con Wälchli** (`experiments/25_overlap_check.py`; `tablas/overlap_*.csv`; `results/overlap_check/25_overlap_check.png`)", "",
              f"Se comparan clase (GBM frente a IDH nativo, LGG frente a IDH mutado), sexo y edad (tolerancia de {tol:g} año) entre los {len(w)} pacientes de Wälchli y los {len(b)} de Bejarano con RNA-seq en masa. "
              f"**Parejas compatibles: {obs}; esperadas por azar: {nul.mean():.1f} (p = {p:.2f}).** "
              f"{int((per['n_parejas'] == 0).sum())} de los {len(per)} pacientes de Wälchli no tienen ninguna pareja compatible, así que no pueden estar repetidos.", "",
              "No prueba que haya o no solapamiento (con 14 y 18 pacientes hay parejas compatibles por azar); señala a quién preguntar:", "",
              to_markdown(pairs.round(1)) if len(pairs) else "_(ninguna pareja compatible)_", ""]
        inject_block(readme, "OVERLAP", "\n".join(md))
    return pairs, per, nul


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--xie", default="data/xie/xie_endothelial.h5ad")
    ap.add_argument("--ages", default="configs/bejarano_bulk_ages.csv")
    ap.add_argument("--out", default="results/overlap_check")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--tol", type=float, default=1.0)
    a = ap.parse_args()
    main(load_config(a.config), a.xie, a.ages, a.out, a.readme, a.tol)
