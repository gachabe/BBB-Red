"""Pseudobulks (propuesta de los coautores): promediar paquetes de células del mismo paciente y repetir el análisis.

Tres análisis, cada uno con varios tamaños de paquete y varias particiones aleatorias (semillas):
  A. Dentro de Walchli, dejando un paciente fuera (todos los pseudobulks del paciente quedan juntos en el mismo pliegue).
  B. Entrenar con los pseudobulks de Walchli y aplicar a los de Xie (todo, solo core, solo periferia), sin reajustar nada.
  C. Diagnóstico de estudio: ¿se sigue identificando Walchli frente a Xie con pseudobulks?
Las métricas ponderan cada paciente por igual y además se resumen con UNA predicción por paciente (columnas *_patient).
Análisis principal fijado de antemano: tamaño 50 y regresión de profundidad según la configuración.
"""
from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.batch import common_genes, domain_cv
from src.data import GROUP, build_dataset, load_config
from src.evaluation import classification_metrics, describe_figure, per_patient_table, to_markdown
from src.features import lopo_folds, task_classes
from src.final_model import align_genes, apply_all, train_bundle
from src.models import make_model
from src.pseudobulk import make_pseudobulks, patient_level
from src.rom import covariates, fit_representations

TASK = "glioma_type"


def _summarise(y, pred, pats, classes, w):
    m = classification_metrics(y, pred, classes, w)
    pp = per_patient_table(y, pred, pats, classes)
    row = dict(macro_f1=m["macro_f1"], balanced_accuracy=m["balanced_accuracy"],
               patient_acc=float((pp["label"] == pp["pred_majority"]).mean()), n_patients=len(pp), n_pseudobulks=len(y))
    row.update(patient_level(pp, classes))
    return row, pp


def within_lopo(Xp, meta, cfg, rom, cap_pb, models):
    classes = task_classes(meta, TASK)
    y = np.searchsorted(classes, meta[TASK].astype(str).to_numpy())
    w = 1.0 / meta.groupby(GROUP)["cell"].transform("size").to_numpy()
    C = covariates(meta, rom.get("regress_out"))
    preds = {}
    for f in lopo_folds(meta, cap_pb, cfg["seed"]):
        assert meta[GROUP].iloc[f.test_idx].nunique() == 1 and f.held_out not in set(meta[GROUP].iloc[f.train_idx])
        r = {**rom, "components": [c for c in rom["components"] if c < min(len(f.train_idx), rom["hvg"])]}
        Ctr, Cte = (None, None) if C is None else (C[f.train_idx], C[f.test_idx])
        for rep in fit_representations(Xp[f.train_idx], r, Ctr):
            Ztr, Zte = rep.transform(Xp[f.train_idx], Ctr), rep.transform(Xp[f.test_idx], Cte)
            for m in models:
                mod = make_model(m, cfg["seed"], cfg.get("ml", {}).get("rf_trees", 200)).fit(Ztr, y[f.train_idx])
                preds.setdefault((rep.name, rep.n_features, m), []).append((f.test_idx, mod.predict(Zte)))
    rows, pps = [], []
    for (rn, nf, m), parts in preds.items():
        idx = np.concatenate([p[0] for p in parts])
        pred = np.concatenate([p[1] for p in parts])
        row, pp = _summarise(y[idx], pred, meta[GROUP].to_numpy()[idx], classes, w[idx])
        rows.append(dict(model=m, representation=rn, n_features=nf, **row))
        pps.append(pp.assign(representation=rn, model=m))
    return pd.DataFrame(rows), pd.concat(pps, ignore_index=True)


def external(ds_pb, cfg, rom, cap_pb, Xx, genes_x, mx):
    cfg_pb = {**cfg, "rom": rom, "ml": {**cfg.get("ml", {}), "max_train_cells_per_patient": cap_pb}}
    with contextlib.redirect_stdout(io.StringIO()):
        b = train_bundle(ds_pb, cfg_pb)
    Xa, _ = align_genes(Xx, genes_x, b["genes"])
    C = covariates(mx, b["regress_out"])
    subsets = {"todo": np.ones(len(mx), bool)}
    if "site" in mx.columns:
        subsets["core"] = (mx["site"] == "core").to_numpy()
        subsets["periferia"] = (mx["site"] == "peripheral").to_numpy()
    rows, pps = [], []
    for name, mask in subsets.items():
        if mask.sum() == 0 or mx.loc[mask, TASK].nunique() < 2:
            continue
        res, pp = apply_all(b, Xa[mask], mx[mask].reset_index(drop=True), None if C is None else C[mask], GROUP, TASK)
        for (rn, m), g in pp.groupby(["representation", "model"], sort=False):
            r = res[(res["representation"] == rn) & (res["model"] == m)].iloc[0]
            rows.append(dict(subset=name, model=m, representation=rn, n_features=int(r["n_features"]), macro_f1=r["macro_f1"],
                             balanced_accuracy=r["balanced_accuracy"], patient_acc=r["patient_acc"], n_patients=len(g),
                             n_pseudobulks=int(r["n_cells"]), **patient_level(g, b["classes"])))
        pps.append(pp.assign(subset=name))
    return pd.DataFrame(rows), pd.concat(pps, ignore_index=True)


def one_run(ds, xie, cfg, size, seed, regress, models, include_all, cap_cells=1000):
    cfg_s = {**cfg, "seed": cfg["seed"] + seed}
    rom = {**cfg["rom"], "include_all_genes": include_all, "regress_out": cfg["rom"].get("regress_out") if regress else None}
    cap_pb = max(5, int(np.ceil(cfg.get("ml", {}).get("max_train_cells_per_patient", cap_cells) / size)))
    Xw, mw, _ = make_pseudobulks(ds.X, ds.meta, size, cfg_s["seed"])
    Xe, obs, genes_x = xie
    gcols = ["Patient", "sample"] if "sample" in obs.columns else ["Patient"]
    Xx, mx, _ = make_pseudobulks(Xe, obs, size, cfg_s["seed"], group_cols=gcols, keep_cols=("Patient", TASK, "sample", "site"))
    cnt = mw.groupby(GROUP).size().clip(upper=cap_pb)
    rom = {**rom, "components": [c for c in rom["components"] if c < int(cnt.sum() - cnt.max())]}     # PCA-r necesita más filas que r
    assert set(mw[GROUP]) == set(ds.meta[GROUP]) and set(mx[GROUP]) == set(obs[GROUP]), "se perdió algún paciente al hacer pseudobulks"
    tag = dict(size=size, seed=seed, regress_depth=regress)
    w_res, w_pp = within_lopo(Xw, mw, cfg_s, rom, cap_pb, models)
    e_res, e_pp = external(SimpleNamespace(X=Xw, meta=mw, genes=ds.genes), cfg_s, rom, cap_pb, Xx, genes_x, mx)
    Xa, Xb, _ = common_genes(Xw, ds.genes, Xx, genes_x)
    meta = pd.concat([mw[["cell", GROUP, "nCount_RNA", "nFeature_RNA"]].assign(study="Walchli"),
                      mx[["cell", GROUP, "nCount_RNA", "nFeature_RNA"]].assign(study="Xie")], ignore_index=True)
    b_res, _ = domain_cv(sp.vstack([Xa, Xb]).tocsr(), meta, rom, models, cap_pb, cfg_s["seed"], cfg.get("ml", {}).get("rf_trees", 200),
                         regress, log=lambda *_: None)
    out = {k: v.assign(**tag) for k, v in dict(within=w_res, external=e_res, batch=b_res, within_pp=w_pp, external_pp=e_pp).items()}
    out["n_pb"] = dict(walchli=len(mw), xie=len(mx))
    return out


def figure(summ: dict, primary, out: Path, classes_note: str):
    import matplotlib.pyplot as plt
    panels = [("Dentro de Walchli (LOPO)", summ["within"], "macro_f1"), ("Xie, todo", summ["external"].query("subset == 'todo'"), "macro_f1"),
              ("Xie, solo core", summ["external"].query("subset == 'core'"), "macro_f1"),
              ("Xie, solo periferia", summ["external"].query("subset == 'periferia'"), "macro_f1"),
              ("Identificar el estudio (batch)", summ["batch"], "balanced_accuracy")]
    panels = [p for p in panels if len(p[1])]
    fig, ax = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4.4), squeeze=False)
    lines = ["17_pseudobulk.png",
             f"Qué expresa: resultado con pseudobulks de {primary[0]} células (análisis principal; regresión de profundidad: {'sí' if primary[1] else 'no'}), "
             "media y desviación entre particiones aleatorias de los paquetes. Un panel por análisis: dentro de Wälchli, aplicado a Xie (todo, core, periferia) "
             "y capacidad de identificar el estudio (balanced accuracy; cuanto más alto, más efecto de estudio). Línea punteada: 0,5.", ""]
    for a_, (title, d, metric) in zip(ax[0], panels):
        d = d[(d["size"] == primary[0]) & (d["regress_depth"] == primary[1])]
        for model, g in d.groupby("model"):
            g = g.sort_values("n_features")
            a_.errorbar(range(len(g)), g[metric + "_mean"], yerr=g[metric + "_std"].fillna(0), marker="o", capsize=3, label=model)
            a_.set_xticks(range(len(g)), g["representation"], rotation=35, fontsize=7)
            best = g.loc[g[metric + "_mean"].idxmax()]
            sd = 0 if pd.isna(best[metric + "_std"]) else best[metric + "_std"]
            lines.append(f"{title} | {model}: mejor {best['representation']} con {metric} {best[metric + '_mean']:.3f} (±{sd:.3f})")
        a_.axhline(0.5, color="gray", ls=":")
        a_.set(title=title, ylim=(0, 1.02), ylabel=metric)
        a_.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "17_pseudobulk.png", dpi=130)
    plt.close(fig)
    describe_figure(out, "17_pseudobulk.png", lines)
    print("\n".join(lines))


def main(cfg: dict, xie_path, out, sizes=(25, 50, 100), n_seeds=5, regress="both", primary_size=50, models=None,
         include_all=None, force=False) -> dict:
    out = Path(out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    models = list(models or cfg["models"])
    include_all = cfg["rom"].get("include_all_genes", True) if include_all is None else include_all
    variants = {"both": [True, False], "yes": [True], "no": [False]}[regress]
    if not cfg["rom"].get("regress_out"):
        variants = [False]
    ds = build_dataset(cfg)
    a = ad.read_h5ad(xie_path)
    Xe = a.X.astype(np.float32).tocsr() if sp.issparse(a.X) else sp.csr_matrix(np.asarray(a.X, dtype=np.float32))
    xie = (Xe, a.obs.reset_index(drop=True), np.asarray(a.var_names.astype(str)))
    order = sorted(sizes, key=lambda s: (s != primary_size, s))          # el principal primero
    runs = []
    for size in order:
        for reg in variants:
            for s in range(n_seeds):
                f = out / "runs" / f"s{size}_seed{s}_reg{int(reg)}.joblib"
                if f.exists() and not force:
                    runs.append(joblib.load(f))
                    continue
                r = one_run(ds, xie, cfg, size, s, reg, models, include_all)
                joblib.dump(r, f)
                runs.append(r)
                print(f"  hecho: tamaño {size}, semilla {s}, regresión de profundidad {reg} "
                      f"({r['n_pb']['walchli']} pseudobulks de Walchli, {r['n_pb']['xie']} de Xie)", flush=True)
    allr = {k: pd.concat([r[k] for r in runs], ignore_index=True) for k in ("within", "external", "batch", "within_pp", "external_pp")}
    for k in ("within", "external", "batch"):
        allr[k].to_csv(out / f"pseudobulk_{k}_all_runs.csv", index=False)

    def agg(d, keys):
        num = [c for c in d.columns if d[c].dtype.kind == "f" and c not in ("seed",)]
        g = d.groupby(keys + ["size", "regress_depth"])[num].agg(["mean", "std"])
        g.columns = [f"{c}_{f}" for c, f in g.columns]
        return g.reset_index()
    summ = {"within": agg(allr["within"], ["model", "representation", "n_features"]),
            "external": agg(allr["external"], ["subset", "model", "representation", "n_features"]),
            "batch": agg(allr["batch"], ["model", "representation", "n_features"])}
    for k, v in summ.items():
        v.to_csv(out / f"pseudobulk_{k}_summary.csv", index=False)
        (out / f"pseudobulk_{k}_summary.md").write_text(to_markdown(v.round(3)) + "\n", encoding="utf-8")
    prim = (primary_size if primary_size in sizes else order[0], variants[0])
    # acierto por paciente medio entre particiones (análisis principal): permite el bootstrap de 13_bootstrap
    for key, name, filt in (("within_pp", "within", None), ("external_pp", "external_todo", "todo"), ("external_pp", "external_core", "core")):
        d = allr[key]
        d = d[(d["size"] == prim[0]) & (d["regress_depth"] == prim[1])]
        if filt:
            d = d[d["subset"] == filt]
        if len(d):
            g = d.groupby(["Patient", "label", "representation", "model"], sort=False, as_index=False).agg(
                accuracy=("accuracy", "mean"), n_cells=("n_cells", "mean"))
            g["pred_majority"] = np.where((g["accuracy"] >= 0.5), g["label"], g["label"].map(lambda v: "LGG" if v == "GBM" else "GBM"))
            g.to_csv(out / f"results_per_patient_pb_{name}.csv", index=False)
    figure(summ, prim, out, "")
    print(f"\nResultados en {out}")
    return dict(summary=summ, all=allr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--xie", default="data/xie/xie_endothelial.h5ad")
    ap.add_argument("--out", default="results/pseudobulk")
    ap.add_argument("--sizes", type=int, nargs="+", default=[25, 50, 100])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--regress", choices=["both", "yes", "no"], default="both")
    ap.add_argument("--primary-size", type=int, default=50)
    ap.add_argument("--no-all-genes", action="store_true", help="omite 'All genes' (mucho más rápido)")
    ap.add_argument("--quick", action="store_true", help="prueba rápida: tamaño 50, 2 semillas, solo con regresión, sin All genes")
    ap.add_argument("--force", action="store_true", help="rehace aunque ya exista el resultado de esa ejecución")
    a = ap.parse_args()
    if a.quick:
        a.sizes, a.seeds, a.regress, a.no_all_genes = [50], 2, "yes", True
    main(load_config(a.config), a.xie, a.out, tuple(a.sizes), a.seeds, a.regress, a.primary_size,
         include_all=False if a.no_all_genes else None, force=a.force)
