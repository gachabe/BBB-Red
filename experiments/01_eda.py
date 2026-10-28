"""EDA: cohorte, composición por tipo de vaso, fuga de información y normalización."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data import CELLTYPE, GROUP, leakage_report, library_size_check, load_config
from src.evaluation import fig_cohort


def run(ds, cfg: dict, outdir: Path) -> dict:
    meta, outdir = ds.meta, Path(outdir)

    ct = pd.crosstab(meta[GROUP], meta[CELLTYPE])
    ct.columns = [f"ec:{c}" for c in ct.columns]
    cohort = (
        meta.groupby(GROUP, sort=True).agg(glioma_type=("glioma_type", "first"), n_cells=("cell", "size")).reset_index()
        .merge(ct.reset_index(), on=GROUP)
    )
    cohort.to_csv(outdir / "cohort_table.csv", index=False)
    comp = pd.crosstab(meta[CELLTYPE], meta["glioma_type"])
    comp_frac = comp.div(comp.sum(axis=0), axis=1).add_prefix("fraction_")
    pd.concat([comp.add_prefix("n_"), comp_frac], axis=1).to_csv(outdir / "composition_by_glioma_type.csv")
    print("\nCélulas por paciente:\n", cohort[[GROUP, "glioma_type", "n_cells"]].to_string(index=False))
    print("\nComposición de ECclusters por tipo de glioma (fracción de las células de cada tipo):\n",
          comp_frac.round(3).to_string())
    fig_cohort(cohort, outdir)

    leak = {}
    for task in cfg.get("tasks", ["glioma_type"]):
        rep = leakage_report(meta.drop(columns=["cell"]), task)
        rep.to_csv(outdir / f"leakage_{task}.csv", index=False)
        leak[task] = rep[rep["note"] == "DETERMINA el objetivo"]["column"].tolist()
        print(f"\nFuga ({task}): columnas de obs que por sí solas determinan la etiqueta: {leak[task]}"
              "\n  -> ninguna se usa como característica (las características son solo X).")

    report = {"library_size": library_size_check(ds.X, seed=cfg["seed"]), "columns_that_determine_target": leak}
    (outdir / "eda_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    from src.features import prepare

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    out = Path(cfg.get("results_dir", f"results/{cfg['mode']}"))
    out.mkdir(parents=True, exist_ok=True)
    ds, _ = prepare(cfg, out)
    run(ds, cfg, out)
