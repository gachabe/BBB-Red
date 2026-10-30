"""Regenera las figuras desde los CSV/JSON ya guardados (sin recargar datos ni reentrenar)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_config
from src.evaluation import (
    fig_by_cluster,
    fig_cohort,
    fig_components_seeds,
    fig_components_vs_performance,
    fig_confusions,
    fig_pca_variance,
    fig_per_patient,
    fig_permutation,
)


def main(cfg: dict) -> None:
    out = Path(cfg.get("results_dir", f"results/{cfg['mode']}"))

    def csv(name):
        p = out / name
        return pd.read_csv(p) if p.exists() else None

    if (d := csv("cohort_table.csv")) is not None:
        fig_cohort(d, out)
    if (v := csv("pca_variance.csv")) is not None:
        fig_pca_variance(v, cfg["rom"]["components"], out)
    for t in cfg.get("tasks", ["glioma_type"]):
        if (r := csv(f"results_{t}.csv")) is not None:
            fig_components_vs_performance(r, t, out)
        if (p := csv(f"results_per_patient_{t}.csv")) is not None:
            fig_per_patient(p, t, out)
        if (b := csv(f"results_by_cluster_{t}.csv")) is not None:
            fig_by_cluster(b, t, out)
        if (s := csv(f"results_seeds_summary_{t}.csv")) is not None:
            fig_components_seeds(s, t, out)
        cm_path = out / f"confusion_matrices_{t}.json"
        if cm_path.exists():
            j = json.loads(cm_path.read_text(encoding="utf-8"))
            cms = {tuple(k.split("|")): np.array(c) for k, c in j["real"].items()}
            fig_confusions(cms, j["classes"], t, out)
    perm, res = out / "permutation" / "perm_stats.csv", out / "permutation_results_glioma_type.csv"
    if perm.exists() and res.exists():
        st = pd.read_csv(perm).drop_duplicates(["id", "representation"])
        null_max = st.pivot(index="id", columns="representation", values="macro_f1").max(axis=1).to_numpy()
        pr = pd.read_csv(res)
        fig_permutation(null_max, dict(zip(pr["representation"], pr["real_macro_f1"])), "glioma_type", out)
    print(f"Figuras y descripciones regeneradas en {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(load_config(ap.parse_args().config))
