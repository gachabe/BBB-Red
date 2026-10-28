"""Repite ROM + ML con varias semillas (submuestreo por paciente, RF, permutación) y agrega media ± std.

Con dejar-un-paciente-fuera el reparto no cambia entre semillas: la desviación mide el ruido del
submuestreo, del random forest y de la permutación, NO la incertidumbre por tener pocos pacientes.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import build_dataset, load_config
from src.evaluation import fig_components_seeds, to_markdown
from src.features import lopo_folds

METRICS = ("macro_f1", "balanced_accuracy", "patient_acc", "cell_macro_f1")


def _fmt(u, v) -> str:
    return "--" if np.isnan(u) else (f"{u:.3f}" if np.isnan(v) else f"{u:.3f} ± {v:.3f}")


def main(cfg: dict, n_seeds: int = 10) -> dict[str, pd.DataFrame]:
    rom = importlib.import_module("experiments.02_rom")
    ml = importlib.import_module("experiments.03_ml")
    out = Path(cfg.get("results_dir", f"results/{cfg['mode']}"))
    out.mkdir(parents=True, exist_ok=True)
    ds = build_dataset(cfg)  # se carga una sola vez
    cap = cfg.get("ml", {}).get("max_train_cells_per_patient")

    runs: dict[str, list] = {}
    for i in range(n_seeds):
        seed = cfg["seed"] + i
        folds = lopo_folds(ds.meta, cap, seed)
        fits = rom.run(ds, folds, cfg, out, plot=False)
        for t, res in ml.run(ds, fits, cfg, out, seed=seed, save=False).items():
            runs.setdefault(t, []).append(res.assign(seed=seed))
        print(f"semilla {seed} ({i + 1}/{n_seeds}) hecha")

    aggs = {}
    for t, lst in runs.items():
        raw = pd.concat(lst, ignore_index=True)
        raw.to_csv(out / f"results_seeds_raw_{t}.csv", index=False)
        g = raw.groupby(["model", "variant", "representation", "n_features"], sort=False)[list(METRICS)].agg(["mean", "std"])
        g.columns = [f"{m}_{s}" for m, s in g.columns]
        agg = g.reset_index()
        agg.to_csv(out / f"results_seeds_summary_{t}.csv", index=False)
        parts = []
        for (model, variant), a in agg.groupby(["model", "variant"], sort=False):
            tab = a[["representation", "n_features"]].copy()
            for m in METRICS:
                tab[m] = [_fmt(u, v) for u, v in zip(a[f"{m}_mean"], a[f"{m}_std"])]
            parts.append(f"### {t} — {model} — {variant} ({n_seeds} semillas)\n\n" + to_markdown(tab))
        (out / f"results_seeds_summary_{t}.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
        fig_components_seeds(agg, t, out)
        aggs[t] = agg
    print(f"\nListo. Resultados en {out}/ (results_seeds_summary_*.md, 06_*.png)")
    return aggs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-seeds", type=int, default=10)
    a = ap.parse_args()
    main(load_config(a.config), a.n_seeds)
