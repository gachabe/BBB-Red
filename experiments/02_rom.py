"""ROM: genes variables + escalado + PCA, ajustados en cada ronda solo con los pacientes de entrenamiento."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data import GROUP, load_config
from src.evaluation import fig_pca_variance
from src.rom import FoldFit, covariates, fit_representations


def run(ds, folds, cfg: dict, outdir: Path, plot: bool = True) -> list[FoldFit]:
    assert ds.X.shape[1] == len(ds.genes)  # las características son solo X
    C = covariates(ds.meta, cfg["rom"].get("regress_out"))
    fits = []
    for f in tqdm(folds, desc="ROM (ajuste por ronda)"):
        assert (ds.meta[GROUP].iloc[f.train_idx] != f.held_out).all()
        reps = fit_representations(ds.X[f.train_idx], cfg["rom"], None if C is None else C[f.train_idx])
        pca = next((r.pca for r in reps if r.kind == "pca"), None)
        fits.append(FoldFit(f, reps, None if pca is None else pca.explained_variance_ratio_))
    if plot and fits[0].var_ratio is not None:
        var = pd.DataFrame(
            [dict(held_out=ff.fold.held_out, component=k + 1, ratio=float(v))
             for ff in fits for k, v in enumerate(ff.var_ratio)]
        )
        var.to_csv(Path(outdir) / "pca_variance.csv", index=False)
        fig_pca_variance(var, cfg["rom"]["components"], Path(outdir))
    return fits


if __name__ == "__main__":
    from src.features import prepare

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    out = Path(cfg.get("results_dir", f"results/{cfg['mode']}"))
    out.mkdir(parents=True, exist_ok=True)
    ds, folds = prepare(cfg, out)
    run(ds, folds, cfg, out)
