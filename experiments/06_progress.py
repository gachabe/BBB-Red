"""Acierto según el avance de la transición y según la severidad (un split, semilla del YAML)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.data import SCENARIOS, SEVERITIES, build_dataset, load_config
from src.evaluation import fig_progress, to_markdown
from src.features import rows, split_realizations
from src.models import make_model
from src.rom import fit_representation


def _m(y, p) -> dict:
    return {
        "macro_f1": float(f1_score(y, p, labels=list(SCENARIOS), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "n_rows": len(y),
    }


def main(cfg: dict, bins: int = 5) -> None:
    out = Path(cfg.get("results_dir", f"results/{cfg['mode']}"))
    out.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    ds = build_dataset(cfg)
    split = split_realizations(ds.realizations, cfg.get("split", {}).get("train_fraction", 0.8), seed)
    X_fit, _ = rows(ds, split.train)
    X_tr, y_tr = rows(ds, split.train, [cfg.get("ml", {}).get("train_phase", "developed")])
    cap = cfg.get("ml", {}).get("max_train_rows")
    if cap and len(X_tr) > cap:
        keep = np.random.default_rng(seed).choice(len(X_tr), cap, replace=False)
        X_tr, y_tr = X_tr[keep], y_tr[keep]

    te = ds.meta["realization"].isin(split.test).to_numpy().copy()
    Xte, mte = ds.X[te], ds.meta[te].reset_index(drop=True)
    y = mte["scenario"].to_numpy()
    phase, sev = mte["phase"].to_numpy(), mte["severity"].to_numpy()
    prog = (mte["days_since_start"] / mte["transition_end"]).to_numpy()
    bin_id = np.clip((prog * bins).astype(int), 0, bins - 1)

    reps = [fit_representation(X_fit, None)] + [fit_representation(X_fit, r) for r in cfg["rom"]["components"]]
    by_bin, by_sev = [], []
    for model in cfg["models"]:
        for rep in reps:
            clf = make_model(model, seed).fit(rep.transform(X_tr), y_tr)
            pred = clf.predict(rep.transform(Xte))
            label = "Raw" if rep.pca is None else f"PCA-{rep.n_components}"
            for b in range(bins):
                s = (phase == "transition") & (bin_id == b)
                by_bin.append(dict(model=model, representation=label, components=rep.n_components,
                                   progress_pct=f"{100 * b // bins}-{100 * (b + 1) // bins}", bin=b, **_m(y[s], pred[s])))
            for ph in ("developed", "transition"):
                for sv in SEVERITIES:
                    s = (phase == ph) & (sev == sv)
                    by_sev.append(dict(model=model, representation=label, components=rep.n_components,
                                       phase=ph, severity=sv, **_m(y[s], pred[s])))
    b_df, s_df = pd.DataFrame(by_bin), pd.DataFrame(by_sev)
    b_df.to_csv(out / "progress_by_bin.csv", index=False)
    s_df.to_csv(out / "progress_by_severity.csv", index=False)

    parts = []
    for model, g in b_df.groupby("model", sort=False):
        t = g.pivot(index="representation", columns="progress_pct", values="balanced_accuracy").reset_index()
        parts.append(f"### {model} — balanced accuracy por tramo de transición (% del avance)\n\n" + to_markdown(t))
    for (model, ph), g in s_df.groupby(["model", "phase"], sort=False):
        t = g.pivot(index="representation", columns="severity", values="balanced_accuracy").reset_index()
        parts.append(f"### {model} — {ph}: balanced accuracy por severidad\n\n" + to_markdown(t))
    (out / "progress_summary.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")

    fig_progress(b_df, out)
    print(f"\nListo. Resultados en {out}/ (progress_summary.md, 07_progress.png)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--bins", type=int, default=5)
    a = ap.parse_args()
    main(load_config(a.config), a.bins)
