"""Ejecuta toda la pipeline (según el YAML) y escribe resultados + metadatos."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import json
import subprocess
import sys
from pathlib import Path

from src.data import load_config
from src.features import prepare


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _versions() -> dict:
    out = {}
    for p in ("numpy", "pandas", "scipy", "scikit-learn", "anndata", "matplotlib"):
        try:
            out[p] = md.version(p)
        except Exception:
            out[p] = "unknown"
    return out


def main(cfg: dict) -> dict:
    out = Path(cfg.get("results_dir", f"results/{cfg['mode']}"))
    out.mkdir(parents=True, exist_ok=True)
    ds, folds = prepare(cfg, out)
    eda = importlib.import_module("experiments.01_eda").run(ds, cfg, out)
    fits = importlib.import_module("experiments.02_rom").run(ds, folds, cfg, out)
    results = importlib.import_module("experiments.03_ml").run(ds, fits, cfg, out)

    summary = {
        "mode": cfg["mode"],
        "note": "El modo demo es un test de integridad, no una fuente de conclusiones." if cfg["mode"] == "demo" else "",
        "n_patients": len(folds),
        "n_cells": len(ds.meta),
        "n_genes": len(ds.genes),
        "results": {t: df.round(4).to_dict(orient="records") for t, df in results.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    meta = {
        "mode": cfg["mode"],
        "seed": cfg["seed"],
        "git_commit": _git_commit(),
        "python_version": sys.version.split()[0],
        "library_versions": _versions(),
        "dataset": cfg.get("dataset"),
        "data_file": Path(cfg["data"]["path"]).name,
        "tasks": cfg.get("tasks", ["glioma_type"]),
        "hvg": cfg["rom"]["hvg"],
        "components": cfg["rom"]["components"],
        "include_all_genes": cfg["rom"].get("include_all_genes", True),
        "scale_clip": cfg["rom"].get("scale_clip"),
        "regress_out": cfg["rom"].get("regress_out"),
        "max_train_cells_per_patient": cfg.get("ml", {}).get("max_train_cells_per_patient"),
        "models": cfg["models"],
        "control_permuted_labels": cfg.get("ml", {}).get("control", True),
        "patients": ds.patients.to_dict(orient="records"),
        "folds_held_out": [f.held_out for f in folds],
        "eda": eda,
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nListo. Resultados en {out}/")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(load_config(ap.parse_args().config))
