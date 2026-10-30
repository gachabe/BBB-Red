"""Entrenamiento final con TODOS los pacientes. Guarda models/<modo>/bundle.joblib para aplicarlo a otro estudio."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data import build_dataset, load_config
from src.final_model import save_bundle, train_bundle


def main(cfg: dict) -> Path:
    out = Path(cfg.get("models_dir", f"models/{cfg['mode']}"))
    ds = build_dataset(cfg)
    print(f"{len(ds.patients)} pacientes, {ds.X.shape[0]} células, {ds.X.shape[1]} genes")
    bundle = train_bundle(ds, cfg)
    path = out / "bundle.joblib"
    save_bundle(bundle, path)
    meta = {k: bundle[k] for k in ("classes", "regress_out", "seed", "cap", "patients", "n_train_cells", "mode", "versions")}
    meta["models"] = sorted(f"{r} / {m}" for r, m in bundle["models"])
    (out / "bundle_info.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado {path} ({path.stat().st_size / 1e6:.0f} MB)")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(load_config(ap.parse_args().config))
