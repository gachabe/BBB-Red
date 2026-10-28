"""Validación del dataset (`make check`) y verificación de la pipeline (`make verify`)."""
from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np

from src.data import GROUP, REQUIRED_OBS, leakage_report, library_size_check, load_config, read_source, study_id_columns


def check_dataset(cfg: dict) -> bool:
    results: list[tuple[str, str, str]] = []

    def rec(ok, name, detail=""):
        results.append(("PASS" if ok else "FAIL", name, "" if ok else detail))

    def info(name, detail):
        results.append(("INFO", name, detail))

    dcfg = cfg["data"]
    X, obs, genes = read_source(dcfg)
    exp = dcfg.get("expected_shape")
    if exp:
        rec(tuple(X.shape) == tuple(exp), f"forma {tuple(exp)} (solo Walchli, sin Xie mezclado)", f"la forma real es {X.shape}")
    rec(all(c in obs.columns for c in REQUIRED_OBS), "columnas de obs esperadas", f"obs tiene {list(obs.columns)}")
    rec(obs.index.is_unique, "códigos de célula únicos")
    rec(bool(np.isfinite(X.data).all() and X.data.min() >= 0), "X finita y no negativa")
    sample = X.data[: min(len(X.data), 2_000_000)]
    rec(not np.all(sample == np.round(sample)), "X no son conteos enteros (está normalizada)")
    rec(bool((obs.groupby(GROUP)["glioma_type"].nunique() == 1).all()), "cada paciente tiene un único glioma_type")
    pt = obs.groupby(GROUP)["glioma_type"].first().value_counts()
    rec(len(pt) == 2 and pt.min() >= 2, "2 clases con >=2 pacientes cada una (necesario para dejar un paciente fuera)",
        f"pacientes por clase: {pt.to_dict()}")
    rec(bool((obs["PATH"] == obs["glioma_type"]).all()), "PATH idéntica a glioma_type (columna redundante)",
        f"{int((obs['PATH'] != obs['glioma_type']).sum())} células donde difieren")
    ids = study_id_columns(obs.columns)
    rec(not ids, "sin columnas de identificador de estudio/muestra en obs", f"{ids}")

    lib = library_size_check(X)
    info("normalización", f"suma de expm1(X) por célula: mediana {lib['median_expm1_rowsum']:.1f}, CV {lib['cv']:.4f}, "
         f"max(X) {lib['x_max']:.2f}; log1p con total fijo: {lib['consistent_with_log1p_fixed_total']}")
    rep = leakage_report(obs, "glioma_type")
    info("columnas que determinan glioma_type (no se usan como característica)",
         str(rep[rep["note"] == "DETERMINA el objetivo"]["column"].tolist()))

    print("\nValidación del dataset\n======================\n")
    for st, name, detail in results:
        print(f"[{st}] {name}" + (f"  -> {detail}" if detail else ""))
    ok = all(st != "FAIL" for st, _, _ in results)
    print("\nRESULT:", "DATASET OK" if ok else "DATASET INVALID")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--run-tests", action="store_true")
    a = ap.parse_args()
    if a.run_tests:
        print("bbb verification\n================\n")
        rc = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"]).returncode
        print("\nRESULT:", "PIPELINE READY" if rc == 0 else "PIPELINE NOT READY")
        sys.exit(rc)
    sys.exit(0 if check_dataset(load_config(a.config)) else 1)


if __name__ == "__main__":
    main()
