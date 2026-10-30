"""Une las 13 muestras 10x de Xie (GSE242044), normaliza como Walchli y se queda con el endotelio.

Salida: data/xie/xie_endothelial.h5ad (+ xie_prepare_report.json). Después: experiments.08_external.
Umbral endotelial: percentil 5 de la puntuación endotelial en Walchli (si se da --reference-config); si no, 0,5.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data import load_config, read_source
from src.xie import (FALLBACK_THRESHOLD, add_depth_and_normalize, build_counts, calibrate_threshold,
                     select_endothelial)


def main(folder, samples_csv, out, reference_config=None, min_genes=200, sites=None) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(samples_csv)
    if sites:
        samples = samples[samples["site"].isin(sites)]
    print(f"{len(samples)} muestras, {samples['patient'].nunique()} pacientes")
    a = build_counts(Path(folder), samples, out / "extracted")
    n_raw = a.obs["sample"].value_counts().to_dict()
    a = add_depth_and_normalize(a, min_genes)
    n_qc = a.obs["sample"].value_counts().to_dict()
    if reference_config:
        Xr, _, gr = read_source(load_config(reference_config)["data"])
        thr, how = calibrate_threshold(Xr, gr), "percentil 5 en Walchli"
    else:
        thr, how = FALLBACK_THRESHOLD, "fijo (sin referencia)"
    a = select_endothelial(a, thr)
    e = a[a.obs["is_endothelial"].to_numpy()].copy()
    per = pd.DataFrame(dict(celulas_crudas=pd.Series(n_raw), tras_QC=pd.Series(n_qc),
                            endoteliales=e.obs["sample"].value_counts())).fillna(0).astype(int)
    per["frac_endotelial"] = (per["endoteliales"] / per["tras_QC"]).round(3)
    print("\n" + per.to_string())
    e.write_h5ad(out / "xie_endothelial.h5ad")
    rep = dict(umbral_endotelial=thr, umbral_origen=how, min_genes=min_genes, sitios=sorted(samples["site"].unique()),
               por_muestra=per.reset_index(names="sample").to_dict("records"),
               pacientes=e.obs.groupby("Patient")["glioma_type"].first().to_dict(),
               celulas_por_paciente=e.obs["Patient"].value_counts().to_dict(),
               mediana_nCount_RNA=float(e.obs["nCount_RNA"].median()), mediana_nFeature_RNA=float(e.obs["nFeature_RNA"].median()))
    (out / "xie_prepare_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nUmbral endotelial {thr:.3f} ({how}). Guardado {out / 'xie_endothelial.h5ad'}")
    return out / "xie_endothelial.h5ad"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="carpeta con los .tar.gz")
    ap.add_argument("--samples", default="configs/xie_samples.csv")
    ap.add_argument("--out", default="data/xie")
    ap.add_argument("--reference-config", default=None, help="p. ej. configs/full.yaml (calibra el umbral con Walchli)")
    ap.add_argument("--min-genes", type=int, default=200)
    ap.add_argument("--sites", nargs="*", default=None, help="core y/o peripheral (por defecto, ambos)")
    a = ap.parse_args()
    main(a.folder, a.samples, a.out, a.reference_config, a.min_genes, a.sites)
