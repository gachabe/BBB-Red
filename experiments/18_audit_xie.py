"""Auditoría de Xie a nivel de muestra: mapeo archivos-muestras-pacientes-diagnóstico-sitio, recuento de células, marcadores y contaminación."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.evaluation import to_markdown
from src.xie import ENDO, GLIA, IMMUNE, MURAL


def _mean_expr(X, genes, names):
    pos = {str(g).upper(): i for i, g in enumerate(genes)}
    idx = [pos[n] for n in names if n in pos]
    if not idx:
        return np.full(X.shape[0], np.nan)
    return np.asarray(X[:, idx].mean(axis=1)).ravel()


def main(xie_path, samples_csv, report_json, out, published_csv=None) -> pd.DataFrame:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(xie_path)
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    genes = np.asarray(a.var_names.astype(str))
    smp = pd.read_csv(samples_csv)
    rep = json.loads(Path(report_json).read_text(encoding="utf-8"))
    per = pd.DataFrame(rep["por_muestra"]).set_index("sample")
    pub = None
    if published_csv and Path(published_csv).exists():
        pub = pd.read_csv(published_csv)
    rows = []
    for r in smp.itertuples():
        m = (a.obs["sample"] == r.sample).to_numpy()
        row = dict(study_id="Xie", patient_id=r.patient, sample_id=r.sample, diagnosis=r.glioma_type, anatomical_site=r.site,
                   n_cells_before_qc=per["celulas_crudas"].get(r.sample, np.nan), n_cells_after_qc=per["tras_QC"].get(r.sample, np.nan),
                   n_endothelial_selected=int(m.sum()),
                   n_endothelial_published=int((pub["sample"] == r.sample).sum()) if pub is not None else np.nan,
                   median_total_counts=float(a.obs.loc[m, "nCount_RNA"].median()) if m.any() else np.nan,
                   median_detected_genes=float(a.obs.loc[m, "nFeature_RNA"].median()) if m.any() else np.nan)
        if m.any():
            Xs = X[m]
            for g in ENDO:
                j = np.flatnonzero(genes == g)
                row[f"mean_{g}"] = float(Xs[:, j[0]].mean()) if len(j) else np.nan
                row[f"frac_{g}_detected"] = float((Xs[:, j[0]] > 0).mean()) if len(j) else np.nan
            for name, gl in (("mural", MURAL), ("immune", IMMUNE), ("glia", GLIA)):
                row[f"mean_{name}_score"] = float(np.nanmean(_mean_expr(Xs, genes, gl)))
            row["mean_endo_score"] = float(np.nanmean(_mean_expr(Xs, genes, ENDO)))
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(out / "xie_audit.csv", index=False)
    (out / "xie_audit.md").write_text(to_markdown(t.round(3)) + "\n", encoding="utf-8")
    issues = []
    for p, g in smp.groupby("patient"):
        if g["glioma_type"].nunique() != 1:
            issues.append(f"el paciente {p} tiene más de un diagnóstico en el mapeo")
    missing = sorted(set(smp["sample"]) - set(per.index))
    if missing:
        issues.append(f"muestras del mapeo que no están en el informe de preparación: {missing}")
    extra = sorted(set(a.obs["sample"].astype(str)) - set(smp["sample"]))
    if extra:
        issues.append(f"muestras en los datos que no están en el mapeo: {extra}")
    no_endo = t.loc[t["n_endothelial_selected"] == 0, "sample_id"].tolist()
    if no_endo:
        issues.append(f"muestras sin células endoteliales seleccionadas: {no_endo}")
    print(t[["patient_id", "sample_id", "diagnosis", "anatomical_site", "n_cells_before_qc", "n_cells_after_qc", "n_endothelial_selected",
             "n_endothelial_published", "median_total_counts", "median_detected_genes"]].to_string(index=False))
    mk = [c for c in t.columns if c.startswith(("mean_PECAM1", "mean_CLDN5", "mean_VWF", "mean_FLT1", "mean_EGFL7", "mean_ESAM", "mean_mural", "mean_immune", "mean_glia", "mean_endo"))]
    print("\nMarcadores y contaminación (media de expresión log-normalizada de las células endoteliales seleccionadas):")
    print(t[["sample_id"] + mk].round(2).to_string(index=False))
    print("\nComprobaciones del mapeo: " + ("sin incidencias" if not issues else "; ".join(issues)))
    print("Anotación endotelial publicada: " + ("cargada" if pub is not None else "NO disponible (columna n_endothelial_published vacía)"))
    print("Recuento del artículo de Xie: no verificado aquí; la discrepancia con el recuento publicado no se puede explicar sin esa cifra.")
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xie", default="data/xie/xie_endothelial.h5ad")
    ap.add_argument("--samples", default="configs/xie_samples.csv")
    ap.add_argument("--report", default="data/xie/xie_prepare_report.json")
    ap.add_argument("--published", default=None, help="CSV opcional con columnas sample,cell de la anotación endotelial publicada")
    ap.add_argument("--out", default="results/pseudobulk_counts")
    a = ap.parse_args()
    main(a.xie, a.samples, a.report, a.out, a.published)
