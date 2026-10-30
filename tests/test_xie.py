import gzip
import importlib
import tarfile

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

from src.xie import ENDO, MURAL, calibrate_threshold, marker_scores

GENES = ENDO + MURAL + ["PTPRC", "GFAP"] + [f"G{i}" for i in range(80)]


def _sample(rng, n_endo, n_mural):
    G = len(GENES)
    rows = []
    for k in range(n_endo + n_mural):
        lam = rng.gamma(0.5, 0.6, G)
        lam[: len(ENDO)] += 6.0 if k < n_endo else 0.0
        lam[len(ENDO): len(ENDO) + len(MURAL)] += 0.0 if k < n_endo else 6.0
        rows.append(rng.poisson(lam * 3))
    return sp.csr_matrix(np.array(rows).T)             # genes x células, como 10x


def _write_tar(tmp, gsm, sample, M):
    d = tmp / "pack" / sample
    d.mkdir(parents=True)
    scipy.io.mmwrite(str(d / "matrix.mtx"), M)
    with open(d / "matrix.mtx", "rb") as f, gzip.open(d / "matrix.mtx.gz", "wb") as g:
        g.write(f.read())
    (d / "matrix.mtx").unlink()
    pd.DataFrame({0: [f"ENSG{i}" for i in range(len(GENES))], 1: GENES, 2: "Gene Expression"}).to_csv(
        d / "features.tsv.gz", sep="\t", header=False, index=False)
    pd.DataFrame({0: [f"BC{i}-1" for i in range(M.shape[1])]}).to_csv(d / "barcodes.tsv.gz", sep="\t", header=False, index=False)
    out = tmp / "datos" / f"{gsm}_{sample}.tar.gz"
    out.parent.mkdir(exist_ok=True)
    with tarfile.open(out, "w:gz") as t:
        t.add(d, arcname=sample)


def test_prepare_xie_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    samples = pd.DataFrame([
        dict(gsm="GSM1", sample="T001", patient="P001", site="core", glioma_type="GBM", geo_patient=1),
        dict(gsm="GSM2", sample="N001", patient="P001", site="peripheral", glioma_type="GBM", geo_patient=1),
        dict(gsm="GSM3", sample="T002", patient="P002", site="core", glioma_type="LGG", geo_patient=2)])
    for r in samples.itertuples():
        _write_tar(tmp_path, r.gsm, r.sample, _sample(rng, 80, 40))
    csv = tmp_path / "s.csv"
    samples.to_csv(csv, index=False)
    mod = importlib.import_module("experiments.10_prepare_xie")
    path = mod.main(tmp_path / "datos", csv, tmp_path / "out", min_genes=5)
    a = ad.read_h5ad(path)
    assert {"Patient", "glioma_type", "nCount_RNA", "nFeature_RNA", "site"} <= set(a.obs.columns)
    assert dict(a.obs.groupby("Patient")["glioma_type"].first()) == {"P001": "GBM", "P002": "LGG"}
    # se quedan (casi) todas las endoteliales y (casi) ninguna mural: 80 de cada 120 por muestra
    assert 0.55 < len(a) / 360 < 0.75
    tot = np.asarray(np.expm1(a.X).sum(axis=1)).ravel()
    assert np.allclose(tot, 1e4, rtol=1e-3)                       # normalizado como Walchli


def test_calibrate_threshold_uses_reference_cells():
    rng = np.random.default_rng(1)
    X = sp.csr_matrix(rng.random((500, len(GENES))).astype(np.float32))
    thr = calibrate_threshold(X, GENES, n=300)
    s = marker_scores(X, GENES)["endo_score"].to_numpy()
    assert 0.0 < thr < s.max() and (s >= thr).mean() > 0.9
