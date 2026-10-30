"""Prepara el dataset de Xie et al. (GSE242044, 10x por muestra) para aplicarle el paquete entrenado con Walchli.

Pasos: descomprimir .tar.gz -> leer matrices 10x -> unir -> QC mínimo -> normalizar (10.000 por célula + log1p, como
Walchli) -> quedarse con las células endoteliales (las muestras están enriquecidas en CD31+ y PDGFRB+).
"""
from __future__ import annotations

import tarfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

ENDO = ["PECAM1", "CLDN5", "VWF", "FLT1", "EGFL7", "ESAM"]
MURAL = ["PDGFRB", "RGS5", "ACTA2", "MYH11", "TAGLN", "NOTCH3"]
IMMUNE = ["PTPRC", "C1QA", "CD74", "LYZ"]
GLIA = ["GFAP", "SNAP25", "PLP1", "OLIG1"]
FALLBACK_THRESHOLD = 0.5


def extract_sample(folder: Path, gsm: str, sample: str, workdir: Path) -> Path:
    """Devuelve el directorio con matrix/features/barcodes de la muestra (descomprime si hace falta)."""
    dest = workdir / f"{gsm}_{sample}"
    if not dest.exists() or not any(dest.rglob("matrix.mtx*")):
        dest.mkdir(parents=True, exist_ok=True)
        arch = list(Path(folder).rglob(f"{gsm}_{sample}.tar.gz"))
        if arch:
            with tarfile.open(arch[0]) as t:
                t.extractall(dest, filter="data")
        else:                                   # ya descomprimida en la carpeta de origen
            src = [p.parent for p in Path(folder).rglob("matrix.mtx*") if gsm in str(p)]
            assert src, f"no encuentro {gsm}_{sample}.tar.gz ni una carpeta descomprimida en {folder}"
            return src[0]
    hits = [p.parent for p in dest.rglob("matrix.mtx*")]
    assert len(hits) == 1, f"{gsm}: se esperaba 1 matriz 10x y hay {len(hits)}"
    return hits[0]


def _find(d: Path, *names) -> Path:
    for n in names:
        for suffix in ("", ".gz"):
            p = d / (n + suffix)
            if p.exists():
                return p
    raise FileNotFoundError(f"falta {names} en {d}")


def read_10x(d: Path):
    """(X células x genes CSR enteros, nombres de gen, códigos de barras)."""
    M = scipy.io.mmread(_find(d, "matrix.mtx")).tocsr()          # genes x células
    feat = pd.read_csv(_find(d, "features.tsv", "genes.tsv"), sep="\t", header=None)
    genes = feat.iloc[:, 1 if feat.shape[1] > 1 else 0].astype(str).to_numpy()
    bars = pd.read_csv(_find(d, "barcodes.tsv"), sep="\t", header=None).iloc[:, 0].astype(str).to_numpy()
    assert M.shape == (len(genes), len(bars)), f"{d}: forma {M.shape} no encaja con {len(genes)} genes y {len(bars)} códigos"
    return M.T.tocsr().astype(np.float32), genes, bars


def build_counts(folder: Path, samples: pd.DataFrame, workdir: Path, log=print) -> ad.AnnData:
    parts, genes0 = [], None
    for r in samples.itertuples():
        X, genes, bars = read_10x(extract_sample(folder, r.gsm, r.sample, workdir))
        if genes0 is None:
            genes0 = genes
        assert len(genes) == len(genes0) and (genes == genes0).all(), f"{r.sample}: genes distintos a los de la primera muestra"
        obs = pd.DataFrame(dict(sample=r.sample, gsm=r.gsm, Patient=r.patient, site=r.site, glioma_type=r.glioma_type),
                           index=[f"{r.sample}_{b}" for b in bars])
        parts.append(ad.AnnData(X=X, obs=obs))
        log(f"  {r.sample}: {X.shape[0]} células")
    a = ad.concat(parts, join="outer")
    a.var_names = genes0
    a.var_names_make_unique()
    return a


def add_depth_and_normalize(a: ad.AnnData, min_genes: int = 200) -> ad.AnnData:
    """nCount/nFeature desde los conteos crudos; QC mínimo; normaliza a 10.000 por célula y log1p."""
    X = a.X.tocsr()
    a.obs["nCount_RNA"] = np.asarray(X.sum(axis=1)).ravel()
    a.obs["nFeature_RNA"] = np.asarray((X > 0).sum(axis=1)).ravel()
    keep = (a.obs["nFeature_RNA"] >= min_genes).to_numpy()
    a = a[keep].copy()
    X = a.X.tocsr()
    Xn = sp.diags(1e4 / a.obs["nCount_RNA"].to_numpy()).dot(X).tocsr()
    Xn.data = np.log1p(Xn.data)
    a.X = Xn.astype(np.float32)
    return a


def _mean_expr(X, genes, markers) -> np.ndarray:
    idx = [i for i, g in enumerate(pd.Index(genes).str.upper()) if g in set(markers)]
    if not idx:
        return np.zeros(X.shape[0])
    return np.asarray(X[:, idx].mean(axis=1)).ravel()


def marker_scores(X, genes) -> pd.DataFrame:
    endo = _mean_expr(X, genes, ENDO)
    other = np.max([_mean_expr(X, genes, m) for m in (MURAL, IMMUNE, GLIA)], axis=0)
    return pd.DataFrame(dict(endo_score=endo, other_score=other))


def calibrate_threshold(ref_X, ref_genes, n=20000, q=5, seed=0) -> float:
    """Umbral = percentil q de la puntuación endotelial en las células de Walchli (todas endoteliales)."""
    rows = np.random.default_rng(seed).choice(ref_X.shape[0], min(n, ref_X.shape[0]), replace=False)
    return float(np.percentile(marker_scores(ref_X[np.sort(rows)], ref_genes)["endo_score"], q))


def select_endothelial(a: ad.AnnData, threshold: float) -> ad.AnnData:
    s = marker_scores(a.X, a.var_names)
    a.obs["endo_score"], a.obs["other_score"] = s["endo_score"].to_numpy(), s["other_score"].to_numpy()
    mask = ((a.obs["endo_score"] >= threshold) & (a.obs["endo_score"] > a.obs["other_score"])).to_numpy()
    a.obs["is_endothelial"] = mask
    return a
