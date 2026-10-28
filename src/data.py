"""Carga del .h5ad (Walchli et al. 2024, células endoteliales de glioma) y construcción del Dataset.

Las características son SOLO la matriz X (genes). Las columnas de obs sirven para etiquetas,
agrupación por paciente y descripción; nunca entran como característica.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

GROUP = "Patient"        # unidad independiente de muestreo
CELLTYPE = "ECclusters"  # subtipo de célula endotelial (descriptivo, no es característica)
BATCH_COL_HINTS = ("study", "dataset", "batch", "sample", "library", "orig", "source", "run")
REQUIRED_OBS = ("ECclusters", "PATH", "Patient", "sex", "age", "glioma_type", "nCount_RNA", "nFeature_RNA")
STRING_OBS = ("ECclusters", "PATH", "Patient", "sex", "age", "glioma_type")


def load_config(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- lectura (+ caché)


def _file_key(path: Path) -> dict:
    st = path.stat()
    return {"name": path.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def read_source(dcfg: dict):
    """Devuelve (X csr float32, obs, genes). Con data.cache=true guarda una copia rápida en data/processed."""
    path = Path(dcfg["path"])
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Revisa data.path en el YAML.")
    cache = Path(dcfg.get("cache_dir", "data/processed")) / path.stem if dcfg.get("cache", True) else None
    key = _file_key(path)

    if cache is not None and (cache / "source.json").exists():
        if json.loads((cache / "source.json").read_text(encoding="utf-8")) == key:
            X = sp.load_npz(cache / "X.npz").tocsr()
            obs = pd.read_csv(cache / "obs.csv", index_col=0, keep_default_na=False)
            obs.index = obs.index.astype(str)
            for c in STRING_OBS:
                obs[c] = obs[c].astype(str)
            genes = np.array((cache / "genes.txt").read_text(encoding="utf-8").split("\n"))
            return X, obs, genes

    import anndata as ad

    adata = ad.read_h5ad(path)
    X = adata.X
    X = X.astype(np.float32).tocsr() if sp.issparse(X) else sp.csr_matrix(np.asarray(X, dtype=np.float32))
    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)
    for c in STRING_OBS:
        if c in obs.columns:
            obs[c] = obs[c].astype(str)
    genes = np.asarray(adata.var_names.astype(str))
    del adata

    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        sp.save_npz(cache / "X.npz", X, compressed=False)
        obs.to_csv(cache / "obs.csv")
        (cache / "genes.txt").write_text("\n".join(genes), encoding="utf-8")
        (cache / "source.json").write_text(json.dumps(key), encoding="utf-8")  # al final: caché completa
    return X, obs, genes


# --------------------------------------------------------------------------- comprobaciones


def check_integrity(X, obs: pd.DataFrame, genes) -> None:
    assert X.shape == (len(obs), len(genes)), f"X {X.shape} no cuadra con obs ({len(obs)}) y genes ({len(genes)})"
    missing = [c for c in REQUIRED_OBS if c not in obs.columns]
    assert not missing, f"Faltan columnas en obs: {missing}"
    assert obs.index.is_unique, "Hay códigos de célula duplicados"
    assert obs[GROUP].notna().all(), "Hay células sin paciente"
    assert X.nnz > 0 and np.isfinite(X.data).all(), "X vacía o con valores no finitos"
    assert X.data.min() >= 0, "X tiene valores negativos"
    sample = X.data[: min(len(X.data), 2_000_000)]
    assert not np.all(sample == np.round(sample)), (
        "X parece contener conteos enteros; esta pipeline asume datos normalizados (log)."
    )
    n_types = obs.groupby(GROUP)["glioma_type"].nunique()
    assert (n_types == 1).all(), f"Pacientes con más de un glioma_type: {list(n_types[n_types > 1].index)}"


def library_size_check(X, n: int = 500, seed: int = 0) -> dict:
    """Si X = log1p(conteos normalizados a un total fijo), expm1(X) suma lo mismo en cada célula."""
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(X.shape[0], size=min(n, X.shape[0]), replace=False))
    sums = np.asarray(X[rows].expm1().sum(axis=1)).ravel()
    cv = float(sums.std() / sums.mean())
    return {
        "median_expm1_rowsum": float(np.median(sums)),
        "min": float(sums.min()),
        "max": float(sums.max()),
        "cv": cv,
        "consistent_with_log1p_fixed_total": bool(cv < 0.01),
        "x_min": float(X.data.min()),
        "x_max": float(X.data.max()),
    }


def leakage_report(obs: pd.DataFrame, target: str, max_levels: int = 100) -> pd.DataFrame:
    """Para cada columna de obs: ¿determina por sí sola el objetivo? purity = 1.0 significa que sí."""
    rows = []
    for c in obs.columns:
        if c == target:
            continue
        n = obs[c].nunique()
        if n > max_levels:
            rows.append(dict(column=c, n_levels=n, purity=np.nan, note="continua: no evaluable"))
            continue
        ct = pd.crosstab(obs[c], obs[target])
        purity = float(ct.max(axis=1).sum() / ct.to_numpy().sum())
        rows.append(dict(column=c, n_levels=n, purity=purity, note="DETERMINA el objetivo" if purity == 1.0 else ""))
    return pd.DataFrame(rows).sort_values("purity", ascending=False, na_position="last").reset_index(drop=True)


# --------------------------------------------------------------------------- submuestreo (demo/mini)


def select_subset(obs: pd.DataFrame, dcfg: dict, seed: int) -> np.ndarray:
    """Posiciones de fila a conservar (solo demo/mini): n pacientes por tipo de glioma y tope de células."""
    rng = np.random.default_rng(seed)
    pat = obs[GROUP].to_numpy()
    patients = sorted(set(pat))
    ppc = dcfg.get("patients_per_class")
    if ppc:
        ptype = obs.groupby(GROUP)["glioma_type"].first()
        keep = []
        for cls in sorted(ptype.unique()):
            ids = sorted(ptype[ptype == cls].index)
            keep += rng.choice(ids, size=min(int(ppc), len(ids)), replace=False).tolist()
        patients = sorted(keep)
    cap = dcfg.get("max_cells_per_patient")
    sel = []
    for p in patients:
        idx = np.flatnonzero(pat == p)
        if cap and len(idx) > cap:
            idx = rng.choice(idx, size=int(cap), replace=False)
        sel.append(idx)
    return np.sort(np.concatenate(sel))


# --------------------------------------------------------------------------- Dataset


@dataclass
class Dataset:
    X: sp.csr_matrix          # (N, G) float32, log-normalizada; filas = células
    meta: pd.DataFrame        # N filas alineadas con X: cell + columnas de obs
    patients: pd.DataFrame    # una fila por paciente: Patient, n_cells, una columna por tarea
    genes: np.ndarray         # (G,) nombres de gen


def patient_table(meta: pd.DataFrame, tasks) -> pd.DataFrame:
    g = meta.groupby(GROUP, sort=True)
    t = g.size().rename("n_cells").to_frame()
    for task in tasks:
        n = g[task].nunique()
        t[task] = np.where(n == 1, g[task].first(), "mixed")
    return t.reset_index()


def build_dataset(cfg: dict) -> Dataset:
    dcfg = cfg["data"]
    tasks = list(cfg.get("tasks", ["glioma_type"]))
    X, obs, genes = read_source(dcfg)
    check_integrity(X, obs, genes)
    missing = [t for t in tasks if t not in obs.columns]
    assert not missing, f"Tareas sin columna en obs: {missing}"
    exp = dcfg.get("expected_shape")
    if exp and not dcfg.get("patients_per_class") and not dcfg.get("max_cells_per_patient"):
        assert tuple(X.shape) == tuple(exp), f"Forma {X.shape} distinta de la esperada {tuple(exp)}"
    pos = select_subset(obs, dcfg, cfg["seed"])
    X = X[pos]
    meta = obs.iloc[pos].rename_axis("cell").reset_index()
    assert len(meta) == X.shape[0] and X.shape[1] == len(genes)
    return Dataset(X=X, meta=meta, patients=patient_table(meta, tasks), genes=np.asarray(genes))


def study_id_columns(obs_columns) -> list[str]:
    """Columnas de obs cuyo nombre sugiere identificador de estudio/muestra (indicio de dos datasets mezclados)."""
    return [c for c in obs_columns if any(h in str(c).lower() for h in BATCH_COL_HINTS)]
