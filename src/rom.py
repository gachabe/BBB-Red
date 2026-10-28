"""ROM: genes variables + escalado + PCA, ajustados SOLO con las células de entrenamiento.

Opcional (rom.regress_out): se quita por regresión lineal el efecto de la profundidad de secuenciación
(log1p de nCount_RNA y nFeature_RNA) de cada gen, con coeficientes ajustados solo con entrenamiento.

Escalera de representaciones (de menos a más dimensiones):
  PCA-r (r componentes) < HVG-N (N genes más variables) < All genes (todos, matriz dispersa)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.sparsefuncs import mean_variance_axis


def covariates(meta, cols):
    """Matriz (N, k) de log1p de las columnas de profundidad, o None si no se pide regresión."""
    if not cols:
        return None
    C = np.log1p(meta[list(cols)].to_numpy(dtype=np.float64))
    assert np.isfinite(C).all(), "covariables de profundidad con valores no finitos"
    return C


@dataclass
class Regressor:
    """Residuo de cada gen tras quitar su dependencia lineal de las covariables (ajustado con entrenamiento)."""
    mu: np.ndarray        # (G,) media por gen en entrenamiento
    c_mean: np.ndarray    # (k,)
    c_std: np.ndarray     # (k,)
    B: np.ndarray         # (k, G) coeficientes

    @classmethod
    def fit(cls, X, C):
        assert C.shape[0] == X.shape[0]
        mu = np.asarray(X.mean(axis=0), dtype=np.float64).ravel()
        c_mean, c_std = C.mean(axis=0), C.std(axis=0)
        assert (c_std > 0).all(), "una covariable de profundidad es constante en entrenamiento"
        Cc = (C - c_mean) / c_std
        CX = np.asarray(X.T @ Cc).T if sp.issparse(X) else Cc.T @ np.asarray(X)  # sin densificar X
        B = np.linalg.pinv(Cc.T @ Cc) @ CX
        return cls(mu.astype(np.float32), c_mean, c_std, B.astype(np.float32))

    def residual(self, X, C):
        Cc = ((C - self.c_mean) / self.c_std).astype(np.float32)
        out = X.toarray() if sp.issparse(X) else np.array(X, dtype=np.float32)
        out = out.astype(np.float32, copy=False)
        out -= self.mu
        out -= Cc @ self.B
        return out


@dataclass
class Representation:
    name: str                       # "PCA-10", "HVG-2000", "All genes"
    kind: str                       # "pca" | "hvg" | "all"
    n_features: int
    scaler: StandardScaler
    gene_idx: np.ndarray | None     # genes variables (None = todos)
    pca: PCA | None = None
    r: int | None = None
    clip: float | None = None
    explained_variance: float = float("nan")
    regress: Regressor | None = None

    def _scaled(self, X, C=None):
        Xs = X if self.kind == "all" else X[:, self.gene_idx]
        if self.regress is not None:
            assert C is not None, "falta la matriz de covariables de profundidad"
            Z = self.scaler.transform(self.regress.residual(Xs, C))  # densa
            if self.clip:
                np.clip(Z, -self.clip, self.clip, out=Z)
            return Z
        if self.kind == "all":
            Z = self.scaler.transform(Xs)  # dispersa; solo se divide por la desviación (sin centrar)
            if self.clip:
                np.minimum(Z.data, self.clip, out=Z.data)
            return Z
        Z = self.scaler.transform(Xs.toarray() if sp.issparse(Xs) else np.asarray(Xs))
        if self.clip:
            np.clip(Z, -self.clip, self.clip, out=Z)
        return Z

    def transform(self, X, C=None):
        Z = self._scaled(X, C)
        if self.kind != "pca":
            return Z
        out = (Z - self.pca.mean_) @ self.pca.components_[: self.r].T
        assert out.shape[1] == self.n_features
        return out


@dataclass
class FoldFit:
    fold: object                    # features.Fold
    reps: list
    var_ratio: np.ndarray | None    # explained_variance_ratio_ de la PCA (longitud = máx. de componentes)


def select_hvg(X_train, n: int) -> np.ndarray:
    """Los n genes de mayor varianza (valores log) en el entrenamiento. Índices ordenados."""
    _, var = mean_variance_axis(X_train, axis=0)
    order = np.argsort(-np.asarray(var, dtype=np.float64), kind="stable")[:n]
    return np.sort(order)


def fit_representations(X_train, rom_cfg: dict, C_train=None) -> list[Representation]:
    assert sp.issparse(X_train)
    n_genes = X_train.shape[1]
    n_hvg = int(rom_cfg["hvg"])
    comps = sorted(int(r) for r in rom_cfg["components"])
    clip = rom_cfg.get("scale_clip")
    use_reg = bool(rom_cfg.get("regress_out"))
    assert not use_reg or C_train is not None, "rom.regress_out pide covariables y no se pasaron"
    assert n_hvg <= n_genes, f"hvg={n_hvg} > genes={n_genes}"
    assert comps[-1] <= n_hvg and comps[-1] <= X_train.shape[0], "components no puede superar hvg ni nº de células"

    idx = select_hvg(X_train, n_hvg)
    Xh = X_train[:, idx]
    reg_h = Regressor.fit(Xh, C_train) if use_reg else None
    Xh = reg_h.residual(Xh, C_train) if use_reg else Xh.toarray()
    scaler = StandardScaler().fit(Xh)
    Z = scaler.transform(Xh)
    if clip:
        np.clip(Z, -clip, clip, out=Z)
    pca = PCA(n_components=comps[-1], svd_solver="full").fit(Z)  # PCA-r = primeras r componentes

    reps = [
        Representation(f"PCA-{r}", "pca", r, scaler, idx, pca, r, clip, float(pca.explained_variance_ratio_[:r].sum()), reg_h)
        for r in comps
    ]
    reps.append(Representation(f"HVG-{n_hvg}", "hvg", n_hvg, scaler, idx, None, None, clip, regress=reg_h))
    if rom_cfg.get("include_all_genes", True):
        if use_reg:
            reg_a = Regressor.fit(X_train, C_train)
            sc_all = StandardScaler().fit(reg_a.residual(X_train, C_train))
        else:
            reg_a, sc_all = None, StandardScaler(with_mean=False).fit(X_train)
        reps.append(Representation("All genes", "all", n_genes, sc_all, None, None, None, clip, regress=reg_a))
    return reps
