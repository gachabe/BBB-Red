"""Pseudobulks de conteos crudos (guía de los coautores): suma de conteos, TMM, log-CPM, PCA y regresión logística.

Todo lo que depende de los datos (filtro de genes, referencia TMM, escalado, HVG, PCA, clasificador) se ajusta SOLO con los
pacientes de entrenamiento. Los paquetes de un paciente nunca se reparten entre entrenamiento y prueba y las métricas se
calculan una vez por paciente (probabilidad media de sus paquetes). Clase positiva = GBM.
"""
from __future__ import annotations

import itertools
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

# --------------------------------------------------------------------------- conteos crudos


def reconstruct_counts(X, total, tol: float = 0.05, chunk: int = 10000):
    """Conteos crudos desde log1p(conteos / total * 1e4): conteos = expm1(X) * total / 1e4. Falla si no salen enteros."""
    X = sp.csr_matrix(X)
    total = np.asarray(total, dtype=np.float64)
    parts, worst = [], 0.0
    for s in range(0, X.shape[0], chunk):
        B = X[s: s + chunk].astype(np.float64)
        B.data = np.expm1(B.data)
        B = (sp.diags(total[s: s + chunk] / 1e4) @ B).tocsr()
        if len(B.data):
            worst = max(worst, float(np.abs(B.data - np.round(B.data)).max()))
        B.data = np.round(B.data)
        parts.append(B.astype(np.float32))
    assert worst < tol, f"los conteos reconstruidos no son enteros (desviación máxima {worst:.3f}); hacen falta los conteos crudos"
    C = sp.vstack(parts).tocsr()
    C.eliminate_zeros()
    return C, worst


def age_from_obs(obs: pd.DataFrame) -> np.ndarray:
    if "age" not in obs.columns:
        return np.full(len(obs), np.nan)
    return np.array([float(m.group()) if (m := re.search(r"\d+(\.\d+)?", str(v))) else np.nan for v in obs["age"]])


# --------------------------------------------------------------------------- paquetes


def make_packets(C, obs: pd.DataFrame, size, R: int, seed: int, unit_col: str, cols: dict, comp_col=None, min_cells: int = 100,
                 mode: str = "random", strata=None, umi=None, max_disjoint: int = 40):
    """Paquetes de células dentro de cada unidad (paciente o muestra). `cols` mapea nombre -> columna de obs.
    Modos (size significa cosas distintas):
      random   (por defecto) `size` células SIN reemplazo dentro del paquete; R paquetes por unidad que pueden solaparse. size='whole' = todas (mínimo min_cells).
      disjoint paquetes de `size` células SIN solaparse (hasta max_disjoint por unidad; mínimo 3).
      fraction `size` es la fracción (0-1) de las células de la unidad que entra en cada paquete (mínimo 30 células); R paquetes.
      umi      `size` es el nº de UMIs objetivo: se añaden células al azar hasta alcanzar ese total (iguala profundidad, no nº de células).
      balanced `size` células repartidas a partes iguales entre los estratos de `strata` (vector por célula); R paquetes.
    Devuelve (X, meta, no_factibles)."""
    obs = obs.reset_index(drop=True)
    uvals = obs[unit_col].astype(str).to_numpy()
    units = sorted(set(uvals))
    if mode in ("umi",):
        assert umi is not None, "el modo umi necesita el total de UMIs por célula"
        umi = np.asarray(umi, dtype=np.float64)
    if mode == "balanced":
        assert strata is not None, "el modo balanced necesita estratos"
        strata = np.asarray(strata).astype(str)
        levels = sorted(set(strata))
    if comp_col:
        cl = pd.Categorical(obs[comp_col].astype(str))
        onehot = sp.csr_matrix((np.ones(len(obs), dtype=np.float32), (np.arange(len(obs)), cl.codes)), shape=(len(obs), len(cl.categories)))
    skey = int(round(float(size) * 1000)) if mode in ("fraction",) else (int(size) if size != "whole" else 0)
    packs, metas, infeasible = [], [], []
    for ui, u in enumerate(units):
        idx = np.flatnonzero(uvals == u)
        rng = np.random.default_rng([int(seed), skey, ui])
        if size == "whole":
            if len(idx) < min_cells:
                infeasible.append((u, len(idx)))
                continue
            sets = [idx]
        elif mode == "random":
            if len(idx) < int(size):
                infeasible.append((u, len(idx)))
                continue
            sets = [np.sort(rng.choice(idx, size=int(size), replace=False)) for _ in range(R)]
        elif mode == "disjoint":
            k = len(idx) // int(size)
            if k < 3:
                infeasible.append((u, len(idx)))
                continue
            perm = rng.permutation(idx)
            sets = [np.sort(perm[i * int(size): (i + 1) * int(size)]) for i in range(min(k, max_disjoint))]
        elif mode == "fraction":
            n = int(np.ceil(float(size) * len(idx)))
            if n < 30:
                infeasible.append((u, len(idx)))
                continue
            sets = [np.sort(rng.choice(idx, size=n, replace=False)) for _ in range(R)]
        elif mode == "umi":
            if umi[idx].sum() < float(size):
                infeasible.append((u, len(idx)))
                continue
            sets = []
            for _ in range(R):
                perm = rng.permutation(idx)
                n = int(np.searchsorted(np.cumsum(umi[perm]), float(size))) + 1
                sets.append(np.sort(perm[:n]))
        elif mode == "balanced":
            per = int(size) // len(levels)
            by = {l: idx[strata[idx] == l] for l in levels}
            if sum(min(per, len(v)) for v in by.values()) < int(size) // 2:
                infeasible.append((u, len(idx)))
                continue
            sets = [np.sort(np.concatenate([rng.choice(v, size=min(per, len(v)), replace=False) for v in by.values() if len(v)])) for _ in range(R)]
        else:
            raise ValueError(f"modo desconocido: {mode}")
        for r, p in enumerate(sets):
            packs.append(p)
            m = {k: obs[c].iloc[idx[0]] for k, c in cols.items()}
            m.update(unit=u, size=size, replicate=r, seed=int(seed), n_cells=len(p))
            metas.append(m)
    assert packs, "ninguna unidad alcanza el tamaño de paquete"
    indptr = np.r_[0, np.cumsum([len(p) for p in packs])]
    G = sp.csr_matrix((np.ones(indptr[-1], dtype=np.float32), np.concatenate(packs), indptr), shape=(len(packs), len(obs)))
    X = (G @ C).tocsr().astype(np.float32)
    meta = pd.DataFrame(metas)
    meta["lib_size"] = np.asarray(X.sum(axis=1), dtype=np.float64).ravel()
    meta["detected_genes"] = np.asarray((X > 0).sum(axis=1)).ravel()
    if comp_col:
        comp = np.asarray((G @ onehot).todense()) / meta["n_cells"].to_numpy()[:, None]
        for j, name in enumerate(cl.categories):
            meta[f"comp_{name}"] = comp[:, j]
    return X, meta, infeasible


def thin_counts(X, target: float, seed: int = 0):
    """Adelgazamiento binomial: cada fila con más de `target` conteos se reduce al azar a ~target (iguala la profundidad). Devuelve denso float32."""
    X = np.asarray(X.todense() if sp.issparse(X) else X)
    rng = np.random.default_rng(seed)
    lib = X.sum(axis=1)
    out = np.empty_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        out[i] = X[i] if lib[i] <= target else rng.binomial(X[i].astype(np.int64), target / lib[i])
    return out


# --------------------------------------------------------------------------- normalización TMM + log-CPM


def _tmm_factor(obs, ref, lo, lr, trim=0.3, sum_trim=0.05) -> float:
    nz = (obs > 0) & (ref > 0)
    o, r = obs[nz], ref[nz]
    if len(o) < 10:
        return 1.0
    lR = np.log2((o / lo) / (r / lr))
    aE = 0.5 * (np.log2(o / lo) + np.log2(r / lr))
    v = np.maximum((lo - o) / lo / o + (lr - r) / lr / r, 1e-12)
    n = len(lR)
    loL, loS = int(np.floor(n * trim)) + 1, int(np.floor(n * sum_trim)) + 1
    rR, rE = rankdata(lR), rankdata(aE)
    k = (rR >= loL) & (rR <= n + 1 - loL) & (rE >= loS) & (rE <= n + 1 - loS)
    if not k.any():
        return 1.0
    return float(2.0 ** (np.sum(lR[k] / v[k]) / np.sum(1.0 / v[k])))


class TMM:
    """Filtro de genes (CPM > min_cpm en al menos min_frac de los pseudobulks de entrenamiento), factores TMM frente a una
    referencia de entrenamiento y log2-CPM (prior 2, como edgeR). Los factores de prueba se calculan frente a la MISMA referencia."""

    def __init__(self, min_cpm: float = 1.0, min_frac: float = 0.25, prior: float = 2.0):
        self.min_cpm, self.min_frac, self.prior = min_cpm, min_frac, prior

    def fit(self, C):
        C = np.asarray(C, dtype=np.float64)
        cpm = C / C.sum(axis=1, keepdims=True) * 1e6
        self.keep = (cpm > self.min_cpm).mean(axis=0) >= self.min_frac
        assert self.keep.sum() >= 10, "el filtro de genes deja menos de 10 genes"
        Ck = C[:, self.keep]
        lib = Ck.sum(axis=1)
        f75 = np.array([np.quantile(Ck[i] / lib[i], 0.75) for i in range(len(Ck))])
        ref = int(np.argmin(np.abs(f75 - f75.mean())))
        self.ref, self.ref_lib = Ck[ref], float(lib[ref])
        f = np.array([_tmm_factor(Ck[i], self.ref, lib[i], self.ref_lib) for i in range(len(Ck))])
        self.geo = float(np.exp(np.mean(np.log(f))))
        self.mean_lib = float(lib.mean())
        return self

    def transform(self, C):
        Ck = np.asarray(C, dtype=np.float64)[:, self.keep]
        lib = Ck.sum(axis=1)
        f = np.array([_tmm_factor(Ck[i], self.ref, lib[i], self.ref_lib) for i in range(len(Ck))]) / self.geo
        eff = lib * f
        pr = self.prior * eff / self.mean_lib
        return np.log2((Ck + pr[:, None]) / (eff + 2 * pr)[:, None] * 1e6).astype(np.float32)


class DenseReps:
    """Escalado de genes, HVG y PCA con medias/desviaciones de entrenamiento. Representaciones: PCA-r, HVG-n y 'All genes' (todos los filtrados)."""

    def __init__(self, components, n_hvg: int = 2000, clip: float = 10.0):
        self.components, self.n_hvg, self.clip = sorted(components), n_hvg, clip

    def fit(self, L):
        self.mu, self.sd = L.mean(axis=0), L.std(axis=0) + 1e-6
        S = self._scale(L)
        self.hvg = np.sort(np.argsort(-L.var(axis=0), kind="stable")[: min(self.n_hvg, L.shape[1])])
        self.comps = [c for c in self.components if c < min(L.shape[0], len(self.hvg))]
        self.pca = PCA(n_components=max(self.comps), svd_solver="full").fit(S[:, self.hvg]) if self.comps else None
        return self

    def _scale(self, L):
        return np.clip((L - self.mu) / self.sd, -self.clip, self.clip)

    def transform(self, L) -> dict:
        S = self._scale(L)
        out = {}
        if self.pca is not None:
            P = (S[:, self.hvg] - self.pca.mean_) @ self.pca.components_.T
            out.update({f"PCA-{r}": P[:, :r] for r in self.comps})
        out[f"HVG-{len(self.hvg)}"] = S[:, self.hvg]
        out["All genes"] = S
        return out


def fold_features(Ctr, Cte, rom: dict):
    """Normalización y representaciones ajustadas con entrenamiento; devuelve (Ztr, Zte) como dict por representación."""
    tmm = TMM(rom.get("min_cpm", 1.0), rom.get("min_frac", 0.25)).fit(Ctr)
    Ltr, Lte = tmm.transform(Ctr), tmm.transform(Cte)
    reps = DenseReps(rom["components"], rom.get("hvg", 2000), rom.get("scale_clip", 10.0)).fit(Ltr)
    return reps.transform(Ltr), reps.transform(Lte)


def fit_lr(Ztr, y, w, C: float = 1.0):
    assert len(set(y)) == 2, "el entrenamiento solo tiene una clase"
    return LogisticRegression(class_weight="balanced", solver="liblinear", C=C).fit(Ztr, y, sample_weight=w)  # penalización L2 (por defecto)


def fit_model(Ztr, y, w, model: str = "lr"):
    """lr = regresión logística (principal); rf = random forest (solo sensibilidad)."""
    if model == "lr":
        return fit_lr(Ztr, y, w)
    if model.startswith("lr:"):                         # lr:<C>, solo sensibilidad a la regularización
        return fit_lr(Ztr, y, w, C=float(model[3:]))
    if model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        assert len(set(y)) == 2, "el entrenamiento solo tiene una clase"
        return RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", min_samples_leaf=2, random_state=0, n_jobs=1).fit(Ztr, y, sample_weight=w)
    raise ValueError(model)


def packet_weights(patients) -> np.ndarray:
    """Peso 1/R_p: cada paciente suma 1 en el entrenamiento aunque tenga distinto número de paquetes."""
    s = pd.Series(patients)
    return (1.0 / s.groupby(s).transform("size")).to_numpy()


def _extra_block(E_tr, E_te):
    mu, sd = E_tr.mean(axis=0), E_tr.std(axis=0) + 1e-6
    return (E_tr - mu) / sd, (E_te - mu) / sd


def predict_block(Ctr, ytr, ptr, Cte, rom, mode="expr", Etr=None, Ete=None, model: str = "lr") -> dict:
    """P(clase positiva) de los paquetes de prueba por representación. mode: expr | extra (solo Etr/Ete) | both (expresión + Etr)."""
    w = packet_weights(ptr)
    if mode == "extra":
        a, b = _extra_block(Etr, Ete)
        return {"extra": fit_model(a, ytr, w, model).predict_proba(b)[:, 1]}
    Ztr, Zte = fold_features(Ctr, Cte, rom)
    out = {}
    for name in Ztr:
        A, B = Ztr[name], Zte[name]
        if mode == "both":
            ea, eb = _extra_block(Etr, Ete)
            A, B = np.hstack([A, ea]), np.hstack([B, eb])
        out[name] = fit_model(A, ytr, w, model).predict_proba(B)[:, 1]
    return out


def lopo(Xp, meta: pd.DataFrame, y, rom: dict, group: str = "patient", mode: str = "expr", extra=None, keep_cache: bool = False, model: str = "lr"):
    """Dejar un grupo (paciente) fuera. y: 1 = clase positiva. Devuelve la tabla por paquete (y la caché de características)."""
    C = np.asarray(Xp.todense(), dtype=np.float64) if sp.issparse(Xp) else np.asarray(Xp, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    g = meta[group].astype(str).to_numpy()
    rows, cache = [], []
    for q in sorted(set(g)):
        te, tr = g == q, g != q
        assert q not in set(g[tr])
        Etr, Ete = (None, None) if extra is None else (extra[tr], extra[te])
        if keep_cache and mode == "expr":
            Ztr, Zte = fold_features(C[tr], C[te], rom)
            cache.append(dict(held=q, tr_group=g[tr], Ztr=Ztr, Zte=Zte, te_idx=np.flatnonzero(te)))
            w = packet_weights(g[tr])
            pr = {n: fit_lr(Ztr[n], y[tr], w).predict_proba(Zte[n])[:, 1] for n in Ztr}
        else:
            pr = predict_block(C[tr], y[tr], g[tr], C[te], rom, mode, Etr, Ete, model)
        for name, p in pr.items():
            rows.append(pd.DataFrame(dict(packet=np.flatnonzero(te), group=q, y=y[te], representation=name, p=p)))
    df = pd.concat(rows, ignore_index=True)
    return (df, cache) if keep_cache else df


# --------------------------------------------------------------------------- agregación por paciente y métricas


def trimmed_mean(v, trim=0.1, min_n=20):
    v = np.sort(np.asarray(v, dtype=float))
    if len(v) < min_n:
        return float(v.mean())
    k = int(np.floor(trim * len(v)))
    return float(v[k: len(v) - k].mean())


def to_patient(df: pd.DataFrame, keys=("group", "representation")) -> pd.DataFrame:
    """Una fila por paciente: probabilidad media (principal), mediana y media recortada; y dispersión entre sus paquetes."""
    g = df.groupby(list(keys), sort=True)["p"]
    out = g.agg(p_mean="mean", p_median="median", p_sd="std", p_min="min", p_max="max", n_packets="size").reset_index()
    out["p_trim"] = g.apply(trimmed_mean).to_numpy()
    out["p_q05"] = g.quantile(0.05).to_numpy()
    out["p_q95"] = g.quantile(0.95).to_numpy()
    out["p_iqr"] = (g.quantile(0.75) - g.quantile(0.25)).to_numpy()
    out["frac_packets_gbm"] = g.apply(lambda v: float((v >= 0.5).mean())).to_numpy()
    y = df.groupby(list(keys))["y"].first().to_numpy()
    out["y"] = y
    return out


def patient_metrics(y, p, thr: float = 0.5) -> dict:
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    f1p = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f1n = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
    both = len(set(y)) == 2
    return dict(macro_f1=(f1p + f1n) / 2, balanced_accuracy=np.nanmean([sens, spec]), sensitivity=sens, specificity=spec,
                auroc=float(roc_auc_score(y, p)) if both else np.nan, auprc=float(average_precision_score(y, p)) if both else np.nan,
                tp=tp, fn=fn, fp=fp, tn=tn, n_patients=len(y))


def bootstrap_patients(y, p, B: int = 5000, seed: int = 0) -> dict:
    """Intervalo al 95 % remuestreando pacientes con reemplazo y manteniendo cuántos hay de cada clase."""
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    i0, i1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    keys = ("macro_f1", "balanced_accuracy", "auroc")
    boots = np.empty((B, len(keys)))
    for b in range(B):
        idx = np.concatenate([rng.choice(i0, len(i0)), rng.choice(i1, len(i1))])
        m = patient_metrics(y[idx], p[idx])
        boots[b] = [m[k] for k in keys]
    point = patient_metrics(y, p)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return {k: (point[k], lo[j], hi[j]) for j, k in enumerate(keys)}


def icc1(values, groups) -> float:
    """Correlación intraclase (ANOVA de un factor) de las probabilidades de los paquetes anidados en pacientes."""
    d = pd.DataFrame(dict(v=np.asarray(values, float), g=np.asarray(groups)))
    k = d["g"].nunique()
    n = len(d)
    gm = d.groupby("g")["v"].agg(["mean", "size"])
    ssb = float((gm["size"] * (gm["mean"] - d["v"].mean()) ** 2).sum())
    ssw = float(((d["v"] - d["g"].map(gm["mean"])) ** 2).sum())
    msb, msw = ssb / (k - 1), ssw / (n - k)
    n0 = (n - (gm["size"] ** 2).sum() / n) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw)) if msb + (n0 - 1) * msw > 0 else np.nan


# --------------------------------------------------------------------------- permutación exacta entre pacientes


def _perm_chunk(cache, patients, gbm_sets, reps):
    """Para cada asignación (conjunto de pacientes GBM) devuelve el macro-F1 por paciente y representación."""
    out = np.zeros((len(gbm_sets), len(reps)))
    for a, gset in enumerate(gbm_sets):
        lab = {p: int(p in gset) for p in patients}
        probs = {r: [] for r in reps}
        ytrue = np.array([lab[f["held"]] for f in cache])
        for f in cache:
            ytr = np.array([lab[x] for x in f["tr_group"]])
            w = packet_weights(f["tr_group"])
            for r in reps:
                probs[r].append(float(fit_lr(f["Ztr"][r], ytr, w).predict_proba(f["Zte"][r])[:, 1].mean()))
        for j, r in enumerate(reps):
            out[a, j] = patient_metrics(ytrue, np.array(probs[r]))["macro_f1"]
    return out


def exact_permutation(cache, y_by_patient: dict, reps, n_jobs: int = 1, chunk: int = 50) -> pd.DataFrame:
    """Todas las asignaciones posibles de etiquetas entre pacientes (mismo nº de GBM). El p ajustado usa el máximo entre representaciones."""
    from joblib import Parallel, delayed
    patients = sorted(y_by_patient)
    n_gbm = sum(y_by_patient.values())
    sets = [frozenset(c) for c in itertools.combinations(patients, n_gbm)]
    real = frozenset(p for p, v in y_by_patient.items() if v == 1)
    parts = Parallel(n_jobs=n_jobs)(delayed(_perm_chunk)(cache, patients, sets[i: i + chunk], reps) for i in range(0, len(sets), chunk))
    stat = np.vstack(parts)
    obs = stat[sets.index(real)]
    mx = stat.max(axis=1)
    return pd.DataFrame(dict(representation=reps, macro_f1_observed=obs, n_assignments=len(sets),
                             p_exact=[(stat[:, j] >= obs[j] - 1e-12).mean() for j in range(len(reps))],
                             p_adjusted_maxT=[(mx >= obs[j] - 1e-12).mean() for j in range(len(reps))]))
