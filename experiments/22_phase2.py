"""Fase 2 de los pseudobulks de conteos: robustez del resultado `core` de Xie y análisis de la guía que faltaban.

Todo es reanudable (cada resultado se guarda al terminar; relanzar continúa donde se quedó) y cada etapa falla sin tumbar las demás
(los errores van a phase2_errors.log). Salida: results/pseudobulk_phase2 + phase2_summary.md con un veredicto automático.

Etapas (en este orden, de más barata a más cara):
  holm        Holm entre las permutaciones de la fase 1.
  age_prior   Qué predeciría un modelo solo de edad (Wälchli) con las edades medias de Xie publicadas (LGG 49,3; GBM 61,3).
  converge    Cuántos paquetes hacen falta (convergencia de réplicas) y componentes de varianza (ANOVA), sobre las ejecuciones de la fase 1.
  periphery   Diagnóstico del umbral en la periferia: desplazamiento núcleo-periferia, AUROC y centrado sin etiquetas.
  variants    Sensibilidad del resultado: random forest, paquetes disjuntos, por fracción, igualados por UMI, adelgazados a igual profundidad,
              solo genes comunes, equilibrados por estado endotelial (interno), y sin los genes que más distinguen los estudios (con
              refinamiento automático si el resultado se rompe).
  state       Paquetes específicos de cada estado endotelial (ECclusters), solo dentro de Wälchli.
  reverse     Transferencia inversa (entrenar con Xie, predecir Wälchli) y señal dentro de Xie.
  training    Remuestreo del entrenamiento (100 subconjuntos de 10 pacientes de Wälchli) y quitar un paciente de entrenamiento cada vez.
  followups   Pruebas condicionadas a lo anterior (paquetes mayores si una variante rompe el resultado).
  summary     Tablas, figuras y veredicto.
"""
from __future__ import annotations

import argparse
import importlib
import math
import os
import time
import traceback
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from scipy.stats import ttest_ind

from src.data import load_config
from src.evaluation import describe_figure, to_markdown
from src.final_model import align_genes
from src.xie import ENDO, GLIA, IMMUNE, MURAL
from src.pbcount import lopo, make_packets, patient_metrics, predict_block, thin_counts, to_patient

warnings.filterwarnings("ignore")
M19 = importlib.import_module("experiments.19_pseudobulk_counts")
COLS_W, COLS_X = M19.COLS_W, M19.COLS_X
REPS_MAIN = ["PCA-2", "PCA-5", "HVG-2000"]
OK_F1 = 0.9
PRIMARY, PERIPHERY_SIZES = 100, (100, 250, 500)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _dense(X):
    return np.asarray(X.todense(), dtype=np.float64) if sp.issparse(X) else np.asarray(X, dtype=np.float64)


# --------------------------------------------------------------------------- una variante = un conjunto de decisiones

_MASKS: dict = {}


def xie_mask(S, kind: str) -> np.ndarray:
    """Filtros de células de Xie más estrictos que los de la preparación original (QC mínimo de 200 genes, sin filtro mitocondrial ni de dobletes):
      qc_paper  <10 % de conteos mitocondriales (criterio del artículo; los 200 genes ya se aplicaron);
      pure      qc_paper + al menos 2 de los marcadores endoteliales detectados y puntuación endotelial > puntuaciones mural, inmune y glial."""
    key = (id(S[3]), kind)
    if key in _MASKS:
        return _MASKS[key]
    Cx, ox, gx = S[3], S[4], np.char.upper(np.asarray(S[5]).astype(str))
    tot = np.asarray(Cx.sum(axis=1)).ravel()
    mt = np.flatnonzero(np.char.startswith(gx, "MT-"))
    pct_mt = np.asarray(Cx[:, mt].sum(axis=1)).ravel() / np.maximum(tot, 1) if len(mt) else np.zeros(Cx.shape[0])
    keep = pct_mt < 0.10
    if kind == "pure":
        pos = {g: i for i, g in enumerate(gx)}

        def score(names):
            idx = [pos[n] for n in names if n in pos]
            if not idx:
                return np.zeros(Cx.shape[0]), np.zeros(Cx.shape[0])
            sub = Cx[:, idx].tocsr()
            L = sub.multiply(1e4 / np.maximum(tot, 1)[:, None]).tocsr()
            L.data = np.log1p(L.data)
            return np.asarray(L.mean(axis=1)).ravel(), np.asarray((sub > 0).sum(axis=1)).ravel()

        e, n_e = score(ENDO)
        others = np.max([score(g)[0] for g in (MURAL, IMMUNE, GLIA)], axis=0)
        keep &= (n_e >= 2) & (e > others)
    _MASKS[key] = keep
    return keep



def build(S, spec: dict, seed: int, need_ext: bool = True):
    """Paquetes de ambos estudios con las decisiones de `spec`; devuelve matrices densas alineadas a los genes de Wälchli."""
    Cw, ow, gw, Cx, ox, gx = S
    if spec.get("xie_filter"):
        km = xie_mask(S, spec["xie_filter"])
        Cx, ox = Cx[km], ox[km].reset_index(drop=True)
    size, mode, R = spec["size"], spec.get("mode", "random"), spec.get("R", 20)
    kw = dict(mode=mode)
    if mode == "umi":
        kw_w, kw_x = dict(kw, umi=ow["nCount_RNA"].to_numpy()), dict(kw, umi=ox["nCount_RNA"].to_numpy())
    elif mode == "balanced":
        kw_w, kw_x = dict(kw, strata=ow["ECclusters"].to_numpy()), None
    else:
        kw_w = kw_x = kw
    Xw, mw, infw = make_packets(Cw, ow, size, R, seed, "Patient", COLS_W, None, spec.get("min_whole", 100), **kw_w)
    out = dict(mw=mw, infw=infw, y=(mw["diagnosis"] == "GBM").astype(int).to_numpy())
    W = _dense(Xw)
    A = mx = None
    if need_ext and kw_x is not None:
        Xx, mx, infx = make_packets(Cx, ox, size, R, seed, "sample", COLS_X, None, spec.get("min_whole", 100), **kw_x)
        Xa, present = align_genes(Xx, gx, gw)
        A = _dense(Xa)
        out.update(mx=mx, infx=infx)
        keep = present.copy() if spec.get("common") else np.ones(W.shape[1], dtype=bool)
    else:
        keep = np.ones(W.shape[1], dtype=bool)
    if spec.get("drop") is not None:
        keep[np.asarray(spec["drop"], dtype=int)] = False
    W = W[:, keep]
    A = A[:, keep] if A is not None else None
    if spec.get("thin"):
        lib = np.r_[W.sum(axis=1), A.sum(axis=1)] if A is not None else W.sum(axis=1)
        target = float(np.percentile(lib, 5)) if spec["thin"] == "auto" else float(spec["thin"])
        W, A = thin_counts(W, target, seed).astype(np.float64), (thin_counts(A, target, seed + 1).astype(np.float64) if A is not None else None)
        out["thin_target"] = target
    out.update(W=W, A=A, n_genes=int(keep.sum()))
    return out


def run_variant(S, rom, spec: dict, seed: int) -> dict:
    b = build(S, spec, seed, need_ext=not spec.get("internal_only"))
    model = spec.get("model", "lr")
    mw, y, W, A = b["mw"], b["y"], b["W"], b["A"]
    internal = lopo(W, mw, y, rom, model=model)
    pat = to_patient(internal, keys=("group", "representation"))
    rows = []
    for rep, g in pat.groupby("representation"):
        rows.append(dict(analysis="internal", representation=rep, **patient_metrics(g["y"], g["p_mean"])))
    core_pat = pd.DataFrame()
    if A is not None:
        mx = b["mx"]
        pr = predict_block(W, y, mw["patient"].to_numpy(), A, rom, model=model)
        ext = pd.concat([pd.DataFrame(dict(packet=np.arange(len(mx)), patient=mx["patient"], sample=mx["sample"], site=mx["site"],
                                           y=(mx["diagnosis"] == "GBM").astype(int), representation=n, p=p)) for n, p in pr.items()], ignore_index=True)
        ext["seed"], ext["size"] = seed, spec["size"]
        lv = M19.specimen_levels(ext)
        for name in ("core", "periphery", "all"):
            for rep, g in lv[name].groupby("representation"):
                rows.append(dict(analysis=f"xie_{name}", representation=rep, **patient_metrics(g["y"], g["p_mean"])))
        core_pat = lv["core"][["representation", "patient", "y", "p_mean"]]
    m = pd.DataFrame(rows).assign(seed=seed)
    info = dict(n_genes=b["n_genes"], n_cells_median_w=float(mw["n_cells"].median()), lib_median_w=float(mw["lib_size"].median()),
                n_cells_median_x=float(b["mx"]["n_cells"].median()) if A is not None else np.nan,
                lib_median_x=float(b["mx"]["lib_size"].median()) if A is not None else np.nan, thin_target=b.get("thin_target", np.nan),
                n_infeasible_w=len(b["infw"]), n_infeasible_x=len(b.get("infx", [])))
    return dict(metrics=m, core_patients=core_pat, info=info)


def run_named(S, rom, out: Path, name: str, spec: dict, seeds, force=False) -> pd.DataFrame:
    """Ejecuta (o recupera) una variante para varias semillas. Devuelve las métricas apiladas."""
    (out / "variants").mkdir(parents=True, exist_ok=True)
    ms, cps, infos = [], [], []
    for s in seeds:
        f = out / "variants" / f"{name}_seed{s}.joblib"
        if f.exists() and not force:
            r = joblib.load(f)
        else:
            t0 = time.time()
            r = run_variant(S, rom, spec, s)
            joblib.dump(r, f)
            _log(f"  variante {name} semilla {s} hecha ({time.time() - t0:.0f} s)")
        ms.append(r["metrics"])
        cps.append(r["core_patients"].assign(seed=s))
        infos.append(dict(r["info"], seed=s))
    m = pd.concat(ms, ignore_index=True).assign(variant=name)
    pd.concat(cps, ignore_index=True).assign(variant=name).to_csv(out / "variants" / f"{name}_core_patients.csv", index=False)
    pd.DataFrame(infos).assign(variant=name).to_csv(out / "variants" / f"{name}_info.csv", index=False)
    return m


def mean_f1(m: pd.DataFrame, analysis: str, rep: str) -> float:
    d = m[(m["analysis"] == analysis) & (m["representation"] == rep)]
    return float(d["macro_f1"].mean()) if len(d) else np.nan


# --------------------------------------------------------------------------- etapas baratas


def stage_holm(res1: Path, out: Path):
    f = res1 / "permutation_results.csv"
    if not f.exists():
        raise FileNotFoundError(f"falta {f}; ejecuta antes 19_pseudobulk_counts")
    d = pd.read_csv(f)
    p = d["p_exact"].to_numpy()
    order = np.argsort(p)
    m = len(p)
    holm = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        holm[i] = min(1.0, run)
    d["p_holm_across_all_tests"] = holm
    d["n_tests_in_family"] = m
    d.to_csv(out / "holm_permutation.csv", index=False)
    return d


def stage_age_prior(S, out: Path, age_lgg=49.3, age_gbm=61.3):
    """Un modelo que solo conoce la edad (entrenado con los 14 pacientes de Wälchli) aplicado a las edades medias de Xie."""
    from sklearn.linear_model import LogisticRegression
    ow = S[1]
    t = ow.groupby("Patient").agg(age=("age_num", "first"), dx=("glioma_type", "first")).dropna().reset_index()
    y = (t["dx"] == "GBM").astype(int).to_numpy()
    mu, sd = t["age"].mean(), t["age"].std() + 1e-9
    clf = LogisticRegression(class_weight="balanced").fit(((t["age"] - mu) / sd).to_numpy()[:, None], y)
    rows = [dict(group="Wälchli LGG (media)", age=t[t["dx"] == "LGG"]["age"].mean()), dict(group="Wälchli GBM (media)", age=t[t["dx"] == "GBM"]["age"].mean()),
            dict(group="Xie LGG (media publicada)", age=age_lgg), dict(group="Xie GBM (media publicada)", age=age_gbm)]
    for r in rows:
        r["p_gbm_age_only"] = float(clf.predict_proba(np.array([[(r["age"] - mu) / sd]]))[0, 1])
    d = pd.DataFrame(rows)
    d["edad_min_walchli"], d["edad_max_walchli"] = t["age"].min(), t["age"].max()
    d.to_csv(out / "age_prior_xie.csv", index=False)
    t.to_csv(out / "walchli_ages.csv", index=False)
    return d


def stage_converge(res1: Path, out: Path, size=None, ks=(1, 2, 3, 5, 10, 15, 20), draws=300):
    """Con cuántos paquetes se estabiliza la probabilidad de un paciente (sobre las ejecuciones guardadas de la fase 1)."""
    size = size or PRIMARY
    rng = np.random.default_rng(0)
    rows, comp = [], []
    for f in sorted((res1 / "runs").glob(f"seed*_size{size}.joblib")):
        r = joblib.load(f)
        seed = int(f.stem.split("_")[0].replace("seed", ""))
        blocks = [("internal", r["internal"].rename(columns={"group": "unit"}))]
        e = r["external"]
        blocks.append(("xie_core", e[e["site"] == "core"].rename(columns={"sample": "unit"})))
        for analysis, d in blocks:
            for rep, gr in d.groupby("representation"):
                sds = {k: [] for k in ks}
                flips = {k: [] for k in ks}
                for u, gu in gr.groupby("unit"):
                    v = gu["p"].to_numpy()
                    full = v.mean()
                    for k in ks:
                        if k > len(v):
                            continue
                        means = np.array([v[rng.choice(len(v), k, replace=False)].mean() for _ in range(draws)])
                        sds[k].append(means.std())
                        flips[k].append(float(((means >= 0.5) != (full >= 0.5)).mean()))
                for k in ks:
                    if sds[k]:
                        rows.append(dict(seed=seed, analysis=analysis, representation=rep, n_packets=k, median_sd_of_patient_mean=float(np.median(sds[k])),
                                         max_sd_of_patient_mean=float(np.max(sds[k])), mean_prob_class_flip_vs_full=float(np.mean(flips[k]))))
                if analysis == "internal":
                    g = gr.groupby("unit")["p"]
                    k_ = g.ngroups
                    n_ = len(gr)
                    ssb = float((g.size() * (g.mean() - gr["p"].mean()) ** 2).sum())
                    ssw = float(((gr["p"] - g.transform("mean")) ** 2).sum())
                    msb, msw = ssb / (k_ - 1), ssw / (n_ - k_)
                    n0 = (n_ - (g.size() ** 2).sum() / n_) / (k_ - 1)
                    comp.append(dict(seed=seed, representation=rep, var_between_patients=max((msb - msw) / n0, 0.0), var_within_patient=msw,
                                     share_between=max((msb - msw) / n0, 0.0) / (max((msb - msw) / n0, 0.0) + msw)))
    pd.DataFrame(rows).groupby(["analysis", "representation", "n_packets"]).mean(numeric_only=True).drop(columns="seed").reset_index().to_csv(out / "replicate_convergence.csv", index=False)
    pd.DataFrame(comp).groupby("representation").mean(numeric_only=True).drop(columns="seed").reset_index().to_csv(out / "variance_components_anova.csv", index=False)


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def stage_periphery(res1: Path, out: Path, sizes=None):
    """Periferia: ¿es un desplazamiento constante de las probabilidades? (diagnóstico posterior, no una validación)."""
    rows, shift = [], []
    for z in (sizes or PERIPHERY_SIZES):
        for f in sorted((res1 / "runs").glob(f"seed*_size{z}.joblib")):
            r = joblib.load(f)
            seed = int(f.stem.split("_")[0].replace("seed", ""))
            lv = M19.specimen_levels(r["external"])
            core, per = lv["core"], lv["periphery"]
            for rep in sorted(set(per["representation"])):
                c, p = core[core["representation"] == rep], per[per["representation"] == rep]
                if p["y"].nunique() < 2:
                    continue
                lp = _logit(p["p_mean"])
                base = patient_metrics(p["y"], p["p_mean"])
                thr_c = patient_metrics(p["y"], 1 / (1 + np.exp(-(lp - lp.mean()))))             # centrado sin etiquetas (transductivo)
                order = np.sort(np.unique(p["p_mean"]))
                best = max((patient_metrics(p["y"], p["p_mean"], thr=t)["macro_f1"], t) for t in order)  # umbral óptimo a posteriori (techo)
                rows.append(dict(size=z, seed=seed, representation=rep, n_patients=len(p), auroc=base["auroc"], macro_f1_thr05=base["macro_f1"],
                                 macro_f1_centered_unlabeled=thr_c["macro_f1"], macro_f1_best_threshold_posthoc=best[0], mean_p_periphery=p["p_mean"].mean()))
                mm = c.merge(p, on="patient", suffixes=("_core", "_per"))
                if len(mm):
                    shift.append(dict(size=z, seed=seed, representation=rep, mean_logit_shift_per_minus_core=float((_logit(mm["p_mean_per"]) - _logit(mm["p_mean_core"])).mean()),
                                      frac_patients_per_lower_than_core=float((mm["p_mean_per"] < mm["p_mean_core"]).mean()), n=len(mm)))
    pd.DataFrame(rows).groupby(["size", "representation"]).mean(numeric_only=True).drop(columns="seed").reset_index().to_csv(out / "periphery_threshold_diagnostic.csv", index=False)
    pd.DataFrame(shift).groupby(["size", "representation"]).mean(numeric_only=True).drop(columns="seed").reset_index().to_csv(out / "periphery_logit_shift.csv", index=False)


# --------------------------------------------------------------------------- variantes


def study_gene_ranking(S, rom, out: Path):
    """Genes ordenados por cuánto distinguen los estudios (t de Welch sobre la media de log-CPM por paciente/muestra; 100 células, semilla 0)."""
    f = out / "study_gene_ranking.csv"
    Cw, ow, gw, Cx, ox, gx = S
    b = build(S, dict(size=100), 0)
    W, A, mw, mx = b["W"], b["A"], b["mw"], b["mx"]
    _, present = align_genes(sp.csr_matrix((1, len(gx))), gx, gw)

    def unit_means(M, units):
        L = np.log2(M / np.maximum(M.sum(axis=1, keepdims=True), 1) * 1e6 + 1)
        return pd.DataFrame(L).groupby(np.asarray(units)).mean().to_numpy()

    uw, ux = unit_means(W, mw["unit"]), unit_means(A, mx["unit"])
    t = np.zeros(W.shape[1])
    ok = present & ((W.sum(axis=0) > 0) & (A.sum(axis=0) > 0))
    t[ok] = ttest_ind(uw[:, ok], ux[:, ok], equal_var=False)[0]
    t = np.nan_to_num(t)
    order = np.argsort(-np.abs(t), kind="stable")
    order = order[ok[order]]
    pd.DataFrame(dict(gene=np.asarray(gw)[order], position=order, abs_t=np.abs(t[order]), t=t[order])).to_csv(f, index=False)
    return order


BASE = dict(size=100)
ROBUSTNESS = {                                           # nombre -> (spec, comentario)
    "base_lr_100": (dict(BASE), "referencia (igual que la fase 1, mismo código)"),
    "rf_100": (dict(BASE, model="rf"), "random forest en lugar de regresión logística"),
    "disjoint_50": (dict(size=50, mode="disjoint"), "paquetes disjuntos de 50 células"),
    "disjoint_100": (dict(size=100, mode="disjoint"), "paquetes disjuntos de 100 células"),
    "fraction_010": (dict(size=0.10, mode="fraction"), "el 10 % de las células de cada paciente"),
    "fraction_025": (dict(size=0.25, mode="fraction"), "el 25 %"),
    "fraction_050": (dict(size=0.50, mode="fraction"), "el 50 %"),
    "umi_150k": (dict(size=150000, mode="umi"), "paquetes con ~150.000 UMIs (iguala la profundidad, no el nº de células)"),
    "thin_auto": (dict(BASE, thin="auto"), "conteos adelgazados al azar a la misma profundidad (percentil 5)"),
    "common_genes": (dict(BASE, common=True), "solo genes presentes en ambos estudios (sin ceros rellenados)"),
    "balanced_100": (dict(BASE, mode="balanced", internal_only=True), "equilibrado por estado endotelial (solo interno)"),
    "xie_qc_paper": (dict(BASE, xie_filter="qc_paper"), "Xie con filtro mitocondrial <10 % (criterio del artículo)"),
    "xie_pure": (dict(BASE, xie_filter="pure"), "Xie solo células claramente endoteliales (≥2 marcadores y endotelial > mural/inmune/glía)"),
    "xie_pure_strictest": (dict(BASE, xie_filter="pure", common=True, thin="auto"), "Xie puro + genes comunes + misma profundidad"),
}


def stage_xie_filters(S, out: Path):
    """Cuántas células de Xie sobreviven a cada filtro, por muestra (el artículo publica 103.230 células y 57.324 endoteliales en total)."""
    ox = S[4]
    rows = []
    for kind in ("qc_paper", "pure"):
        km = xie_mask(S, kind)
        for smp, g in ox.assign(k=km).groupby("sample"):
            rows.append(dict(filter=kind, sample=smp, patient=g["Patient"].iloc[0], site=g["site"].iloc[0], n_cells_loaded=len(g), n_cells_kept=int(g["k"].sum())))
    d = pd.DataFrame(rows)
    d.to_csv(out / "xie_filter_counts.csv", index=False)
    tot = d.groupby("filter")[["n_cells_loaded", "n_cells_kept"]].sum().reset_index()
    tot.assign(published_total_cells=103230, published_endothelial=57324).to_csv(out / "xie_filter_totals.csv", index=False)


def stage_variants(S, rom, out: Path, seeds, force=False):
    all_m = []
    try:
        stage_xie_filters(S, out)
    except Exception:
        _err(out, "filtros de Xie")
    for name, (spec, _) in ROBUSTNESS.items():
        try:
            all_m.append(run_named(S, rom, out, name, spec, seeds, force))
        except Exception:
            _err(out, f"variante {name}")
    # sin los genes que más distinguen los estudios, con parada y refinamiento automáticos
    try:
        order = study_gene_ranking(S, rom, out)
        ks, broke, last_ok = [500, 2000, 5000, 10000], None, 0
        for k in ks:
            if k >= len(order):
                break
            m = run_named(S, rom, out, f"dropstudy_{k}", dict(BASE, drop=order[:k]), seeds, force)
            all_m.append(m)
            f1 = mean_f1(m, "xie_core", "HVG-2000")
            if f1 < OK_F1:
                broke = k
                break
            last_ok = k
        extra = []
        if broke is not None:
            extra = [int(round(last_ok + (broke - last_ok) * f)) for f in (0.25, 0.5, 0.75)]
        elif last_ok and last_ok + 5000 < len(order):
            extra = [last_ok + 5000]
        for k in extra:
            all_m.append(run_named(S, rom, out, f"dropstudy_{k}", dict(BASE, drop=order[:k]), seeds, force))
        pd.DataFrame(dict(k_breaks=[broke if broke is not None else -1], k_last_ok=[last_ok], refinement=[str(extra)])).to_csv(out / "dropstudy_breakpoint.csv", index=False)
    except Exception:
        _err(out, "genes de estudio")
    if all_m:
        pd.concat(all_m, ignore_index=True).to_csv(out / "variants_all_metrics.csv", index=False)


def stage_state(S, rom, out: Path, seeds, size=50, R=20):
    """Paquetes de un solo estado endotelial (ECclusters), solo dentro de Wälchli."""
    Cw, ow, gw, *_ = S
    rows = []
    for st in sorted(ow["ECclusters"].astype(str).unique()):
        for s in seeds:
            mask = (ow["ECclusters"].astype(str) == st).to_numpy()
            try:
                Xw, mw, inf = make_packets(Cw[mask], ow[mask].reset_index(drop=True), size, R, s, "Patient", COLS_W, None, 100)
            except AssertionError:
                continue
            y = (mw["diagnosis"] == "GBM").astype(int).to_numpy()
            if len(set(y)) < 2 or min((mw.groupby("patient")["diagnosis"].first() == "GBM").sum(), (mw.groupby("patient")["diagnosis"].first() == "LGG").sum()) < 2:
                continue
            pat = to_patient(lopo(Xw, mw, y, rom), keys=("group", "representation"))
            for rep, g in pat.groupby("representation"):
                rows.append(dict(state=st, seed=s, size=size, representation=rep, n_infeasible=len(inf), **patient_metrics(g["y"], g["p_mean"])))
        _log(f"  estado {st} hecho")
    pd.DataFrame(rows).to_csv(out / "state_specific_internal.csv", index=False)


def stage_reverse(S, rom, out: Path, seeds, size=100, R=20):
    """Entrenar con Xie y predecir Wälchli, y señal dentro de Xie (dejar un paciente fuera)."""
    Cw, ow, gw, Cx, ox, gx = S
    rows = []
    for s in seeds:
        Xw, mw, _ = make_packets(Cw, ow, size, R, s, "Patient", COLS_W, None, 100)
        Xx, mx, _ = make_packets(Cx, ox, size, R, s, "sample", COLS_X, None, 100)
        Xa, _ = align_genes(Xx, gx, gw)
        W, A = _dense(Xw), _dense(Xa)
        yw = (mw["diagnosis"] == "GBM").astype(int).to_numpy()
        yx = (mx["diagnosis"] == "GBM").astype(int).to_numpy()
        for site_name, sel in (("core", mx["site"] == "core"), ("todo", mx["site"].notna())):
            sel = sel.to_numpy()
            if len(set(yx[sel])) < 2:
                continue
            pr = predict_block(A[sel], yx[sel], mx["patient"].to_numpy()[sel], W, rom)
            for rep, p in pr.items():
                pat = pd.DataFrame(dict(y=yw, p=p, g=mw["patient"].to_numpy())).groupby("g").agg(y=("y", "first"), p_mean=("p", "mean"))
                rows.append(dict(analysis=f"inverso_entrena_xie_{site_name}", seed=s, representation=rep, **patient_metrics(pat["y"], pat["p_mean"])))
        for site_name, sel in (("core", mx["site"] == "core"), ("periferia", mx["site"] == "peripheral"), ("todo", mx["site"].notna())):
            sel = sel.to_numpy()
            m_ = mx[sel].reset_index(drop=True)
            if m_["patient"].nunique() < 4 or len(set(yx[sel])) < 2:
                continue
            d = lopo(A[sel], m_, yx[sel], rom)
            pat = to_patient(d, keys=("group", "representation"))
            for rep, g in pat.groupby("representation"):
                rows.append(dict(analysis=f"dentro_de_xie_{site_name}", seed=s, representation=rep, **patient_metrics(g["y"], g["p_mean"])))
        _log(f"  inverso/dentro de Xie, semilla {s}")
    pd.DataFrame(rows).to_csv(out / "reverse_and_within_xie.csv", index=False)


def _fit_subset(W, y, pw, A_core, ycore, pcore, rom, train_patients):
    keep = np.isin(pw, train_patients)
    pr = predict_block(W[keep], y[keep], pw[keep], A_core, rom)
    rows = []
    for rep, p in pr.items():
        d = pd.DataFrame(dict(p=p, pt=pcore, y=ycore)).groupby("pt").agg(y=("y", "first"), p_mean=("p", "mean"))
        rows.append(dict(representation=rep, **patient_metrics(d["y"], d["p_mean"])))
    return rows


def stage_training(S, rom, out: Path, n_draws=100, n_jobs=-1, size=None):
    """¿Depende el resultado `core` de qué pacientes de Wälchli se usan para entrenar?"""
    b = build(S, dict(size=size or PRIMARY), 0)
    W, A, mw, mx, y = b["W"], b["A"], b["mw"], b["mx"], b["y"]
    pw = mw["patient"].to_numpy()
    core = (mx["site"] == "core").to_numpy()
    A_core, ycore, pcore = A[core], (mx["diagnosis"] == "GBM").astype(int).to_numpy()[core], mx["patient"].to_numpy()[core]
    gbm = sorted(mw[mw["diagnosis"] == "GBM"]["patient"].unique())
    lgg = sorted(mw[mw["diagnosis"] == "LGG"]["patient"].unique())
    rng = np.random.default_rng(0)
    draws, seen = [], set()
    if len(gbm) < 4 or len(lgg) < 3:
        raise RuntimeError(f"muy pocos pacientes para remuestrear (GBM {len(gbm)}, LGG {len(lgg)})")
    n_possible = math.comb(len(gbm), len(gbm) - 2) * math.comb(len(lgg), len(lgg) - 2)
    n_draws = min(n_draws, n_possible)
    while len(draws) < n_draws:
        t = tuple(sorted(list(rng.choice(gbm, len(gbm) - 2, replace=False)) + list(rng.choice(lgg, len(lgg) - 2, replace=False))))
        if t not in seen:
            seen.add(t)
            draws.append(list(t))
    res = Parallel(n_jobs=n_jobs)(delayed(_fit_subset)(W, y, pw, A_core, ycore, pcore, rom, t) for t in draws)
    rows = [dict(draw=i, train=";".join(t), **r) for i, (t, rr) in enumerate(zip(draws, res)) for r in rr]
    pd.DataFrame(rows).to_csv(out / "training_subsample_core.csv", index=False)
    _log("  remuestreo del entrenamiento hecho")
    patients = sorted(set(pw))
    res2 = Parallel(n_jobs=n_jobs)(delayed(_fit_subset)(W, y, pw, A_core, ycore, pcore, rom, [q for q in patients if q != p]) for p in patients)
    rows2 = [dict(dropped=p, dropped_dx=("GBM" if p in gbm else "LGG"), **r) for p, rr in zip(patients, res2) for r in rr]
    pd.DataFrame(rows2).to_csv(out / "training_drop_one_core.csv", index=False)


def stage_followups(S, rom, out: Path, seeds):
    """Pruebas condicionadas: si una variante deja el `core` por debajo de 0,9, se repite con paquetes mayores (¿se rescata?)."""
    f = out / "variants_all_metrics.csv"
    if not f.exists():
        return
    m = pd.read_csv(f)
    done = []
    for name, g in m.groupby("variant"):
        if name.startswith("dropstudy") or name in ("base_lr_100", "balanced_100") or name not in ROBUSTNESS:
            continue
        if mean_f1(g, "xie_core", "HVG-2000") < OK_F1 or mean_f1(g, "xie_core", "PCA-5") < OK_F1:
            spec = dict(ROBUSTNESS[name][0])
            for z, label in ((250, "250"), (500, "500")):
                s2 = dict(spec)
                s2["size"] = {"umi": z * 1500, "fraction": spec["size"]}.get(spec.get("mode"), z)
                if spec.get("mode") == "disjoint" and z == 500:
                    continue
                try:
                    run_named(S, rom, out, f"followup_{name}_{label}", s2, seeds)
                    done.append(f"{name}_{label}")
                except Exception:
                    _err(out, f"followup {name} {label}")
    (out / "followups_run.txt").write_text("\n".join(done) if done else "ninguna (ninguna variante rompió el resultado)\n", encoding="utf-8")


# --------------------------------------------------------------------------- resumen


def _err(out: Path, where: str):
    with open(out / "phase2_errors.log", "a", encoding="utf-8") as fh:
        fh.write(f"=== {where} ===\n{traceback.format_exc()}\n")
    _log(f"  ERROR en {where} (detalle en phase2_errors.log)")


def stage_summary(out: Path):
    lines = ["# Fase 2: robustez del resultado `core` y análisis de la guía\n"]
    verdict = []
    f = out / "variants_all_metrics.csv"
    if f.exists():
        m = pd.read_csv(f)
        fol = [p.stem.replace("_core_patients", "") for p in (out / "variants").glob("followup_*_core_patients.csv")]
        for nm in fol:
            if not (m["variant"] == nm).any():
                m = pd.concat([m, run_named_light(out, nm)], ignore_index=True)
        t = []
        for v, g in m.groupby("variant", sort=False):
            r = dict(variante=v)
            for a, lab in (("xie_core", "core"), ("xie_all", "todo"), ("xie_periphery", "periferia"), ("internal", "interno")):
                for rep in REPS_MAIN:
                    r[f"{lab} {rep}"] = round(mean_f1(g, a, rep), 3)
            t.append(r)
        tab = pd.DataFrame(t)
        tab.to_csv(out / "robustness_table.csv", index=False)
        lines += ["## Sensibilidad (macro-F1 medio entre semillas)\n", to_markdown(tab), "\n"]
        base = tab[tab["variante"] == "base_lr_100"]
        core_cols = [f"core {r}" for r in REPS_MAIN]
        ext = tab[~tab["variante"].isin(["base_lr_100", "balanced_100"]) & ~tab["variante"].str.startswith("dropstudy") & ~tab["variante"].str.startswith("followup")]
        bad = ext[(ext[["core HVG-2000", "core PCA-5"]] < OK_F1).any(axis=1)]["variante"].tolist()
        verdict.append("Variantes que rompen el `core` (macro-F1 < 0,9 en HVG-2000 o PCA-5): " + (", ".join(bad) if bad else "ninguna"))
        ds = tab[tab["variante"].str.startswith("dropstudy")]
        if len(ds):
            ok_k = [int(v.split("_")[1]) for v, h in zip(ds["variante"], ds["core HVG-2000"]) if h >= OK_F1]
            verdict.append(f"Quitando los genes que más distinguen los estudios, el `core` aguanta hasta k = {max(ok_k) if ok_k else 0} genes eliminados (HVG-2000 ≥ 0,9)")
    for fn, title in (("training_subsample_core.csv", "Remuestreo del entrenamiento (100 subconjuntos de 6 GBM + 4 LGG)"),):
        p = out / fn
        d = _read(p)
        if d is not None:
            s = d.groupby("representation").agg(f1_mediana=("macro_f1", "median"), f1_p05=("macro_f1", lambda v: v.quantile(0.05)), prop_perfecto=("macro_f1", lambda v: (v >= 0.999).mean()),
                                                 auroc_mediana=("auroc", "median")).reset_index().round(3)
            lines += [f"## {title}\n", to_markdown(s), "\n"]
            h = s[s["representation"] == "HVG-2000"]["prop_perfecto"]
            if len(h):
                verdict.append(f"Con subconjuntos de entrenamiento, HVG-2000 separa el `core` sin error en el {100 * float(h.iloc[0]):.0f} % de los casos")
            fig_training(d, out)
    for fn, title in (("training_drop_one_core.csv", "Quitar un paciente de entrenamiento cada vez (macro-F1 en `core`)"),
                      ("reverse_and_within_xie.csv", "Transferencia inversa y señal dentro de Xie"),
                      ("state_specific_internal.csv", "Paquetes específicos de cada estado endotelial (interno)"),
                      ("periphery_threshold_diagnostic.csv", "Periferia: umbral"), ("periphery_logit_shift.csv", "Periferia: desplazamiento de logit frente al núcleo"),
                      ("replicate_convergence.csv", "Convergencia de réplicas"), ("variance_components_anova.csv", "Componentes de varianza (ANOVA)"),
                      ("age_prior_xie.csv", "Modelo solo de edad con las edades medias de Xie"), ("holm_permutation.csv", "Holm sobre las permutaciones de la fase 1")):
        p = out / fn
        d = _read(p)
        if d is not None:
            if fn == "training_drop_one_core.csv":
                d = d[d["representation"].isin(REPS_MAIN)][["dropped", "dropped_dx", "representation", "macro_f1", "auroc"]]
            if fn == "reverse_and_within_xie.csv":
                d = d.groupby(["analysis", "representation"]).agg(macro_f1=("macro_f1", "mean"), auroc=("auroc", "mean")).reset_index()
            if fn == "state_specific_internal.csv":
                d = d[d["representation"].isin(REPS_MAIN)].groupby(["state", "representation"]).agg(macro_f1=("macro_f1", "mean"), auroc=("auroc", "mean"), n=("n_patients", "min")).reset_index()
            if fn == "replicate_convergence.csv":
                d = d[d["representation"].isin(["PCA-5", "HVG-2000"])]
            lines += [f"## {title}\n", to_markdown(d.round(3)), "\n"]
    a = _read(out / "age_prior_xie.csv")
    if a is not None:
        xl, xg = a[a["group"].str.contains("Xie LGG")]["p_gbm_age_only"].iloc[0], a[a["group"].str.contains("Xie GBM")]["p_gbm_age_only"].iloc[0]
        verdict.append(f"Un modelo solo de edad daría P(GBM) = {xl:.2f} a la edad media de los LGG de Xie y {xg:.2f} a la de los GBM: " +
                       ("ambos grupos se llamarían GBM, así que la edad por sí sola no explicaría la separación perfecta" if xl >= 0.5 else "la edad sí separa los grupos de Xie; la confusión con la edad no se descarta"))
    d = _read(out / "reverse_and_within_xie.csv")
    if d is not None:
        d = d.groupby(["analysis", "representation"])["auroc"].mean().reset_index()
        for a_, g in d.groupby("analysis"):
            verdict.append(f"{a_}: AUROC medio {g['auroc'].mean():.2f} (HVG-2000: {g[g['representation'] == 'HVG-2000']['auroc'].mean():.2f})")
    lines.insert(1, "## Veredicto automático (revisar con las tablas)\n\n" + "\n".join(f"- {v}" for v in verdict) + "\n")
    (out / "phase2_summary.md").write_text("\n".join(lines), encoding="utf-8")
    fig_robustness(out)
    return verdict


def _read(p: Path):
    try:
        d = pd.read_csv(p)
        return d if len(d) else None
    except Exception:
        return None


def run_named_light(out: Path, name: str) -> pd.DataFrame:
    seeds_files = sorted((out / "variants").glob(f"{name}_seed*.joblib"))
    return pd.concat([joblib.load(f)["metrics"] for f in seeds_files], ignore_index=True).assign(variant=name) if seeds_files else pd.DataFrame()


def fig_robustness(out: Path):
    f = out / "robustness_table.csv"
    if not f.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = pd.read_csv(f)
    t = t[t["core HVG-2000"].notna()]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.32 * len(t) + 1.5)))
    ypos = np.arange(len(t))
    for rep, mk, off in (("HVG-2000", "o", -0.2), ("PCA-5", "s", 0.0), ("PCA-2", "^", 0.2)):
        ax.scatter(t[f"core {rep}"], ypos + off, marker=mk, label=rep, s=28)
    ax.axvline(OK_F1, color="grey", ls="--", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(t["variante"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.05)
    ax.set_xlabel("macro-F1 en Xie `core` (6 pacientes)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "22_robustness.png", dpi=150)
    plt.close(fig)
    describe_figure(out, "22_robustness.png", ["22_robustness.png",
                    "Macro-F1 en Xie `core` (6 pacientes, modelo congelado entrenado con Wälchli) para cada variante del análisis: tipo de modelo, forma de los paquetes, igualación de profundidad, genes comunes y eliminación de genes que distinguen los estudios.",
                    "Cada punto es la media de las semillas; la línea gris marca 0,9. Eje X: macro-F1; eje Y: variante. Tres símbolos por variante: HVG-2000, PCA-5 y PCA-2.",
                    "Con 6 pacientes cada paciente mal clasificado baja el macro-F1 en unos 0,17, así que las diferencias pequeñas no son interpretables."])


def fig_training(d: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    reps = [r for r in REPS_MAIN if r in set(d["representation"])]
    fig, axes = plt.subplots(1, len(reps), figsize=(3.2 * len(reps), 3), sharey=True)
    for ax, rep in zip(np.atleast_1d(axes), reps):
        v = d[d["representation"] == rep]["macro_f1"]
        ax.hist(v, bins=np.linspace(0, 1, 11), color="#4C72B0")
        ax.set_title(rep, fontsize=9)
        ax.set_xlabel("macro-F1 en `core`")
    np.atleast_1d(axes)[0].set_ylabel("subconjuntos de entrenamiento")
    fig.tight_layout()
    fig.savefig(out / "22_training_subsample.png", dpi=150)
    plt.close(fig)
    describe_figure(out, "22_training_subsample.png", ["22_training_subsample.png",
                    "Distribución del macro-F1 en Xie `core` cuando el modelo se entrena con subconjuntos aleatorios de 10 de los 14 pacientes de Wälchli (6 GBM y 4 LGG), 100 subconjuntos distintos.",
                    "Cada panel es una representación; eje X: macro-F1 (con 6 pacientes solo toma unos pocos valores); eje Y: número de subconjuntos. Si casi todos están en 1,0 el resultado no depende de qué pacientes se usan.",
                    "Los subconjuntos comparten pacientes entre sí, así que no son independientes."])


# --------------------------------------------------------------------------- principal


def main(cfg: dict, xie_path, out, res1="results/pseudobulk_counts", seeds=3, n_jobs=-1, stages=None, n_draws=100, force=False) -> dict:
    out, res1 = Path(out), Path(res1)
    out.mkdir(parents=True, exist_ok=True)
    rom = dict(components=[2, 5, 10, 20], hvg=cfg["rom"].get("hvg", 2000), scale_clip=cfg["rom"].get("scale_clip", 10.0), min_cpm=1.0, min_frac=0.25)
    todo = stages or ["holm", "age_prior", "converge", "periphery", "variants", "state", "reverse", "training", "followups", "summary"]
    t0 = time.time()
    S = None
    if set(todo) & {"age_prior", "variants", "state", "reverse", "training", "followups"}:
        S = M19.load_studies(cfg, xie_path)
    if n_jobs == -1:
        n_jobs = min(os.cpu_count() or 1, 6)                      # cada proceso carga matrices densas de ~100 MB y hace PCA: sin pasarse
    sd = list(range(seeds))
    jobs = dict(holm=lambda: stage_holm(res1, out), age_prior=lambda: stage_age_prior(S, out), converge=lambda: stage_converge(res1, out),
                periphery=lambda: stage_periphery(res1, out), variants=lambda: stage_variants(S, rom, out, sd, force),
                state=lambda: stage_state(S, rom, out, sd[:2]), reverse=lambda: stage_reverse(S, rom, out, sd),
                training=lambda: stage_training(S, rom, out, n_draws, n_jobs), followups=lambda: stage_followups(S, rom, out, sd),
                summary=lambda: stage_summary(out))
    done = {}
    for st in todo:
        _log(f"=== etapa: {st} ===")
        t1 = time.time()
        try:
            done[st] = jobs[st]()
        except Exception:
            _err(out, st)
        _log(f"=== etapa {st} terminada en {time.time() - t1:.0f} s (total {time.time() - t0:.0f} s) ===")
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--xie", default="data/xie/xie_endothelial.h5ad")
    ap.add_argument("--out", default="results/pseudobulk_phase2")
    ap.add_argument("--res1", default="results/pseudobulk_counts")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--n-draws", type=int, default=100)
    ap.add_argument("--stages", nargs="+")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(load_config(a.config), a.xie, a.out, a.res1, a.seeds, a.n_jobs, a.stages, a.n_draws, a.force)
