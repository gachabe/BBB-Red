"""Reparto por paciente (dejar un paciente fuera), submuestreo de entrenamiento y etiquetas permutadas."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import GROUP, Dataset, build_dataset


@dataclass
class Fold:
    held_out: str            # paciente de test de esta ronda
    train_idx: np.ndarray    # filas de entrenamiento (otros pacientes, con tope por paciente)
    test_idx: np.ndarray     # TODAS las células del paciente de test


def _by_patient(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    return {p: np.asarray(i) for p, i in meta.groupby(GROUP, sort=True).indices.items()}


def capped_indices(meta: pd.DataFrame, cap, seed: int) -> dict[str, np.ndarray]:
    """Submuestra fija por paciente (tope de células) para que un paciente grande no domine el ajuste."""
    rng = np.random.default_rng(seed)
    out = {}
    for p, idx in _by_patient(meta).items():
        if cap and len(idx) > cap:
            idx = np.sort(rng.choice(idx, size=int(cap), replace=False))
        out[p] = idx
    return out


def lopo_folds(meta: pd.DataFrame, cap, seed: int) -> list[Fold]:
    full = _by_patient(meta)
    train_by = capped_indices(meta, cap, seed)
    folds = []
    for p in sorted(full):
        tr = np.concatenate([train_by[q] for q in sorted(full) if q != p])
        te = full[p]
        assert p not in set(meta[GROUP].iloc[tr]), "fuga: el paciente de test está en train"
        assert set(tr).isdisjoint(te)
        folds.append(Fold(p, tr, te))
    assert sorted(f.held_out for f in folds) == sorted(full)  # cada paciente es test exactamente una vez
    assert sum(len(f.test_idx) for f in folds) == len(meta)
    return folds


def task_classes(meta: pd.DataFrame, task: str) -> list[str]:
    assert meta[task].notna().all(), f"La tarea {task} tiene etiquetas vacías"
    return sorted(meta[task].astype(str).unique())


def check_task(meta: pd.DataFrame, task: str, folds: list[Fold]) -> list[str]:
    classes = task_classes(meta, task)
    assert len(classes) == 2, f"{task}: se esperaban 2 clases y hay {classes}"
    y = meta[task].astype(str).to_numpy()
    for f in folds:
        assert set(y[f.train_idx]) == set(classes), f"{task}: al dejar fuera {f.held_out}, train pierde una clase"
    return classes


def task_is_patient_constant(meta: pd.DataFrame, task: str) -> bool:
    return bool((meta.groupby(GROUP)[task].nunique() == 1).all())


def permute_labels(meta: pd.DataFrame, task: str, seed: int) -> np.ndarray:
    """Control: etiquetas permutadas ENTRE PACIENTES (la etiqueta es del paciente), conservando
    cuántos pacientes hay de cada clase. Si no fuera constante por paciente, se permuta dentro de cada paciente."""
    rng = np.random.default_rng(seed)
    y = meta[task].astype(str).to_numpy()
    if task_is_patient_constant(meta, task):
        per = meta.groupby(GROUP)[task].first().astype(str)
        for _ in range(100):
            perm = rng.permutation(per.to_numpy())
            if not np.array_equal(perm, per.to_numpy()):
                break
        return meta[GROUP].map(dict(zip(per.index, perm))).astype(str).to_numpy()
    out = y.copy()
    for idx in _by_patient(meta).values():
        out[idx] = rng.permutation(y[idx])
    return out


def prepare(cfg: dict, outdir: Path) -> tuple[Dataset, list[Fold]]:
    ds = build_dataset(cfg)
    cap = cfg.get("ml", {}).get("max_train_cells_per_patient")
    folds = lopo_folds(ds.meta, cap, cfg["seed"])
    for t in cfg.get("tasks", ["glioma_type"]):
        check_task(ds.meta, t, folds)
    sp_dir = Path(outdir) / "splits"
    sp_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [dict(held_out=f.held_out, n_train_cells=len(f.train_idx), n_test_cells=len(f.test_idx)) for f in folds]
    ).to_csv(sp_dir / "folds.csv", index=False)
    ds.patients.to_csv(sp_dir / "patients.csv", index=False)
    return ds, folds
