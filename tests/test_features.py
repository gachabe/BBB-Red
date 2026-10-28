import numpy as np
import pytest

from src.data import GROUP
from src.features import check_task, lopo_folds, permute_labels


def test_folds_leave_one_patient_out(dataset_and_folds):
    ds, folds = dataset_and_folds
    pats = ds.meta[GROUP].to_numpy()
    assert len(folds) == 9 and sorted(f.held_out for f in folds) == sorted(set(pats))
    seen = np.zeros(len(pats), int)
    for f in folds:
        assert set(pats[f.test_idx]) == {f.held_out}
        assert f.held_out not in set(pats[f.train_idx]) and set(f.train_idx).isdisjoint(f.test_idx)
        seen[f.test_idx] += 1
    assert (seen == 1).all()   # cada célula es test exactamente una vez


def test_train_cap_and_full_test(dataset_and_folds):
    ds, folds = dataset_and_folds
    f = next(f for f in folds if f.held_out == "LGG1")
    counts = ds.meta[GROUP].iloc[f.train_idx].value_counts()
    assert counts.max() <= 50                                           # tope por paciente
    assert len(f.test_idx) == (ds.meta[GROUP] == "LGG1").sum()          # test usa todas las células


def test_folds_deterministic(dataset_and_folds):
    ds, _ = dataset_and_folds
    a, b = lopo_folds(ds.meta, 50, 7), lopo_folds(ds.meta, 50, 7)
    assert all((x.train_idx == y.train_idx).all() for x, y in zip(a, b))
    c = lopo_folds(ds.meta, 50, 8)
    assert any(len(x.train_idx) != len(z.train_idx) or (x.train_idx != z.train_idx).any() for x, z in zip(a, c))


def test_check_task_two_classes(dataset_and_folds):
    ds, folds = dataset_and_folds
    assert check_task(ds.meta, "glioma_type", folds) == ["GBM", "LGG"]
    bad = ds.meta.copy()
    bad["t3"] = np.where(bad[GROUP].isin(["GBM1", "GBM2"]), "X", bad["glioma_type"])
    with pytest.raises(AssertionError):
        check_task(bad, "t3", folds)


def test_permuted_labels_are_per_patient_and_keep_balance(dataset_and_folds):
    ds, _ = dataset_and_folds
    y = permute_labels(ds.meta, "glioma_type", 1)
    per = ds.meta.assign(y=y).groupby(GROUP)["y"].nunique()
    assert (per == 1).all()                                             # una etiqueta por paciente
    real = ds.meta.groupby(GROUP)["glioma_type"].first().value_counts().to_dict()
    perm = ds.meta.assign(y=y).groupby(GROUP)["y"].first().value_counts().to_dict()
    assert real == perm                                                 # mismos pacientes por clase
    assert (y != ds.meta["glioma_type"].to_numpy()).any()
