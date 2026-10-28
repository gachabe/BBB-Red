import numpy as np
import pytest

from src.data import GROUP, build_dataset, check_integrity, leakage_report, library_size_check, read_source


def test_build_dataset_shapes(cfg):
    ds = build_dataset(cfg)
    assert ds.X.shape == (len(ds.meta), len(ds.genes)) and ds.X.dtype == np.float32
    assert set(ds.patients["glioma_type"]) == {"GBM", "LGG"} and len(ds.patients) == 9


def test_cache_roundtrip(cfg, tmp_path):
    d = {**cfg["data"], "cache": True, "cache_dir": str(tmp_path)}
    X1, o1, g1 = read_source(d)   # crea la caché
    X2, o2, g2 = read_source(d)   # lee la caché
    assert (X1 != X2).nnz == 0 and list(g1) == list(g2) and list(o1.index) == list(o2.index)
    assert list(o1["Patient"]) == list(o2["Patient"]) and (o1["nCount_RNA"].to_numpy() == o2["nCount_RNA"].to_numpy()).all()


def test_integrity_rejects_integer_counts(cfg):
    X, obs, genes = read_source(cfg["data"])
    bad = X.copy()
    bad.data = np.round(bad.data).astype(np.float32)
    with pytest.raises(AssertionError):
        check_integrity(bad, obs, genes)


def test_integrity_rejects_patient_with_two_types(cfg):
    X, obs, genes = read_source(cfg["data"])
    obs = obs.copy()
    first = obs.index[obs[GROUP] == "GBM1"][0]
    obs.loc[first, "glioma_type"] = "LGG"
    with pytest.raises(AssertionError):
        check_integrity(X, obs, genes)


def test_leakage_report_flags_redundant_columns(cfg):
    _, obs, _ = read_source(cfg["data"])
    rep = leakage_report(obs, "glioma_type").set_index("column")
    assert rep.loc["PATH", "note"] == "DETERMINA el objetivo"
    assert rep.loc["Patient", "note"] == "DETERMINA el objetivo"   # el paciente implica el tipo
    assert rep.loc["ECclusters", "purity"] < 1.0


def test_library_size_check_on_log_normalised(cfg):
    X, _, _ = read_source(cfg["data"])
    lib = library_size_check(X)
    assert lib["consistent_with_log1p_fixed_total"] and abs(lib["median_expm1_rowsum"] - 1e4 - X.shape[1] * 0) < 50
