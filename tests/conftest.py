import pytest

from tests.synthetic import make_synthetic


@pytest.fixture(scope="session")
def synth_path(tmp_path_factory):
    return make_synthetic(tmp_path_factory.mktemp("synth"))


@pytest.fixture(scope="session")
def cfg(synth_path, tmp_path_factory):
    return {
        "mode": "demo", "seed": 42, "dataset": "synthetic",
        "results_dir": str(tmp_path_factory.mktemp("results")),
        "data": {"path": str(synth_path), "cache": False, "patients_per_class": None, "max_cells_per_patient": None},
        "tasks": ["glioma_type"],
        "rom": {"hvg": 60, "components": [2, 5], "include_all_genes": True, "scale_clip": 10.0,
                "regress_out": ["nCount_RNA", "nFeature_RNA"]},
        "ml": {"max_train_cells_per_patient": 50, "rf_trees": 20, "control": True, "by_cluster": True},
        "permutation": {"n": "all", "representations": ["PCA-2", "PCA-5"], "chunk": 20, "n_jobs": 1},
        "models": ["logistic_regression", "random_forest"],
    }


@pytest.fixture(scope="session")
def dataset_and_folds(cfg, tmp_path_factory):
    from src.features import prepare
    return prepare(cfg, tmp_path_factory.mktemp("prep"))
