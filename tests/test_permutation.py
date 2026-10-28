import importlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.permutation import enumerate_assignments, p_values, stats_from_fraction


def test_enumeration_exact_and_montecarlo():
    combos, exact = enumerate_assignments(9, 4, "all", 0, (0, 1, 2, 3))
    assert exact and len(combos) == math.comb(9, 4) == len(set(combos)) and (0, 1, 2, 3) in combos
    mc, exact2 = enumerate_assignments(9, 4, 20, 0, (0, 1, 2, 3))
    assert not exact2 and len(mc) == 20 and (0, 1, 2, 3) not in mc and len(set(mc)) == 20


def test_stats_and_pvalues():
    a = np.array([0, 0, 1, 1])
    assert stats_from_fraction(np.array([0.0, 0.0, 1.0, 1.0]), a) == (1.0, 1.0, 1.0)
    m, b, acc = stats_from_fraction(np.array([1.0, 1.0, 0.0, 0.0]), a)
    assert m == 0.0 and acc == 0.0
    null = np.array([[0.2], [0.5], [0.9], [0.7]])
    p, pm = p_values(np.array([0.9]), null, exact=True)
    assert p[0] == 0.25 and pm[0] == 0.25
    p2, _ = p_values(np.array([0.9]), null[:3], exact=False)
    assert p2[0] == (1 + 1) / (1 + 3)


def test_permutation_pipeline_matches_real_and_resumes(cfg):
    ml = importlib.import_module("experiments.03_ml")
    perm = importlib.import_module("experiments.06_permutation")
    final = importlib.import_module("experiments.04_final")
    final.main(cfg)  # resultados reales (con etiquetas reales)
    real = pd.read_csv(Path(cfg["results_dir"]) / "results_glioma_type.csv")
    real = real[(real["variant"] == "real") & (real["model"] == "logistic_regression")].set_index("representation")
    t = perm.main(cfg).set_index("representation")
    # la asignación real, con el mismo procedimiento, reproduce el análisis real
    for rn in ("PCA-2", "PCA-5"):
        assert abs(t.loc[rn, "real_macro_f1"] - real.loc[rn, "macro_f1"]) < 1e-6
        assert abs(t.loc[rn, "real_patient_acc"] - real.loc[rn, "patient_acc"]) < 1e-9
    assert t["exact"].all() and (t["n_assignments"] == math.comb(9, 4)).all()
    assert ((t["p_unadjusted"] > 0) & (t["p_unadjusted"] <= 1)).all()
    assert (t["p_adjusted_max"] >= t["p_unadjusted"] - 1e-12).all()
    assert t["p_unadjusted"].min() < 0.2                      # hay señal en el sintético
    assert (Path(cfg["results_dir"]) / "08_permutation_glioma_type.png").exists()
    # reanudar: no recalcula nada y da lo mismo
    n_rows = len(pd.read_csv(Path(cfg["results_dir"]) / "permutation" / "perm_stats.csv"))
    t2 = perm.main(cfg)
    assert len(pd.read_csv(Path(cfg["results_dir"]) / "permutation" / "perm_stats.csv")) == n_rows
    assert np.allclose(t2["p_unadjusted"].to_numpy(), t["p_unadjusted"].to_numpy())
