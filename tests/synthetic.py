"""AnnData sintético con la estructura del objeto real (log-normalizado, 'Patient', 'ECclusters'...)."""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

CLUSTERS = ["Capillary", "Venule", "Arteriole", "Angiogenic capillary"]


def make_synthetic(root, n_gbm=5, n_lgg=4, n_genes=300, base_cells=60, seed=0) -> Path:
    rng = np.random.default_rng(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    patients = [f"GBM{i + 1}" for i in range(n_gbm)] + [f"LGG{i + 1}" for i in range(n_lgg)]
    cells_per = {p: base_cells + int(rng.integers(0, 40)) for p in patients}
    cells_per[patients[0]] = 220  # un paciente grande, como GBM5 en el dato real
    ages = dict(zip(patients, list(rng.choice(np.arange(48, 83), n_gbm, replace=False))
                    + list(rng.choice(np.arange(25, 47), n_lgg, replace=False))))

    base = rng.gamma(0.6, 0.6, n_genes)                       # media por gen (muchos ceros)
    signal = np.zeros(n_genes)
    signal[:40] = 1.0                                         # genes que distinguen GBM de LGG
    clus_eff = rng.normal(0, 0.4, (len(CLUSTERS), n_genes)) * (rng.random((len(CLUSTERS), n_genes)) < 0.1)
    rows, obs_rows = [], []
    for p in patients:
        is_gbm = p.startswith("GBM")
        pat_eff = rng.normal(0, 0.3, n_genes) * (rng.random(n_genes) < 0.2)  # efecto de paciente (lote)
        for k in range(cells_per[p]):
            c = int(rng.choice(len(CLUSTERS), p=[0.4, 0.2, 0.1, 0.3] if is_gbm else [0.3, 0.3, 0.3, 0.1]))
            mu = base * np.exp(0.9 * signal * is_gbm + pat_eff + clus_eff[c])
            counts = rng.poisson(mu * rng.uniform(0.7, 1.3) * 2.0)
            if counts.sum() == 0:
                counts[0] = 1
            rows.append(np.log1p(counts / counts.sum() * 1e4).astype(np.float32))
            obs_rows.append(dict(
                cell=f"{p[:3]}_{p}_{k:04d}-1", ECclusters=CLUSTERS[c], PATH="GBM" if is_gbm else "LGG", Patient=p,
                sex="Male" if sum(map(ord, p)) % 2 else "Female", age=f"{ages[p]} years",
                glioma_type="GBM" if is_gbm else "LGG", nCount_RNA=float(counts.sum()),
                nFeature_RNA=int((counts > 0).sum()),
            ))
    X = sp.csc_matrix(np.vstack(rows))
    obs = pd.DataFrame(obs_rows).set_index("cell")
    genes = [f"GENE{i:04d}" for i in range(n_genes)]
    path = root / "synthetic.h5ad"
    ad.AnnData(X=X, obs=obs, var=pd.DataFrame({"name": genes}, index=genes)).write_h5ad(path)
    return path
