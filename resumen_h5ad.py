import sys
import numpy as np
import anndata as ad
a = ad.read_h5ad(sys.argv[1], backed="r")
print("Forma (celulas x genes):", a.shape)
print("Tipo de X:", type(a.X))
print("Capas:", list(a.layers.keys()))
print("obs (columnas por celula):", list(a.obs.columns))
print("var (columnas por gen):", list(a.var.columns))
print("obsm:", list(a.obsm.keys()), "| uns:", list(a.uns.keys()))
print("\nPrimeras filas de obs:\n", a.obs.head(5).to_string())
print("\nPrimeras filas de var:\n", a.var.head(5).to_string())
sub = a.X[:200, :]
sub = sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)
print("\nValores de X (200 celulas): min", sub.min(), "max", sub.max(), "| son enteros:", bool(np.allclose(sub, sub.round())))
for c in a.obs.columns:
    s = a.obs[c]
    if s.nunique() <= 30:
        print("\n--", c, "--")
        print(s.value_counts(dropna=False).to_string())
    else:
        print("\n--", c, "--", s.nunique(), "valores distintos")
