"""Métricas, tablas y figuras."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

MODEL_NAMES = {"logistic_regression": "Logistic regression", "random_forest": "Random forest"}


def _name(model: str) -> str:
    return MODEL_NAMES.get(model, model)


# --------------------------------------------------------------------------- métricas


def classification_metrics(y_true, y_pred, classes, weights=None) -> dict:
    """Macro-F1 y balanced accuracy con pesos opcionales (peso 1/nº de células del paciente = cada paciente cuenta igual)."""
    assert len(y_true) == len(y_pred)
    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels, sample_weight=weights)
    tp, support, predicted = np.diag(cm), cm.sum(axis=1), cm.sum(axis=0)
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros(len(labels)), where=denom > 0)
    recall = np.divide(tp, support, out=np.zeros(len(labels)), where=support > 0)
    m = {
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall[support > 0].mean()),
        **{f"f1_{c}": float(v) for c, v in zip(classes, f1)},
    }
    assert all(np.isfinite(v) for v in m.values())
    return m


def weighted_confusion(y_true, y_pred, classes, weights=None) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(len(classes))), sample_weight=weights)


def per_patient_table(y_true, y_pred, patients, classes) -> pd.DataFrame:
    """Acierto por célula dentro de cada paciente y predicción mayoritaria del paciente."""
    df = pd.DataFrame({"Patient": patients, "y": y_true, "p": y_pred})
    rows = []
    for pat, g in df.groupby("Patient", sort=True):
        rows.append(
            dict(
                Patient=pat,
                label=classes[int(g["y"].mode().iloc[0])],
                pred_majority=classes[int(g["p"].mode().iloc[0])],
                n_cells=len(g),
                accuracy=float((g["y"] == g["p"]).mean()),
            )
        )
    return pd.DataFrame(rows)


def cluster_breakdown(y_true, y_pred, patients, clusters, classes) -> pd.DataFrame:
    """Balanced accuracy dentro de cada tipo de célula endotelial (ECclusters), cada paciente pesa igual."""
    df = pd.DataFrame({"Patient": patients, "y": y_true, "p": y_pred, "cluster": clusters})
    rows = []
    for cl, g in df.groupby("cluster", sort=True):
        w = 1.0 / g.groupby("Patient")["y"].transform("size").to_numpy()
        recalls = []
        for k in range(len(classes)):
            sel = (g["y"] == k).to_numpy()
            if sel.any():
                recalls.append(float(w[sel & (g["p"] == k).to_numpy()].sum() / w[sel].sum()))
        n_pat = g.groupby("y")["Patient"].nunique()
        rows.append(
            dict(
                cluster=cl,
                n_cells=len(g),
                **{f"n_patients_{classes[k]}": int(n_pat.get(k, 0)) for k in range(len(classes))},
                balanced_accuracy=float(np.mean(recalls)) if len(recalls) == len(classes) else np.nan,
            )
        )
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "--" if np.isnan(v) else f"{v:.3f}")
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join(map(str, r)) + " |" for r in d.to_numpy()]
    return "\n".join([head, sep, *body])


def save_tables(results: pd.DataFrame, task: str, outdir: Path) -> None:
    results.to_csv(outdir / f"results_{task}.csv", index=False)
    f1_cols = [c for c in results.columns if c.startswith("f1_")]
    cols = ["representation", "n_features", "macro_f1", "balanced_accuracy", "patient_acc",
            "cell_macro_f1", *f1_cols, "explained_variance"]
    parts = []
    for (model, variant), g in results.groupby(["model", "variant"], sort=False):
        parts.append(f"### {task} — {model} — {variant}\n\n" + to_markdown(g[cols]))
    (outdir / f"summary_{task}.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- figuras


def _legend_below(fig, ax, ncol: int = 3) -> None:
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncol, fontsize=8, frameon=False)


def _rep_label(name: str, n: int) -> str:
    return f"All genes\n({n:,})" if name == "All genes" else name


def _rep_order(df: pd.DataFrame):
    d = df.drop_duplicates("representation")[["representation", "n_features"]].sort_values("n_features", kind="stable")
    return list(d["representation"]), dict(zip(d["representation"], d["n_features"]))


def fig_cohort(cohort: pd.DataFrame, outdir: Path) -> None:
    """cohort: Patient, glioma_type, n_cells, columnas 'ec:<cluster>' con nº de células."""
    d = cohort.sort_values(["glioma_type", "Patient"]).reset_index(drop=True)
    ec = [c for c in d.columns if c.startswith("ec:")]
    classes = sorted(d["glioma_type"].unique())
    colors = dict(zip(classes, plt.cm.Set1(range(len(classes)))))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].bar(d["Patient"], d["n_cells"], color=[colors[c] for c in d["glioma_type"]])
    ax[0].set(yscale="log", ylabel="Endothelial cells per patient (log scale)", xlabel="Patient")
    ax[0].tick_params(axis="x", labelrotation=60)
    ax[0].legend([plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in classes], classes, title="Glioma type")
    frac = d[ec].div(d[ec].sum(axis=1), axis=0)
    bottom = np.zeros(len(d))
    cmap = plt.cm.tab20(np.linspace(0, 1, len(ec)))
    for c, col in zip(ec, cmap):
        ax[1].bar(d["Patient"], frac[c], bottom=bottom, color=col, label=c[3:])
        bottom += frac[c].to_numpy()
    ax[1].set(ylabel="Fraction of cells", xlabel="Patient", ylim=(0, 1))
    ax[1].tick_params(axis="x", labelrotation=60)
    ax[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), title="ECclusters")
    fig.tight_layout()
    fig.savefig(outdir / "01_cohort.png", dpi=130)
    plt.close(fig)


def fig_pca_variance(var_df: pd.DataFrame, components, outdir: Path) -> None:
    """var_df: held_out, component, ratio (una PCA por ronda)."""
    cum = var_df.sort_values(["held_out", "component"]).assign(cum=lambda d: d.groupby("held_out")["ratio"].cumsum())
    g = cum.groupby("component")["cum"].agg(["mean", "min", "max"])
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.plot(g.index, g["mean"], "o-", ms=3, label="Mean over folds")
    ax.fill_between(g.index, g["min"], g["max"], alpha=0.2, label="Min–max over folds")
    for r in components:
        ax.axvline(r, color="gray", ls=":")
    ax.set(xlabel="Number of components", ylabel="Cumulative explained variance (scaled HVG)", ylim=(0, 1.02))
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "02_pca_variance.png", dpi=130)
    plt.close(fig)


def fig_confusions(cms: dict, classes, task: str, outdir: Path) -> None:
    """cms[(model, representation)] = matriz 2x2 (con pesos); se normaliza por filas."""
    models = list(dict.fromkeys(k[0] for k in cms))
    reps = list(dict.fromkeys(k[1] for k in cms))
    fig, ax = plt.subplots(len(models), len(reps), figsize=(2.7 * len(reps), 2.9 * len(models)), squeeze=False)
    for i, mo in enumerate(models):
        for j, rp in enumerate(reps):
            cm = np.asarray(cms[(mo, rp)], dtype=float)
            norm = cm / cm.sum(axis=1, keepdims=True).clip(1e-12)
            a = ax[i, j]
            a.imshow(norm, vmin=0, vmax=1, cmap="Blues")
            for (u, v), val in np.ndenumerate(norm):
                a.text(v, u, f"{val:.2f}", ha="center", va="center", fontsize=8)
            a.set_xticks(range(len(classes)), classes, fontsize=8)
            a.set_yticks(range(len(classes)), classes, fontsize=8)
            a.set_title(f"{_name(mo)}\n{rp}", fontsize=8)
            if i == len(models) - 1:
                a.set_xlabel("Predicted class", fontsize=8)
            if j == 0:
                a.set_ylabel("True class", fontsize=8)
    fig.suptitle(f"{task}: confusion matrices (rows normalised; each patient weighted equally)", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / f"03_confusion_{task}.png", dpi=130)
    plt.close(fig)


def _components_axes(ax, order, n_feat) -> None:
    for a in ax:
        a.set_xticks(range(len(order)), [_rep_label(r, n_feat[r]) for r in order], fontsize=8)
        a.set_xlabel("Representation (increasing number of features)")
        a.set_ylim(0, 1.02)


def fig_components_vs_performance(results: pd.DataFrame, task: str, outdir: Path) -> None:
    order, n_feat = _rep_order(results)
    panels = [("macro_f1", "Macro-F1 (each patient weighted equally)")]
    if results["patient_acc"].notna().any():
        panels.append(("patient_acc", "Patient-level accuracy (majority vote)"))
    fig, ax = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5), squeeze=False, sharex=True)
    ax = ax[0]
    for (model, variant), g in results.groupby(["model", "variant"], sort=False):
        g = g.set_index("representation").loc[order]
        for a, (col, _) in zip(ax, panels):
            a.plot(range(len(order)), g[col].to_numpy(), marker="o", ls="--" if variant == "permuted" else "-",
                   label=f"{_name(model)} / {variant}")
    _components_axes(ax, order, n_feat)
    for a, (_, lab) in zip(ax, panels):
        a.set_ylabel(lab)
    fig.suptitle(task, fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    _legend_below(fig, ax[0])
    fig.savefig(outdir / f"04_components_vs_performance_{task}.png", dpi=130)
    plt.close(fig)


def fig_components_seeds(agg: pd.DataFrame, task: str, outdir: Path) -> None:
    order, n_feat = _rep_order(agg)
    panels = [("macro_f1", "Macro-F1 (mean ± std over seeds)")]
    if agg["patient_acc_mean"].notna().any():
        panels.append(("patient_acc", "Patient-level accuracy (mean ± std over seeds)"))
    fig, ax = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5), squeeze=False, sharex=True)
    ax = ax[0]
    for (model, variant), g in agg.groupby(["model", "variant"], sort=False):
        g = g.set_index("representation").loc[order]
        for a, (col, _) in zip(ax, panels):
            a.errorbar(range(len(order)), g[f"{col}_mean"], yerr=g[f"{col}_std"].fillna(0), marker="o", capsize=3,
                       ls="--" if variant == "permuted" else "-", label=f"{_name(model)} / {variant}")
    _components_axes(ax, order, n_feat)
    for a, (_, lab) in zip(ax, panels):
        a.set_ylabel(lab)
    fig.suptitle(task, fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    _legend_below(fig, ax[0])
    fig.savefig(outdir / f"06_components_vs_performance_seeds_{task}.png", dpi=130)
    plt.close(fig)


def _heatmap_panels(d, row_key, col_key, val_key, row_order, title_for, cbar_label, path, fmt_nan=False):
    models = list(dict.fromkeys(d["model"]))
    reps = list(dict.fromkeys(d[col_key]))
    fig, ax = plt.subplots(1, len(models), figsize=(1.0 * len(reps) * len(models) + 4, 0.42 * len(row_order) + 2),
                           squeeze=False)
    im = None
    for a, mo in zip(ax[0], models):
        piv = d[d["model"] == mo].pivot(index=row_key, columns=col_key, values=val_key).loc[row_order, reps]
        arr = piv.to_numpy(dtype=float)
        im = a.imshow(arr, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        for (u, v), val in np.ndenumerate(arr):
            a.text(v, u, "n/a" if np.isnan(val) else f"{val:.2f}", ha="center", va="center", fontsize=7)
        a.set_xticks(range(len(reps)), reps, rotation=40, ha="right", fontsize=8)
        a.set_yticks(range(len(row_order)), title_for(row_order), fontsize=8)
        a.set_title(_name(mo), fontsize=9)
    fig.colorbar(im, ax=ax[0].tolist(), shrink=0.6, label=cbar_label)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_per_patient(pp: pd.DataFrame, task: str, outdir: Path) -> None:
    d = pp[pp["variant"] == "real"]
    order = d.drop_duplicates("Patient").sort_values(["label", "Patient"])
    lab = dict(zip(order["Patient"], order["label"]))
    _heatmap_panels(d, "Patient", "representation", "accuracy", list(order["Patient"]),
                    lambda rows: [f"{p} ({lab[p]})" for p in rows], "Fraction of cells correctly classified",
                    outdir / f"05_per_patient_{task}.png")


def fig_by_cluster(bc: pd.DataFrame, task: str, outdir: Path) -> None:
    d = bc[bc["variant"] == "real"]
    _heatmap_panels(d, "cluster", "representation", "balanced_accuracy", sorted(d["cluster"].unique()),
                    lambda rows: list(rows), "Balanced accuracy within cluster", outdir / f"07_by_cluster_{task}.png")


def fig_permutation(null_max: np.ndarray, real: dict, task: str, outdir: Path) -> None:
    """null_max: mejor Macro-F1 entre representaciones en cada permutación; real: {representación: Macro-F1 real}."""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.hist(null_max, bins=30, color="lightgray", edgecolor="gray")
    for (rn, v), c in zip(real.items(), plt.cm.tab10(range(len(real)))):
        ax.axvline(v, color=c, label=f"{rn} (real {v:.3f})")
    ax.set(xlabel="Best Macro-F1 across representations (patient-level permutations)",
           ylabel="Number of label assignments")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / f"08_permutation_{task}.png", dpi=130)
    plt.close(fig)
