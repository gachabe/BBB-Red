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


def _describe(outdir, png_name: str, lines) -> None:
    """Escribe results/<modo>/descriptions/<nombre_de_la_figura>.txt con lo que expresa la figura y sus cifras clave."""
    d = Path(outdir) / "descriptions"
    d.mkdir(parents=True, exist_ok=True)
    (d / (Path(png_name).stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")


def describe_figure(outdir, png_name: str, lines) -> None:
    _describe(outdir, png_name, lines)


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
    lines = ["01_cohort.png",
             "Qué expresa: la cohorte. Izquierda: células endoteliales por paciente (escala logarítmica), coloreadas por tipo de glioma. "
             "Derecha: composición de cada paciente por tipo de célula endotelial (ECclusters), como fracción de sus células.", ""]
    for c in classes:
        g = d[d["glioma_type"] == c]
        pooled = (g[ec].sum() / g[ec].sum().sum()).sort_values(ascending=False).head(3)
        lines.append(f"{c}: {len(g)} pacientes, {int(g['n_cells'].sum()):,} células; células por paciente de {int(g['n_cells'].min()):,} "
                     f"({g.loc[g['n_cells'].idxmin(), 'Patient']}) a {int(g['n_cells'].max()):,} ({g.loc[g['n_cells'].idxmax(), 'Patient']}). "
                     "Clústeres más abundantes: " + "; ".join(f"{k[3:]} {v:.1%}" for k, v in pooled.items()) + ".")
    lines.append("Lectura: los pacientes aportan números de células muy distintos y la composición por clúster difiere entre clases, "
                 "por eso las métricas pesan igual a cada paciente y se mira el acierto dentro de cada clúster.")
    _describe(outdir, "01_cohort.png", lines)
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
    lines = ["02_pca_variance.png",
             "Qué expresa: varianza acumulada que explican las primeras componentes de la PCA (sobre los genes variables escalados), "
             "media sobre las rondas de validación (banda = mínimo y máximo entre rondas). Las líneas verticales marcan los tamaños de PCA evaluados.", ""]
    for r in components:
        if r in g.index:
            lines.append(f"{r} componentes: {g.loc[r, 'mean']:.1%} de la varianza (rango entre rondas {g.loc[r, 'min']:.1%} a {g.loc[r, 'max']:.1%}).")
    lines.append("Lectura: pocas componentes explican una parte pequeña de la varianza total; la señal es difusa, no está concentrada en 2 ejes.")
    _describe(outdir, "02_pca_variance.png", lines)
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
    lines = [f"03_confusion_{task}.png",
             f"Qué expresa: matrices de confusión (filas = clase real, columnas = predicha; normalizadas por filas; cada paciente pesa igual) "
             f"para cada modelo y representación. La diagonal es la proporción de acierto de cada clase ({classes[0]} y {classes[1]}).", ""]
    for (mo, rp), cm in cms.items():
        cm = np.asarray(cm, dtype=float)
        rec = np.diag(cm / cm.sum(axis=1, keepdims=True).clip(1e-12))
        lines.append(f"{_name(mo)} | {rp}: acierto {classes[0]} {rec[0]:.2f}, acierto {classes[1]} {rec[1]:.2f}")
    _describe(outdir, f"03_confusion_{task}.png", lines)
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
    lines = [f"04_components_vs_performance_{task}.png",
             "Qué expresa: rendimiento (Macro-F1 con cada paciente pesando igual, y acierto por paciente por voto mayoritario) según la "
             "representación, ordenada de menos a más características. Línea continua = datos reales; discontinua = control con etiquetas permutadas "
             "(debería quedar en torno al azar).", ""]
    for (model, variant), g2 in results.groupby(["model", "variant"], sort=False):
        g2 = g2.set_index("representation").loc[order]
        lines.append(f"{_name(model)} / {variant}: " + "; ".join(
            f"{r} F1 {row['macro_f1']:.3f}" + (f" (pac. {row['patient_acc']:.3f})" if pd.notna(row.get("patient_acc")) else "")
            for r, row in g2.iterrows()))
    _describe(outdir, f"04_components_vs_performance_{task}.png", lines)
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
    lines = [f"06_components_vs_performance_seeds_{task}.png",
             "Qué expresa: igual que la figura 04 pero con la media ± desviación sobre varias semillas. La desviación mide el ruido del "
             "submuestreo de células y del random forest, NO la incertidumbre por tener pocos pacientes.", ""]
    for (model, variant), g2 in agg.groupby(["model", "variant"], sort=False):
        g2 = g2.set_index("representation").loc[order]
        lines.append(f"{_name(model)} / {variant}: " + "; ".join(
            f"{r} F1 {row['macro_f1_mean']:.3f} ± {0 if pd.isna(row['macro_f1_std']) else row['macro_f1_std']:.3f}" for r, row in g2.iterrows()))
    _describe(outdir, f"06_components_vs_performance_seeds_{task}.png", lines)
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
    lines = [f"05_per_patient_{task}.png",
             "Qué expresa: fracción de células de cada paciente (filas) bien clasificadas con cada representación (columnas), para cada modelo. "
             "Verde = acierto alto, rojo = bajo. Entre paréntesis, la clase real del paciente.", ""]
    for mo, g2 in d.groupby("model", sort=False):
        mean_acc = g2.groupby("Patient")["accuracy"].mean()
        bad = [f"{p} ({lab[p]}) {v:.2f}" for p, v in mean_acc.sort_values().items() if v < 0.5]
        lines.append(f"{_name(mo)}: acierto medio por paciente (media sobre representaciones) entre {mean_acc.min():.2f} ({mean_acc.idxmin()}) y "
                     f"{mean_acc.max():.2f} ({mean_acc.idxmax()}). Pacientes por debajo de 0,50: " + ("; ".join(bad) if bad else "ninguno") + ".")
    _describe(outdir, f"05_per_patient_{task}.png", lines)


def fig_by_cluster(bc: pd.DataFrame, task: str, outdir: Path) -> None:
    d = bc[bc["variant"] == "real"]
    _heatmap_panels(d, "cluster", "representation", "balanced_accuracy", sorted(d["cluster"].unique()),
                    lambda rows: list(rows), "Balanced accuracy within cluster", outdir / f"07_by_cluster_{task}.png")
    lines = [f"07_by_cluster_{task}.png",
             "Qué expresa: balanced accuracy dentro de cada tipo de célula endotelial (ECclusters), para cada representación y modelo; "
             "responde a si la señal GBM/LGG persiste comparando solo células del mismo tipo. n/a = el clúster no tiene pacientes de ambas clases.", ""]
    for mo, g2 in d.groupby("model", sort=False):
        m = g2.groupby("cluster")["balanced_accuracy"].mean().dropna().sort_values(ascending=False)
        lines.append(f"{_name(mo)} (media sobre representaciones): " + "; ".join(f"{k} {v:.2f}" for k, v in m.items()))
    _describe(outdir, f"07_by_cluster_{task}.png", lines)


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
    lines = [f"08_permutation_{task}.png",
             "Qué expresa: distribución nula del mejor Macro-F1 entre representaciones cuando las etiquetas se permutan entre pacientes "
             "(todas las asignaciones posibles). Las líneas verticales son el Macro-F1 real de cada representación; "
             "cuanto más a la derecha del grueso del histograma, menos probable por azar.", "",
             f"Asignaciones: {len(null_max)}; mejor F1 nulo: media {np.mean(null_max):.3f}, percentil 95 {np.percentile(null_max, 95):.3f}, máximo {np.max(null_max):.3f}."]
    for rn, v in real.items():
        lines.append(f"{rn}: F1 real {v:.3f}; p ajustado (proporción de asignaciones con mejor F1 >= real) = {(np.asarray(null_max) >= v - 1e-12).mean():.3f}")
    _describe(outdir, f"08_permutation_{task}.png", lines)
    fig.savefig(outdir / f"08_permutation_{task}.png", dpi=130)
    plt.close(fig)


STUDY_COLORS = {"Walchli": "#0072B2", "Xie": "#E69F00"}
GLIOMA_COLORS = {"GBM": "#D55E00", "LGG": "#009E73"}


def _scatter_two_panels(df, xcol, ycol, xlabel, ylabel, title, path, boundary=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    order = df.sample(frac=1.0, random_state=0)
    for s_, c in STUDY_COLORS.items():
        g = order[order["study"] == s_]
        ax[0].scatter(g[xcol], g[ycol], s=3, alpha=0.25, color=c, label=s_, rasterized=True)
    for gt, c in GLIOMA_COLORS.items():
        g = order[order["glioma_type"] == gt]
        for s_, mk in (("Walchli", "o"), ("Xie", "^")):
            h = g[g["study"] == s_]
            ax[1].scatter(h[xcol], h[ycol], s=3, alpha=0.25, color=c, marker=mk, label=f"{gt} ({s_})", rasterized=True)
    ax[0].set_title("Colour = study", fontsize=10)
    ax[1].set_title("Colour = glioma type (circle = Walchli, triangle = Xie)", fontsize=10)
    for a in ax:
        a.set(xlabel=xlabel)
        if boundary is not None:
            w, b = boundary
            xs, ylim = np.array(a.get_xlim()), a.get_ylim()
            if abs(w[1]) > 1e-9:
                a.plot(xs, -(w[0] * xs + b) / w[1], "k--", lw=1.2, label="decision boundary (final logistic regression)")
            elif abs(w[0]) > 1e-9:
                a.axvline(-b / w[0], color="k", ls="--", lw=1.2, label="decision boundary (final logistic regression)")
            a.set_xlim(xs)
            a.set_ylim(ylim)                  # la frontera no debe cambiar la escala de los puntos
        lg = a.legend(fontsize=8, markerscale=4, loc="best")
    ax[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_batch_model_space(df, boundary, classes, outdir: Path) -> None:
    name = "09_batch_model_space.png"
    _scatter_two_panels(df, "PC1", "PC2", "PC1 (PCA fitted on Walchli)", "PC2",
                        "Cells in the space seen by the trained PCA-2 model", Path(outdir) / name, boundary)
    lines = [name,
             "Qué expresa: cada punto es una célula endotelial proyectada sobre las 2 componentes de la PCA-2 del modelo final, "
             "que se ajustó solo con Walchli. Izquierda: color = estudio. Derecha: color = tipo de glioma. "
             "La línea discontinua es la frontera de decisión de la regresión logística final: a un lado predice "
             f"{classes[1]} y al otro {classes[0]}. Si Xie ocupa otra zona que Walchli, el modelo se enfrenta a datos fuera de lo que aprendió.", ""]
    for s_, g in df.groupby("study"):
        lines.append(f"{s_}: PC1 media {g['PC1'].mean():.2f} (sd {g['PC1'].std():.2f}), PC2 media {g['PC2'].mean():.2f} (sd {g['PC2'].std():.2f}); "
                     f"{(g['side'] == classes[1]).mean():.1%} de las células caen en el lado {classes[1]}.")
    for (s_, gt), g in df.groupby(["study", "glioma_type"]):
        lines.append(f"  {s_} / {gt}: {(g['side'] == classes[1]).mean():.1%} en el lado {classes[1]} ({len(g):,} células)"
                     + (" [datos de entrenamiento, descriptivo]" if s_ == "Walchli" else " [datos externos]"))
    w_, x_ = df[df["study"] == "Walchli"], df[df["study"] == "Xie"]
    for c in ("PC1", "PC2"):
        sd = np.sqrt((w_[c].var() + x_[c].var()) / 2)
        lo, hi = np.percentile(w_[c], [2.5, 97.5])
        out = ((x_[c] < lo) | (x_[c] > hi)).mean()
        lines.append(f"{c}: diferencia Xie - Walchli = {(x_[c].mean() - w_[c].mean()) / sd:+.2f} desviaciones; "
                     f"{out:.1%} de las células de Xie quedan fuera del rango central (95 %) de Walchli (esperado por azar: 5 %).")
    _describe(outdir, name, lines)


def fig_batch_joint(df, eta, var_expl, outdir: Path) -> None:
    name = "10_batch_joint_pca.png"
    _scatter_two_panels(df, "PC1", "PC2", "PC1 (PCA fitted on both studies)", "PC2",
                        "Cells in a PCA fitted on Walchli and Xie together", Path(outdir) / name)
    lines = [name,
             "Qué expresa: PCA ajustada con las células de ambos estudios a la vez (solo para visualizar). Izquierda: color = estudio; "
             "si los colores se separan, el estudio es una dirección principal de variación. Derecha: color = tipo de glioma.",
             f"Las 5 primeras componentes explican {var_expl:.1%} de la varianza.", "",
             "Porcentaje de varianza de cada componente asociado a cada variable (eta²; el paciente está anidado en el estudio, por lo que su eta² siempre es mayor):"]
    for r in eta.itertuples():
        lines.append(f"  {r.component}: estudio {r.eta2_study:.3f} | tipo de glioma {r.eta2_glioma_type:.3f} | paciente {r.eta2_patient:.3f}")
    _describe(outdir, name, lines)


def fig_batch_domain(res, depth, outdir: Path) -> None:
    name = "11_batch_domain_classifier.png"
    reps = res.drop_duplicates("representation").sort_values("n_features", kind="stable")["representation"].tolist()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for (m, rg), g in res.groupby(["model", "regress_depth"], sort=False):
        g = g.set_index("representation").loc[reps]
        for a, col in zip(ax, ("balanced_accuracy", "auc")):
            a.plot(range(len(reps)), g[col].to_numpy(), marker="o", ls="-" if rg else "--",
                   label=f"{_name(m)} / {'with' if rg else 'without'} depth regression")
    for a, col, lab in zip(ax, ("balanced_accuracy", "auc"), ("Balanced accuracy (study, new patients)", "AUC (study, new patients)")):
        a.axhline(0.5, color="gray", ls=":", label="chance")
        a.axhline(depth[col], color="red", ls="-.", lw=1, label="depth only (nCount, nFeature)")
        a.set_xticks(range(len(reps)), reps, fontsize=8)
        a.set(ylabel=lab, ylim=(0.4, 1.02), xlabel="Representation")
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    _legend_below(fig, ax[0], ncol=3)
    fig.savefig(Path(outdir) / name, dpi=130)
    plt.close(fig)
    lines = [name,
             "Qué expresa: capacidad de un clasificador para adivinar el ESTUDIO (Walchli o Xie) de las células de un paciente que no ha visto "
             "(dejar un paciente fuera: una ronda por paciente de ambos estudios), según la representación. 0,5 = azar; 1,0 = el estudio se identifica sin error. "
             "Línea roja: lo mismo usando solo la profundidad de secuenciación. Continua = con regresión de profundidad; discontinua = sin ella.", ""]
    for (m, rg), g in res.groupby(["model", "regress_depth"], sort=False):
        g = g.set_index("representation").loc[reps]
        lines.append(f"{_name(m)} / {'con' if rg else 'sin'} regresión de profundidad: " + "; ".join(
            f"{r} bal.acc {row['balanced_accuracy']:.3f}, AUC {row['auc']:.3f}, pacientes {row['patient_acc']:.2f}" for r, row in g.iterrows()))
    lines.append(f"Solo profundidad: bal.acc {depth['balanced_accuracy']:.3f}, AUC {depth['auc']:.3f}, pacientes {depth['patient_acc']:.2f}.")
    lines.append("Lectura: valores cercanos a 1,0 indican un efecto de estudio fuerte (batch) que un modelo puede aprovechar en vez de la biología.")
    _describe(outdir, name, lines)
