"""Hiperparámetros con validación anidada + modelo final ajustado con los 14 pacientes.

1) Fuera: dejar un paciente fuera (14 rondas). Dentro de cada una, los hiperparámetros se eligen con LOPO interno
   solo entre los otros pacientes -> estimación honrada del rendimiento (results/<modo>/tuning/).
2) Final: LOPO interno con los 14 pacientes, elección de hiperparámetros y bundle en models/<modo>_tuned/.
Se puede interrumpir y relanzar: reanuda por rondas (results/<modo>/tuning/outer_<paciente>.pkl).
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import GROUP, build_dataset, load_config
from src.evaluation import classification_metrics, per_patient_table, to_markdown
from src.features import task_is_patient_constant
from src.final_model import save_bundle
from src.tuning import (fit_final, hp_key, inner_scores, make_bundle, make_ctx, predict_patient, select_best)

TASK = "glioma_type"


def main(cfg: dict) -> pd.DataFrame:
    out = Path(cfg.get("results_dir", f"results/{cfg['mode']}")) / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    ds = build_dataset(cfg)
    assert task_is_patient_constant(ds.meta, TASK)
    ds.meta.attrs["genes"] = np.asarray(ds.genes).astype(str)
    ctx = make_ctx(ds, cfg)
    pats = sorted(ctx.full)
    print(f"{len(pats)} pacientes; rejilla: {cfg.get('tuning', 'por defecto')}")

    # 1) validación anidada
    for q in pats:
        f = out / f"outer_{q}.pkl"
        if f.exists():
            continue
        t0 = time.time()
        train_p = [p for p in pats if p != q]
        print(f"[externa] paciente de test {q}: eligiendo hiperparámetros con los otros {len(train_p)}", flush=True)
        scores = inner_scores(ctx, train_p)
        chosen = select_best(scores, cfg)
        reps, models, _ = fit_final(ctx, train_p, chosen)
        preds = predict_patient(ctx, reps, models, q)
        with open(f, "wb") as fh:
            pickle.dump(dict(q=q, chosen=chosen, preds=preds), fh)
        print(f"[externa] {q} hecho en {(time.time() - t0) / 60:.1f} min", flush=True)

    outer = [pickle.load(open(out / f"outer_{q}.pkl", "rb")) for q in pats]
    idx = np.concatenate([ctx.full[q] for q in pats])
    y, w, pat = ctx.y[idx], ctx.w[idx], ctx.meta[GROUP].to_numpy()[idx]
    rows, per_pat = [], []
    for key in outer[0]["preds"]:
        pred = np.concatenate([o["preds"][key] for o in outer])
        met = classification_metrics(y, pred, ctx.classes, w)
        pp = per_patient_table(y, pred, pat, ctx.classes)
        freq = Counter(hp_key(o["chosen"].get(key, {})) or "defecto" for o in outer)
        rows.append(dict(representation=key[0], model=key[1], macro_f1=met["macro_f1"],
                         balanced_accuracy=met["balanced_accuracy"],
                         patient_acc=float((pp["label"] == pp["pred_majority"]).mean()),
                         hiperparametros_elegidos="; ".join(f"{k} ({n})" for k, n in freq.most_common())))
        per_pat.append(pp.assign(representation=key[0], model=key[1]))
    nested = pd.DataFrame(rows)
    nested.to_csv(out / f"results_nested_{TASK}.csv", index=False)
    pd.concat(per_pat).to_csv(out / f"results_nested_per_patient_{TASK}.csv", index=False)
    (out / f"results_nested_{TASK}.md").write_text(
        "### Validación anidada (hiperparámetros elegidos sin ver al paciente de test)\n\n" + to_markdown(nested) + "\n",
        encoding="utf-8")
    print("\n" + nested.round(3).to_string(index=False))

    # 2) modelo final con los 14 pacientes
    print("\n[final] eligiendo hiperparámetros con los 14 pacientes", flush=True)
    scores = inner_scores(ctx, pats)
    scores.to_csv(out / "tuning_scores_all_patients.csv", index=False)
    chosen = select_best(scores, cfg)
    reps, models, tr = fit_final(ctx, pats, chosen)
    bundle = make_bundle(ctx, reps, models, chosen, len(tr))
    bdir = Path(cfg.get("models_dir", f"models/{cfg['mode']}_tuned"))
    save_bundle(bundle, bdir / "bundle.joblib")
    info = {k: bundle[k] for k in ("classes", "regress_out", "seed", "cap", "patients", "n_train_cells", "mode", "hparams", "versions")}
    (bdir / "bundle_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado {bdir / 'bundle.joblib'}\nHiperparámetros finales: {bundle['hparams']}")
    return nested


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(load_config(ap.parse_args().config))
