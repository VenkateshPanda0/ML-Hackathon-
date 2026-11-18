"""Train and compare candidate classifiers.

- 5-fold GroupKFold cross-validation on the 80 training trips (groups = trip, so no trip leaks across folds)
- refit on all training trips, evaluate on the 10 student-testing trips
- classifier-only vs classifier+HMM route reconstruction on student-testing trips
Outputs: results/model_comparison.csv, results/cv_folds.csv, results/hmm_by_model.csv,
         artifacts/models/<name>.joblib, results/figures/*.png
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GroupKFold

from pathsnap.features import FEATURES, TripContext, build_candidates
from pathsnap.io import list_trips, load_trip
from pathsnap.metrics import (candidate_metrics, candidate_miss_rate, continuity, pointwise_accuracy,
                              route_metrics, top1_accuracy)
from pathsnap.models import MODEL_ZOO, SEED, fit_model, predict_proba
from pathsnap import hmm

only = sys.argv[1:] or list(MODEL_ZOO)
np.random.seed(SEED)
os.makedirs("artifacts/models", exist_ok=True); os.makedirs("results/figures", exist_ok=True)

tr = pd.read_csv("results/features_train.csv.gz", dtype={"trip": str})
te = pd.read_csv("results/features_student_test.csv.gz", dtype={"trip": str})
Xtr, ytr, gtr = tr[FEATURES].to_numpy(), tr["label"].to_numpy(), tr["trip"].to_numpy()
Xte, yte = te[FEATURES].to_numpy(), te["label"].to_numpy()

# student-test trip contexts for route reconstruction (same candidate generation as the feature file)
test_trips = []
for tid in list_trips("data/student_test"):
    ctx = TripContext(load_trip(os.path.join("data/student_test", tid)))
    cand, ref = build_candidates(ctx)
    test_trips.append((ctx, cand, ref))


def route_eval(model):
    rows = []
    for ctx, cand, ref in test_trips:
        p = model.predict_proba(cand[FEATURES].to_numpy())[:, 1]
        ref_obs = ref[cand["obs"].unique()]
        out = {"trip": ctx.trip.trip_id}
        for mode, fn in (("clf", hmm.classifier_only), ("hmm", hmm.decode)):
            t0 = time.time()
            obs_arcs, route, _ = fn(ctx, cand, p)
            out[f"{mode}_time_s"] = time.time() - t0
            out[f"{mode}_pointwise_acc"] = pointwise_accuracy(obs_arcs, ref_obs)
            out[f"{mode}_continuity"] = continuity(obs_arcs, ctx.trip.arcs)
            for k, v in route_metrics(route, ctx.trip.route, ctx.trip.arc_len).items():
                out[f"{mode}_{k}"] = v
        rows.append(out)
    return pd.DataFrame(rows)


summary, folds, hmm_rows = [], [], []
gkf = GroupKFold(n_splits=5)
for name in only:
    family, _, nmax, cfg = MODEL_ZOO[name]
    print(f"=== {name}", flush=True)
    for f, (a, b) in enumerate(gkf.split(Xtr, ytr, gtr)):
        m, ft = fit_model(name, Xtr[a], ytr[a])
        p, it = predict_proba(m, Xtr[b])
        r = candidate_metrics(ytr[b], p)
        r.update(model=name, fold=f, top1=top1_accuracy(tr.iloc[b], p), fit_time_s=ft, predict_time_s=it)
        folds.append(r)
        print(f"  fold {f}: macroF1={r['macro_f1']:.4f} top1={r['top1']:.4f} logloss={r['log_loss']:.4f}", flush=True)
    model, fit_time = fit_model(name, Xtr, ytr)
    joblib.dump(model, f"artifacts/models/{name}.joblib")
    p_tr, _ = predict_proba(model, Xtr)
    p_te, inf_time = predict_proba(model, Xte)
    rt = candidate_metrics(yte, p_te)
    fd = pd.DataFrame([x for x in folds if x["model"] == name])
    re = route_eval(model); re["model"] = name; hmm_rows.append(re)
    summary.append(dict(
        model=name, family=family, config=cfg,
        cv_macro_f1=fd.macro_f1.mean(), cv_macro_f1_std=fd.macro_f1.std(),
        cv_top1=fd.top1.mean(), cv_log_loss=fd.log_loss.mean(),
        train_macro_f1=candidate_metrics(ytr, p_tr)["macro_f1"], train_top1=top1_accuracy(tr, p_tr),
        test_accuracy=rt["accuracy"], test_precision=rt["precision"], test_recall=rt["recall"],
        test_f1=rt["f1"], test_macro_f1=rt["macro_f1"], test_log_loss=rt["log_loss"],
        test_top1=top1_accuracy(te, p_te),
        clf_only_continuity=re.clf_continuity.mean(), hmm_continuity=re.hmm_continuity.mean(),
        clf_only_route_f1=re.clf_route_f1.mean(), hmm_route_f1=re.hmm_route_f1.mean(),
        hmm_pointwise_acc=re.hmm_pointwise_acc.mean(),
        fit_time_s=fit_time, inference_ms_per_1k_candidates=1000 * inf_time / len(Xte) * 1000,
    ))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary[-1].items()}, flush=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(yte, (p_te >= 0.5).astype(int), ax=ax, colorbar=False,
                                            display_labels=["not ref", "ref arc"])
    ax.set_title(f"{name} - student test"); fig.tight_layout()
    fig.savefig(f"results/figures/confusion_{name}.png", dpi=130); plt.close(fig)


def merge_save(df, path, key):
    if os.path.exists(path) and len(only) < len(MODEL_ZOO):
        old = pd.read_csv(path)
        df = pd.concat([old[~old["model"].isin(only)], df], ignore_index=True)
    df.to_csv(path, index=False)
    return df


S = merge_save(pd.DataFrame(summary), "results/model_comparison.csv", "model")
merge_save(pd.DataFrame(folds), "results/cv_folds.csv", "model")
merge_save(pd.concat(hmm_rows, ignore_index=True), "results/hmm_by_model.csv", "model")
json.dump({"seed": SEED, "cv": "GroupKFold(5) by trip", "features": FEATURES,
           "models": {k: v[3] for k, v in MODEL_ZOO.items()}}, open("results/logs/train_config.json", "w"), indent=2)
print(S[["model", "cv_macro_f1", "cv_top1", "test_macro_f1", "test_top1", "test_log_loss", "hmm_route_f1"]])
