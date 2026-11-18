"""Bayesian hyperparameter optimisation (Optuna TPE) of the selected classifier.

Objective : minimise mean log loss over 3-fold GroupKFold (groups = trip) on the 80 training trips.
            Log loss is used because the HMM consumes the classifier's probabilities directly, so
            calibrated probabilities matter more than a thresholded F1.
Outputs   : results/bayes_opt_trials.csv, results/bayes_opt_best.json, results/tuned_vs_untuned.csv,
            artifacts/final_model.joblib, results/figures/bayes_opt_*.png
Usage     : python scripts/04_bayes_opt.py [n_trials]
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from pathsnap.features import FEATURES, TripContext, build_candidates
from pathsnap.io import list_trips, load_trip
from pathsnap.metrics import candidate_metrics, continuity, pointwise_accuracy, route_metrics, top1_accuracy
from pathsnap.models import SEED
from pathsnap import hmm

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
tr = pd.read_csv("results/features_train.csv.gz", dtype={"trip": str})
te = pd.read_csv("results/features_student_test.csv.gz", dtype={"trip": str})
X, y, g = tr[FEATURES].to_numpy(), tr["label"].to_numpy(), tr["trip"].to_numpy()
Xte, yte = te[FEATURES].to_numpy(), te["label"].to_numpy()
folds = list(GroupKFold(n_splits=3).split(X, y, g))

SPACE = {
    "learning_rate": "log-uniform [0.01, 0.3]",
    "max_iter": "int [100, 600]",
    "max_leaf_nodes": "int [15, 127]",
    "min_samples_leaf": "int [10, 300]",
    "l2_regularization": "log-uniform [1e-4, 10]",
    "max_features": "uniform [0.5, 1.0]",
}


def make(params):
    return HistGradientBoostingClassifier(random_state=SEED, **params)


def cv_eval(params):
    ll, f1, top = [], [], []
    for a, b in folds:
        m = make(params).fit(X[a], y[a])
        p = m.predict_proba(X[b])[:, 1]
        r = candidate_metrics(y[b], p)
        ll.append(r["log_loss"]); f1.append(r["macro_f1"]); top.append(top1_accuracy(tr.iloc[b], p))
    return np.mean(ll), np.mean(f1), np.mean(top)


def objective(trial):
    params = dict(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_iter=trial.suggest_int("max_iter", 100, 600),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 127),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 300),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-4, 10, log=True),
        max_features=trial.suggest_float("max_features", 0.5, 1.0),
    )
    t0 = time.time()
    ll, f1, top = cv_eval(params)
    trial.set_user_attr("cv_macro_f1", f1); trial.set_user_attr("cv_top1", top)
    trial.set_user_attr("seconds", time.time() - t0)
    print(f"trial {trial.number}: logloss={ll:.4f} macroF1={f1:.4f} top1={top:.4f}", flush=True)
    return ll


UNTUNED = dict(learning_rate=0.1, max_iter=200, max_leaf_nodes=31)   # the configuration from 03_train_models
t0 = time.time()
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
                            study_name="pathsnap_hgb")
study.enqueue_trial({**UNTUNED, "min_samples_leaf": 20, "l2_regularization": 1e-4, "max_features": 1.0})
study.optimize(objective, n_trials=N_TRIALS)
search_time = time.time() - t0
trials = study.trials_dataframe()
trials.to_csv("results/bayes_opt_trials.csv", index=False)
best = study.best_params
json.dump({"objective": "min mean log loss, GroupKFold(3) by trip, 80 training trips",
           "sampler": "TPE (seed 42)", "n_trials": N_TRIALS, "search_space": SPACE,
           "best_params": best, "best_cv_log_loss": study.best_value,
           "search_time_s": search_time}, open("results/bayes_opt_best.json", "w"), indent=2)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(trials["number"], trials["value"], "o", alpha=0.6, label="trial")
ax[0].plot(trials["number"], trials["value"].cummin(), "r-", label="best so far")
ax[0].set_xlabel("trial"); ax[0].set_ylabel("CV log loss"); ax[0].set_title("Bayesian optimisation history"); ax[0].legend()
imp = optuna.importance.get_param_importances(study)
ax[1].barh(list(imp)[::-1], list(imp.values())[::-1]); ax[1].set_title("Hyperparameter importance (fANOVA)")
fig.tight_layout(); fig.savefig("results/figures/bayes_opt_history.png", dpi=130); plt.close(fig)

# ---- untuned vs tuned on identical partitions
test_trips = []
for tid in list_trips("data/student_test"):
    ctx = TripContext(load_trip(os.path.join("data/student_test", tid)))
    cand, ref = build_candidates(ctx)
    test_trips.append((ctx, cand, ref))

rows = []
for name, params in [("HGB untuned", UNTUNED), ("HGB tuned (Bayesian)", best)]:
    ll, f1, top = cv_eval(params)
    t0 = time.time(); m = make(params).fit(X, y); fit_t = time.time() - t0
    t0 = time.time(); p = m.predict_proba(Xte)[:, 1]; inf_t = time.time() - t0
    r = candidate_metrics(yte, p)
    rf, cont, pw, dec_t = [], [], [], 0
    for ctx, cand, ref in test_trips:
        pp = m.predict_proba(cand[FEATURES].to_numpy())[:, 1]
        t1 = time.time(); oa, route, _ = hmm.decode(ctx, cand, pp); dec_t += time.time() - t1
        rf.append(route_metrics(route, ctx.trip.route, ctx.trip.arc_len)["route_f1"])
        cont.append(continuity(oa, ctx.trip.arcs)); pw.append(pointwise_accuracy(oa, ref[cand.obs.unique()]))
    rows.append(dict(model=name, cv_log_loss=ll, cv_macro_f1=f1, cv_top1=top, test_top1=top1_accuracy(te, p),
                     test_macro_f1=r["macro_f1"], test_log_loss=r["log_loss"], hmm_route_f1=np.mean(rf),
                     hmm_continuity=np.mean(cont), hmm_pointwise_acc=np.mean(pw), fit_time_s=fit_t,
                     inference_s_test_candidates=inf_t, hmm_decode_s_10_trips=dec_t))
    if "tuned" in name and "untuned" not in name:
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(m, "artifacts/final_model.joblib")
cmp_ = pd.DataFrame(rows); cmp_.to_csv("results/tuned_vs_untuned.csv", index=False)
print(cmp_.T)
