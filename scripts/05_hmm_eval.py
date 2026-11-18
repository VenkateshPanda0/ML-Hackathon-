"""Decoder tuning + classifier-only vs classifier+HMM comparison.

1. Out-of-fold (GroupKFold 5) probabilities of the final tuned classifier on the training trips, so the
   decoder is tuned on probabilities the model did not see during fitting.
2. Small grid search of decoder parameters (beta, emission_weight, jump_penalty) on 25 training trips,
   objective = mean route F1.
3. Evaluate classifier-only and classifier+HMM on the 10 student-testing trips with the chosen decoder.
Outputs: results/decoder_grid.csv, artifacts/decoder_config.json, results/hmm_vs_classifier_by_trip.csv,
         results/hmm_vs_classifier_summary.csv, results/figures/hmm_vs_classifier.png
"""
import itertools, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from pathsnap import hmm
from pathsnap.features import CFG, FEATURES, TripContext, build_candidates
from pathsnap.io import list_trips, load_trip
from pathsnap.metrics import continuity, pointwise_accuracy, route_continuity, route_metrics

model = joblib.load("artifacts/final_model.joblib")

# ---- 1. OOF probabilities on 25 training trips
train_ids = list_trips("data/train")
rng = np.random.default_rng(42)
tune_ids = sorted(rng.choice(train_ids, 25, replace=False))
tr = pd.read_csv("results/features_train.csv", dtype={"trip": str})
X, y, g = tr[FEATURES].to_numpy(), tr["label"].to_numpy(), tr["trip"].to_numpy()
oof = np.zeros(len(tr))
for a, b in GroupKFold(5).split(X, y, g):
    oof[b] = clone(model).fit(X[a], y[a]).predict_proba(X[b])[:, 1]

tune = []
for tid in tune_ids:
    ctx = TripContext(load_trip(f"data/train/{tid}"))
    cand, ref = build_candidates(ctx)
    p = oof[(tr["trip"] == tid).to_numpy()]
    assert len(p) == len(cand)
    tune.append((ctx, cand, p))

# ---- 2. grid search
grid = dict(beta=[5.0, 10.0, 20.0], emission_weight=[0.5, 1.0, 2.0], jump_penalty=[20.0, 40.0])
rows = []
for vals in itertools.product(*grid.values()):
    params = dict(zip(grid, vals))
    f1 = [route_metrics(hmm.decode(c, cd, p, params)[1], c.trip.route, c.trip.arc_len)["route_f1"]
          for c, cd, p in tune]
    rows.append({**params, "train_oof_route_f1": np.mean(f1)})
    print(rows[-1], flush=True)
grid_df = pd.DataFrame(rows).sort_values("train_oof_route_f1", ascending=False)
grid_df.to_csv("results/decoder_grid.csv", index=False)
best = {k: float(grid_df.iloc[0][k]) for k in grid}
decoder = {**hmm.DECODER_DEFAULTS, **best}
json.dump({"decoder": decoder, "candidates": CFG, "tuned_on": "25 training trips, out-of-fold probabilities",
           "objective": "mean route F1"}, open("artifacts/decoder_config.json", "w"), indent=2)

# ---- 3. student-test comparison
out = []
for tid in list_trips("data/student_test"):
    ctx = TripContext(load_trip(f"data/student_test/{tid}"))
    cand, ref = build_candidates(ctx)
    p = model.predict_proba(cand[FEATURES].to_numpy())[:, 1]
    ref_obs = ref[cand.obs.unique()]
    r = {"trip": tid, "observations": len(ref_obs)}
    for mode, fn in (("classifier_only", hmm.classifier_only), ("classifier_hmm", hmm.decode)):
        t0 = time.time()
        oa, route, _ = fn(ctx, cand, p, decoder)
        r[f"{mode}_time_s"] = time.time() - t0
        r[f"{mode}_pointwise_acc"] = pointwise_accuracy(oa, ref_obs)
        r[f"{mode}_observation_continuity"] = continuity(oa, ctx.trip.arcs)
        r[f"{mode}_continuity"] = route_continuity(route, ctx.trip.arcs)  # final-route connectivity
        r.update({f"{mode}_{k}": v for k, v in route_metrics(route, ctx.trip.route, ctx.trip.arc_len).items()})
    out.append(r)
    print(tid, round(r["classifier_only_route_f1"], 3), "->", round(r["classifier_hmm_route_f1"], 3), flush=True)
df = pd.DataFrame(out)
df.to_csv("results/hmm_vs_classifier_by_trip.csv", index=False)
summ = pd.DataFrame({m: {k: df[f"{m}_{k}"].mean() for k in
                         ["pointwise_acc", "continuity", "observation_continuity", "route_precision", "route_recall", "route_f1",
                          "route_mismatch", "exact_match", "time_s"]}
                     for m in ["classifier_only", "classifier_hmm"]})
summ.to_csv("results/hmm_vs_classifier_summary.csv")
print(summ)

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
x = np.arange(len(df)); w = 0.38
ax[0].bar(x - w / 2, df.classifier_only_route_f1, w, label="classifier only")
ax[0].bar(x + w / 2, df.classifier_hmm_route_f1, w, label="classifier + HMM")
ax[0].set_xticks(x, df.trip, rotation=45); ax[0].set_ylim(0.5, 1.01); ax[0].set_title("Route F1 per student-test trip"); ax[0].legend()
keys = ["pointwise_acc", "continuity", "route_f1"]
ax[1].bar(np.arange(3) - w / 2, [summ.loc[k, "classifier_only"] for k in keys], w, label="classifier only")
ax[1].bar(np.arange(3) + w / 2, [summ.loc[k, "classifier_hmm"] for k in keys], w, label="classifier + HMM")
ax[1].set_xticks(range(3), ["point-wise accuracy", "continuity", "route F1"]); ax[1].set_ylim(0, 1.05)
ax[1].set_title("Mean over student-test trips"); ax[1].legend()
fig.tight_layout(); fig.savefig("results/figures/hmm_vs_classifier.png", dpi=130); plt.close(fig)
