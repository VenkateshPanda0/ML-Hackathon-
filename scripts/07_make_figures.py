"""Model-comparison plots from results/model_comparison.csv and results/cv_folds.csv."""
import os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

S = pd.read_csv("results/model_comparison.csv")
F = pd.read_csv("results/cv_folds.csv")
order = S.sort_values("cv_macro_f1", ascending=False)["model"].tolist()
S = S.set_index("model").loc[order]
os.makedirs("results/figures", exist_ok=True)

fig, ax = plt.subplots(2, 2, figsize=(13, 8))
ax[0, 0].bar(order, S.cv_macro_f1, yerr=S.cv_macro_f1_std, capsize=4, color="#4C72B0")
ax[0, 0].set_ylim(0.85, 0.97); ax[0, 0].set_title("CV Macro F1 (GroupKFold 5, train) ± std")
x = np.arange(len(order)); w = 0.38
ax[0, 1].bar(x - w / 2, S.test_top1, w, label="Top-1 accuracy"); ax[0, 1].bar(x + w / 2, S.test_macro_f1, w, label="Macro F1")
ax[0, 1].set_xticks(x, order); ax[0, 1].set_ylim(0.85, 0.97); ax[0, 1].legend(); ax[0, 1].set_title("Student-test Top-1 accuracy and Macro F1")
ax[1, 0].bar(x - w / 2, S.cv_log_loss, w, label="CV"); ax[1, 0].bar(x + w / 2, S.test_log_loss, w, label="student test")
ax[1, 0].set_xticks(x, order); ax[1, 0].legend(); ax[1, 0].set_title("Log loss (lower is better)")
ax[1, 1].bar(x - w / 2, S.clf_only_route_f1, w, label="classifier only"); ax[1, 1].bar(x + w / 2, S.hmm_route_f1, w, label="classifier + HMM")
ax[1, 1].set_xticks(x, order); ax[1, 1].set_ylim(0.85, 1.0); ax[1, 1].legend(); ax[1, 1].set_title("Student-test route F1 (length-weighted)")
for a in ax.ravel(): a.tick_params(axis="x", rotation=25)
fig.tight_layout(); fig.savefig("results/figures/model_comparison.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
F.boxplot(column="macro_f1", by="model", ax=ax[0], rot=25); ax[0].set_title("CV Macro F1 per fold"); ax[0].set_xlabel("")
ax[1].bar(order, S.fit_time_s, color="#C44E52"); ax[1].set_yscale("log"); ax[1].set_title("Training time on all 80 trips (s, log)")
ax[2].bar(order, S.inference_ms_per_1k_candidates, color="#8172B2"); ax[2].set_yscale("log"); ax[2].set_title("Inference time (ms per 1k candidates, log)")
for a in ax: a.tick_params(axis="x", rotation=25)
plt.suptitle(""); fig.tight_layout(); fig.savefig("results/figures/model_stability_cost.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x - w / 2, S.clf_only_continuity, w, label="classifier only"); ax.bar(x + w / 2, S.hmm_continuity, w, label="classifier + HMM")
ax.set_xticks(x, order, rotation=25); ax.set_ylim(0, 1.15); ax.legend(loc="upper center", ncol=2); ax.set_title("Route continuity on student test")
fig.tight_layout(); fig.savefig("results/figures/continuity_by_model.png", dpi=130); plt.close(fig)

# train vs CV gap (overfitting check)
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x - w / 2, S.train_macro_f1, w, label="train (refit, in-sample)"); ax.bar(x + w / 2, S.cv_macro_f1, w, label="CV (out-of-fold)")
ax.set_xticks(x, order, rotation=25); ax.set_ylim(0.85, 1.0); ax.legend(); ax.set_title("Train vs CV Macro F1 (overfitting check)")
fig.tight_layout(); fig.savefig("results/figures/train_vs_cv.png", dpi=130); plt.close(fig)
print("figures written")
