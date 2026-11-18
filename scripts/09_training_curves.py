"""Training / validation curves for the models that keep an epoch-wise (iteration-wise) history.

Only MLP (sklearn MLPClassifier) and HistGradientBoosting keep one. The other four models
(Logistic Regression solver, Gaussian NB, KNN, Random Forest) have no iteration history worth plotting,
so no curve is drawn for them.

Reads the fitted models in artifacts/models/ and plots what is stored inside them. Nothing is re-trained.
Output: results/figures/training_curves.png, results/training_curves.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mlp = joblib.load("artifacts/models/MLP.joblib").steps[-1][1]
hgb = joblib.load("artifacts/models/HistGradientBoosting.joblib")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))

# MLP: training loss per epoch, and validation accuracy on the 10% hold-out used for early stopping
epochs = np.arange(1, len(mlp.loss_curve_) + 1)
ax[0].plot(epochs, mlp.loss_curve_, color="tab:blue", label="training loss (log loss)")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training loss")
ax0b = ax[0].twinx()
ax0b.plot(epochs, mlp.validation_scores_, color="tab:orange", label="validation accuracy")
ax0b.set_ylabel("validation accuracy")
ax[0].set_title("MLP (64, 32): stopped after %d epochs" % mlp.n_iter_)
h1, l1 = ax[0].get_legend_handles_labels(); h2, l2 = ax0b.get_legend_handles_labels()
ax[0].legend(h1 + h2, l1 + l2, loc="center right")

# HistGradientBoosting: log loss on the training part and on the internal validation split, per boosting round
rows = []
if hasattr(hgb, "train_score_") and len(getattr(hgb, "validation_score_", [])) > 0:
    it = np.arange(len(hgb.train_score_))
    ax[1].plot(it, -hgb.train_score_, label="training loss (log loss)")
    ax[1].plot(it, -hgb.validation_score_, label="validation loss (log loss)")
    ax[1].set_xlabel("boosting iteration"); ax[1].set_ylabel("log loss")
    ax[1].set_title("HistGradientBoosting: %d iterations" % hgb.n_iter_); ax[1].legend()
    rows += [dict(model="HistGradientBoosting", step=i, train=-a, validation=-b)
             for i, (a, b) in enumerate(zip(hgb.train_score_, hgb.validation_score_))]
else:
    ax[1].text(0.5, 0.5, "no validation history stored", ha="center"); ax[1].set_axis_off()
rows += [dict(model="MLP", step=i + 1, train=a, validation=b)
         for i, (a, b) in enumerate(zip(mlp.loss_curve_, mlp.validation_scores_))]

fig.tight_layout(); fig.savefig("results/figures/training_curves.png", dpi=130); plt.close(fig)
pd.DataFrame(rows).to_csv("results/training_curves.csv", index=False)
print("MLP epochs:", mlp.n_iter_, "| final train loss %.4f | best val acc %.4f" % (mlp.loss_curve_[-1], max(mlp.validation_scores_)))
print("HGB iterations:", hgb.n_iter_)
