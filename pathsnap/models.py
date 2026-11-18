"""Model zoo for the candidate classifier + shared evaluation helpers."""
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42

# name -> (family, factory, max training rows (None = all; subsampling keeps slow models tractable), config text)
MODEL_ZOO = {
    "LogisticRegression": ("Linear / probabilistic",
                           lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)),
                           None, "StandardScaler, L2, C=1.0"),
    "GaussianNB": ("Probabilistic (generative)",
                   lambda: GaussianNB(var_smoothing=1e-9), None, "var_smoothing=1e-9"),
    "KNN": ("Instance-based",
            lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance")),
            150_000, "StandardScaler, k=25, distance-weighted, 150k-row subsample"),
    "RandomForest": ("Ensemble (bagging)",
                     lambda: RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_leaf=5,
                                                    max_samples=0.3, n_jobs=-1, random_state=SEED),
                     None, "150 trees, max_depth=14, min_leaf=5, max_samples=0.3"),
    "HistGradientBoosting": ("Boosting",
                             lambda: HistGradientBoostingClassifier(learning_rate=0.1, max_iter=200,
                                                                     max_leaf_nodes=31, random_state=SEED),
                             None, "lr=0.1, 200 iters, 31 leaves"),
    "MLP": ("Neural network",
            lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4,
                                                                  early_stopping=True, max_iter=60,
                                                                  random_state=SEED)),
            250_000, "StandardScaler, (64,32) ReLU, Adam, early stopping, 250k-row subsample"),
}


def subsample(X, y, n, seed=SEED):
    if n is None or len(X) <= n:
        return X, y
    rng = np.random.default_rng(seed)
    ix = rng.choice(len(X), n, replace=False)
    return X[ix], y[ix]


def fit_model(name, X, y):
    _, factory, nmax, _ = MODEL_ZOO[name]
    m = factory()
    Xs, ys = subsample(X, y, nmax)
    t0 = time.time()
    m.fit(Xs, ys)
    return m, time.time() - t0


def predict_proba(model, X):
    t0 = time.time()
    p = model.predict_proba(X)[:, 1]
    return p, time.time() - t0
