"""Static (matplotlib) route plots for the report: GPS, nearby network, reference and predicted route."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from pathsnap.io import load_trip
from pathsnap.pipeline import PathSnap, evaluate_against_reference
from pathsnap import hmm
from pathsnap.features import FEATURES

engine = PathSnap()
for tid in sys.argv[1:] or ["00000091", "00000065"]:
    folder = f"data/student_test/{tid}"
    pred = engine.predict(folder)
    met = evaluate_against_reference(pred, folder)
    t = pred["ctx"].trip
    ref = load_trip(folder).route
    p = engine.model.predict_proba(pred["cand"][FEATURES].to_numpy())[:, 1]
    _, clf_route, _ = hmm.classifier_only(pred["ctx"], pred["cand"], p, engine.decoder)
    mid = (t.nodes_xy[t.arcs[:, 0]] + t.nodes_xy[t.arcs[:, 1]]) / 2
    near = np.where(cKDTree(t.track_xy[::5]).query(mid)[0] < 150)[0]
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    for a_, (route, name) in zip(ax, [(clf_route, "classifier only"), (pred["route"], "classifier + HMM")]):
        def draw(ids, **kw):
            for i, a in enumerate(ids):
                p0, p1 = t.nodes_ll[t.arcs[a, 0]], t.nodes_ll[t.arcs[a, 1]]
                a_.plot([p0[0], p1[0]], [p0[1], p1[1]], label=kw.pop("label", None) if i == 0 else None, **kw)
        draw(near, color="#cccccc", lw=0.6, label="road network")
        draw(ref, color="#2ca02c", lw=6, alpha=0.5, label="reference route")
        draw(route, color="#d62728", lw=1.8, label=f"predicted ({name})")
        a_.scatter(t.track_ll[:, 0], t.track_ll[:, 1], s=2, color="#1f77b4", label="GPS", zorder=5)
        a_.set_title(f"Student-test trip {tid}: {name}"); a_.legend(loc="best", fontsize=8)
        a_.set_xlabel("longitude"); a_.set_ylabel("latitude")
    fig.tight_layout(); fig.savefig(f"results/figures/route_{tid}.png", dpi=120); plt.close(fig)
    print(tid, met["route_f1"])
