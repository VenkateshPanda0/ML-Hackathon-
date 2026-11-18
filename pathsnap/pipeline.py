"""End-to-end inference used by the API: raw trip folder -> candidate features -> classifier -> HMM route."""
import json
import os
import time

import joblib

from . import hmm
from .features import FEATURES, TripContext, build_candidates
from .io import load_trip
from .metrics import continuity, pointwise_accuracy, route_continuity, route_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PathSnap:
    def __init__(self, model_path=None, config_path=None):
        self.model = joblib.load(model_path or os.path.join(ROOT, "artifacts", "final_model.joblib"))
        cfg = json.load(open(config_path or os.path.join(ROOT, "artifacts", "decoder_config.json")))
        self.decoder, self.cand_cfg = cfg["decoder"], cfg["candidates"]

    def predict(self, trip_folder):
        """Prediction uses only .track/.nodes/.arcs - the reference route is NOT loaded here."""
        t0 = time.time()
        trip = load_trip(trip_folder, with_route=False)
        ctx = TripContext(trip, self.cand_cfg)
        cand, _ = build_candidates(ctx, with_labels=False)
        p = self.model.predict_proba(cand[FEATURES].to_numpy())[:, 1]
        obs_arcs, route, steps = hmm.decode(ctx, cand, p, self.decoder)
        return dict(ctx=ctx, cand=cand, obs_arcs=obs_arcs, route=route, steps=steps,
                    seconds=time.time() - t0)


def evaluate_against_reference(pred, trip_folder):
    """Server-side scoring: load the reference route separately, after prediction is complete."""
    ref_trip = load_trip(trip_folder, with_route=True)
    ctx = pred["ctx"]
    m = route_metrics(pred["route"], ref_trip.route, ctx.trip.arc_len)
    # route connectivity of the FINAL returned route (consecutive arcs share a node)
    m["continuity"] = route_continuity(pred["route"], ctx.trip.arcs)
    # older observation-level measure (raw per-GPS-point picks), kept for comparison
    m["observation_continuity"] = continuity(pred["obs_arcs"], ctx.trip.arcs)
    # observation-level accuracy vs. the reference arc each observation aligns to
    ctx.trip.route = ref_trip.route
    from .features import reference_labels
    ref_obs = reference_labels(ctx)[0][pred["cand"].obs.unique()]
    ctx.trip.route = None
    m["pointwise_accuracy"] = pointwise_accuracy(pred["obs_arcs"], ref_obs)
    return m
