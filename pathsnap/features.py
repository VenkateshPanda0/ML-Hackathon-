"""Observation selection, candidate generation, feature engineering and reference labels.

One row per (GPS observation, candidate arc). The classifier learns P(candidate is the reference
arc | features); the HMM then uses these probabilities as emission scores.
"""
import numpy as np
import pandas as pd

from .geometry import ArcIndex, downsample, project, track_headings
from .graph import RoadGraph

CFG = dict(min_dist=10.0, radius=50.0, k_max=8)

FEATURES = [
    "dist", "dist_rank", "gap_to_nearest", "dist_ratio", "cos_heading", "abs_heading_diff",
    "moving", "speed", "end_frac", "log_arc_len", "n_cands", "is_nearest",
]


class TripContext:
    """Everything needed to process one trip (projection, spatial index, routing graph)."""

    def __init__(self, trip, cfg=None):
        self.cfg = {**CFG, **(cfg or {})}
        self.trip = project(trip) if trip.nodes_xy is None else trip
        self.index = ArcIndex(self.trip)
        self.graph = RoadGraph(self.trip)
        xy = self.trip.track_xy
        self.obs = downsample(xy, self.cfg["min_dist"])            # indices of kept GPS observations
        self.heading = track_headings(xy)
        t = self.trip.times
        sp = np.zeros(len(xy))
        sp[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(np.diff(t), 1e-6)
        self.speed = pd.Series(sp).rolling(5, center=True, min_periods=1).median().to_numpy()


def reference_labels(ctx):
    """Assign each kept observation the reference arc it lies on, via a monotone alignment of the
    observation sequence to the ordered reference route (dynamic programming: minimise total
    point-to-arc distance with a non-decreasing route position)."""
    trip, idx = ctx.trip, ctx.index
    route = trip.route
    P = trip.track_xy[ctx.obs]
    R = len(route)
    a, d = idx.a[route], idx.d[route]
    dd = np.maximum((d * d).sum(1), 1e-9)
    prev = None
    back = np.zeros((len(P), R), dtype=np.int32)
    for i, p in enumerate(P):
        t = np.clip(((p - a) * d).sum(1) / dd, 0, 1)
        cost = np.linalg.norm(a + d * t[:, None] - p, axis=1)
        if prev is None:
            cur = cost
            back[i] = np.arange(R)
        else:
            # running minimum of prev over positions <= j
            best = np.minimum.accumulate(prev)
            changes = np.r_[True, prev[1:] < best[:-1]]
            arg = np.maximum.accumulate(np.where(changes, np.arange(R), 0)).astype(np.int32)
            cur = cost + best
            back[i] = arg
        prev = cur
    pos = np.zeros(len(P), dtype=np.int64)
    pos[-1] = int(np.argmin(prev))
    for i in range(len(P) - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    return route[pos], pos


def build_candidates(ctx, with_labels=True):
    """Return (DataFrame of candidate rows, ref_arc per observation or None)."""
    trip, idx, c = ctx.trip, ctx.index, ctx.cfg
    ref = reference_labels(ctx)[0] if (with_labels and trip.route is not None) else None
    rows = []
    for k, i in enumerate(ctx.obs):
        arcs, dist, frac = idx.candidates(trip.track_xy[i], c["radius"], c["k_max"])
        n = len(arcs)
        if n == 0:
            continue
        h = ctx.heading[i]
        moving = 0.0 if np.isnan(h) else 1.0
        diff = np.zeros(n) if np.isnan(h) else np.abs(np.angle(np.exp(1j * (idx.bearing[arcs] - h))))
        rank = np.argsort(np.argsort(dist))
        df = pd.DataFrame({
            "trip": trip.trip_id, "obs": k, "pt": int(i), "arc": arcs, "frac": frac,
            "dist": dist, "dist_rank": rank, "gap_to_nearest": dist - dist.min(),
            "dist_ratio": dist / (dist.min() + 1.0),
            "cos_heading": np.cos(diff) if moving else np.zeros(n), "abs_heading_diff": diff,
            "moving": moving, "speed": ctx.speed[i], "end_frac": np.minimum(frac, 1 - frac),
            "log_arc_len": np.log1p(idx.len[arcs]), "n_cands": n, "is_nearest": (rank == 0).astype(float),
        })
        if ref is not None:
            df["label"] = (arcs == ref[k]).astype(int)
            df["ref_arc"] = ref[k]
        rows.append(df)
    cand = pd.concat(rows, ignore_index=True)
    return cand, ref
