"""Projection to local metres, arc geometry, candidate search."""
import numpy as np
from scipy.spatial import cKDTree

R_EARTH = 6371008.8


def project(trip):
    """Equirectangular projection around the track centre (accurate to <0.1% at city scale)."""
    lon0, lat0 = trip.track_ll.mean(axis=0)
    k = np.cos(np.radians(lat0))

    def f(ll):
        x = np.radians(ll[:, 0] - lon0) * R_EARTH * k
        y = np.radians(ll[:, 1] - lat0) * R_EARTH
        return np.column_stack([x, y])

    trip.nodes_xy = f(trip.nodes_ll)
    trip.track_xy = f(trip.track_ll)
    a, b = trip.nodes_xy[trip.arcs[:, 0]], trip.nodes_xy[trip.arcs[:, 1]]
    trip.arc_len = np.maximum(np.linalg.norm(b - a, axis=1), 1e-3)
    return trip


class ArcIndex:
    """Spatial index over straight-line arcs (samples every `step` metres + exact projection)."""

    def __init__(self, trip, step=10.0):
        self.trip = trip
        a = trip.nodes_xy[trip.arcs[:, 0]]
        b = trip.nodes_xy[trip.arcs[:, 1]]
        self.a, self.b = a, b
        self.d = b - a
        self.len = trip.arc_len
        self.bearing = np.arctan2(self.d[:, 1], self.d[:, 0])
        n = np.maximum(np.ceil(self.len / step).astype(int), 1)
        ids = np.repeat(np.arange(len(a)), n + 1)
        offs = np.concatenate([np.arange(k + 1) / k for k in n])
        pts = a[ids] + self.d[ids] * offs[:, None]
        self.ids = ids
        self.tree = cKDTree(pts)
        self.step = step

    def project_point(self, p, arc_ids):
        d = self.d[arc_ids]
        t = np.einsum("ij,ij->i", p - self.a[arc_ids], d) / np.maximum((d * d).sum(1), 1e-9)
        t = np.clip(t, 0.0, 1.0)
        proj = self.a[arc_ids] + d * t[:, None]
        return np.linalg.norm(proj - p, axis=1), t

    def candidates(self, p, radius, k_max, expand=0):
        """Arcs within `radius` of point p -> (arc_ids, dist, frac), nearest first, at most k_max."""
        r = radius
        for _ in range(expand + 1):
            hit = self.tree.query_ball_point(p, r + self.step)
            if hit:
                arcs = np.unique(self.ids[hit])
                dist, frac = self.project_point(p, arcs)
                m = dist <= r
                if m.any():
                    arcs, dist, frac = arcs[m], dist[m], frac[m]
                    o = np.argsort(dist)[:k_max]
                    return arcs[o], dist[o], frac[o]
            r *= 2
        return np.empty(0, int), np.empty(0), np.empty(0)


def track_headings(xy, window=3):
    """Direction of travel at each point from neighbours (radians); NaN when stationary."""
    n = len(xy)
    h = np.full(n, np.nan)
    for i in range(n):
        j0, j1 = max(0, i - window), min(n - 1, i + window)
        v = xy[j1] - xy[j0]
        if np.hypot(*v) > 3.0:
            h[i] = np.arctan2(v[1], v[0])
    return h


def downsample(xy, min_dist):
    """Keep a point only once it is >= min_dist from the last kept point (removes jitter while stopped)."""
    keep = [0]
    last = xy[0]
    for i in range(1, len(xy)):
        if np.hypot(*(xy[i] - last)) >= min_dist:
            keep.append(i)
            last = xy[i]
    if keep[-1] != len(xy) - 1:
        keep.append(len(xy) - 1)
    return np.array(keep)
