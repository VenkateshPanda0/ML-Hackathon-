"""Loading trip folders: .nodes, .arcs, .track, .route."""
import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Trip:
    trip_id: str
    nodes_ll: np.ndarray          # (N, 2) lon, lat
    arcs: np.ndarray              # (A, 2) from_node, to_node  (arc id = row index, 0-based)
    track_ll: np.ndarray          # (T, 2) lon, lat
    times: np.ndarray             # (T,)
    route: np.ndarray = None      # (R,) ordered arc ids (ground truth, if present)
    # filled by geometry.project()
    nodes_xy: np.ndarray = field(default=None, repr=False)
    track_xy: np.ndarray = field(default=None, repr=False)
    arc_len: np.ndarray = field(default=None, repr=False)


def _load(path, **kw):
    return np.loadtxt(path, ndmin=kw.pop("ndmin", 2), **kw)


def load_trip(folder, with_route=True):
    trip_id = os.path.basename(os.path.normpath(folder))
    p = os.path.join(folder, trip_id)
    nodes = _load(p + ".nodes")
    arcs = _load(p + ".arcs", dtype=np.int64)
    track = _load(p + ".track")
    route = None
    if with_route and os.path.exists(p + ".route"):
        route = np.loadtxt(p + ".route", dtype=np.int64, ndmin=1)
    return Trip(trip_id, nodes[:, :2], arcs[:, :2], track[:, :2], track[:, 2], route)


def list_trips(split_dir):
    return sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d)))


def write_route(route, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(str(int(a)) for a in route) + "\n")
