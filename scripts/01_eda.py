"""Dataset exploration for the training and student-testing partitions.

Outputs (results/eda/): inventory CSV, data-quality CSV, summary stats, plots, and a Folium map.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap.geometry import ArcIndex, project
from pathsnap.io import list_trips, load_trip

OUT = "results/eda"
os.makedirs(OUT, exist_ok=True)
rows = []
for split in ["train", "student_test"]:
    for tid in list_trips(f"data/{split}"):
        folder = f"data/{split}/{tid}"
        files = {f"has_{ext}": os.path.exists(f"{folder}/{tid}.{ext}") for ext in ["track", "nodes", "arcs", "route"]}
        sizes = {f"{ext}_kb": os.path.getsize(f"{folder}/{tid}.{ext}") / 1024 for ext in ["track", "nodes", "arcs", "route"] if files[f"has_{ext}"]}
        raw = np.loadtxt(f"{folder}/{tid}.track", ndmin=2)
        t = project(load_trip(folder))
        dt = np.diff(t.times)
        step = np.linalg.norm(np.diff(t.track_xy, axis=0), axis=1)
        n = len(t.nodes_xy)
        G = csr_matrix((np.ones(len(t.arcs)), (t.arcs[:, 0], t.arcs[:, 1])), shape=(n, n))
        ncomp_w, lab = connected_components(G, directed=True, connection="weak")
        ncomp_s, _ = connected_components(G, directed=True, connection="strong")
        deg = np.bincount(t.arcs.ravel(), minlength=n)
        # distance of each GPS point to the reference route (how noisy the GPS is)
        idx = ArcIndex(t)
        sample = t.track_xy[:: max(1, len(t.track_xy) // 300)]
        d_route = np.array([idx.project_point(p, t.route)[0].min() for p in sample])
        rows.append(dict(
            split=split, trip=tid, **files, **sizes,
            gps_points=len(raw), missing_values=int(np.isnan(raw).sum()),
            duplicate_timestamps=int((dt == 0).sum()), non_increasing_timestamps=int((dt < 0).sum()),
            pct_1s_steps=100 * np.mean(dt == 1), max_gap_s=dt.max(), gaps_over_5s=int((dt > 5).sum()),
            duration_min=(t.times[-1] - t.times[0]) / 60, track_km=step.sum() / 1000,
            max_jump_m=step.max(), median_speed_mps=np.median(step / np.maximum(dt, 1e-6)),
            lon_min=raw[:, 0].min(), lon_max=raw[:, 0].max(), lat_min=raw[:, 1].min(), lat_max=raw[:, 1].max(),
            nodes=n, arcs=len(t.arcs), isolated_nodes=int((deg == 0).sum()),
            weak_components=ncomp_w, strong_components=ncomp_s,
            route_arcs=len(t.route), route_km=t.arc_len[t.route].sum() / 1000,
            route_connected=bool(np.all(t.arcs[t.route[:-1], 1] == t.arcs[t.route[1:], 0])),
            route_repeated_arcs=len(t.route) - len(set(t.route.tolist())),
            gps_to_route_median_m=np.median(d_route), gps_to_route_p95_m=np.percentile(d_route, 95),
        ))
        print(split, tid, flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/dataset_inventory.csv", index=False)
num = df.select_dtypes("number").columns
df.groupby("split")[num].describe().T.to_csv(f"{OUT}/summary_stats_by_split.csv")
df.groupby("split").agg(trips=("trip", "count"), gps_points=("gps_points", "sum"),
                        missing_values=("missing_values", "sum"), gaps_over_5s=("gaps_over_5s", "sum"),
                        trips_with_disconnected_route=("route_connected", lambda s: int((~s).sum())),
                        mean_gps_to_route_m=("gps_to_route_median_m", "mean")).to_csv(f"{OUT}/partition_overview.csv")

# ---- plots (training partition unless stated)
trn = df[df.split == "train"]
fig, ax = plt.subplots(2, 3, figsize=(14, 7))
ax[0, 0].hist(trn.gps_points, bins=30, color="#4C72B0"); ax[0, 0].set_title("GPS points per trip (train)")
ax[0, 1].hist(trn.route_km, bins=30, color="#55A868"); ax[0, 1].set_title("Reference route length, km (train)")
ax[0, 2].hist(trn.pct_1s_steps, bins=30, color="#C44E52"); ax[0, 2].set_title("% of 1-second sampling steps (train)")
ax[1, 0].scatter(trn.arcs, trn.nodes, s=12); ax[1, 0].set_xscale("log"); ax[1, 0].set_yscale("log")
ax[1, 0].set_title("Road network size: arcs vs nodes"); ax[1, 0].set_xlabel("arcs"); ax[1, 0].set_ylabel("nodes")
ax[1, 1].hist(trn.gps_to_route_median_m, bins=30, color="#8172B2"); ax[1, 1].set_title("Median GPS-to-route distance, m")
ax[1, 2].hist(trn.max_gap_s.clip(upper=60), bins=30, color="#937860"); ax[1, 2].set_title("Max time gap per trip, s (clipped 60)")
fig.tight_layout(); fig.savefig(f"{OUT}/eda_distributions.png", dpi=130); plt.close(fig)

# trip locations (trips are spread across the world, so plot each trip's centroid)
fig, ax = plt.subplots(figsize=(9, 5))
for split, c in [("train", "#4C72B0"), ("student_test", "#DD8452")]:
    sub = df[df.split == split]
    ax.scatter((sub.lon_min + sub.lon_max) / 2, (sub.lat_min + sub.lat_max) / 2, s=25, color=c, label=split, alpha=0.8)
ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.legend(); ax.grid(alpha=0.3)
ax.set_title("Trip locations (centre of each trajectory's bounding box)")
fig.tight_layout(); fig.savefig(f"{OUT}/eda_all_trajectories.png", dpi=130); plt.close(fig)

# one trip: road network + GPS + reference route
tid = "00000039"
t = project(load_trip(f"data/train/{tid}"))
fig, ax = plt.subplots(figsize=(8, 8))
a, b = t.nodes_ll[t.arcs[:, 0]], t.nodes_ll[t.arcs[:, 1]]
for p, q in zip(a, b):
    ax.plot([p[0], q[0]], [p[1], q[1]], color="#bbbbbb", lw=0.5)
for r in t.route:
    p, q = t.nodes_ll[t.arcs[r, 0]], t.nodes_ll[t.arcs[r, 1]]
    ax.plot([p[0], q[0]], [p[1], q[1]], color="green", lw=2.5)
ax.scatter(t.track_ll[:, 0], t.track_ll[:, 1], s=4, color="blue", zorder=3)
ax.set_title(f"Train trip {tid}: road network (grey), reference route (green), GPS (blue)")
fig.tight_layout(); fig.savefig(f"{OUT}/eda_trip_{tid}.png", dpi=130); plt.close(fig)
print(df.groupby("split")[["gps_points", "missing_values", "gaps_over_5s", "pct_1s_steps", "gps_to_route_median_m"]].mean())
