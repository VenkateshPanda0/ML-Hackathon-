"""Build the candidate table (one row per GPS observation x candidate arc) for a data partition.

Usage: python scripts/02_build_features.py train
       python scripts/02_build_features.py student_test
Output: results/features_<split>.csv (uncompressed) and results/candidate_stats_<split>.csv
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from pathsnap.io import load_trip, list_trips
from pathsnap.features import TripContext, build_candidates, CFG

split = sys.argv[1]
root = os.path.join("data", split)
frames, stats = [], []
t0 = time.time()
for tid in list_trips(root):
    ctx = TripContext(load_trip(os.path.join(root, tid)))
    cand, ref = build_candidates(ctx)
    frames.append(cand)
    g = cand.groupby("obs")
    stats.append(dict(trip=tid, gps_points=len(ctx.trip.track_xy), observations_used=len(ctx.obs),
                      observations_with_candidates=cand.obs.nunique(), candidates=len(cand),
                      mean_candidates_per_obs=g.size().mean(),
                      candidate_miss_rate=1 - g["label"].max().mean()))
    print(f"{tid}: {len(cand)} candidates, miss={stats[-1]['candidate_miss_rate']:.3f}", flush=True)
df = pd.concat(frames, ignore_index=True)
os.makedirs("results", exist_ok=True)
df.to_csv(f"results/features_{split}.csv", index=False)
pd.DataFrame(stats).to_csv(f"results/candidate_stats_{split}.csv", index=False)
print(f"{split}: {len(df)} rows, positives={df.label.mean():.3f}, config={CFG}, {time.time()-t0:.0f}s")
