"""Folium maps of GPS observations, nearby road network, reference route and predicted route.

Usage: python scripts/06_folium_maps.py [split] [trip ids...]
       default: one training trip (EDA) + all 10 student-testing trips (predictions)
Output: results/maps/<split>_<trip>.html
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.io import list_trips, load_trip
from pathsnap.geometry import project
from pathsnap.pipeline import PathSnap, evaluate_against_reference
from pathsnap.viz import trip_map

os.makedirs("results/maps", exist_ok=True)
split = sys.argv[1] if len(sys.argv) > 1 else "student_test"
ids = sys.argv[2:] or list_trips(f"data/{split}")

# EDA map: raw GPS + reference route of a training trip
t = project(load_trip("data/train/00000039"))
trip_map(t, reference=t.route, title="Train trip 00000039 (EDA)").save("results/maps/train_00000039_eda.html")

engine = PathSnap()
for tid in ids:
    folder = f"data/{split}/{tid}"
    pred = engine.predict(folder)
    m = evaluate_against_reference(pred, folder)
    ref = load_trip(folder).route
    title = f"{split} trip {tid} - route F1 {m['route_f1']:.3f}"
    trip_map(pred["ctx"].trip, predicted=pred["route"], reference=ref, title=title).save(f"results/maps/{split}_{tid}.html")
    print(title, flush=True)
