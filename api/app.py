"""PathSnap FastAPI backend (Laptop 1).

Run:  uvicorn api.app:app --host 0.0.0.0 --port 8000
Env:  PATHSNAP_DATA_DIR  folder holding the evaluation trips (default: data/evaluator_test,
                         falls back to data/student_test if that folder is empty)
"""
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap.io import list_trips
from pathsnap.pipeline import PathSnap, evaluate_against_reference

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")


def data_dir():
    d = os.environ.get("PATHSNAP_DATA_DIR", os.path.join(ROOT, "data", "evaluator_test"))
    if not os.path.isdir(d) or not list_trips(d):
        d = os.path.join(ROOT, "data", "student_test")
    return d


app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
engine = PathSnap()
log.info("Model and decoder loaded. Serving trips from %s", data_dir())


class PredictRequest(BaseModel):
    trip_id: str


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.time()
    resp = await call_next(request)
    log.info("%s %s from %s -> %d (%.2fs)", request.method, request.url.path,
             request.client.host if request.client else "?", resp.status_code, time.time() - t0)
    return resp


@app.get("/health")
def health():
    return {"status": "ok", "data_dir": data_dir(), "trips": len(list_trips(data_dir()))}


@app.get("/trips")
def trips():
    return {"trips": list_trips(data_dir())}


@app.post("/predict")
def predict(req: PredictRequest):
    d = data_dir()
    if req.trip_id not in list_trips(d):
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    folder = os.path.join(d, req.trip_id)
    log.info("PREDICT trip=%s: running candidate generation -> classifier -> HMM/Viterbi", req.trip_id)
    pred = engine.predict(folder)                       # uses .track/.nodes/.arcs only
    metrics = evaluate_against_reference(pred, folder)  # reference route read server-side only
    trip = pred["ctx"].trip
    nl = trip.nodes_ll
    segs = [[[nl[trip.arcs[a, 0], 1], nl[trip.arcs[a, 0], 0]], [nl[trip.arcs[a, 1], 1], nl[trip.arcs[a, 1], 0]]]
            for a in pred["route"]]
    step = max(1, len(trip.track_ll) // 2000)
    log.info("RESULT trip=%s route_arcs=%d route_f1=%.4f pointwise_acc=%.4f continuity=%.4f time=%.2fs",
             req.trip_id, len(pred["route"]), metrics["route_f1"], metrics["pointwise_accuracy"],
             metrics["continuity"], pred["seconds"])
    return {
        "trip_id": req.trip_id,
        "gps_points": [[float(y), float(x)] for x, y in trip.track_ll[::step]],
        "predicted_route_arcs": [int(a) for a in pred["route"]],
        "predicted_route_segments": [[[float(c) for c in p] for p in s] for s in segs],
        "observations": int(len(pred["obs_arcs"])),
        "inference_seconds": round(pred["seconds"], 3),
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
    }
