# PathSnap: GPS Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon, PES University (RR campus).

**Team:** Varun Rao (PES1UG24CS520), Vedanth Kiran (PES1UG24CS523), Venkatesh Panda (PES1UG24CS524), Varshit Gude (PES1UG24CS518)

PathSnap takes noisy GPS observations plus the road network of a trip and reconstructs the sequence of road segments (arcs) the vehicle actually drove:

```
.track + .nodes + .arcs
   -> candidate generation (arcs within 50 m of each GPS observation)
   -> feature engineering (distance, heading agreement, speed, ...)
   -> candidate classifier (P(candidate is the true arc))
   -> HMM + Viterbi decoding over the road graph
   -> connected route (ordered arc ids)
```

## Repository layout

| Folder | Contents |
|---|---|
| `data/` | `train/` (80 trips), `student_test/` (10 trips), `evaluator_test/` (put the 10 evaluator trips here on Laptop 1) |
| `pathsnap/` | Library code: `io` (file parsing), `geometry` (projection, spatial index), `graph` (road graph / Dijkstra), `features` (candidates, features, reference labels), `models` (model zoo), `hmm` (HMM + Viterbi), `metrics`, `viz` (Folium), `pipeline` (end-to-end inference) |
| `scripts/` | Numbered experiment scripts (see below) |
| `api/` | FastAPI backend (Laptop 1) |
| `ui/` | Evaluator web UI (Laptop 2) |
| `artifacts/` | `final_model.joblib` (tuned classifier), `decoder_config.json` (HMM + candidate config), `models/` (all compared models) |
| `results/` | CSV results, experiment logs, Optuna trials, figures (`figures/`), EDA outputs (`eda/`), Folium maps (`maps/`) |
| `report/` | Technical report (PDF + source) |

## Installation

Python 3.11+ (scikit-learn is pinned to the version the saved models were trained with).

```bash
git clone https://github.com/VenkateshPanda0/ML-Hackathon-.git
cd ML-Hackathon-
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the experiments

Run from the repository root, in order:

```bash
python scripts/01_eda.py                          # dataset exploration -> results/eda/
python scripts/02_build_features.py train         # candidate table for training trips
python scripts/02_build_features.py student_test  # candidate table for student-testing trips
python scripts/03_train_models.py                 # 6 models, GroupKFold CV, student-test eval, HMM per model
python scripts/04_bayes_opt.py 30                 # Bayesian (Optuna TPE) tuning of HistGradientBoosting
python scripts/05_hmm_eval.py                     # decoder tuning + classifier-only vs classifier+HMM
python scripts/06_folium_maps.py                  # interactive maps -> results/maps/
python scripts/07_make_figures.py                 # model-comparison plots
```

Random seed is fixed at 42 everywhere. Results are written to `results/`, and every number in the report comes from those files.

## Deployment (two laptops)

**Laptop 1: model + API**

```bash
# copy the 10 evaluator trips into data/evaluator_test/<trip_id>/ (or set PATHSNAP_DATA_DIR)
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Find the laptop's IP (`ipconfig` on Windows, `ip a` or `ifconfig` on Linux/macOS) and allow port 8000 through the firewall. Every request is logged in this terminal: the trip id, the route F1 and the timing.

Endpoints:
- `GET /health`: status and data folder
- `GET /trips`: the evaluation trip ids
- `POST /predict` with body `{"trip_id": "..."}`: runs the full pipeline and returns the GPS points, predicted route and metrics

**Laptop 2: UI**

```bash
cd ui
python -m http.server 8080
# open http://localhost:8080, enter http://<laptop1-ip>:8000, click "Load evaluation trips"
```

The UI needs internet access to load the Leaflet map tiles.

**Ground-truth separation:** `PathSnap.predict()` loads the trip with `with_route=False`, so the reference route never reaches the model or the decoder. The API reads the reference route only afterwards, server-side, to compute metrics. The UI receives only the predicted route and the metric values, never the reference geometry. Nothing is hard-coded per trip.

## Main results

See `results/model_comparison.csv`, `results/tuned_vs_untuned.csv`, `results/hmm_vs_classifier_summary.csv` and the report.
