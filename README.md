# AI-Based Predictive Traffic Signal Control for Alternative Roads During Unexpected Incidents

Code, data and paper for our ICAISC 2027 submission (3rd International Conference on Advanced Innovations in Smart Cities, Jeddah International College).

**Authors:** Atheer Bahussein, Mnar Alnaghes — Department of CSIT, Jeddah International College

## Overview

A closed loop that couples:

1. **Incident detection** from roadside video — YOLO + ByteTrack + a stopped-vehicle rule (`detection/`)
2. **Incident-conditioned graph forecasting** of link queues 2 minutes ahead (`simulation/simcore.py`)
3. **Predictive max-pressure signal control** with a spillback guard (`simulation/simcore.py`)

The forecasting and control stages are evaluated in the **SUMO** microscopic simulator, with detection latency and missed detections emulated in the loop.

## Main findings

| Finding | Result |
|---|---|
| Forecast error near the incident (boosted model vs. persistence) | −16.2% |
| Proposed controller vs. reactive max-pressure (delay) | within 1.2 s/veh; not better |
| Best controller in every scenario | SUMO actuated (14–16% below max-pressure under full closure) |
| Congestion shift to far links | none (|Δshift| ≤ 0.7 veh·h) |
| Unseen incident durations | forecaster does not generalise to longer incidents; controller equal or worse |

All numbers come from the files in `simulation/results/` (8 random seeds per scenario, incident location unseen in training).

## Repository structure

```
paper/                 LaTeX source (IEEE conference template), figures, compiled PDF
simulation/
  simcore.py           network graph, features, forecasters, controllers, incident injection, metrics
  experiments.py       forecaster training data + baselines (fixed-time, actuated, reactive MP)
  collect120.py        training data for the boosted forecaster (h = 120 s)
  run_final.py         forecast accuracy on test runs
  run_final2.py        predictive-MP variants + detection latency / missed detection (resumable)
  run_dur.py           unseen incident durations (600 s, 1800 s)
  analysis.py          tables, statistics (Wilcoxon), summary.json
  fig_ieee.py          paper figures (delay, latency)
  make_timeseries.py   paper figure (forecast vs. actual)
  network/             4x4 SUMO grid (static + actuated), background routes per seed, corridor flows
  results/             raw results (JSON/JSONL), trained models, figures
detection/
  tads_detect.py       YOLO + ByteTrack accident detection and evaluation on TADS
  README.md            how to run on Google Colab / GPU
```

## Reproduce the simulation results

```bash
pip install -r requirements.txt

cd simulation
python experiments.py train      # linear forecasters (h = 60/120/300 s)
python experiments.py control    # baselines on test seeds 1-8
python collect120.py             # training data for the boosted model
python - <<'EOF'                 # fit the boosted forecaster
import numpy as np, pickle
from sklearn.ensemble import HistGradientBoostingRegressor as H
z = np.load('results/train_h120.npz'); X, y, g = z['X'], z['y'], z['g']
idx = np.random.default_rng(0).choice(np.where(g != 106)[0], 300000, replace=False)
pickle.dump(H(max_iter=200, learning_rate=0.1, max_leaf_nodes=31).fit(X[idx][:, 1:], y[idx]),
            open('results/hgb_h120.pkl', 'wb'))
EOF
python run_final.py              # stage 1: forecast accuracy (stages 2-3 are in run_final2.py)
python run_final2.py
python run_dur.py
python analysis.py
python fig_ieee.py
python make_timeseries.py
```

The trained models are already included in `simulation/results/`, so the control experiments can be rerun without retraining. Runs are deterministic given the seed (SUMO 1.28).

Note: the baseline results in `results/v1_bug/control_main.json` were produced before a fix to the *predictive* controller only; the fixed-time, actuated and reactive-MP rows are unaffected and are reused (verified identical on re-run). The predictive-MP rows from that file are not used.

## Detection (TADS)

The detector is implemented but not yet evaluated: the TADS frame archive is 77.1 GB. See `detection/README.md`.

Dataset: Y. Chai, J. Fang, H. Liang, W. Silamu, "TADS: a novel dataset for road traffic accident detection from a surveillance perspective," *The Journal of Supercomputing*, 80(18):26226–26249, 2024. https://github.com/cyc-gh/TADS

## Requirements

Python 3.10+, `eclipse-sumo` (pip), see `requirements.txt`. A GPU is recommended for the detection stage only.

## License

MIT — see `LICENSE`.
