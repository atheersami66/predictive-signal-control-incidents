"""Robustness to incident duration unseen in training (600 s, 1800 s); full closure; test seeds."""
import os, json, itertools, pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import experiments as E, simcore
OUT = E.OUT; path = os.path.join(OUT, "duration.jsonl")
hgb = pickle.load(open(os.path.join(OUT, "hgb_h120.pkl"), "rb"))

def job(j):
    if j["controller"] == "mp":
        buf = []
        r = simcore.run("mp", j["seed"], j["demand"], E.TEST_LOC, 2, inc_dur=j["dur"], collect=buf,
                        forecaster=simcore.RidgeForecaster(h=120))
        X = np.concatenate([b[0] for b in buf]); y = np.concatenate([b[1] for b in buf])
        near = X[:, 7:10].sum(1) > 0
        r["fc_pers"] = float(np.abs(y[near]).mean()); r["fc_hgb"] = float(np.abs(hgb.predict(X[near][:, 1:]) - y[near]).mean())
        r["fc_n"] = int(near.sum())
    elif j["controller"] == "pmp":
        r = simcore.run("pmp", j["seed"], j["demand"], E.TEST_LOC, 2, inc_dur=j["dur"], guard=True,
                        forecaster=simcore.HGBForecaster(hgb, 120))
    else:
        r = simcore.run(j["controller"], j["seed"], j["demand"], E.TEST_LOC, 2, inc_dur=j["dur"])
    r.update(j); return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in r.items()}

jobs = [dict(controller=c, seed=s, demand=d, dur=dur) for dur in (600, 1800) for d in ("med", "high")
        for s in E.TEST_SEEDS for c in ("actuated", "mp", "pmp")]
done = set()
if os.path.exists(path):
    done = {json.dumps({k: json.loads(l)[k] for k in ("controller", "seed", "demand", "dur")}, sort_keys=True) for l in open(path)}
todo = [j for j in jobs if json.dumps(j, sort_keys=True) not in done]
print("todo", len(todo), flush=True)
with ProcessPoolExecutor(2) as ex:
    for f in as_completed([ex.submit(job, j) for j in todo]):
        with open(path, "a") as fh: fh.write(json.dumps(f.result()) + "\n")
print("ALL DONE", flush=True)
