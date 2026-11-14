"""Final test-set runs (seeds 1-8, unseen incident location B1C1).
Proposed forecaster (HGB) was selected on validation forecast accuracy only."""
import os, json, itertools, pickle, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import experiments as E, simcore

OUT = E.OUT
t0 = time.time()

def _fc_eval(args):
    seed, d, b = args
    buf = []
    simcore.run("mp", seed, d, E.TEST_LOC, b, collect=buf, forecaster=simcore.RidgeForecaster(h=120))
    X = np.concatenate([x[0] for x in buf]); y = np.concatenate([x[1] for x in buf])
    return X, y

# 1) forecast accuracy on test runs
jobs = [(s, d, b) for s in E.TEST_SEEDS for d in ("med", "high") for b in (1, 2)]
with ProcessPoolExecutor(2) as ex:
    parts = list(ex.map(_fc_eval, jobs))
X = np.concatenate([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
w = np.load(os.path.join(OUT, "fc_w_h120.npy")); hgb = pickle.load(open(os.path.join(OUT, "hgb_h120.pkl"), "rb"))
near = X[:, 7:10].sum(1) > 0
fc = {}
for name, m in (("all", np.ones(len(y), bool)), ("near", near)):
    yy, Xv = y[m], X[m]
    fc[name] = dict(n=int(m.sum()), persistence=float(np.abs(yy).mean()),
                    ridge=float(np.abs(Xv @ w - yy).mean()), hgb=float(np.abs(hgb.predict(Xv[:, 1:]) - yy).mean()))
json.dump(fc, open(os.path.join(OUT, "forecast_test.json"), "w"), indent=1)
print("forecast", fc, round(time.time() - t0), flush=True)

# 2) control: proposed variants (baselines fixed/actuated/mp from control_main.json are unaffected by the fix)
jobs = []
for d, b, s in itertools.product(("med", "high"), (1, 2), E.TEST_SEEDS):
    base = dict(seed=s, demand=d, inc_edge=E.TEST_LOC, lanes_blocked=b, controller="pmp")
    jobs += [dict(base, guard=True, fc="hgb"), dict(base, guard=False, fc="hgb"), dict(base, guard=True, fc="ridge")]
with ProcessPoolExecutor(2) as ex:
    rs = list(ex.map(E._ctl, jobs))
base = [r for r in json.load(open(os.path.join(OUT, "v1_bug", "control_main.json"))) if r["controller"] != "pmp"]
json.dump(base + rs, open(os.path.join(OUT, "control_final.json"), "w"), indent=0)
print("control", len(rs), round(time.time() - t0), flush=True)

# 3) robustness to detection latency / missed detection (full closure)
jobs = []
for d, s in itertools.product(("med", "high"), E.TEST_SEEDS):
    base = dict(seed=s, demand=d, inc_edge=E.TEST_LOC, lanes_blocked=2, controller="pmp", guard=True, fc="hgb")
    jobs += [dict(base, latency=L) for L in (60, 120, 300)] + [dict(base, miss=True)]
with ProcessPoolExecutor(2) as ex:
    rs = list(ex.map(E._ctl, jobs))
for j, r in zip(jobs, rs):
    r["miss"] = j.get("miss", False)
json.dump(rs, open(os.path.join(OUT, "robustness_final.json"), "w"), indent=0)
print("robustness", len(rs), round(time.time() - t0), flush=True)
