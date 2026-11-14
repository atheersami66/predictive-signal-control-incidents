"""Run all experiments: forecaster training/evaluation and closed-loop control.
Outputs go to results/ as JSON/CSV."""
import os, sys, json, random, itertools, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import simcore

OUT = os.path.join(simcore.HERE, "results"); os.makedirs(OUT, exist_ok=True)
TRAIN_SEEDS = [101, 102, 103, 104, 105, 106]
TEST_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]
TRAIN_LOCS = ["A1B1", "C1D1", "B2C2", "C2B2", "B1B2"]   # test location B1C1 is unseen
TEST_LOC = "B1C1"
WORKERS = 2


def _collect(args):
    seed, loc, blocked, demand, h = args
    fc = simcore.RidgeForecaster(h=h)
    buf = []
    simcore.run("mp", seed, demand, loc, blocked, collect=buf, forecaster=fc)
    X = np.concatenate([b[0] for b in buf]); y = np.concatenate([b[1] for b in buf])
    return X, y


def train(h):
    rng = random.Random(h)
    jobs = [(s, l, b, rng.choice(["med", "high"]), h) for s in TRAIN_SEEDS for l in TRAIN_LOCS for b in (1, 2)]
    with ProcessPoolExecutor(WORKERS) as ex:
        parts = list(ex.map(_collect, jobs))
    X = np.concatenate([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
    fc = simcore.RidgeForecaster(h=h); fc.fit(X, y, lam=10.0)
    np.save(os.path.join(OUT, f"fc_w_h{h}.npy"), fc.w)
    return fc, len(y)


def _eval_fc(args):
    seed, demand, blocked, h = args
    w = np.load(os.path.join(OUT, f"fc_w_h{h}.npy"))
    fc = simcore.RidgeForecaster(w, h)
    buf = []
    simcore.run("mp", seed, demand, TEST_LOC, blocked, collect=buf, forecaster=fc)
    G = simcore.Graph(os.path.join(simcore.NET, "grid.net.xml"))
    cap = np.array([G.cap[e] for e in G.edges])
    hop = G.hops(TEST_LOC); near = np.array([hop[e] <= 2 for e in G.edges])
    rows = []
    for X, dy, cur in buf:
        true = cur + dy; pred = fc.predict(X, cur)
        rows.append((np.abs(pred - true)[near].mean(), np.abs(cur - true)[near].mean(),
                     pred / cap, true / cap, near))
    return rows


def _ctl(args):
    kw = dict(args)
    h = kw.pop("h", 120)
    fc = kw.pop("fc", "ridge")
    if kw["controller"] == "pmp":
        if fc == "hgb":
            import pickle
            kw["forecaster"] = simcore.HGBForecaster(pickle.load(open(os.path.join(OUT, f"hgb_h{h}.pkl"), "rb")), h)
        else:
            kw["forecaster"] = simcore.RidgeForecaster(np.load(os.path.join(OUT, f"fc_w_h{h}.npy")), h)
    r = simcore.run(**kw); r["h"] = h; r["fc"] = fc
    return {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in r.items()}


def main(stage):
    t0 = time.time()
    if stage in ("train", "all"):
        info = {}
        for h in (60, 120, 300):
            _, n = train(h); info[h] = n; print("trained h", h, n, round(time.time() - t0))
        json.dump(info, open(os.path.join(OUT, "train_info.json"), "w"))
    if stage in ("fceval", "all"):
        res = {}
        for h in (60, 120, 300):
            jobs = [(s, d, b, h) for s in TEST_SEEDS[:4] for d in ("med", "high") for b in (1, 2)]
            with ProcessPoolExecutor(WORKERS) as ex:
                rows = [r for part in ex.map(_eval_fc, jobs) for r in part]
            mae = float(np.mean([r[0] for r in rows])); mae_p = float(np.mean([r[1] for r in rows]))
            # at-risk classification (rho > 0.5 used here because 0.85 is rarely reached)
            out = {"mae_model": mae, "mae_persistence": mae_p}
            for th in (0.3, 0.5):
                tp = fp = fn = 0
                for r in rows:
                    p, t, m = r[2] > th, r[3] > th, r[4]
                    tp += int((p & t & m).sum()); fp += int((p & ~t & m).sum()); fn += int((~p & t & m).sum())
                out[f"prec@{th}"] = tp / max(tp + fp, 1); out[f"rec@{th}"] = tp / max(tp + fn, 1)
            res[h] = out; print("fc", h, out, round(time.time() - t0))
        json.dump(res, open(os.path.join(OUT, "forecast_eval.json"), "w"), indent=1)
    if stage in ("control", "all"):
        jobs = []
        for d, b, s in itertools.product(("med", "high"), (1, 2), TEST_SEEDS):
            base = dict(seed=s, demand=d, inc_edge=TEST_LOC, lanes_blocked=b)
            for c in ("fixed", "actuated", "mp"):
                jobs.append(dict(base, controller=c))
            jobs.append(dict(base, controller="pmp", guard=True))
            jobs.append(dict(base, controller="pmp", guard=False))
        with ProcessPoolExecutor(WORKERS) as ex:
            rs = list(ex.map(_ctl, jobs))
        json.dump(rs, open(os.path.join(OUT, "control_main.json"), "w"), indent=0)
        print("control main done", len(rs), round(time.time() - t0))
    if stage in ("ablation", "all"):
        jobs = []
        for d, b, s in itertools.product(("med", "high"), (2,), TEST_SEEDS):
            base = dict(seed=s, demand=d, inc_edge=TEST_LOC, lanes_blocked=b, controller="pmp", guard=True)
            for L in (60, 120, 300):
                jobs.append(dict(base, latency=L))
            jobs.append(dict(base, miss=True))
            for h in (60, 300):
                jobs.append(dict(base, h=h))
        with ProcessPoolExecutor(WORKERS) as ex:
            rs = list(ex.map(_ctl, jobs))
        for j, r in zip(jobs, rs):
            r["miss"] = j.get("miss", False)
        json.dump(rs, open(os.path.join(OUT, "ablation.json"), "w"), indent=0)
        print("ablation done", len(rs), round(time.time() - t0))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
