"""Resumable stages 2-3 of run_final.py: each finished run is appended to a .jsonl file."""
import os, json, itertools, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import experiments as E

OUT = E.OUT
def jid(j): return json.dumps(j, sort_keys=True)
def run_stage(jobs, path):
    done = set()
    if os.path.exists(path):
        for line in open(path):
            done.add(json.loads(line)["_job"])
    todo = [j for j in jobs if jid(j) not in done]
    print(path, "todo", len(todo), flush=True)
    with ProcessPoolExecutor(2) as ex:
        futs = {ex.submit(E._ctl, j): j for j in todo}
        for f in as_completed(futs):
            r = f.result(); j = futs[f]; r["_job"] = jid(j); r["miss"] = j.get("miss", False)
            with open(path, "a") as fh: fh.write(json.dumps(r) + "\n")

jobs = []
for d, b, s in itertools.product(("med", "high"), (1, 2), E.TEST_SEEDS):
    base = dict(seed=s, demand=d, inc_edge=E.TEST_LOC, lanes_blocked=b, controller="pmp")
    jobs += [dict(base, guard=True, fc="hgb"), dict(base, guard=False, fc="hgb"), dict(base, guard=True, fc="ridge")]
run_stage(jobs, os.path.join(OUT, "control_pmp.jsonl"))
jobs = []
for d, s in itertools.product(("med", "high"), E.TEST_SEEDS):
    base = dict(seed=s, demand=d, inc_edge=E.TEST_LOC, lanes_blocked=2, controller="pmp", guard=True, fc="hgb")
    jobs += [dict(base, latency=L) for L in (60, 120, 300)] + [dict(base, miss=True)]
run_stage(jobs, os.path.join(OUT, "robustness.jsonl"))
print("ALL DONE", flush=True)
