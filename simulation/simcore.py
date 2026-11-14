"""Closed-loop incident simulation in SUMO: controllers, incident injection,
detection-latency emulation, incident-conditioned graph forecaster hooks, metrics."""
import os, sys, uuid, xml.etree.ElementTree as ET
from collections import deque, defaultdict
import numpy as np
import sumo
os.environ.setdefault("SUMO_HOME", sumo.SUMO_HOME)
import traci, sumolib

HERE = os.path.dirname(os.path.abspath(__file__))
NET = os.path.join(HERE, "network")
STEP_END = 3000          # simulated seconds
MIN_GREEN, YELLOW, MAX_GREEN = 10, 3, 60
VEH_LEN = 7.5
BG = {"med": "low", "high": "high"}   # background random-trip file per demand level            # m per queued vehicle (length + gap)


# ---------------------------------------------------------------- network graph
class Graph:
    def __init__(self, netfile):
        net = sumolib.net.readNet(netfile)
        self.edges = [e.getID() for e in net.getEdges() if not e.getID().startswith(":")]
        self.idx = {e: i for i, e in enumerate(self.edges)}
        self.lanes = {e: [l.getID() for l in net.getEdge(e).getLanes()] for e in self.edges}
        self.length = {e: net.getEdge(e).getLength() for e in self.edges}
        self.cap = {e: self.length[e] * len(self.lanes[e]) / VEH_LEN for e in self.edges}
        self.up = {e: [i.getID() for i in net.getEdge(e).getIncoming()] for e in self.edges}
        self.down = {e: [o.getID() for o in net.getEdge(e).getOutgoing()] for e in self.edges}

    def hops(self, src):
        """undirected hop distance from edge src (edges adjacent through nodes)."""
        dist = {src: 0}; q = deque([src])
        while q:
            e = q.popleft()
            for n in self.up[e] + self.down[e]:
                if n not in dist:
                    dist[n] = dist[e] + 1; q.append(n)
        return {e: dist.get(e, 99) for e in self.edges}


# ---------------------------------------------------------------- forecaster features
N_LAG = 3                 # current, -30 s, -60 s
def features(G, hist, t_idx, inc):
    """Feature matrix (n_edges x F). hist: list of arrays of halting counts per edge
    sampled every 10 s; inc: dict or None (incident as KNOWN to the controller)."""
    E = len(G.edges)
    cur = hist[-1]
    lag3 = hist[-4] if len(hist) >= 4 else hist[0]
    lag6 = hist[-7] if len(hist) >= 7 else hist[0]
    up = np.array([sum(cur[G.idx[u]] for u in G.up[e]) for e in G.edges])
    dn = np.array([sum(cur[G.idx[d]] for d in G.down[e]) for e in G.edges])
    capv = np.array([G.cap[e] for e in G.edges])
    occ = cur / capv
    if inc is None:
        f_inc = np.zeros((E, 5))
    else:
        hop = inc["hops"]
        sev = inc["blocked_frac"]
        el = min(inc["elapsed"] / 600.0, 3.0)
        h = np.array([hop[e] for e in G.edges])
        f_inc = np.stack([(h == 0) * sev, (h == 1) * sev, (h == 2) * sev,
                          (h == 0) * sev * el, (h == 1) * sev * el], 1)
    X = np.column_stack([np.ones(E), cur, cur - lag3, cur - lag6, up, dn, occ,
                         f_inc, f_inc[:, :3] * cur[:, None], f_inc[:, :3] * up[:, None]])
    return X


class RidgeForecaster:
    """Incident-conditioned graph autoregressive model: one weight vector shared by all
    edges (like a 1-hop graph convolution), predicting change in halting vehicles at t+h."""
    def __init__(self, w=None, h=120):
        self.w, self.h = w, h

    def fit(self, X, y, lam=1.0):
        A = X.T @ X + lam * np.eye(X.shape[1]); A[0, 0] -= lam
        self.w = np.linalg.solve(A, X.T @ y)

    def predict(self, X, cur):
        return np.maximum(cur + X @ self.w, 0.0)


# ---------------------------------------------------------------- controllers
class MaxPressure:
    """Reactive (forecaster=None) or predictive max-pressure with optional spillback guard."""
    def __init__(self, G, forecaster=None, guard=False, theta=0.85):
        self.G, self.fc, self.guard, self.theta = G, forecaster, guard, theta
        self.tls = traci.trafficlight.getIDList()
        self.links, self.greens, self.state = {}, {}, {}
        for t in self.tls:
            self.links[t] = traci.trafficlight.getControlledLinks(t)
            logic = traci.trafficlight.getAllProgramLogics(t)[0]
            gs = [p.state for p in logic.phases if "y" not in p.state and ("G" in p.state or "g" in p.state)]
            self.greens[t] = gs
            self.state[t] = dict(cur=0, t_in=0, yellow=0, nxt=None)
            traci.trafficlight.setRedYellowGreenState(t, gs[0])
        self.lane_edge = {l: e for e in G.edges for l in G.lanes[e]}

    def step(self, t, q_lane, pred_edge=None):
        G = self.G
        for tl in self.tls:
            s = self.state[tl]; s["t_in"] += 1
            if s["yellow"] > 0:
                s["yellow"] -= 1
                if s["yellow"] == 0:
                    s["cur"], s["t_in"] = s["nxt"], 0
                    traci.trafficlight.setRedYellowGreenState(tl, self.greens[tl][s["cur"]])
                continue
            if s["t_in"] < MIN_GREEN:
                continue
            press = []
            for st in self.greens[tl]:
                p = 0.0
                for i, ch in enumerate(st):
                    if ch not in "Gg" or not self.links[tl][i]:
                        continue
                    lin, lout, _ = self.links[tl][i][0]
                    ein, eout = self.lane_edge.get(lin), self.lane_edge.get(lout)
                    qi, qo = q_lane.get(lin, 0), q_lane.get(lout, 0)
                    if pred_edge is not None and ein in pred_edge:
                        # forecast correction: lane measurement + forecast change of its link (per lane)
                        cur_e, pred_e = pred_edge[ein]
                        qi = max(qi + (pred_e - cur_e) / len(G.lanes[ein]), 0.0)
                        if eout in pred_edge:
                            cur_o, pred_o = pred_edge[eout]
                            qo = max(qo + (pred_o - cur_o) / len(G.lanes[eout]), 0.0)
                            if self.guard and pred_o / G.cap[eout] > self.theta:
                                continue      # spillback guard: never push into a filling link
                    p += qi - qo
                press.append(p)
            best = int(np.argmax(press))
            if (best != s["cur"] and press[best] > press[s["cur"]]) or s["t_in"] >= MAX_GREEN:
                if best == s["cur"]:
                    best = (s["cur"] + 1) % len(self.greens[tl])
                cur = self.greens[tl][s["cur"]]; nxt = self.greens[tl][best]
                ystate = "".join("y" if c in "Gg" and n not in "Gg" else c for c, n in zip(cur, nxt))
                traci.trafficlight.setRedYellowGreenState(tl, ystate)
                s["yellow"], s["nxt"] = YELLOW, best


# ---------------------------------------------------------------- one run
def run(controller, seed, demand="high", inc_edge="B1C1", lanes_blocked=1,
        inc_start=600, inc_dur=1200, latency=0, miss=False, forecaster=None,
        guard=False, theta=0.85, reroute_p=0.5, collect=None):
    """controller in {'fixed','actuated','mp','pmp'}; returns metrics dict.
    collect: list -> append (X, y_delta, cur) training tuples (forecaster training)."""
    net = os.path.join(NET, "grid_act.net.xml" if controller == "actuated" else "grid.net.xml")
    G = Graph(os.path.join(NET, "grid.net.xml"))
    tag = uuid.uuid4().hex[:8]
    trip = f"/tmp/trip_{tag}.xml"
    cmd = [sumolib.checkBinary("sumo"), "-n", net, "-r", ",".join([os.path.join(NET, f"routes_{BG[demand]}_{seed}.rou.xml"), os.path.join(NET, f"corridor_{demand}.rou.xml")]),
           "--seed", str(seed), "--no-step-log", "--no-warnings", "--duration-log.disable",
           "--time-to-teleport", "300", "--tripinfo-output", trip,
           "--device.rerouting.probability", str(reroute_p), "--device.rerouting.period", "60",
           "--device.rerouting.adaptation-interval", "10", "--end", str(STEP_END)]
    traci.start(cmd, label=tag)
    con = traci.getConnection(tag); traci.switch(tag)
    traci.vehicletype.copy("DEFAULT_VEHTYPE", "blocker")
    traci.vehicletype.setColor("blocker", (255, 0, 0, 255))
    ctl = MaxPressure(G, forecaster if controller == "pmp" else None, guard, theta) if controller in ("mp", "pmp") else None
    all_lanes = [l for e in G.edges for l in G.lanes[e]]
    for l in all_lanes:
        traci.lane.subscribe(l, [traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER])
    hop = G.hops(inc_edge)
    nl = len(G.lanes[inc_edge]); blocked_frac = lanes_blocked / nl
    hist, max_q, far_wait, teleports = [], 0, 0.0, 0
    hfc = forecaster.h if forecaster is not None else 120
    pending = []          # (target time index, X, cur) for collection
    pred_edge = None
    for t in range(STEP_END):
        if t == inc_start:
            traci.route.add("inc_r", [inc_edge])
            for k in range(lanes_blocked):
                lane = nl - 1 - k      # block from the inner (left) lane outward
                vid = f"blk{k}"
                traci.vehicle.add(vid, "inc_r", typeID="blocker", departLane=str(lane), departPos="100", departSpeed="0")
                traci.vehicle.setStop(vid, inc_edge, pos=105, laneIndex=lane, duration=inc_dur)
        traci.simulationStep()
        teleports += traci.simulation.getStartingTeleportNumber()
        sub = traci.lane.getAllSubscriptionResults()
        q_lane = {l: sub[l][traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER] for l in all_lanes}
        q_edge = np.array([sum(q_lane[l] for l in G.lanes[e]) for e in G.edges], float)
        max_q = max(max_q, q_edge.max())
        far_wait += sum(q_edge[G.idx[e]] for e in G.edges if hop[e] >= 2)
        known = (not miss) and (inc_start + latency <= t < inc_start + inc_dur)
        inc = dict(hops=hop, blocked_frac=blocked_frac, elapsed=t - inc_start) if known else None
        if t % 10 == 0:
            hist.append(q_edge)
            if collect is not None or (ctl is not None and ctl.fc is not None):
                X = features(G, hist, len(hist) - 1, inc)
                if collect is not None:
                    pending.append((len(hist) - 1 + hfc // 10, X, q_edge.copy()))
                    while pending and pending[0][0] < len(hist):
                        ti, Xp, cp = pending.pop(0)
                        collect.append((Xp, hist[ti] - cp, cp))
                if ctl is not None and ctl.fc is not None:
                    p = ctl.fc.predict(X, q_edge)
                    pred_edge = {e: (q_edge[G.idx[e]], p[G.idx[e]]) for e in G.edges}
        if ctl is not None:
            ctl.step(t, q_lane, pred_edge)
    remaining = traci.vehicle.getIDList()
    rem_loss = sum(traci.vehicle.getTimeLoss(v) for v in remaining if not v.startswith("blk"))
    backlog = traci.simulation.getMinExpectedNumber() - len(remaining)
    traci.close()
    loss, n = 0.0, 0
    for ti in ET.parse(trip).getroot().iter("tripinfo"):
        if ti.get("id").startswith("blk"):
            continue
        loss += float(ti.get("timeLoss")); n += 1
    os.remove(trip)
    nveh = n + len(remaining)
    return dict(controller=controller, seed=seed, demand=demand, inc_edge=inc_edge,
                lanes_blocked=lanes_blocked, inc_dur=inc_dur, latency=latency, guard=guard,
                delay=(loss + rem_loss) / max(nveh, 1), arrived=n,
                throughput=n / (STEP_END / 3600.0), max_queue=float(max_q),
                far_wait_vehh=far_wait / 3600.0, teleports=teleports, backlog=int(backlog))


class HGBForecaster:
    """Gradient-boosted incident-conditioned graph forecaster (same features as RidgeForecaster)."""
    def __init__(self, model, h=120):
        self.m, self.h = model, h

    def predict(self, X, cur):
        return np.maximum(cur + self.m.predict(X[:, 1:]), 0.0)
