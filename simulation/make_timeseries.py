"""Fig. 2 of the paper: actual vs. forecast queues on three links in one test run
(high demand, full closure, seed 1). Writes results/ts_seed1_high2.npz and ../paper/figures/fig_timeseries.pdf."""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import simcore

HERE = os.path.dirname(os.path.abspath(__file__))
G = simcore.Graph(os.path.join(simcore.NET, "grid.net.xml"))
buf = []
simcore.run("mp", 1, "high", "B1C1", 2, collect=buf, forecaster=simcore.RidgeForecaster(h=120))
m = pickle.load(open(os.path.join(HERE, "results", "hgb_h120.pkl"), "rb"))
T = np.arange(len(buf)) * 10
true = np.array([b[2] + b[1] for b in buf]); cur = np.array([b[2] for b in buf])
pred = np.array([np.maximum(b[2] + m.predict(b[0][:, 1:]), 0) for b in buf])
np.savez(os.path.join(HERE, "results", "ts_seed1_high2.npz"), T=T, true=true, cur=cur, pred=pred, edges=np.array(G.edges))

T = T + 120
E = list(G.edges)
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(3, 1, figsize=(3.45, 3.9), sharex=True)
for ax, (e, lab) in zip(axs, [("B1C1", "Incident link"), ("A1B1", "Upstream link (1 hop)"), ("A0A1", "Diversion feeder (2 hops)")]):
    i = E.index(e); ax.axvspan(600, 1800, color="#e6e5e1", lw=0)
    ax.plot(T, true[:, i], color="#0b0b0b", lw=0.9, label="Actual")
    ax.plot(T, cur[:, i], color="#2a78d6", lw=0.8, ls="--", label="Persistence")
    ax.plot(T, pred[:, i], color="#eb6834", lw=0.9, label="Boosted forecast")
    ax.set_title(lab, fontsize=7, pad=2); ax.set_ylabel("Halting veh.")
axs[-1].set_xlabel("Time (s)")
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, ncol=3, fontsize=6.3, frameon=False, loc="upper center")
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.95))
fig.savefig(os.path.join(HERE, "..", "paper", "figures", "fig_timeseries.pdf"))
