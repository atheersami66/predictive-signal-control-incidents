"""Aggregate final results -> LaTeX tables + figures + summary.json (all numbers in the paper come from here)."""
import json, os
import numpy as np
from scipy.stats import wilcoxon, t as tdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG = os.path.join(R, "fig"); os.makedirs(FIG, exist_ok=True)
main = [r for r in json.load(open(os.path.join(R, "v1_bug", "control_main.json"))) if r["controller"] != "pmp"]
main += [json.loads(l) for l in open(os.path.join(R, "control_pmp.jsonl"))]
rob = [json.loads(l) for l in open(os.path.join(R, "robustness.jsonl"))]
fct = json.load(open(os.path.join(R, "forecast_test.json")))

def key(r):
    if r["controller"] == "pmp":
        return "pmp_" + r.get("fc", "ridge") + ("_guard" if r["guard"] else "_noguard")
    return r["controller"]

NAMES = {"fixed": "Fixed-time", "actuated": "Actuated", "mp": "Reactive MP",
         "pmp_ridge_guard": "Predictive MP (linear fc.)", "pmp_hgb_noguard": "Predictive MP (no guard)",
         "pmp_hgb_guard": "Predictive MP + guard (proposed)"}
ORDER = ["fixed", "actuated", "mp", "pmp_ridge_guard", "pmp_hgb_noguard", "pmp_hgb_guard"]
SCEN = [("med", 1), ("med", 2), ("high", 1), ("high", 2)]

def ci(x):
    x = np.asarray(x, float); n = len(x)
    return float(x.mean()), float(tdist.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0

def get(rows, ctl, d, b, m):
    rs = sorted([r for r in rows if key(r) == ctl and r["demand"] == d and r["lanes_blocked"] == b], key=lambda r: r["seed"])
    assert len(rs) == 8, (ctl, d, b, len(rs))
    return np.array([r[m] for r in rs], float)

S = {}
lines = []
for d, b in SCEN:
    lines.append(r"\multicolumn{5}{l}{\textit{%s demand, %s}}\\" % ("Medium" if d == "med" else "High",
                 "1 of 2 lanes blocked" if b == 1 else "full closure (2 of 2 lanes)"))
    far_mp = get(main, "mp", d, b, "far_wait_vehh"); del_mp = get(main, "mp", d, b, "delay")
    for c in ORDER:
        v = get(main, c, d, b, "delay"); tp = get(main, c, d, b, "throughput")
        sh = get(main, c, d, b, "far_wait_vehh") - far_mp
        S[f"{d}{b}_{c}_delay"] = ci(v); S[f"{d}{b}_{c}_tp"] = ci(tp); S[f"{d}{b}_{c}_shift"] = ci(sh)
        S[f"{d}{b}_{c}_tele"] = ci(get(main, c, d, b, "teleports"))
        p = float(wilcoxon(v, del_mp).pvalue) if c != "mp" else float("nan")
        S[f"{d}{b}_{c}_p"] = p
        mark = "" if c == "mp" else (r"$^{\dagger}$" if p < 0.05 else "")
        bold = c == "pmp_hgb_guard"
        name = NAMES[c]
        row = [f"{ci(v)[0]:.1f} $\\pm$ {ci(v)[1]:.1f}{mark}", f"{ci(tp)[0]:.0f} $\\pm$ {ci(tp)[1]:.0f}",
               "--" if c == "mp" else f"{ci(sh)[0]:+.1f} $\\pm$ {ci(sh)[1]:.1f}", f"{ci(get(main, c, d, b, 'teleports'))[0]:.1f}"]
        lines.append(("\\quad " + name) + " & " + " & ".join(row) + r"\\")
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
open(os.path.join(R, "tab_control.tex"), "w").write("\n".join(lines) + "\n")

# robustness table
rl = []
for d in ("med", "high"):
    cells = [ci(get(main, "mp", d, 2, "delay"))[0], ci(get(main, "pmp_hgb_guard", d, 2, "delay"))[0]]
    S[f"rob_{d}_0"] = ci(get(main, "pmp_hgb_guard", d, 2, "delay"))
    for L in (60, 120, 300):
        v = [r["delay"] for r in rob if r["demand"] == d and r["latency"] == L and not r["miss"]]
        assert len(v) == 8; S[f"rob_{d}_{L}"] = ci(v); cells.append(ci(v)[0])
    v = [r["delay"] for r in rob if r["demand"] == d and r["miss"]]
    assert len(v) == 8; S[f"rob_{d}_miss"] = ci(v); cells.append(ci(v)[0])
    rl.append(("Medium" if d == "med" else "High") + " & " + " & ".join(f"{x:.1f}" for x in cells) + r"\\")
open(os.path.join(R, "tab_robust.tex"), "w").write("\n".join(rl) + "\n")

fl = []
for k, lab in (("all", "All links"), ("near", "Links $\\leq$2 hops from incident")):
    f = fct[k]
    fl.append(f"{lab} & {f['n']:,} & {f['persistence']:.3f} & {f['ridge']:.3f} & {f['hgb']:.3f} & "
              f"{100*(1-f['hgb']/f['persistence']):.1f}\\%\\\\")
    S[f"fc_{k}"] = f
open(os.path.join(R, "tab_forecast.tex"), "w").write("\n".join(fl) + "\n")
json.dump(S, open(os.path.join(R, "summary.json"), "w"), indent=1)

# ---------------- figures (print-safe: color + hatch, recessive axes)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
COL = {"fixed": "#9b9a97", "actuated": "#eda100", "mp": "#2a78d6", "pmp_hgb_guard": "#eb6834"}
HAT = {"fixed": "", "actuated": "....", "mp": "", "pmp_hgb_guard": "////"}
SHOW = ["fixed", "actuated", "mp", "pmp_hgb_guard"]
SHORT = {"fixed": "Fixed-time", "actuated": "Actuated", "mp": "Reactive MP", "pmp_hgb_guard": "Predictive MP + guard"}
fig, ax = plt.subplots(figsize=(5.4, 2.5))
w = 0.2
for i, c in enumerate(SHOW):
    mus = [S[f"{d}{b}_{c}_delay"][0] for d, b in SCEN]; hs = [S[f"{d}{b}_{c}_delay"][1] for d, b in SCEN]
    x = np.arange(len(SCEN)) + (i - 1.5) * w
    ax.bar(x, mus, w * 0.9, yerr=hs, color=COL[c], hatch=HAT[c], edgecolor="white", linewidth=0.4,
           error_kw=dict(lw=0.6, capsize=1.5, ecolor="#52514e"), label=SHORT[c])
ax.set_xticks(range(len(SCEN))); ax.tick_params(axis="x", labelsize=7)
ax.set_xticklabels([f"{'Medium' if d=='med' else 'High'} demand\n{'1 of 2 lanes' if b==1 else 'full closure'}" for d, b in SCEN])
ax.set_ylabel("Average delay (s/veh)")
ax.yaxis.grid(True, color="#e6e5e1", lw=0.5); ax.set_axisbelow(True)
ax.legend(ncol=4, fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15))
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_delay.pdf")); fig.savefig(os.path.join(FIG, "fig_delay.png"), dpi=200)

fig, axs = plt.subplots(1, 2, figsize=(4.8, 1.9))
for ax, d in zip(axs, ("med", "high")):
    Ls = [0, 60, 120, 300]
    mus = [S[f"rob_{d}_{L}"][0] for L in Ls]; hs = [S[f"rob_{d}_{L}"][1] for L in Ls]
    ax.errorbar(Ls, mus, yerr=hs, color=COL["pmp_hgb_guard"], marker="o", ms=4, lw=1.5, capsize=2, label="Predictive MP + guard")
    ax.axhline(S[f"{d}2_mp_delay"][0], color=COL["mp"], lw=1.2, ls="--", label="Reactive MP")
    ax.axhline(S[f"{d}2_actuated_delay"][0], color=COL["actuated"], lw=1.2, ls=":", label="Actuated")
    ax.set_title(f"{'Medium' if d=='med' else 'High'} demand, full closure", fontsize=8)
    ax.set_xlabel("Detection latency (s)"); ax.set_xticks(Ls)
    ax.yaxis.grid(True, color="#e6e5e1", lw=0.5); ax.set_axisbelow(True)
axs[0].set_ylabel("Avg. delay (s/veh)")
axs[1].legend(fontsize=6, frameon=False, loc="center right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_latency.pdf")); fig.savefig(os.path.join(FIG, "fig_latency.png"), dpi=200)
print(open(os.path.join(R, "tab_control.tex")).read()); print(open(os.path.join(R, "tab_robust.tex")).read())
print(open(os.path.join(R, "tab_forecast.tex")).read())
for k, v in S.items():
    if k.endswith("_p"): print(k, round(v, 4) if v == v else v)
