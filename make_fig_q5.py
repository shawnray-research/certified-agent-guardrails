"""Figure for Q5 (single column, 3.5 in): per-decision cost of exact enforcement versus judging.
Gate: median (dot) and 99th percentile (bar) per-decision latency of the compiled P2+P3 keyed-counter
gate, by decade of position in one 10^6-proposal stream over 10^6 recipients (q5_enforcement_cost.py).
Naive replay monitor: recomputes the policy from the committed history for every proposal (same stream).
Judges: median and interquartile range of one hosted judge call, sequential, network included
(q5_judge_latency.py).
Usage: python make_fig_q5.py [q5_enforcement_cost.json] [q5_judge_latency.json] [OUT.pdf]"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import paper_style as ps
ps.apply()
plt.rcParams.update({"savefig.bbox": "tight", "savefig.pad_inches": 0.012, "xtick.labelsize": 8.0, "axes.labelsize": 8.0})
A = sys.argv[1] if len(sys.argv) > 1 else os.path.join(H, "q5_enforcement_cost.json")
B = sys.argv[2] if len(sys.argv) > 2 else os.path.join(H, "q5_judge_latency.json")
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(H, "paper", "figures", "q5_cost.pdf")
c, j = json.load(open(A)), json.load(open(B))

fig, ax = plt.subplots(figsize=(ps.COL_W, 2.35))
fig.subplots_adjust(left=0.16, right=0.985, top=0.975, bottom=0.17)
ax.set_xscale("log"); ax.set_yscale("log")

# hosted judges: median line + IQR band across the whole x range
X0, X1 = 1, 1.6e6
judges = [(k, v) for k, v in j.items() if not k.startswith("_") and v.get("ms")]
judges.sort(key=lambda kv: np.median(kv[1]["ms"]))
for k, v in judges:
    q1, med, q3 = np.percentile(np.array(v["ms"]) * 1e6, [25, 50, 75])        # ms -> ns
    ax.fill_between([X0, X1], q1, q3, color=ps.ORANGE, alpha=0.10, lw=0, zorder=1)
    ax.plot([X0, X1], [med, med], color=ps.ORANGE, lw=0.8, zorder=2)
meds = [np.median(v["ms"]) * 1e6 for _, v in judges]
ax.text(1.35, min(q[0] for q in [np.percentile(np.array(v["ms"]) * 1e6, [25]) for _, v in judges]) / 1.7,
        "one hosted judge call (Claude Haiku, Sonnet, Opus 5.5;\n"
        f"GPT-5.5): medians {min(meds) / 1e9:.2f}–{max(meds) / 1e9:.2f} s, bands = IQR",
        ha="left", va="top", fontsize=8.0, color=ps.INK, linespacing=1.1)

# naive replay monitor (recomputes the policy from the committed history)
rc = np.array(c["B_stream"]["reference_curve"], float)
rc = rc[rc[:, 0] >= 1]
ax.plot(rc[:, 0], rc[:, 1], "o", ms=1.5, color="#5f5e5a", alpha=0.9, zorder=3, mew=0)
ax.text(40, 6e5, "naive replay\nmonitor", ha="left", va="center", fontsize=8.0,
        color=ps.INK, linespacing=1.1)

# compiled gate: per-decade median and p99
gb = c["B_stream"]["gate_bins"]
xm = np.array([np.sqrt(b["from"] * b["to"]) for b in gb]); med = np.array([b["median_ns"] for b in gb])
p99 = np.array([b["p99_ns"] for b in gb])
ax.vlines(xm, med, p99, color=ps.BLUE, lw=0.8, zorder=4)
ax.plot(xm, med, "o", ms=3.4, color=ps.BLUE, mec="white", mew=0.4, zorder=5)
ax.text(9e2, 2.6e3, "compiled T1 gate (P2+P3):\nmedian, p99", ha="left", va="bottom", fontsize=8.0,
        color=ps.INK, linespacing=1.1)

ax.set_xlim(X0, X1); ax.set_ylim(1e2, 1e10)
ax.set_xticks([1, 1e2, 1e4, 1e6]); ax.set_xticklabels(["1", "$10^2$", "$10^4$", "$10^6$"])
ax.set_yticks([1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10])
ax.set_yticklabels(["100 ns", "1 µs", "10 µs", "100 µs", "1 ms", "10 ms", "100 ms", "1 s", "10 s"], fontsize=8.0)
ax.minorticks_off()
ax.set_xlabel("proposals processed so far")
ax.set_ylabel("time per decision")
ax.grid(True, axis="y")
fig.savefig(OUT)
print("wrote", OUT)
for b in gb:
    print(f"gate {b['from']}-{b['to']}: median {b['median_ns']:.0f} ns, p99 {b['p99_ns']:.0f} ns, keys {b['active_keys_at_end']}")
for k, v in judges:
    print(k, "median ms", round(float(np.median(v["ms"])), 0), "IQR", np.percentile(v["ms"], [25, 75]).round(0))
