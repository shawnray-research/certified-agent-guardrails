"""Figure 3 (single column, 3.5 in): paired AUC differences from Gemma-2-9B over the 33-run family with
max-T simultaneous 95% intervals, the predeclared +-.05 equivalence margin, and each judge's AUC.
Input: the JSON written by
    python analyze_agentdojo_inference.py --family complete --json forest33.json
Usage: python make_fig_forest.py [forest33.json] [OUT.pdf]"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import paper_style as ps
ps.apply()
plt.rcParams.update({"savefig.bbox": "tight", "savefig.pad_inches": 0.012, "xtick.labelsize": 8.0, "axes.labelsize": 8.0})
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(H, "forest33.json")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(H, "paper", "figures", "agentdojo_forest_col.pdf")
COMPACT = len(sys.argv) > 3 and sys.argv[3] == "compact"     # main-paper version: 3.3 in tall
d = json.load(open(SRC))
names, auc = d["names"], d["auc"]
ref_name, ref_auc = names[0], auc[0]
nice = lambda n: n.replace("Claude-Opus-", "Claude Opus ").replace("Claude-Sonnet-", "Claude Sonnet ").replace("Claude-Haiku-", "Claude Haiku ").replace("Claude-Fable-", "Claude Fable ").replace("GPT-6.1-", "GPT-6.1 ")
TIED = {"Llama-3.1-8B"}   # 87 of 132 scores exactly .5: AUC dominated by ties (marked with a dagger; see caption)
lab = lambda n: nice(n) + (r"$^{\dagger}$" if n in TIED else "")
rows = sorted(zip([lab(n) for n in names[1:]], auc[1:], d["d"], d["lo"], d["hi"]), key=lambda r: r[2])

fig, ax = plt.subplots(figsize=(ps.COL_W + 0.43, 4.30 if COMPACT else 4.75))   # tight crop trims ~0.43 in -> 3.5 in, 1:1
fig.subplots_adjust(left=0.37, right=0.875, top=0.962 if COMPACT else 0.955, bottom=0.075 if COMPACT else 0.085)
ax.axvspan(-0.05, 0.05, color=ps.LIGHTGRAY, alpha=0.55, lw=0, zorder=0)
for x0 in (-0.05, 0.05):
    ax.axvline(x0, color=ps.GRAY, lw=0.5, ls=(0, (2.5, 1.5)), zorder=1)
ax.axvline(0, color=ps.INK, lw=0.6, zorder=1)
for i, (n, a, dd, lo, hi) in enumerate(rows):
    sup, inf = lo > 0, hi < 0
    col = ps.BLUE if sup else ps.ORANGE if inf else ps.GRAY
    ax.plot([lo, hi], [i, i], color=col, lw=1.0 if (sup or inf) else 0.8, solid_capstyle="butt", zorder=2)
    ax.plot(dd, i, "o", ms=3.0, color=col, mfc=col if (sup or inf) else "white", mec=col, mew=0.7, zorder=3)   # filled = excludes 0
    ax.text(1.035, i, f"{a:.3f}"[1:], transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=8.0, color=ps.INK if (sup or inf) else ps.INK2)
ax.text(1.035, len(rows) - 0.1, "AUC", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
        fontsize=8.0, color=ps.INK)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8.0)
ax.tick_params(axis="y", length=0)
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_xlim(-0.33, 0.235)
ax.set_xticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2]); ax.set_xticklabels(["−.3", "−.2", "−.1", "0", ".1", ".2"])
ax.set_xlabel(f"AUC difference from {ref_name} (AUC {ref_auc:.3f})".replace("0.", ".", 1))
ax.text(0.0, len(rows) - 0.1, "±.05 margin", ha="center", va="bottom", fontsize=8.0, color=ps.INK2)
ax.grid(True, axis="x")
fig.savefig(OUT)
print("wrote", OUT, len(rows), "contrasts; reference", ref_name, round(ref_auc, 3))
