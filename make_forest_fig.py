"""Figure 3 (single column): paired AUC differences from Gemma-2-9B with max-T simultaneous 95%
intervals, drawn at true column width so labels print at ~6.5pt.
Input: JSON written by `python analyze_agentdojo_inference.py --family paper --json forest.json`."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as _style; _style.apply_style()
plt.rcParams.update({"font.size": 6.6, "axes.labelsize": 6.8, "xtick.labelsize": 6.2, "ytick.labelsize": 6.4,
                     "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "axes.grid": False, "lines.linewidth": 0.9, "axes.linewidth": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 0, "axes.spines.right": False, "axes.spines.top": False})
src = sys.argv[1] if len(sys.argv) > 1 else "forest.json"
out = sys.argv[2] if len(sys.argv) > 2 else "agentdojo_forest.pdf"
d = json.load(open(src))
names = d["names"]
if len(names) == len(d["d"]) + 1:      # names[0] is the reference judge (Gemma-2-9B)
    names = names[1:]
rows = list(zip(names, d["d"], d["lo"], d["hi"]))
rows.sort(key=lambda r: r[1])            # most negative at the bottom after inversion below
INK, HI, GRID = "#3a3a3a", "#c4561d", "#d9d9d9"
fig, ax = plt.subplots(figsize=(3.45, 3.25))
for i, (n, dd, lo, hi) in enumerate(rows):
    col = HI if hi < 0 or lo > 0 else INK
    ax.plot([lo, hi], [i, i], color=col, lw=1.0 if col == HI else 0.8, solid_capstyle="butt")
    ax.plot([lo, lo], [i - .18, i + .18], color=col, lw=0.6); ax.plot([hi, hi], [i - .18, i + .18], color=col, lw=0.6)
    ax.plot(dd, i, "o", ms=2.6, color=col, zorder=3)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
ax.axvline(0, color="black", lw=0.7)
for x in (-.05, .05): ax.axvline(x, color="#7a7a7a", lw=0.5, ls=(0, (3, 2)))
ax.set_ylim(-0.7, len(rows) - 0.3)
ax.set_xlabel("AUC difference vs. Gemma-2-9B (95% max-$T$ simultaneous CI)")
ax.xaxis.grid(True, color=GRID, lw=0.4); ax.set_axisbelow(True)
fig.tight_layout(pad=0.15)
fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
print("wrote", out, len(rows), "contrasts")
