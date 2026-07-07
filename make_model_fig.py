"""Draw the enforcement-model schematic: agent -> judge -> gate -> commit,
with the adversary perturbing the judge input. Rendered to figures/model.pdf and
included as a spanning figure in the paper."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import style as _style; _style.apply_style()
plt.rcParams.update({"font.size": 12, "figure.dpi": 200})

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)

fig, ax = plt.subplots(figsize=(7.2, 2.15))
ax.set_xlim(0, 10); ax.set_ylim(0, 3.55); ax.axis("off")

boxes = [
    (0.9, "LLM agent", "proposes\naction $a$"),
    (3.35, "Judge", "score $s\\in[0,1]$"),
    (5.8, "Gate", "$s\\geq\\theta$?\nblock / allow"),
    (8.25, "World", "irreversible\ncommit"),
]
w, h, y = 1.55, 1.15, 1.55
for x, title, sub in boxes:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                                linewidth=1.3, edgecolor="#333333", facecolor="#f2f4f8"))
    ax.text(x + w/2, y + h*0.66, title, ha="center", va="center", fontweight="bold")
    ax.text(x + w/2, y + h*0.28, sub, ha="center", va="center", fontsize=10)

def arrow(x0, x1, color="#333333", label=None, ly=None):
    ax.add_patch(FancyArrowPatch((x0, y + h/2), (x1, y + h/2), arrowstyle="-|>",
                                 mutation_scale=16, linewidth=1.4, color=color))
    if label:
        ax.text((x0 + x1)/2, (ly or y + h/2 + 0.28), label, ha="center", va="bottom", fontsize=9.5)

arrow(0.9 + w, 3.35, label="$a$")
arrow(3.35 + w, 5.8, label="$s$")
arrow(5.8 + w, 8.25, label="commit")

# adversary perturbs the judge input (dashed red shaft + clean solid head, from below).
# Draw the shaft and head as separate patches so the dashed linestyle does not break the
# arrowhead outline (which otherwise renders with a notch/divot).
ax.add_patch(FancyArrowPatch((3.35 + w/2, y - 0.7), (3.35 + w/2, y - 0.16), arrowstyle="-",
                             linewidth=1.4, color="#c0392b", linestyle="dashed"))
ax.add_patch(FancyArrowPatch((3.35 + w/2, y - 0.22), (3.35 + w/2, y), arrowstyle="-|>",
                             mutation_scale=15, linewidth=1.4, color="#c0392b"))
ax.text(3.35 + w/2, y - 0.95, "adversary / injection (budget $\\rho$)",
        ha="center", va="center", color="#c0392b", fontsize=9.5)

# calibrated threshold feeds the gate (from above); same line length as the red arrow
ax.add_patch(FancyArrowPatch((5.8 + w/2, y + h + 0.7), (5.8 + w/2, y + h), arrowstyle="-|>",
                             mutation_scale=15, linewidth=1.4, color="#2471a3"))
ax.text(5.8 + w/2, y + h + 0.75, "conformal $\\theta$ for target $\\delta$",
        ha="center", va="bottom", color="#2471a3", fontsize=9.5)

fig.tight_layout(pad=0.2)
fig.savefig(os.path.join(FIG, "model.pdf"), bbox_inches="tight", pad_inches=0.02)
print("wrote", os.path.join(FIG, "model.pdf"))
