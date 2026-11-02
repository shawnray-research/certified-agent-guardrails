"""Figure 1 (single column): the guarded tool-using agent and where each boundary applies.

Drawn at true print size (3.50 in = IEEE compsoc journal \\columnwidth) in an inch-based
coordinate system, so every font size below is the size on the page (all text >= 7.5 pt; labels 8 pt).
House style from paper_style.py: STIX text and math, BLUE = trusted computing base
(defender side), ORANGE = adversary, GRAY = untrusted or fallible components; text in INK.
Usage: python make_overview_fig.py [out.pdf]"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps

ps.apply()
plt.rcParams["savefig.bbox"] = "standard"  # keep the exact drawn size

OUT = sys.argv[1] if len(sys.argv) > 1 else "overview.pdf"
W, H = ps.COL_W, 2.50
BLUE, ORANGE, GRAY, INK, INK2 = ps.BLUE, ps.ORANGE, ps.GRAY, ps.INK, ps.INK2
TCB_FILL, BOX_FILL, UNTRUSTED_FILL = "#edf3fb", "#ffffff", "#f4f4f2"
LW = 0.6
DASH = (0, (2.6, 1.6))
T, S = 8.5, 8.0          # title and label sizes (pt)

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x0, y0, x1, y1, edge=INK2, fill=BOX_FILL, ls="-", lw=LW, r=0.045, z=2):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
                                linewidth=lw, edgecolor=edge, facecolor=fill, linestyle=ls, zorder=z))


def text(x, y, s, size=S, color=INK, ha="center", va="center", z=5, **kw):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, zorder=z, **kw)


def tab(x, y, label):
    """Small dark tab 'Tk' centered at (x, y): marks where boundary k applies."""
    w, h = 0.21, 0.13
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.025",
                                linewidth=0, facecolor=INK, zorder=6))
    text(x, y - 0.002, label, size=7.5, color="white", z=7)


def arrow(pts, color=INK2, ls="-", lw=LW, z=3):
    path = Path(pts, [Path.MOVETO] + [Path.LINETO] * (len(pts) - 1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=6.5, linewidth=lw,
                                 color=color, linestyle=ls, shrinkA=0, shrinkB=0, joinstyle="miter",
                                 capstyle="butt", zorder=z))


# ---- trusted computing base: the gate --------------------------------------------------------
GX0, GX1, GY0, GY1 = 0.92, 2.71, 0.92, 2.18
box(GX0, GY0, GX1, GY1, edge=BLUE, fill=TCB_FILL, ls=DASH, lw=0.7, r=0.07, z=1)
text(GX0 + 0.06, GY1 - 0.075, "gate (trusted computing base)", size=S, color=BLUE, ha="left")

# policy automaton (T1: what can be enforced; T3: what can be analyzed before deployment)
PX0, PX1 = 0.99, 2.01
box(PX0, 1.42, PX1, 1.98, edge=BLUE)
text((PX0 + PX1) / 2, 1.845, "policy automaton", size=T)
text((PX0 + PX1) / 2, 1.70, "registers, counters", size=S)
text((PX0 + PX1) / 2, 1.565, r"over oracle bits $\Pi$", size=S)
tab(PX0 + 0.135, 1.98, "T1")
tab(PX0 + 0.385, 1.98, "T3")

# score test on the judge's score (T2: frontier and calibrated certificate)
box(PX0, 0.99, PX1, 1.33, edge=BLUE)
text((PX0 + PX1) / 2, 1.215, r"block iff $s\geq\hat\theta$", size=T)
text((PX0 + PX1) / 2, 1.075, r"$\hat\theta$ calibrated for $\delta$", size=S)
tab(PX0 + 0.135, 1.33, "T2")

# decision: allow commits a, block commits the no-op
DX0, DX1, DY0, DY1 = 2.09, 2.64, 1.18, 1.62
box(DX0, DY0, DX1, DY1, edge=BLUE)
text((DX0 + DX1) / 2, 1.475, r"allow: $a$", size=S)
text((DX0 + DX1) / 2, 1.325, r"block: $\bot$", size=S)
arrow([(PX1, 1.70), (DX0, DY1 - 0.07)])
arrow([(PX1, 1.16), (DX0, DY0 + 0.07)])

# ---- agent, untrusted input, judge, world ----------------------------------------------------
AX0, AX1 = 0.02, 0.78
AC = (AX0 + AX1) / 2
box(AX0, 1.20, AX1, 1.72)
text(AC, 1.585, "LLM agent", size=T)
text(AC, 1.44, "proposes", size=S)
text(AC, 1.31, r"tool call $a$", size=S)

box(AX0, 0.44, AX1, 0.96, edge=GRAY, fill=UNTRUSTED_FILL)
text(AC, 0.835, "untrusted", size=T)
text(AC, 0.69, "tool outputs,", size=S)
text(AC, 0.56, "email, web", size=S)
arrow([(AC, 0.96), (AC, 1.20)])

JX0, JX1, JY0, JY1 = 0.99, 2.01, 0.40, 0.84
JC = (JX0 + JX1) / 2
box(JX0, JY0, JX1, JY1, edge=GRAY, fill=UNTRUSTED_FILL)
text(JC, 0.71, "judge (fallible)", size=T)
text(JC, 0.545, r"reads $z=J(\tau,o,a)$", size=S)
arrow([(JC, JY1), (JC, 0.99)])
text(JC + 0.05, 0.915, r"$s$", size=S, ha="left")

# proposal a to the gate; the judge reads (tau, a) from the agent and o from untrusted input
arrow([(AX1, 1.60), (PX0, 1.70)])
text(0.885, 1.735, r"$a$", size=S)
arrow([(0.72, 1.20), (JX0, JY1 - 0.08)])
text(0.80, 1.04, r"$\tau,a$", size=S, ha="right")
arrow([(AX1, 0.62), (JX0, 0.62)])
text((AX1 + JX0) / 2, 0.68, r"$o$", size=S)

WX0, WX1 = 2.79, 3.48
WC = (WX0 + WX1) / 2
box(WX0, 1.14, WX1, 1.66)
text(WC, 1.525, "world", size=T)
text(WC, 1.38, "irreversible", size=S)
text(WC, 1.25, "effects", size=S)
arrow([(DX1, 1.40), (WX0, 1.40)])

# ---- feedback: the committed trace (with the no-op on a block) reaches the agent (T2) --------
FB_Y = 2.28
arrow([(WC, 1.66), (WC, FB_Y), (AC, FB_Y), (AC, 1.72)], color=INK2, ls=DASH)
tab(0.53, FB_Y + 0.11, "T2")
text(0.665, FB_Y + 0.107, r"after a block, the agent sees $\bot$ and may re-propose",
     size=S, ha="left")

# ---- adversary (T2: injected content and a budgeted rewrite of what the judge sees) ----------
text(0.04, 0.12, "adversary", size=T, color=ORANGE, ha="left")
arrow([(0.20, 0.20), (0.20, 0.44)], color=ORANGE, ls=DASH)
text(0.245, 0.32, "injects", size=S, color=ORANGE, ha="left")
arrow([(0.66, 0.12), (1.22, 0.12), (1.22, JY0)], color=ORANGE, ls=DASH)
tab(1.42, 0.20, "T2")
text(1.545, 0.20, r"rewrites $z$, budget $\rho$", size=S, color=ORANGE, ha="left")

fig.savefig(OUT)
print("wrote", OUT)
