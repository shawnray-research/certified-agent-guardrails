"""Supplement figures B.2-B.3 in the house style of paper_style.py, at the journal column width (3.5 in),
recomputed with the experiment modules themselves (same seeds, no inference):
  synthetic_frontier.pdf  -- run_synthetic.check_C2_frontier: swept thresholds vs. the analytic frontier
  controlled_frontier.pdf -- controlled_model: exact (R, C) of every deterministic policy in each class and the
                             lower convex hull (episode-level mixtures), with the delta = .05 budget
  real_certificate.pdf    -- run_real.check_C1 on the cached qwen2.5:3b scores: realized miss (+-1 sd over 400
                             splits) and utility loss against the target delta
Usage: python make_fig_supp.py [OUT_DIR]"""
import os, sys, io, contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
H_ = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H_)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(H_, "paper", "figures")
with contextlib.redirect_stdout(io.StringIO()):
    import run_synthetic as RS, controlled_model as CM, run_real as RR
    import enforce as E
import paper_style as ps

def style():
    ps.apply()
    plt.rcParams.update({"savefig.bbox": "tight", "savefig.pad_inches": 0.012, "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
                         "legend.fontsize": 8.0})

def dec(ax, x=True, y=True):
    fmt = lambda v: ("0" if abs(v) < 1e-12 else (f"{v:.2f}".rstrip("0").rstrip(".").replace("0.", ".", 1)))
    if x: ax.set_xticks(ax.get_xticks()); ax.set_xticklabels([fmt(v) for v in ax.get_xticks()])
    if y: ax.set_yticks(ax.get_yticks()); ax.set_yticklabels([fmt(v) for v in ax.get_yticks()])

# ---- synthetic frontier
with contextlib.redirect_stdout(io.StringIO()):
    _, (R_emp, U_emp, R_base, U_base) = RS.check_C2_frontier()
style()
dd = np.linspace(1e-4, 0.6, 300); U_star = E.predicted_frontier(dd, RS.MU, RS.H)
fig, ax = plt.subplots(figsize=(ps.COL_W, 2.3))
ax.plot(dd, U_star, color=ps.BLUE, lw=1.1, label=r"analytic frontier $U^{\star}\!(\delta)$", zorder=2)
ax.plot(R_emp, U_emp, "o", ms=2.4, color=ps.INK2, mec="white", mew=0.3, label="swept thresholds (simulated)", zorder=3)
ax.plot([R_base], [U_base], "X", ms=5.5, color=ps.ORANGE, mec="white", mew=0.4, label=r"fixed-$\theta$ baseline", zorder=4)
ax.set_xlim(0, 0.6); ax.set_ylim(0, max(0.3, U_base * 1.2))
ax.set_xticks([0, .1, .2, .3, .4, .5, .6]); ax.set_yticks([0, .1, .2, .3, .4, .5, .6]); ax.set_ylim(0, max(0.3, U_base * 1.2))
dec(ax)
ax.set_xlabel(r"episode-unsafe rate $R$ (= target $\delta$ on the frontier)"); ax.set_ylabel(r"utility loss $U$")
ax.grid(True); ax.legend(loc="upper right")
fig.savefig(os.path.join(OUT, "synthetic_frontier.pdf")); plt.close(fig)

# ---- controlled frontier
classes = [("perfect recall", CM.perfect_recall_policies(), ps.BLUE, "o"),
           ("current score and time", CM.score_time_policies(), ps.GRAY, "s"),
           ("stationary threshold", CM.stationary_policies(thresholds_only=True), ps.ORANGE, "^")]
def lower_hull(pts):
    pts = sorted(set(pts)); hull = []
    for p in pts:
        while len(hull) >= 2 and (hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1]) - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0]) <= 0:
            hull.pop()
        hull.append(p)
    out = [hull[0]]                       # keep the nonincreasing part (C falls as R rises)
    for p in hull[1:]:
        if p[1] <= out[-1][1]: out.append(p)
    return out
style()
fig, ax = plt.subplots(figsize=(ps.COL_W, 2.3))
for name, pol, col, mk in classes:
    P = [(float(r), float(c)) for _, (r, c) in pol]
    hull = lower_hull(P)
    ax.plot([p[0] for p in hull], [p[1] for p in hull], color=col, lw=1.0, zorder=2, label=name)
    ax.plot([p[0] for p in P], [p[1] for p in P], mk, ms=2.8, color=col, mec="white", mew=0.3, zorder=3)
d = float(CM.DELTA)
ax.axvline(d, color=ps.INK2, lw=0.6, ls=(0, (2.5, 1.5)), zorder=1)
ax.text(d + 0.006, 1.32, r"$\delta$ = .05", fontsize=8.0, color=ps.INK2, va="top")
for name, pol, col, mk in classes:
    c = float(CM.min_cost(pol)[0])
    ax.plot([d], [c], "o", ms=4.0, mfc="white", mec=col, mew=0.9, zorder=4)
ax.set_xlim(-0.01, 0.46); ax.set_ylim(-0.03, 1.36)
ax.set_xticks([0, .1, .2, .3, .4]); ax.set_yticks([0, .25, .5, .75, 1.0, 1.25]); dec(ax)
ax.set_xlabel("episode-unsafe probability $R$"); ax.set_ylabel("expected compliant blocks $C$")
ax.grid(True); ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, handlelength=1.4, columnspacing=1.0)
fig.savefig(os.path.join(OUT, "controlled_frontier.pdf")); plt.close(fig)

# ---- real certificate (curated probe, qwen2.5:3b)
with contextlib.redirect_stdout(io.StringIO()):
    scores, labels = RR.load_scores()
    _, rows = RR.check_C1(scores, labels)
style()
dl = [r[0] for r in rows]; miss = [r[1] for r in rows]; sd = [r[2] for r in rows]; util = [r[3] for r in rows]
fig, ax = plt.subplots(figsize=(ps.COL_W, 2.3))
fig.subplots_adjust(right=0.86)
ax.plot([0, 0.35], [0, 0.35], color=ps.GRAY, lw=0.6, ls=(0, (2.5, 1.5)), label=r"$y=\delta$")
ax.errorbar(dl, miss, yerr=sd, fmt="o-", color=ps.BLUE, ms=3.0, lw=1.0, capsize=2, elinewidth=0.7, label="realized miss (left)")
ax2 = ax.twinx()
ax2.plot(dl, util, "s--", color=ps.ORANGE, ms=3.0, lw=1.0, label="utility loss $U$ (right)")
ax.set_xlim(0, 0.35); ax.set_ylim(0, 0.35); ax2.set_ylim(0, 1.05)
ax.set_xticks([0, .05, .1, .2, .3]); ax.set_yticks([0, .1, .2, .3]); ax2.set_yticks([0, .25, .5, .75, 1.0]); dec(ax); dec(ax2, x=False)
for sp in ("right",): ax2.spines[sp].set_visible(True)
ax2.spines["top"].set_visible(False)
ax.set_xlabel(r"target miss rate $\delta$"); ax.set_ylabel("realized miss rate"); ax2.set_ylabel("utility loss $U$")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.16), handlelength=1.6, columnspacing=0.9)
ax.grid(True)
fig.savefig(os.path.join(OUT, "real_certificate.pdf")); plt.close(fig)
print("wrote synthetic_frontier.pdf, controlled_frontier.pdf, real_certificate.pdf to", OUT)
print("C1 rows:", [(round(a, 2), round(b, 3), round(c, 3), round(u, 3)) for a, b, c, u, _ in rows])
print("min costs:", {n: float(CM.min_cost(p)[0]) for n, p, _, _ in classes})
