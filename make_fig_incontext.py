"""Figure (single column, 3.5 in): the same seven hosted judges on the same AgentDojo ground-truth tool calls,
scored on the serialized call alone ("action only") versus in their execution context ("in context": user task,
tool outputs, and the call), from frontier2026/toolcall_incontext_summary.json (toolcall_incontext_analyze.py).
  (a) AUC of violating calls (V1) against compliant calls in clean contexts (C0), with bootstrap 95% CIs;
  (b) cross-fitted split-conformal false block at delta = .10 (mean over splits; whiskers: 2.5-97.5 split
      percentiles) on C0, plus the in-context false block on compliant calls that follow an injection (C1).
House style from paper_style.py (8 pt text): GRAY = action only, BLUE = in context, ORANGE = injected context.
Usage: python make_fig_incontext.py [summary.json] [OUT.pdf]"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import paper_style as ps
ps.apply()
plt.rcParams["savefig.bbox"] = "standard"  # keep the exact drawn size (3.5 in = \columnwidth)

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(H, "frontier2026", "toolcall_incontext_summary.json")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(H, "paper", "figures", "incontext.pdf")
S = json.load(open(SRC))
NAME = {"claude-opus-5-5": "Claude Opus 5.5", "claude-fable-5-1": "Claude Fable 5.1", "claude-sonnet-5-5": "Claude Sonnet 5.5",
        "claude-haiku-5-5": "Claude Haiku 5.5", "gpt-5.5": "GPT-5.5", "gpt-6.1-sol": "GPT-6.1 Sol", "gpt-6.1-terra": "GPT-6.1 Terra"}
J = [m for m in S if m in NAME]
assert len(J) == 7
# sort by in-context AUC, then action-only AUC (best at the top)
J.sort(key=lambda m: (S[m]["incontext"]["auc_c0"], S[m]["action"]["auc_c0"]))
G = "0.10"

W, HT = ps.COL_W, 2.20
fig = plt.figure(figsize=(W, HT))
L0, AW, GAP, BW = 0.93, 1.13, 0.22, 1.13          # inches: label margin, panel widths, gap
B0, PH = 0.36, 1.40                                # inches: bottom margin, panel height
axa = fig.add_axes([L0 / W, B0 / HT, AW / W, PH / HT])
axb = fig.add_axes([(L0 + AW + GAP) / W, B0 / HT, BW / W, PH / HT])
y = range(len(J))
LW_CI = 0.8

for i, m in enumerate(J):
    a, c = S[m]["action"], S[m]["incontext"]
    # (a) AUC dumbbell
    axa.plot([a["auc_c0"], c["auc_c0"]], [i, i], color=ps.LIGHTGRAY, lw=1.6, solid_capstyle="round", zorder=1)
    axa.plot(a["auc_c0_ci"], [i - 0.12] * 2, color=ps.GRAY, lw=LW_CI, zorder=2)   # 95% intervals, just off the row
    axa.plot(c["auc_c0_ci"], [i + 0.12] * 2, color=ps.BLUE, lw=LW_CI, zorder=2)
    axa.plot(a["auc_c0"], i, "o", ms=3.6, mfc="white", mec=ps.GRAY, mew=0.9, zorder=3)
    axa.plot(c["auc_c0"], i, "o", ms=3.6, color=ps.BLUE, mec=ps.BLUE, zorder=4)
    # (b) false block at delta = .10
    fa, fc, f1 = a["gates"][G]["fb_c0"], c["gates"][G]["fb_c0"], c["gates"][G]["fb_c1"]
    axb.plot([fa[1], fa[2]], [i - 0.12] * 2, color=ps.GRAY, lw=LW_CI, zorder=2)
    axb.plot([fc[1], fc[2]], [i + 0.12] * 2, color=ps.BLUE, lw=LW_CI, zorder=2)
    axb.plot([f1[1], f1[2]], [i + 0.30] * 2, color=ps.ORANGE, lw=LW_CI, alpha=0.8, zorder=2)
    axb.plot([fc[0], fa[0]], [i, i], color=ps.LIGHTGRAY, lw=1.6, solid_capstyle="round", zorder=1)
    axb.plot(fa[0], i, "o", ms=3.6, mfc="white", mec=ps.GRAY, mew=0.9, zorder=3)
    axb.plot(f1[0], i + 0.30, "D", ms=3.0, color=ps.ORANGE, mec="white", mew=0.3, zorder=6)   # lifted: often near 0 with C0
    axb.plot(fc[0], i, "o", ms=3.6, color=ps.BLUE, mec=ps.BLUE, zorder=5)

axa.set_yticks(list(y)); axa.set_yticklabels([NAME[m] for m in J]); axa.tick_params(axis="y", length=0)
axb.set_yticks(list(y)); axb.set_yticklabels([]); axb.tick_params(axis="y", length=0)
for ax in (axa, axb):
    ax.set_ylim(-0.6, len(J) - 0.4); ax.grid(True, axis="x")
axa.set_xlim(0.5, 1.02); axa.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
axa.set_xticklabels([".5", ".6", ".7", ".8", ".9", "1"])
axa.set_xlabel("AUC (injected vs. clean)")
axb.set_xlim(-0.02, 1.0); axb.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); axb.set_xticklabels(["0", ".25", ".5", ".75", "1"])
axb.set_xlabel(r"false block at $\delta$ = .10")
ps.panel_label(axa, "a", x=-0.03, y=1.01); ps.panel_label(axb, "b", x=-0.03, y=1.01)

# legend above both panels
h1, = axa.plot([], [], "o", ms=3.6, mfc="white", mec=ps.GRAY, mew=0.9, label="action only")
h2, = axa.plot([], [], "o", ms=3.6, color=ps.BLUE, label="in context")
h3, = axa.plot([], [], "D", ms=3.0, color=ps.ORANGE, mec="white", mew=0.3,
               label="in context, compliant call after an injection")
kw = dict(loc="upper left", frameon=False, handletextpad=0.25, columnspacing=1.2, borderaxespad=0.05)
fig.legend(handles=[h1, h2], ncol=2, bbox_to_anchor=(0.005, 1.0), **kw)
fig.legend(handles=[h3], ncol=1, bbox_to_anchor=(0.005, 1.0 - 0.15 / HT), **kw)
fig.savefig(OUT)
print("wrote", OUT)
for m in J[::-1]:
    a, c = S[m]["action"], S[m]["incontext"]
    print(f"{NAME[m]:18s} AUC {a['auc_c0']:.3f} -> {c['auc_c0']:.3f}; fb@.10 C0 {a['gates'][G]['fb_c0'][0]:.3f} -> "
          f"{c['gates'][G]['fb_c0'][0]:.3f}; C1 in context {c['gates'][G]['fb_c1'][0]:.3f}; miss in context {c['gates'][G]['miss'][0]:.3f}")
