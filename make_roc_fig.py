"""Per-judge empirical ROC curves on the curated set and AgentDojo. The paper's
central object is the judge ROC (Thm 2); AUC is only its one-number summary. This
figure shows that judges of similar AUC differ in ROC \emph{shape}, which is what
matters at a target safety level (the low false-alarm region). Colorblind-safe:
each judge has a distinct color (Okabe-Ito), linestyle, and marker; the AgentDojo
panel carries an inset zoomed to the low-false-alarm region."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import style as _style; _style.apply_style()
from agentdojo.task_suite.load_suites import get_suites

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
sc = json.load(open(f"{HERE}/scores_cache.json")); bc = json.load(open(f"{HERE}/bench_scores_cache.json"))

acts = json.load(open(f"{HERE}/actions.json"))["actions"]
cur_t = [a["text"] for a in acts]; cur_l = np.array([a["label"] for a in acts])
suites = get_suites("v1.2.1"); ben_t = []; ben_l = []
for n, s in suites.items():
    for u in s.user_tasks.values(): ben_t.append(u.PROMPT); ben_l.append(0)
    for i in s.injection_tasks.values(): ben_t.append(i.GOAL); ben_l.append(1)
ben_l = np.array(ben_l)

def roc(scores, labels):
    ths = np.linspace(-0.01, 1.01, 200)
    tpr = np.array([np.mean(scores[labels == 1] >= t) for t in ths])
    fpr = np.array([np.mean(scores[labels == 0] >= t) for t in ths])
    return fpr, tpr
def auc(s, l):
    u, v = s[l == 1], s[l == 0]
    return float(np.mean([(x > y) + 0.5 * (x == y) for x in u for y in v]))

# Okabe-Ito colorblind-safe palette + distinct linestyle + marker per judge
JUDGES = [("qwen2.5:3b", "Qwen-3B",  "#0072B2", "-",  "o"),
          ("qwen2.5:7b", "Qwen-7B",  "#E69F00", "--", "s"),
          ("qwen2.5:32b","Qwen-32B", "#009E73", "-.", "^"),
          ("llama3.1:8b","Llama-8B", "#D55E00", ":",  "v"),
          ("gemma2:9b",  "Gemma-9B", "#CC79A7", "-",  "D"),
          ("llama3.3:70b","Llama-70B","#56B4E9", "--", "P")]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.9))
panels = [
    (a1, lambda m: np.array([sc.get(f"{m}::{t}", np.nan) for t in cur_t]), cur_l, "curated", False),
    (a2, lambda m: np.array([bc.get(f"{m}::::{t}", np.nan) for t in ben_t]), ben_l, "AgentDojo", True)]
for ax, getfn, labels, title, inset in panels:
    axin = None
    if inset:
        axin = inset_axes(ax, width="52%", height="52%", loc="lower right", borderpad=1.1)
    for m, name, c, ls, mk in JUDGES:
        s = getfn(m)
        if np.isnan(s).any(): continue
        f, t = roc(s, labels)
        ax.plot(f, t, ls, color=c, lw=1.9, marker=mk, markevery=18, markersize=5,
                label=f"{name} ({auc(s,labels):.2f})")
        if axin is not None:
            axin.plot(f, t, ls, color=c, lw=1.9, marker=mk, markevery=8, markersize=4)
    ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.5)
    ax.set_xlabel("false-alarm rate"); ax.set_ylabel("true-positive rate")
    ax.set_title(title); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9.5, loc="upper left", title="judge (AUC)", title_fontsize=9.5)
    if axin is not None:
        axin.set_xlim(0, 0.20); axin.set_ylim(0, 0.65)
        axin.set_title("low false-alarm region", fontsize=8.5)
        axin.tick_params(labelsize=7.5); axin.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.4)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "roc.pdf")); plt.close()
print("wrote figures/roc.pdf (colorblind-safe, low-FPR inset)")
