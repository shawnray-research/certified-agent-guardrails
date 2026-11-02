"""Supplement B.7: which of the 33 AgentDojo judge runs dominate Llama-Guard-3-8B's single operating point
(higher detection rate at no more false blocks), among judges whose AUC exceeds its binary-point AUC.
Uses the score registry of unify_crossfit.py; cache-only. Usage: python llamaguard_dominance.py [REPO_DIR]"""
import json, os, sys, numpy as np
R = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, R); os.chdir(R)
src = open(os.path.join(R, "unify_crossfit.py")).read()
src = src[:src.index("def crossfit")]
sys.argv = ["x"]; g = {"__file__": os.path.join(R, "unify_crossfit.py")}; exec(src, g)
bench, y, SRC = g["bench"], g["y"], g["SRC"]
lg = json.load(open("llamaguard_bench.json")); l = np.array([lg[t] for t in bench], float)
fpr0 = np.mean(l[y == 0] >= 0.5); tpr0 = np.mean(l[y == 1] >= 0.5); bauc = (1 + tpr0 - fpr0) / 2
print(f"LG point: TPR {tpr0:.3f} FPR {fpr0:.3f} binary AUC {bauc:.3f}")
rows = []
for nm, f in SRC:
    d = f(); s = np.array([d.get(t) for t in bench], float)
    u, v = s[y == 1][:, None], s[y == 0][None, :]; auc = float(np.mean((u > v) + 0.5 * (u == v)))
    best = max(np.mean(s[y == 1] >= t) for t in np.unique(s) if np.mean(s[y == 0] >= t) <= fpr0 + 1e-12) if any(np.mean(s[y == 0] >= t) <= fpr0 + 1e-12 for t in np.unique(s)) else 0.0
    rows.append((nm, auc, best))
hi = [r for r in rows if r[1] > bauc]
dom = [r for r in hi if r[2] > tpr0 + 1e-12]
print(f"judges: {len(rows)}; AUC > binary AUC: {len(hi)}; dominating (TPR > {tpr0:.3f} at FPR <= {fpr0:.3f}): {len(dom)}")
print("not dominating:", [(r[0], round(r[2], 3)) for r in hi if r not in dom])
for nm in ["Qwen-2.5-32B", "Llama-3.3-70B"]:
    print(nm, [round(r[2], 3) for r in rows if r[0] == nm])
