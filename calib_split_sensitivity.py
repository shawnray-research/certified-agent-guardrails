"""Calibration-size sensitivity (Sec. 7.2): 2,000 random 28/7 splits of the 35 AgentDojo violations; the delta=.10
split-conformal threshold from 28 calibration violations, miss on the 7 held out, false block on all 97 compliant
items, for every judge run (score registry of unify_crossfit.py; cache-only). Usage: python calib_split_sensitivity.py [REPO_DIR]"""
import json, os, sys, numpy as np
R = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, R); os.chdir(R)
src = open("unify_crossfit.py").read(); src = src[:src.index("def crossfit")]
sys.argv = ["x"]; g = {"__file__": os.path.join(R, "unify_crossfit.py")}; exec(src, g)
bench, y, SRC, E = g["bench"], g["y"], g["SRC"], g["E"]
res = {}
for nm, f in SRC:
    d = f(); s = np.array([d.get(t) for t in bench], float)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; rng = np.random.default_rng(0); mi, fb = [], []
    for _ in range(2000):
        pp = rng.permutation(pos); cal, te = pp[:28], pp[28:]
        th = E.conformal_threshold(s[cal], 0.10); mi.append(np.mean(s[te] < th)); fb.append(np.mean(s[neg] >= th))
    res[nm] = (float(np.mean(mi)), float(np.mean(fb)))
fbs = sorted(res.items(), key=lambda kv: kv[1][1])
print("lowest FB:", [(k, round(v[0], 3), round(v[1], 3)) for k, v in fbs[:3]])
print("Gemma:", [round(x, 3) for x in res["Gemma-2-9B"]], "Opus5.5:", [round(x, 3) for x in res["Claude-Opus-5.5"]])
print("miss range:", round(min(v[0] for v in res.values()), 3), round(max(v[0] for v in res.values()), 3), "FB range:", round(fbs[0][1][1], 3), round(fbs[-1][1][1], 3))
