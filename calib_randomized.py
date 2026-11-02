"""Calibration-cost decomposition for the context-free judge panel (Supplement; reviewer request).
For every judge run of unify_crossfit.py (same scores, refusals, 500 label-balanced two-fold splits, seed 0):
  (1) deterministic split-conformal operating points (as in the paper), and
  (2) the exact-level randomized (smoothed) split-conformal rule: a test item with score s is allowed with
      probability clip((delta (n+1) - #{c_i < s}) / (1 + #{c_i = s}), 0, 1) given the fold's n calibration
      violation scores c_i, which makes the marginal miss exactly delta under exchangeability; we report
      expected miss and false block (no Monte Carlo over U).
  (3) For Claude Opus 5.5 and Gemma-2-9B: a label-stratified item bootstrap (B = 300 resamples of the 35
      violations and 97 compliant items, 50 splits each, seed 1) of the deterministic cross-fitted false
      block at delta = .10, as a 95% percentile interval.
Usage: python calib_randomized.py [--json OUT]"""
import argparse, json, os, sys, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import enforce as E
ap = argparse.ArgumentParser(); ap.add_argument("--json", default=None); a = ap.parse_args()
src = open(os.path.join(H, "unify_crossfit.py")).read()
start = src.index("L = lambda f:"); stop = src.index("def crossfit(")
ns = {"np": np, "json": json, "os": os, "H": H, "E": E}
exec(src[start:stop], ns)                       # same score loading and refusal rules as unify_crossfit.py
bench, y, SRC = ns["bench"], ns["y"], ns["SRC"]

def allow_prob(cal, s, delta):
    cal = np.sort(cal); n = cal.size
    below = np.searchsorted(cal, s, side="left"); ties = np.searchsorted(cal, s, side="right") - below
    return np.clip((delta * (n + 1) - below) / (1.0 + ties), 0.0, 1.0)

def crossfit(s, yy, delta, splits=500, seed=0):
    rng = np.random.default_rng(seed); pos, neg = np.where(yy == 1)[0], np.where(yy == 0)[0]
    dm, df, rm, rf = [], [], [], []
    for _ in range(splits):
        pp, nn = rng.permutation(pos), rng.permutation(neg)
        P = [pp[: len(pp) // 2], pp[len(pp) // 2:]]; N = [nn[: len(nn) // 2], nn[len(nn) // 2:]]
        for c in (0, 1):
            cal = s[P[c]]; th = E.conformal_threshold(cal, delta)
            dm.append(np.mean(s[P[1 - c]] < th)); df.append(np.mean(s[N[1 - c]] >= th))
            rm.append(np.mean(allow_prob(cal, s[P[1 - c]], delta))); rf.append(1 - np.mean(allow_prob(cal, s[N[1 - c]], delta)))
    return [float(np.mean(v)) for v in (dm, df, rm, rf)]

rows, scores = [], {}
for nm, f in SRC:
    d = f(); s = np.array([d.get(t) for t in bench], float); scores[nm] = s
    r = {"judge": nm}
    for dl in (0.05, 0.10, 0.20):
        dm, df, rm, rf = crossfit(s, y, dl)
        r[f"det_{dl:.2f}"] = [dm, df]; r[f"rand_{dl:.2f}"] = [rm, rf]
    rows.append(r)
    print(f"{nm:27s} " + "  ".join(f"d{dl:.2f} det {r[f'det_{dl:.2f}'][0]:.3f}/{r[f'det_{dl:.2f}'][1]:.3f} rand {r[f'rand_{dl:.2f}'][0]:.3f}/{r[f'rand_{dl:.2f}'][1]:.3f}" for dl in (0.05, 0.10, 0.20)))

boot = {}
rng = np.random.default_rng(1)
pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
for nm in ("Claude-Opus-5.5", "Gemma-2-9B"):
    s = scores[nm]; vals = []
    for b in range(300):
        idx = np.r_[rng.choice(pos, pos.size, replace=True), rng.choice(neg, neg.size, replace=True)]
        vals.append(crossfit(s[idx], y[idx], 0.10, splits=50, seed=b)[1])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    boot[nm] = {"fb_0.10_point": [r for r in rows if r["judge"] == nm][0]["det_0.10"][1], "ci95": [float(lo), float(hi)]}
    print(nm, "deterministic false block at .10:", round(boot[nm]["fb_0.10_point"], 3), "bootstrap 95%:", [round(lo, 3), round(hi, 3)])
if a.json:
    json.dump({"rows": rows, "bootstrap": boot, "splits": 500, "seed": 0}, open(a.json, "w"), indent=1)
