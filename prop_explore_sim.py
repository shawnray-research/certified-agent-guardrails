"""Exact-model check of Proposition prop:explore on the two-system construction of Proposition prop:feedback.

H = 2, uninformative judge. Step 1 violates w.p. q. If step 1 is allowed, both systems next propose a
compliant action. After a block, system A proposes a compliant action and system B one that violates
w.p. p. Evaluated gate pi: block step 1, allow step 2. Truth: episode-unsafe risk R_A(pi) = 0,
R_B(pi) = p; intervention cost C(pi) = 1 - q in both systems.
Logging gates: ungated (never blocks), or epsilon-exploration (blocks each step independently w.p.
epsilon). The per-decision importance-sampling estimate of R(pi) from n logged episodes is computed
for each system; ungated logs have the same law under A and B (post-block dynamics never occur),
so no estimator can separate them, whereas epsilon-logs identify both risks.
Usage: python prop_explore_sim.py [OUT.json]"""
import json, os, sys
import numpy as np
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "prop_explore_sim.json")
q, p, REPS = 0.3, 0.5, 2000

def logs(system, eps, n, rng):
    y1 = rng.random(n) < q
    b1 = rng.random(n) < eps                                    # logging decision at step 1 (True = block)
    y2 = np.where(b1, (rng.random(n) < p) if system == "B" else np.zeros(n, bool), np.zeros(n, bool))
    b2 = rng.random(n) < eps                                    # logging decision at step 2
    return y1, b1, y2, b2

def ipw_risk(y1, b1, y2, b2, eps):
    # pi blocks step 1 (prob 1) and allows step 2 (prob 1); W_t = prod pi/mu over the decisions so far
    w1 = np.where(b1, 1.0 / eps, 0.0)                           # pi(block)=1 / mu(block)=eps
    w2 = w1 * np.where(~b2, 1.0 / (1.0 - eps), 0.0)             # pi(allow)=1 / mu(allow)=1-eps
    r1 = np.zeros_like(y1, float)                               # pi blocks step 1: never unsafe at t=1
    r2 = (y2 & ~b2).astype(float)                               # unsafe iff a violation is allowed at t=2
    return float(np.mean(w1 * r1 + w2 * r2))

rng = np.random.default_rng(20261102)
res = {"q": q, "p": p, "truth": {"R_A": 0.0, "R_B": p, "C": 1 - q}, "reps": REPS, "rows": []}
# ungated logs: identical law under A and B (sufficient statistic: labels of allowed steps only)
y1a, _, y2a, _ = logs("A", 0.0, 10**5, np.random.default_rng(1)); y1b, _, y2b, _ = logs("B", 0.0, 10**5, np.random.default_rng(1))
res["ungated_identical"] = bool(np.array_equal(y1a, y1b) and np.array_equal(y2a, y2b))
for eps in (0.05, 0.2, 0.5):
    for n in (10**2, 10**3, 10**4):
        est = {s: np.array([ipw_risk(*logs(s, eps, n, rng), eps) for _ in range(REPS)]) for s in ("A", "B")}
        row = {"eps": eps, "n": n, "mean_A": float(est["A"].mean()), "mean_B": float(est["B"].mean()),
               "rmse_B": float(np.sqrt(np.mean((est["B"] - p) ** 2))),
               "separated": float(np.mean(est["B"] > p / 2))}  # fraction of replications that rank B riskier than half its risk
        res["rows"].append(row)
        print(f"eps={eps:.2f} n={n:>6d}  mean R_A={row['mean_A']:.3f} (truth 0)  mean R_B={row['mean_B']:.3f} "
              f"(truth {p})  RMSE_B={row['rmse_B']:.3f}  P[R_B est > p/2]={row['separated']:.3f}")
print("ungated logs identical under A and B:", res["ungated_identical"])
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
