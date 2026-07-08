"""Reproduce the C4 good-judge (gemma2:9b) numbers from loop_gemma9b.json.

The 9B that anchors the AgentDojo ceiling was run in the closed loop, sweeping its
operating point theta in {0.90,0.95,0.98} on the two injectable suites, against the
same matched no-gate baselines as loop_allsuites.json (banking 12/144, slack 25/105;
same 7B agent, temperature 0, one fixed endpoint). Prints attack success, gating rate,
utility, and a two-sided Fisher exact test vs baseline for each operating point.
"""
import json, os, sys
from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "loop_gemma9b.json")
BASE = {"banking": (12, 144), "slack": (25, 105)}   # from loop_allsuites.json (no-gate)

print(f"C4 good-judge (gemma2:9b) sweep  [{os.path.basename(path)}]")
print(f'{"suite":9s} {"theta":>5s} {"attack":>9s} {"gated":>6s} {"util":>5s} {"Fisher p vs base":>16s}')
for r in json.load(open(path)):
    s = r["suite"]; k, n = r["atk_succ"], r["n_ep"]
    g = 100 * r.get("blocked", 0) / max(1, r.get("seen", 1))
    util = r["util_succ"] / n
    bk, bn = BASE[s]
    p = fisher_exact([[bk, bn - bk], [k, n - k]])[1]
    print(f'{s:9s} {r["theta"]:>5} {k:>3}/{n:<4} (base {bk}/{bn})  {g:>4.0f}%  {util:>5.2f}  {p:>10.2f}')
print("\nReading: the 9B never significantly reduces attacks (all p>0.6). Its most-blocking")
print("banking point gates 35% yet moves attacks only 12->9 (p=0.65) while utility falls")
print("0.41->0.33; on slack it gates <=3% with no effect. Safety in the loop needs heavy")
print("blocking regardless of judge quality -- the over-blocking price is fundamental.")
