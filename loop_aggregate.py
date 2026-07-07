"""Aggregate the closed-loop results (loop_results.json from the box) pooled across
suites, with Wilson 95% CIs and a Fisher-exact test of no-gate vs gated attack success.
Usage: python3 loop_aggregate.py loop_results.json"""
import sys, json, math
from collections import defaultdict

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (p, max(0, c-h), min(1, c+h))

def fisher_2x2(a, b, c, d):
    # p-value (two-sided) via summing hypergeometric tail; small counts so exact.
    from math import comb
    n = a+b+c+d; r1 = a+b; c1 = a+c
    def hp(x):  # P(A=x) given margins
        return comb(c1, x)*comb(n-c1, r1-x)/comb(n, r1)
    p0 = hp(a); tot = 0.0
    lo = max(0, r1-(n-c1)); hi = min(r1, c1)
    for x in range(lo, hi+1):
        if hp(x) <= p0 + 1e-12: tot += hp(x)
    return min(1.0, tot)

recs = json.load(open(sys.argv[1]))
# pool across suites by (judge, theta)
agg = defaultdict(lambda: {"n_ep":0,"atk":0,"util":0,"blk":0,"seen":0,"suites":set()})
for r in recs:
    key = (r["judge"], r["theta"])
    a = agg[key]
    a["n_ep"]+=r["n_ep"]; a["atk"]+=r["atk_succ"]; a["util"]+=r["util_succ"]
    a["blk"]+=r["blocked"]; a["seen"]+=r["seen"]; a["suites"].add(r["suite"])

print(f"{'judge':12} {'theta':>5} {'suites':>18} {'n':>4} {'attack_succ (95% CI)':>24} {'utility':>8} {'gated%':>7}")
base = None
for (judge, th), a in sorted(agg.items(), key=lambda x:(x[0][0], -x[0][1])):
    p, lo, hi = wilson(a["atk"], a["n_ep"])
    up = a["util"]/a["n_ep"] if a["n_ep"] else float("nan")
    gp = a["blk"]/a["seen"] if a["seen"] else 0
    suites = "+".join(sorted(a["suites"]))
    tag = "  <- no gate" if th > 1.0 else ""
    print(f"{judge:12} {th:>5} {suites:>18} {a['n_ep']:>4} {a['atk']:>2}/{a['n_ep']:<3} {p:.3f}[{lo:.3f},{hi:.3f}] {up:>8.3f} {gp:>6.0%}{tag}")
    if th > 1.0: base = (judge, th, a)

# significance: no-gate baseline vs each 7b/3b gated condition
if base:
    _,_,b = base
    print("\nFisher-exact (attack success), no-gate vs gated:")
    for (judge, th), a in sorted(agg.items()):
        if th > 1.0: continue
        pv = fisher_2x2(b["atk"], b["n_ep"]-b["atk"], a["atk"], a["n_ep"]-a["atk"])
        sig = "SIGNIFICANT" if pv < 0.05 else "n.s."
        print(f"  vs {judge} theta={th}: baseline {b['atk']}/{b['n_ep']} vs {a['atk']}/{a['n_ep']}  p={pv:.4f}  {sig}")
