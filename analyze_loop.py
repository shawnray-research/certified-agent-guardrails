"""Closed-loop multi-suite analysis. Reads loop_allsuites.json (a JSON array of
per-(suite,judge,theta) records from run_loop.py) and produces, per suite and
pooled across suites: end-to-end attack-success and task-utility rates with Wilson
95% intervals, the gating fraction, and two-sided Fisher exact tests of each gate
against its own no-gate baseline. Every condition shares ONE endpoint and a
temperature-0 agent+judge, so the comparison is controlled. Usage:
    python analyze_loop.py loop_allsuites.json
"""
import sys, json, math
from scipy.stats import fisher_exact, norm

def wilson(k, n, z=1.959963985):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return p, max(0.0, c-h), min(1.0, c+h)

def frac(r, key):  # attack/util successes out of episodes
    return r[{"atk":"atk_succ","util":"util_succ"}[key]], r["n_ep"]

def gated(r):
    s = r.get("seen", 0); return (r.get("blocked", 0)/s if s else 0.0), r.get("blocked",0), s

path = sys.argv[1] if len(sys.argv) > 1 else "loop_allsuites.json"
recs = json.load(open(path))

# index: (suite,judge,theta) -> record; baseline is theta>1.0 (no gate)
suites = sorted({r["suite"] for r in recs})
def get(suite, judge, theta):
    for r in recs:
        if r["suite"]==suite and r["judge"]==judge and abs(r["theta"]-theta)<1e-9: return r
    return None
def baseline(suite):
    for r in recs:
        if r["suite"]==suite and r["theta"]>1.0: return r
    return None

CONDS = [("qwen2.5:7b", 0.7), ("qwen2.5:3b", 0.5)]

print("="*94)
print("PER-SUITE  (attack success k/N, Wilson 95%CI; Fisher p vs that suite's no-gate baseline)")
print("="*94)
print(f'{"suite":9s} {"condition":16s} {"N":>4s} {"attack (95% CI)":>22s} {"util":>6s} {"gated":>7s} {"Fisher p":>10s}')
pool = {}  # condition -> [atk_k, atk_n, util_k, blocked, seen]
pool_base = [0,0,0,0,0]
for suite in suites:
    b = baseline(suite)
    if not b: continue
    bk, bn = frac(b, "atk"); bp, blo, bhi = wilson(bk, bn)
    uk, un = frac(b, "util"); up,_,_ = wilson(uk,un)
    print(f'{suite:9s} {"none (baseline)":16s} {bn:>4} {bk:>3}/{bn:<3} {bp:.3f}[{blo:.3f},{bhi:.3f}] {up:>6.3f} {"0%":>7} {"":>10}')
    pool_base[0]+=bk; pool_base[1]+=bn; pool_base[2]+=uk; pool_base[3]+=0; pool_base[4]+=0
    for judge, th in CONDS:
        r = get(suite, judge, th)
        if not r: continue
        k, n = frac(r, "atk"); p, lo, hi = wilson(k, n)
        uk2, un2 = frac(r, "util"); up2,_,_ = wilson(uk2,un2)
        g, blk, seen = gated(r)
        # Fisher: [[atk, n-atk] baseline; [atk, n-atk] gated]
        _, pv = fisher_exact([[bk, bn-bk], [k, n-k]])
        tag = f'{judge.split(":")[1]}@{th}'
        print(f'{suite:9s} {tag:16s} {n:>4} {k:>3}/{n:<3} {p:.3f}[{lo:.3f},{hi:.3f}] {up2:>6.3f} {g*100:>5.0f}% {pv:>10.4f}')
        e = pool.setdefault((judge,th), [0,0,0,0,0])
        e[0]+=k; e[1]+=n; e[2]+=uk2; e[3]+=blk; e[4]+=seen

print("\n"+"="*94)
print(f"POOLED across {len(suites)} suites  (baseline N={pool_base[1]})")
print("="*94)
bk, bn = pool_base[0], pool_base[1]
bp, blo, bhi = wilson(bk, bn); bup,_,_ = wilson(pool_base[2], bn)
print(f'{"none (baseline)":16s} N={bn:<4} attack {bk}/{bn} = {bp:.3f} [{blo:.3f},{bhi:.3f}]   util={bup:.3f}')
for (judge, th), e in pool.items():
    k, n = e[0], e[1]; p, lo, hi = wilson(k, n)
    up,_,_ = wilson(e[2], n); g = e[3]/e[4] if e[4] else 0
    _, pv = fisher_exact([[bk, bn-bk], [k, n-k]])
    tag = f'{judge.split(":")[1]}@{th}'
    print(f'{tag:16s} N={n:<4} attack {k}/{n} = {p:.3f} [{lo:.3f},{hi:.3f}]   util={up:.3f}   gated={g*100:.0f}%   naive-Fisher p={pv:.2e}')

# --- Cochran-Mantel-Haenszel: stratified by suite (the correct multi-suite test) ---
# Each suite is a 2x2 stratum [[gate_atk, gate_noatk],[base_atk, base_noatk]].
# CMH tests the gate's effect on attack success while CONDITIONING on suite, so a
# suite where the agent is barely injectable cannot dilute one where it is. Reports
# the continuity-corrected CMH chi-square, its p, and the Mantel-Haenszel common OR.
def cmh(cond):
    judge, th = cond
    A = E = V = 0.0; num = den = 0.0; used = []
    for suite in suites:
        b = baseline(suite); r = get(suite, judge, th)
        if not b or not r: continue
        a = r["atk_succ"]; bb = r["n_ep"] - a          # gate: atk, no-atk
        c = b["atk_succ"]; d = b["n_ep"] - c            # base: atk, no-atk
        n1, n0 = a+bb, c+d; m1 = a+c; m0 = bb+d; N = n1+n0
        if N < 2 or m1 == 0: continue                    # skip strata with no events
        A += a; E += n1*m1/N; V += n1*n0*m1*m0/(N*N*(N-1))
        num += a*d/N; den += bb*c/N; used.append(suite)
    if V == 0: return None
    chi = (abs(A-E)-0.5)**2 / V
    pv = 2*min(norm.cdf(-math.sqrt(chi)), 0.5)*1  # chi2_1 upper tail = 2*Phi(-sqrt(chi))... use survival
    from scipy.stats import chi2
    pv = chi2.sf(chi, 1)
    mh_or = (num/den) if den > 0 else float('inf')
    return chi, pv, mh_or, used
print("\n"+"-"*94)
print("STRATIFIED by suite (Cochran-Mantel-Haenszel; conditions on suite, no dilution)")
print("-"*94)
for cond in pool.keys():
    res = cmh(cond)
    if not res: continue
    chi, pv, mh_or, used = res
    tag = f'{cond[0].split(":")[1]}@{cond[1]}'
    print(f'{tag:16s} CMH chi2={chi:6.2f}  p={pv:.2e}  MH common OR(gate/base)={mh_or:.3f}  strata={used}')
