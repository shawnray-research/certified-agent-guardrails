"""Q5 (Sec. 7.5): what exact enforcement (T1) and specification analysis (T3) cost. No LLM inference.

(A) Gate overhead. The running example's oracle-checkable policies compiled to one keyed-counter gate:
    P2  cumulative transfer to recipient k in day bucket w never exceeds cap c (a trusted clock
        adapter supplies w), one monotone counter per active key (k, w);
    P3  no send after revoke, one flag.
    Per-decision latency on synthetic proposal streams of 10^6 proposals, against the number of
    active keys.
(B) Differential check. The compiled gate against a naive replay monitor that, for every proposal,
    replays the whole committed history and allows iff the one-step extension of the committed
    effect trace is still a good prefix (the definition of the T1 gate, Def. 4 / Theorem 1).
    Near-cap amounts are oversampled to exercise the boundary. Decisions must agree everywhere; the
    reference's cost per decision grows with the history, the gate's does not.
(C) Specification analysis (T3) as a rule linter. Exact bad-prefix reachability for the keyed-counter
    fragment by cap saturation and one-witness key projection, restricted to the rules that share an
    action with the queried rule (cone of influence). The question asked of each rule R: is there a
    committed (good) history after which some proposal violates R and no other rule? If not, R is
    redundant (never the sole reason for a block). Example: P2 (per-recipient daily cap c) is redundant
    under a per-session transfer limit L <= c, and live when c < L.

Usage: python3 q5_enforcement_cost.py [OUT.json] [--quick]
"""
import json, os, platform, sys, time, tracemalloc
from collections import deque
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
OUT = next((x for x in sys.argv[1:] if x.endswith(".json")), os.path.join(H, "q5_enforcement_cost.json"))
QUICK = "--quick" in sys.argv
ALLOW, BLOCK = 1, 0


class Gate:
    """Compiled (T;Pi)-gate for P2 and P3: O(1) work per proposal, state = one counter per active key."""
    __slots__ = ("cap", "spent", "revoked")

    def __init__(self, cap):
        self.cap, self.spent, self.revoked = cap, {}, False

    def decide(self, a):
        kind = a[0]
        if kind == 0:                                   # transfer(k, v, w)
            key = (a[1], a[3]); s = self.spent.get(key, 0) + a[2]
            if s > self.cap:
                return BLOCK                            # blocking commits the no-op: no state change
            self.spent[key] = s
            return ALLOW
        if kind == 1:                                   # send(k)
            return BLOCK if self.revoked else ALLOW
        self.revoked = True                             # revoke
        return ALLOW


def reference_decide(history, a, cap):
    """Allow iff history + [a] is a good prefix, recomputed from scratch (history is good by induction)."""
    spent, revoked = {}, False
    for b in history + [a]:
        if b[0] == 0:
            key = (b[1], b[3]); spent[key] = spent.get(key, 0) + b[2]
            if spent[key] > cap:
                return BLOCK
        elif b[0] == 1:
            if revoked:
                return BLOCK
        else:
            revoked = True
    return ALLOW


def stream(n, n_recipients, days, seed, p_send=0.18, p_revoke=2e-5, vmax=100):
    rng = np.random.default_rng(seed)
    kind = rng.choice(3, size=n, p=[1 - p_send - p_revoke, p_send, p_revoke])
    rec = rng.integers(0, n_recipients, size=n); amt = rng.integers(1, vmax + 1, size=n)
    day = (np.arange(n) * days) // n
    return [(int(k), int(r), int(v), int(d)) for k, r, v, d in zip(kind, rec, amt, day)]


# ---------------------------------------------------------------- (A) gate overhead
res = {"platform": {"python": platform.python_version(), "machine": platform.machine(),
                    "processor": platform.processor() or "", "node_cpu": ""}}
try:
    res["platform"]["node_cpu"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
except Exception:
    pass
N = 200_000 if QUICK else 1_000_000
CAP, DAYS = 1000, 7
res["A_gate"] = []
for K in ([10**2, 10**4, 10**6] if QUICK else [10**2, 10**3, 10**4, 10**5, 10**6]):
    S = stream(N, K, DAYS, seed=K)
    g = Gate(CAP); dec = g.decide; tick = time.perf_counter_ns
    lat = np.empty(N, dtype=np.int64); blocks = 0
    for i, a in enumerate(S):                          # per-call timing (includes ~50-100 ns timer cost)
        t0 = tick(); d = dec(a); lat[i] = tick() - t0; blocks += d == BLOCK
    g2 = Gate(CAP); dec2 = g2.decide
    t0 = time.perf_counter(); [dec2(a) for a in S]; amort = (time.perf_counter() - t0) / N * 1e9
    tracemalloc.start(); g3 = Gate(CAP)
    for a in S: g3.decide(a)
    mem = tracemalloc.get_traced_memory()[0]; tracemalloc.stop()
    row = {"recipients": K, "proposals": N, "active_keys": len(g.spent), "block_rate": blocks / N,
           "median_ns": float(np.median(lat)), "p99_ns": float(np.percentile(lat, 99)),
           "p999_ns": float(np.percentile(lat, 99.9)), "amortized_ns": amort,
           "state_bytes": mem, "bytes_per_key": mem / max(1, len(g3.spent))}
    res["A_gate"].append(row)
    print(f"[A] recipients={K:>8d} active keys={row['active_keys']:>7d} blocks={row['block_rate']:.3f} "
          f"median={row['median_ns']:.0f} ns p99={row['p99_ns']:.0f} ns amortized={amort:.0f} ns "
          f"state={mem/2**20:.1f} MiB ({row['bytes_per_key']:.0f} B/key)", flush=True)

# ---------------------------------------------------------------- (B) one long stream + differential check
# One stream of N proposals over 10^6 recipients (active keys accumulate): the gate's per-decision latency
# by position in the stream, and the naive replay monitor on the first NREF proposals of the SAME
# stream; then a second, boundary-heavy stream (40 recipients, exact-cap and cap+1 amounts oversampled).
NREF = 3000 if QUICK else 10_000
S = stream(N, 10**6, DAYS, seed=2026)
g = Gate(CAP); lat = np.empty(N, dtype=np.int64); keys_at = {}
edges = [10**e for e in range(1, int(np.log10(N)) + 1)]
for i, a in enumerate(S):
    t0 = time.perf_counter_ns(); g.decide(a); lat[i] = time.perf_counter_ns() - t0
    if i + 1 in edges: keys_at[i + 1] = len(g.spent)
gate_bins = [{"from": lo, "to": hi, "median_ns": float(np.median(lat[lo - 1:hi])),
              "p99_ns": float(np.percentile(lat[lo - 1:hi], 99)), "active_keys_at_end": keys_at[hi]}
             for lo, hi in zip([1] + edges[:-1], edges)]
def differential(SS, n):
    gg = Gate(CAP); hist, dis, nb, rl = [], 0, 0, []
    for i in range(n):
        a = SS[i] if not callable(SS) else SS(gg, i)
        t0 = time.perf_counter_ns(); dref = reference_decide(hist, a, CAP); rl.append((len(hist), time.perf_counter_ns() - t0))
        dg = gg.decide(a); dis += dg != dref; nb += dg == BLOCK
        if dref == ALLOW: hist.append(a)
    return dis, nb, len(hist), rl
dis1, nb1, com1, rl1 = differential(S, NREF)
rngb = np.random.default_rng(7)
def boundary(gg, i):
    u = rngb.random(); k = int(rngb.integers(0, 40)); w = (i * DAYS) // NREF
    if u < 0.75:
        s_ = gg.spent.get((k, w), 0); r = rngb.random()
        v = (CAP - s_) if r < .2 else (CAP - s_ + 1) if r < .4 else int(rngb.integers(1, 400))
        return (0, k, max(1, v), w)
    return (1, k, 0, w) if u < 0.999 else (2, k, 0, w)
dis2, nb2, com2, _ = differential(boundary, NREF)
rl1 = np.array(rl1, dtype=np.int64)
ref_bins = [{"history_from": lo, "history_to": hi, "median_ns": float(np.median(rl1[(rl1[:, 0] >= lo) & (rl1[:, 0] < hi), 1]))}
            for lo, hi in [(0, 10), (10, 100), (100, 1000), (1000, 10000)] if ((rl1[:, 0] >= lo) & (rl1[:, 0] < hi)).any()]
res["B_stream"] = {"proposals": N, "recipients": 10**6, "gate_bins": gate_bins, "reference_bins": ref_bins,
                   "reference_curve": rl1[::max(1, NREF // 400)].tolist()}
res["B_differential"] = {"same_stream": {"proposals": NREF, "disagreements": int(dis1), "blocks": int(nb1), "committed": com1},
                         "boundary_stream": {"proposals": NREF, "disagreements": int(dis2), "blocks": int(nb2), "committed": com2}}
for b in gate_bins:
    print(f"[B] gate proposals {b['from']:>7d}-{b['to']:<7d} median={b['median_ns']:.0f} ns p99={b['p99_ns']:.0f} ns keys={b['active_keys_at_end']}")
for b in ref_bins:
    print(f"[B] reference history {b['history_from']}-{b['history_to']}: median={b['median_ns']/1e3:.1f} us")
print(f"[B] differential: same stream {dis1} disagreements ({nb1} blocks / {NREF}); boundary stream {dis2} disagreements ({nb2} blocks / {NREF})", flush=True)

# ---------------------------------------------------------------- (C) T3 rule linter
def lint_p2_vs_p4(c, L):
    """Rules sharing `transfer`: P2 (witness key counter s, cap c) and P4 (session counter g, limit L).
    Abstract state (s, g) of a committed good history, exact by saturation (good states keep s<=c,
    g<=L) and one-witness projection (a transfer to another key is a transfer to a fresh key: it moves
    g only and is P2-good iff v <= c). Amounts above max(c,L)+1 are equivalent to max(c,L)+1."""
    vmax = max(c, L) + 1; seen = {(0, 0)}; q = deque([(0, 0)]); trans = 0; witness = {"P2": None, "P4": None}
    while q:
        s, gg = q.popleft()
        for target in ("w", "o"):
            for v in range(1, vmax + 1):
                trans += 1
                p2_bad = (s + v > c) if target == "w" else (v > c)
                p4_bad = gg + v > L
                if p2_bad and not p4_bad and witness["P2"] is None: witness["P2"] = (s, gg, target, v)
                if p4_bad and not p2_bad and witness["P4"] is None: witness["P4"] = (s, gg, target, v)
                if p2_bad or p4_bad:
                    continue
                nxt = (s + v, gg + v) if target == "w" else (s, gg + v)
                if nxt not in seen:
                    seen.add(nxt); q.append(nxt)
    return len(seen), trans, witness


def lint_p3_vs_p5(r):
    """Rules sharing `send`: P3 (flag revoked) and P5 (at most r sends per recipient per hour bucket,
    witness key counter n). State (revoked, n); a send to another key is a send to a fresh key."""
    seen = {(False, 0)}; q = deque([(False, 0)]); trans = 0; witness = {"P3": None, "P5": None}
    while q:
        rv, n = q.popleft()
        for act in ("send_w", "send_o", "revoke"):
            trans += 1
            if act == "revoke":
                nxt = (True, n)
            else:
                p3_bad = rv; p5_bad = (n + 1 > r) if act == "send_w" else (1 > r)
                if p3_bad and not p5_bad and witness["P3"] is None: witness["P3"] = (rv, n, act)
                if p5_bad and not p3_bad and witness["P5"] is None: witness["P5"] = (rv, n, act)
                if p3_bad or p5_bad:
                    continue
                nxt = (rv, n + 1) if act == "send_w" else (rv, n)
            if nxt not in seen:
                seen.add(nxt); q.append(nxt)
    return len(seen), trans, witness


res["C_linter"] = []
cases = [(50, 100), (100, 50), (100, 100), (100, 200), (200, 100)] if not QUICK else [(50, 100), (100, 50)]
for c, L in cases:
    t0 = time.perf_counter(); states, trans, wit = lint_p2_vs_p4(c, L); dt = time.perf_counter() - t0
    row = {"query": "P2 vs P4", "c": c, "L": L, "states": states, "transitions": trans, "seconds": dt,
           "P2_live": wit["P2"] is not None, "P4_live": wit["P4"] is not None,
           "P2_witness": wit["P2"], "P4_witness": wit["P4"]}
    res["C_linter"].append(row)
    print(f"[C] P2(c={c}) vs P4(L={L}): states={states} transitions={trans} time={dt*1e3:.1f} ms "
          f"P2 {'live' if row['P2_live'] else 'REDUNDANT'} {wit['P2']}  P4 {'live' if row['P4_live'] else 'REDUNDANT'}", flush=True)
for r in (1, 5, 60):
    t0 = time.perf_counter(); states, trans, wit = lint_p3_vs_p5(r); dt = time.perf_counter() - t0
    row = {"query": "P3 vs P5", "r": r, "states": states, "transitions": trans, "seconds": dt,
           "P3_live": wit["P3"] is not None, "P5_live": wit["P5"] is not None}
    res["C_linter"].append(row)
    print(f"[C] P3 vs P5(r={r}): states={states} time={dt*1e3:.2f} ms P3 live={row['P3_live']} P5 live={row['P5_live']}", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
