#!/usr/bin/env python3
"""Exact enumerator for the two-step retry model (paper Sec. Q2 / Supp. B.2).

Model (as stated in the paper):
  Step 1: score is high (".8") w.p. 2/5, low (".2") w.p. 3/5; violation
          posteriors 4/5 and 1/5.  Scores are taken to equal the ungated
          posteriors (.8 / .2), so the retry's score .2 equals the "low" score.
  Allow at any step -> episode ends; unsafe iff that proposal violates.
  Block at step 1   -> cost += P(compliant) = 1 - p1; a second proposal
          appears with current score .2 in both branches, posterior 3/5
          after a blocked high score and 1/50 after a blocked low score.
  Step 2 (last step): allow -> risk += p2; block -> cost += 1 - p2; ends.
  Cost C = E[# compliant proposals blocked]; risk R = P(violating proposal allowed).

Policy classes (deterministic; randomization = episode-level mixtures, i.e.
convex hull of the deterministic (R,C) points):
  perfect recall      : (d1|H, d1|L, d2|blocked H, d2|blocked L)   2^4 = 16
  current score+time  : f(t, score) on keys (1,.8),(1,.2),(2,.2)    2^3 = 8
  stationary score    : g(score) on keys .8, .2 used at both steps   2^2 = 4
     (monotone thresholds "block iff score >= theta" are 3 of these)
Pure Python + fractions; exact arithmetic.
"""
from fractions import Fraction as F
from itertools import product

A, B = "allow", "block"
DELTA = F(1, 20)

P_FIRST = {"H": F(2, 5), "L": F(3, 5)}        # P(first score)
SCORE1 = {"H": F(4, 5), "L": F(1, 5)}         # first-step score (= posterior)
POST1 = {"H": F(4, 5), "L": F(1, 5)}          # first-step violation posterior
SCORE2 = {"H": F(1, 5), "L": F(1, 5)}         # retry score: .2 in both branches
POST2 = {"H": F(3, 5), "L": F(1, 50)}         # retry posterior


def evaluate(d1, d2):
    """d1[branch], d2[branch] in {A,B}; returns exact (R, C)."""
    R = C = F(0)
    for br, pb in P_FIRST.items():
        p1 = POST1[br]
        if d1[br] == A:
            R += pb * p1
            continue
        C += pb * (1 - p1)
        p2 = POST2[br]
        if d2[br] == A:
            R += pb * p2
        else:
            C += pb * (1 - p2)
    return R, C


def perfect_recall_policies():
    out = []
    for d1H, d1L, d2H, d2L in product((A, B), repeat=4):
        name = f"d1(H)={d1H[0].upper()} d1(L)={d1L[0].upper()} d2(after H)={d2H[0].upper()} d2(after L)={d2L[0].upper()}"
        out.append((name, evaluate({"H": d1H, "L": d1L}, {"H": d2H, "L": d2L})))
    return out


def score_time_policies():
    keys = [(1, F(4, 5)), (1, F(1, 5)), (2, F(1, 5))]
    out = []
    for ds in product((A, B), repeat=len(keys)):
        f = dict(zip(keys, ds))
        d1 = {br: f[(1, SCORE1[br])] for br in P_FIRST}
        d2 = {br: f[(2, SCORE2[br])] for br in P_FIRST}
        name = "f(1,.8)=%s f(1,.2)=%s f(2,.2)=%s" % tuple(x[0].upper() for x in ds)
        out.append((name, evaluate(d1, d2)))
    return out


def stationary_policies(thresholds_only=False):
    out = []
    for gh, gl in product((A, B), repeat=2):
        if thresholds_only and (gh, gl) == (A, B):   # blocks low, allows high: not a threshold
            continue
        g = {F(4, 5): gh, F(1, 5): gl}
        d1 = {br: g[SCORE1[br]] for br in P_FIRST}
        d2 = {br: g[SCORE2[br]] for br in P_FIRST}
        out.append((f"g(.8)={gh[0].upper()} g(.2)={gl[0].upper()}", evaluate(d1, d2)))
    return out


def min_cost(points, delta=DELTA):
    """Exact min C over the convex hull of points subject to R <= delta.
    The optimum lies at a hull vertex with R<=delta or where R=delta cuts a
    segment between two points, so enumerating singles and pairs is exact."""
    best = None
    for i, (ni, (ri, ci)) in enumerate(points):
        if ri <= delta and (best is None or ci < best[0]):
            best = (ci, ((ni, (ri, ci), F(1)),))
    for i, (ni, (ri, ci)) in enumerate(points):
        for j, (nj, (rj, cj)) in enumerate(points):
            if ri < delta < rj:            # i below budget, j above
                w = (rj - delta) / (rj - ri)   # weight on i
                c = w * ci + (1 - w) * cj
                if c < best[0]:
                    best = (c, ((ni, (ri, ci), w), (nj, (rj, cj), 1 - w)))
    return best


def fmt(x):
    return f"{float(x):.6f} ({x})"


def report(title, pts):
    c, mix = min_cost(pts)
    print(f"\n[{title}] {len(pts)} deterministic policies; min C at R<={float(DELTA)}: {fmt(c)}")
    for name, (r, cc), w in mix:
        ties = [n for n, rc in pts if rc == (r, cc)]
        print(f"   weight {fmt(w)} on (R,C)=({float(r):.4f},{float(cc):.4f}) :: {name}")
        if len(ties) > 1:
            print(f"      (same (R,C) point also achieved by: {'; '.join(t for t in ties if t != name)})")
    return c


def behavioral_alternatives(n=20000):
    """Alternative reading: per-step (behavioral, fresh coin per step) randomization
    inside the restricted classes instead of episode-level mixtures. Approximate
    (grid over one probability, exact LP / root in the others)."""
    pH, pL = 0.4, 0.6
    best_st = None   # stationary: a=P(block|.8), b=P(block|.2) at every step
    for k in range(n + 1):
        b = k / n
        # R(a) and C(a) linear in a for fixed b
        def RC(a):
            R = pH * ((1 - a) * .8 + a * (1 - b) * .6) + pL * ((1 - b) * .2 + b * (1 - b) * .02)
            C = pH * a * (.2 + b * .4) + pL * b * (.8 + b * .98)
            return R, C
        for a in (0.0, 1.0):
            R, C = RC(a)
            if R <= 0.05 + 1e-12 and (best_st is None or C < best_st[0]):
                best_st = (C, a, b)
        R0, _ = RC(0.0); R1, _ = RC(1.0)
        if R1 != R0:
            a = (0.05 - R0) / (R1 - R0)
            if 0 <= a <= 1:
                R, C = RC(a)
                if best_st is None or C < best_st[0]:
                    best_st = (C, a, b)
    best_t = None    # score/time: a=P(block|1,.8), b=P(block|1,.2), c=P(block|2,.2)
    for k in range(n + 1):
        c = k / n
        def RC(a, b):
            R = pH * ((1 - a) * .8 + a * (1 - c) * .6) + pL * ((1 - b) * .2 + b * (1 - c) * .02)
            C = pH * a * (.2 + c * .4) + pL * b * (.8 + c * .98)
            return R, C
        cands = [(a, b) for a in (0., 1.) for b in (0., 1.)]
        for a in (0., 1.):   # edges with R = .05
            R0, _ = RC(a, 0.); R1, _ = RC(a, 1.)
            if R1 != R0:
                bb = (0.05 - R0) / (R1 - R0)
                if 0 <= bb <= 1: cands.append((a, bb))
        for b in (0., 1.):
            R0, _ = RC(0., b); R1, _ = RC(1., b)
            if R1 != R0:
                aa = (0.05 - R0) / (R1 - R0)
                if 0 <= aa <= 1: cands.append((aa, b))
        for a, b in cands:
            R, C = RC(a, b)
            if R <= 0.05 + 1e-12 and (best_t is None or C < best_t[0]):
                best_t = (C, a, b, c)
    return best_st, best_t


def main():
    pr = perfect_recall_policies()
    print("Two-step retry model, delta = 1/20. All 16 deterministic perfect-recall policies:")
    print(f"{'#':>2}  {'policy':<62} {'R':>8} {'C':>8}   exact (R, C)")
    for k, (name, (r, c)) in enumerate(pr, 1):
        print(f"{k:>2}  {name:<62} {float(r):8.4f} {float(c):8.4f}   ({r}, {c})")

    c_pr = report("perfect recall, episode-level mixtures", pr)
    c_st = report("current score + time, episode-level mixtures", score_time_policies())
    c_sta = report("stationary score rules (all 4 maps), episode-level mixtures", stationary_policies())
    c_thr = report("stationary monotone thresholds (3), episode-level mixtures", stationary_policies(True))

    bh = evaluate({"H": B, "L": A}, {"H": A, "L": A})  # block .8, allow .2 (retry score .2 allowed)
    print(f"\n'block high' static threshold (block score .8, allow score .2 at both steps): "
          f"(R,C) = ({float(bh[0]):.4f}, {float(bh[1]):.4f}) = ({bh[0]}, {bh[1]}); "
          f"R <= delta? {bh[0] <= DELTA}")

    print("\nSUMMARY (paper claims .551 / .863 / 1.137, block-high (.36,.08)):")
    print(f"  perfect recall      : {float(c_pr):.3f}")
    print(f"  current score+time  : {float(c_st):.3f}")
    print(f"  stationary threshold: {float(c_thr):.3f} (all 4 stationary maps: {float(c_sta):.3f})")
    print(f"  block high (R,C)    : ({float(bh[0]):.2f}, {float(bh[1]):.2f})")

    st, tm = behavioral_alternatives()
    print("\nALTERNATIVE reading (per-step behavioral randomization inside restricted classes; grid-approx):")
    print(f"  stationary, fresh coin per step : min C ~ {st[0]:.4f} at P(block|.8)={st[1]:.4f}, P(block|.2)={st[2]:.4f}")
    print(f"  score+time, fresh coin per step : min C ~ {tm[0]:.4f} at a={tm[1]:.4f}, b={tm[2]:.4f}, c={tm[3]:.4f}")


if __name__ == "__main__":
    main()
