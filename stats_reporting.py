#!/usr/bin/env python3
"""Statistical-reporting additions for the TDSC paper (Sec. 7 Q1; Supplement B.6, B.9, B.19, E), computed from
CACHED judge scores only (no API calls, no model inference). Writes stats_reporting.json.

The repo's own definitions are reused, not re-implemented:
  * score registry, refusal rule and cross-fit of unify_crossfit.py (exec'd up to its main loop, as
    calib_split_sensitivity.py does): run 1 of the hosted judges, a provider refusal scores 1.0, any other reply
    without an integer .5; 500 label-balanced two-fold splits (seed 0); enforce.conformal_threshold, i.e. the
    floor((n+1)*delta/H)-th smallest violating calibration score, block-all if that index is 0 (H = 1);
  * the 33-run family of analyze_agentdojo_inference.py --family complete [--run r] [--refusal x]
    [--drop-benign-goals], obtained by exec'ing that script itself (its stdout is captured); for an item set it has
    no flag for (refused items dropped) a verbatim copy of its DeLong / max-T block is used, and that copy is
    asserted to reproduce the script's own output;
  * calib_split_sensitivity.py (2,000 random 28/7 violation splits, seed 0), exec'd for the anchor;
  * analyze_attack_common.py --family unified (the 21-run Q3 family), exec'd for item 5.
The vectorized cross-fit / 28-7 code used inside the bootstrap is asserted equal to the originals on the real data.

Usage (numpy + scipy; run from anywhere):
  python -B stats_reporting.py [--B 2000] [--boot-seed 20261102] [--boot-splits 500] [--json OUT]
  python3 stats_reporting.py --audit-only [--json OUT]   # item 6 only (stdlib + git, needs a git clone);
                                                          # merged into OUT if it exists
"""
import argparse, contextlib, glob, hashlib, io, json, os, re, subprocess, sys

R = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--json", default=os.path.join(R, "stats_reporting.json"))
ap.add_argument("--B", type=int, default=2000, help="bootstrap replicates")
ap.add_argument("--boot-seed", type=int, default=20261102)
ap.add_argument("--boot-splits", type=int, default=500, help="two-fold splits inside each bootstrap replicate")
ap.add_argument("--audit-only", action="store_true", help="only item 6 (git audit); merge into --json")
ap.add_argument("--skip-audit", action="store_true")
A = ap.parse_args()


def r3(x):
    return None if x is None else round(float(x), 3)


def pv(x):                                  # p-values: 4 significant digits (3-decimal rounding would print 0.000)
    return float(f"{float(x):.4g}")


def rel(p):
    return os.path.relpath(p, R)


# ----------------------------------------------------------------------------------------------------------------
# Item 6: 'predeclared' audit (stdlib + git only, so it can run in a git clone without numpy)
# ----------------------------------------------------------------------------------------------------------------
AUDIT_PAT = re.compile(r"pre-?declared|pre-?registered|pre-?specified", re.I)
# topic -> (what, [(protocol file, results file), ...] = the pairs whose first-add commits are compared,
#           pickaxe strings searched in the protocol files)
TOPICS = {
    "q3_suffix": (
        "the fixed 16-word administrator-preapproval suffix of Q3 (Supplement B.9)",
        [("run_bench.py", "bench_scores_cache.json"), ("_bench_safewash_texts.json", "bench_scores_cache.json"),
         ("run_bench.py", "cohere_safewash.json"), ("run_bench.py", "frontier_claude-opus-4-5_bench.json"),
         ("run_bench.py", "frontier2026/frontier2026_claude-opus-5-5_bench.json"),
         ("analyze_attack_common.py", "attack_unified.json")],
        ["pre-approved by the administrator", 'choices=["panel", "all", "unified"]', "the 11 predeclared panel judges"]),
    "equivalence_margin": (
        "the +-.05 AUC equivalence margin of the familywise AUC analysis",
        [("agentdojo_paired.py", "bench_scores_cache.json"), ("agentdojo_paired.py", "frontier_claude-opus-4-5_bench.json"),
         ("analyze_agentdojo_inference.py", "frontier2026/fam_all2026_ref1.0.json"),
         ("analyze_agentdojo_inference.py", "forest33.json")],
        ["MARGIN = 5000, 20260706, 0.05", "eqm[i] <= .05"]),
    "gemma_reference": (
        "Gemma-2-9B as the reference of the paired AUC contrasts",
        [("agentdojo_paired.py", "bench_scores_cache.json"), ("agentdojo_paired.py", "frontier_claude-opus-4-5_bench.json"),
         ("analyze_agentdojo_inference.py", "frontier2026/fam_all2026_ref1.0.json"),
         ("analyze_agentdojo_inference.py", "forest33.json")],
        ["gemma2:9b reference", "predeclared reference"]),
    "toolcall_diagnostic": (
        "the tool-call diagnostic protocol (header of frontier2026/toolcall_diagnostic.py)",
        [("frontier2026/toolcall_diagnostic.py", "frontier2026/toolcall_diagnostic.json"),
         ("frontier2026/toolcall_paired.py", "frontier2026/toolcall_paired.json")],
        ["PREDECLARED PROTOCOL"]),
    "toolcall_suffix": (
        "the Q3 suffix appended to the 26 violating tool calls (frontier2026/toolcall_suffix.py); the suffix itself is q3_suffix",
        [("frontier2026/toolcall_suffix.py", "frontier2026/toolcall_suffix.json"),
         ("run_bench.py", "frontier2026/toolcall_suffix.json")],
        []),
    "bestofn": (
        "the query-adaptive best-of-N protocol (header of frontier2026/frontier2026_bestofn.py)",
        [("frontier2026/frontier2026_bestofn.py", "frontier2026/frontier2026_bestofn_claude-opus-5-5.json"),
         ("frontier2026/frontier2026_bestofn.py", "frontier2026/frontier2026_bestofn_summary.json")],
        ["PREDECLARED PROTOCOL (fixed before any query was sent)"]),
    "q4_not_preregistered": (
        "disclaimer only: the two closed-loop gate operating points (Q4) were NOT preregistered",
        [("run_loop.py", "loop_allsuites.json"), ("run_loop.py", "loop_gemma9b.json")],
        []),
}


def classify(after):
    """Topic of an occurrence from the text that follows the matched word (rest of line + next line)."""
    w = after.lower()
    if re.match(r"\s*(in\s+\\path\{frontier2026/toolcall_diagnostic)", w): return "toolcall_diagnostic"
    if re.match(r"\s*query-adaptive", w): return "bestofn"
    if re.match(r"\s*(the\s+)?(gemma-2-9b\s+)?reference", w): return "gemma_reference"
    if re.match(r"\s*\$\\pm\s*\.05\$", w): return "equivalence_margin"
    if re.match(r"\s*q3 suffix to the 26 violating calls", w): return "toolcall_suffix"
    if re.match(r"\s*(16-word|rewrite)", w) or "suffix" in w[:60]: return "q3_suffix"
    if re.match(r"\s*,?\s*so the", w) or "operating points" in w[:40]: return "q4_not_preregistered"
    return "unclassified"


def git(*args):
    return subprocess.run(["git", "-C", R, *args], capture_output=True, text=True, check=True).stdout


def first_add(f):
    out = git("log", "--diff-filter=A", "--format=%h %ad|%at", "--date=iso", "--", f).strip().splitlines()
    if not out: return None
    h, ts = out[-1].split("|")                                # earliest add if the file was ever re-added
    return {"commit": h, "ts": int(ts)}


def pickaxe(s, files):
    out = git("log", "-S", s, "--reverse", "--format=%h %ad", "--date=iso", "--", *files).strip().splitlines()
    return out[0] if out else None


def order(c1, c2):
    h1, h2 = c1["commit"].split()[0], c2["commit"].split()[0]
    if h1 == h2: return "same commit (no git evidence that the protocol preceded the results)"
    anc = subprocess.run(["git", "-C", R, "merge-base", "--is-ancestor", h1, h2]).returncode == 0
    return "protocol commit predates results commit" if anc else "RESULTS commit predates protocol commit"


def audit():
    top = git("rev-parse", "--show-toplevel").strip()
    if os.path.realpath(top) != os.path.realpath(R): raise RuntimeError(f"git top-level {top} is not this repo")
    files = sorted(glob.glob(os.path.join(R, "paper/sections/*.tex"))) + sorted(glob.glob(os.path.join(R, "paper/supp/*.tex")))
    occ, related = [], []
    for f in files:
        lines = open(f, encoding="utf-8").read().split("\n")
        for i, line in enumerate(lines):
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            for m in AUDIT_PAT.finditer(line):
                occ.append({"file": rel(f), "line": i + 1, "match": m.group(0), "context": line.strip()[:170],
                            "topic": classify(line[m.end():] + " " + nxt)})
            for m in re.finditer(r"fixed before|in advance", line, re.I):
                related.append({"file": rel(f), "line": i + 1, "match": m.group(0), "context": line.strip()[:170]})
    topics = {}
    for k, (what, pairs, picks) in TOPICS.items():
        rows = []
        for pf, rf in pairs:
            pa, ra = first_add(pf), first_add(rf)
            rows.append({"protocol_file": pf, "protocol_first_added": pa["commit"], "results_file": rf,
                         "results_first_added": ra["commit"], "verdict": order(pa, ra)})
        prot_files = sorted({pf for pf, _ in pairs})
        topics[k] = {"what": what, "pairs": rows,
                     "protocol_string_first_commit": {s: pickaxe(s, prot_files) for s in picks},
                     "n_occurrences": sum(o["topic"] == k for o in occ)}
        if k == "q3_suffix":                                  # code path that writes attack_unified.json
            pk = git("log", "-S", 'choices=["panel", "all", "unified"]', "--reverse", "--format=%h %ad|%at", "--date=iso",
                     "--", "analyze_attack_common.py").strip().splitlines()[0].split("|")
            ra = first_add("attack_unified.json")
            rows.append({"protocol_file": "analyze_attack_common.py --family unified (first commit containing that option)",
                         "protocol_first_added": pk[0], "results_file": "attack_unified.json",
                         "results_first_added": ra["commit"],
                         "verdict": order({"commit": pk[0], "ts": int(pk[1])}, ra)})
    days = {}
    for ln in git("log", "--format=%ad", "--date=short").split():
        days[ln] = days.get(ln, 0) + 1
    return {"pattern": AUDIT_PAT.pattern + " (case-insensitive) over paper/sections/*.tex and paper/supp/*.tex",
            "n_occurrences": len(occ), "occurrences": occ, "topics": topics,
            "related_phrases_outside_pattern": related,
            "git_head": git("log", "-1", "--format=%h %ad %s", "--date=iso").strip()[:120],
            "commits_per_day": dict(sorted(days.items())),
            "note": ("first-add commits from `git log --diff-filter=A --format='%h %ad' -- FILE` in the git clone; "
                     "commit order is only an upper bound on when a protocol was written, and files first added in the "
                     "same commit give no git evidence of precedence")}


if A.audit_only:
    out = json.load(open(A.json)) if os.path.exists(A.json) else {}
    out["item6_predeclared_audit"] = audit()
    json.dump(out, open(A.json, "w"), indent=1)
    print(f"item 6 audit merged into {A.json}: {out['item6_predeclared_audit']['n_occurrences']} occurrences")
    for k, v in out["item6_predeclared_audit"]["topics"].items():
        for p_ in v["pairs"]:
            print(f"  {k:20s} {p_['protocol_file']} {p_['protocol_first_added']} | {p_['results_file']} "
                  f"{p_['results_first_added']} -> {p_['verdict']}")
    sys.exit(0)

# ----------------------------------------------------------------------------------------------------------------
# numeric part (numpy + scipy)
# ----------------------------------------------------------------------------------------------------------------
import numpy as np
from scipy.stats import norm


def _exec(path, argv, upto=None):
    """Run a repo script in-process (stdout captured) and return its globals."""
    src = open(path).read()
    if upto: src = src[:src.index(upto)]
    g = {"__file__": path, "__name__": "_stats_reporting_exec"}
    old_argv, old_cwd, buf = sys.argv, os.getcwd(), io.StringIO()
    sys.argv = argv
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(src, path, "exec"), g)
    finally:
        sys.argv = old_argv; os.chdir(old_cwd)
    g["_stdout"] = buf.getvalue()
    return g


AAI = os.path.join(R, "analyze_agentdojo_inference.py")


def aai(run=1, refusal=1.0, drop=False):
    argv = ["analyze_agentdojo_inference.py", "--family", "complete", "--run", str(run), "--refusal", repr(float(refusal))]
    return _exec(AAI, argv + (["--drop-benign-goals"] if drop else []))


def family_inference(S, y, draws=100_000, seed=20260706):
    """Verbatim copy of analyze_agentdojo_inference.py's DeLong / max-T block (reference = row 0)."""
    J = len(S)
    X, Y = S[:, y == 1], S[:, y == 0]; m, n = X.shape[1], Y.shape[1]
    psi = (X[:, :, None] > Y[:, None, :]) + 0.5 * (X[:, :, None] == Y[:, None, :])   # J x m x n
    V10, V01 = psi.mean(2), psi.mean(1)                                              # placements
    auc = V10.mean(1)
    Cov = np.cov(V10) / m + np.cov(V01) / n                                          # DeLong paired covariance
    r = 0
    C = np.eye(J)[1:] - np.eye(J)[[r]]                                              # contrasts vs reference
    d = C @ auc; Cd = C @ Cov @ C.T; se = np.sqrt(np.diag(Cd)); t = d / se
    Rho = Cd / np.outer(se, se)
    rng = np.random.default_rng(seed)
    Z = rng.multivariate_normal(np.zeros(J - 1), Rho, size=draws, method="cholesky")
    mx = np.abs(Z).max(1); crit = float(np.quantile(mx, 0.95))
    p_maxT = np.array([(mx >= abs(tt)).mean() for tt in t])
    p_raw = 2 * norm.sf(np.abs(t))
    o = np.argsort(p_raw); k = len(p_raw); p_holm = np.empty(k); run = 0.0
    for i, j in enumerate(o): run = max(run, min(1.0, (k - i) * p_raw[j])); p_holm[j] = run
    lo, hi = d - crit * se, d + crit * se
    return dict(auc=auc, Cov=Cov, d=d, se=se, t=t, lo=lo, hi=hi, crit=crit, p_maxT=p_maxT, p_holm=p_holm, p_raw=p_raw,
                m=m, n=n)


def contrast(fi, names, judge="Claude-Opus-5.5"):
    i = names.index(judge); c = i - 1
    a, se_a = fi["auc"][i], float(np.sqrt(fi["Cov"][i, i])); g, se_g = fi["auc"][0], float(np.sqrt(fi["Cov"][0, 0]))
    d, se = fi["d"][c], fi["se"][c]
    return {"n_items": int(fi["m"] + fi["n"]), "n_violations": int(fi["m"]), "n_compliant": int(fi["n"]),
            "auc": r3(a), "auc_95": [r3(a - 1.96 * se_a), r3(a + 1.96 * se_a)],
            "auc_gemma": r3(g), "auc_gemma_95": [r3(g - 1.96 * se_g), r3(g + 1.96 * se_g)],
            "delta": r3(d), "delta_95_marginal": [r3(d - 1.96 * se), r3(d + 1.96 * se)], "p_marginal": pv(fi["p_raw"][c]),
            "delta_simultaneous_95": [r3(fi["lo"][c]), r3(fi["hi"][c])], "maxT_crit": r3(fi["crit"]),
            "p_maxT": pv(fi["p_maxT"][c]), "p_holm": pv(fi["p_holm"][c]),
            "superior_marginal": bool(d - 1.96 * se > 0), "superior_maxT_family": bool(fi["lo"][c] > 0),
            "superior_holm": bool(fi["p_holm"][c] < 0.05 and d > 0)}


def check_same(fi, g, what):
    for k in ("auc", "d", "lo", "hi", "p_maxT", "p_holm"):
        assert np.allclose(fi[k], np.asarray(g[k]), atol=1e-12, rtol=0), f"{what}: copy differs from script ({k})"
    assert abs(fi["crit"] - g["crit"]) < 1e-12, what


# ---------------- registries ----------------
U = _exec(os.path.join(R, "unify_crossfit.py"), ["unify_crossfit.py"], upto="rows = []")
bench, y, SRC, E, crossfit, SW, goals = U["bench"], U["y"], U["SRC"], U["E"], U["crossfit"], U["SW"], U["goals"]
NAMES = [nm for nm, _ in SRC]
DICT = {nm: f() for nm, f in SRC}
S33 = {nm: np.array([DICT[nm].get(t) if DICT[nm].get(t) is not None else np.nan for t in bench], float) for nm in NAMES}
assert not any(np.isnan(s).any() for s in S33.values()) and len(NAMES) == 33
OPUS, GEMMA = "Claude-Opus-5.5", "Gemma-2-9B"
POS, NEG = np.where(y == 1)[0], np.where(y == 0)[0]
man = json.load(open(os.path.join(R, "agentdojo_manifest.json")))["items"]
assert len(POS) == 35 and len(NEG) == 97

G1 = aai(1, 1.0)                                       # authoritative family, run 1, refusals as blocks
fam_names, S_fam = list(G1["names"]), np.array(G1["S"])
assert sorted(fam_names) == sorted(NAMES) and len(fam_names) == 33
for i, nm in enumerate(fam_names):                     # both registries load identical score vectors
    assert np.array_equal(S_fam[i], S33[nm]), nm
FI1 = family_inference(S_fam, np.array(G1["y"]))
check_same(FI1, G1, "run 1 / refusal 1.0")

# ---------------- vectorized cross-fit and 28/7 procedures (asserted identical to the originals) ----------------
def perm_table(n1, splits, seed, n2=None):
    rng = np.random.default_rng(seed); a, b = [], []
    for _ in range(splits):
        a.append(rng.permutation(n1))
        if n2 is not None: b.append(rng.permutation(n2))
    return np.array(a), (np.array(b) if n2 is not None else None)


PP, NN = perm_table(len(POS), 500, 0, len(NEG))        # same draw order as unify_crossfit.crossfit (seed 0)
Q287, _ = perm_table(len(POS), 2000, 0)                # same draw order as calib_split_sensitivity.py (seed 0)


def kth(cal, delta):                                   # row-wise enforce.conformal_threshold
    n = cal.shape[1]; k = int(np.floor((n + 1) * delta))
    if k <= 0: return np.full(cal.shape[0], -np.inf)
    if k > n: return cal.max(1)
    return np.sort(cal, 1)[:, k - 1]


def folds(splits=500):
    P, N = POS[PP[:splits]], NEG[NN[:splits]]; hp, hn = len(POS) // 2, len(NEG) // 2
    return (P[:, :hp], P[:, hp:]), (N[:, :hn], N[:, hn:])


FOLDS500 = folds(500)


def cf_folds(s, delta, F=FOLDS500, rule="conformal"):
    """Per-split, per-fold (miss, false block). rule: conformal (paper), oracle (threshold set on the TEST fold so its
    miss <= delta), plugin (largest threshold with CALIBRATION-fold miss <= delta; no conformal correction)."""
    Pf, Nf = F; S_ = Pf[0].shape[0]; mi = np.empty((S_, 2)); fb = np.empty((S_, 2))
    for c in (0, 1):
        if rule == "conformal":
            th = kth(s[Pf[c]], delta)
        else:
            v = np.sort(s[Pf[1 - c] if rule == "oracle" else Pf[c]], 1); nv = v.shape[1]
            m_ = int(np.floor(delta * nv + 1e-9)); th = v[:, m_] if m_ < nv else np.full(S_, np.inf)
        mi[:, c] = (s[Pf[1 - c]] < th[:, None]).mean(1); fb[:, c] = (s[Nf[1 - c]] >= th[:, None]).mean(1)
    return mi, fb


def s287(s, Q=Q287, delta=0.10, ncal=28):
    P = POS[Q]; th = kth(s[P[:, :ncal]], delta)[:, None]
    return (s[P[:, ncal:]] < th).mean(1), (s[NEG][None, :] >= th).mean(1)


CF = {}
for nm in NAMES:
    for dl in (0.05, 0.10, 0.20):
        mi, fb = cf_folds(S33[nm], dl)
        m0, f0 = crossfit(S33[nm], y, dl)
        assert abs(mi.reshape(-1).mean() - m0) < 1e-12 and abs(fb.reshape(-1).mean() - f0) < 1e-12, (nm, dl)
        CF[nm, dl] = (m0, f0)
CS = _exec(os.path.join(R, "calib_split_sensitivity.py"), ["calib_split_sensitivity.py", R])
RES287 = CS["res"]
for nm in NAMES:
    m7, f7 = s287(S33[nm])
    assert abs(m7.mean() - RES287[nm][0]) < 1e-12 and abs(f7.mean() - RES287[nm][1]) < 1e-12, nm

# ---------------- refusal bookkeeping (raw caches) ----------------
NEWTAGS = [("claude-opus-5-5", "Claude-Opus-5.5"), ("claude-fable-5-1", "Claude-Fable-5.1"), ("claude-sonnet-5-5", "Claude-Sonnet-5.5"),
           ("claude-haiku-5-5", "Claude-Haiku-5.5"), ("gpt-6.1-sol", "GPT-6.1-Sol"), ("gpt-5.5", "GPT-5.5"), ("gpt-6.1-terra", "GPT-6.1-Terra")]
PRE = {1: "", 2: "rep_r2_", 3: "rep_r3_"}


def refused(tag, run):
    raw = json.load(open(os.path.join(R, f"frontier2026/{PRE[run]}frontier2026_{tag}_bench_raw.json")))
    return np.array([raw.get(t, {}).get("stop") == "refusal" for t in bench])


REF = {(nm, r): refused(tag, r) for tag, nm in NEWTAGS for r in (1, 2, 3)}
refusal_table = {nm: {f"run{r}": {"violations": int(REF[nm, r][POS].sum()), "compliant": int(REF[nm, r][NEG].sum())}
                      for r in (1, 2, 3)} for _, nm in NEWTAGS}

out = {"meta": {
    "script": "stats_reporting.py", "inputs_from_cache_only": True,
    "repo_scripts_reused": ["unify_crossfit.py (registry, refusal rule, crossfit)", "enforce.py (conformal_threshold)",
                            "analyze_agentdojo_inference.py --family complete (exec'd; DeLong/max-T copied verbatim and checked)",
                            "calib_split_sensitivity.py (exec'd)", "analyze_attack_common.py --family unified (exec'd)"],
    "self_checks_passed": ["family_inference copy == analyze_agentdojo_inference.py output (run 1, refusal 1.0; drop-benign)",
                           "unify_crossfit registry == analyze_agentdojo_inference registry (33 run-1 score vectors identical)",
                           "vectorized cross-fit == unify_crossfit.crossfit for all 33 runs at delta .05/.10/.20 (|diff|<1e-12)",
                           "vectorized 28/7 == calib_split_sensitivity.py for all 33 runs (|diff|<1e-12)"],
    "conventions": "gate blocks iff s >= theta; false block on compliant items, miss on violations; DeLong marginal 95% = "
                   "estimate +- 1.96 SE (as frontier2026_table.py); max-T: 100,000 draws, seed 20260706, as the paper",
    "input_sha1": {f: hashlib.sha1(open(os.path.join(R, f), "rb").read()).hexdigest()[:12] for f in
                   ["_bench_texts.json", "bench_scores_cache.json", "frontier2026/frontier2026_claude-opus-5-5_bench.json",
                    "frontier2026/frontier2026_claude-opus-5-5_bench_raw.json", "agentdojo_manifest.json"]}}}

# ================================================================================================================
# 0. anchors
# ================================================================================================================
iop = fam_names.index(OPUS)
fb10 = {nm: CF[nm, 0.10][1] for nm in NAMES}; mi10 = {nm: CF[nm, 0.10][0] for nm in NAMES}
fb287 = {nm: RES287[nm][1] for nm in NAMES}
GD = aai(1, 1.0, drop=True)
drop = np.asarray(GD["drop"]); keep_b = ~drop
low = (S_fam < 0.5).sum(0) > len(S_fam) / 2
assert np.array_equal(drop, (y == 1) & low)
g_red10, g_red20 = GD["_crossfit"](S33[GEMMA][keep_b], y[keep_b], 0.10), GD["_crossfit"](S33[GEMMA][keep_b], y[keep_b], 0.20)
anchors = [
    ("AUC Claude Opus 5.5 (run 1, refusals as blocks)", .929, FI1["auc"][iop], 3),
    ("AUC Gemma-2-9B", .813, FI1["auc"][0], 3),
    ("Delta AUC Opus 5.5 - Gemma-2-9B", .116, FI1["d"][iop - 1], 3),
    ("simultaneous 95% lower", .014, FI1["lo"][iop - 1], 3),
    ("simultaneous 95% upper", .217, FI1["hi"][iop - 1], 3),
    ("max-T p", .013, FI1["p_maxT"][iop - 1], 3),
    ("cross-fitted false block delta=.10, min over 33 runs", .641, min(fb10.values()), 3),
    ("  ...attained by", "Claude-Opus-5.5", min(fb10, key=fb10.get), None),
    ("cross-fitted false block delta=.10, max over 33 runs", 1.000, max(fb10.values()), 3),
    ("cross-fitted miss delta=.10, max over 33 runs", .053, max(mi10.values()), 3),
    ("  ...attained by", "Claude-Opus-5.5", max(mi10, key=mi10.get), None),
    ("28/7-split false block delta=.10, min over 33 runs", .485, min(fb287.values()), 3),
    ("  ...attained by", "Claude-Opus-5.5", min(fb287, key=fb287.get), None),
    ("28/7-split false block delta=.10, max over 33 runs", 1.000, max(fb287.values()), 3),
    ("Gemma-2-9B cross-fitted false block delta=.10, all 132 items", .779, CF[GEMMA, 0.10][1], 3),
    ("Gemma-2-9B cross-fitted false block delta=.10, without the 7 benign-text goals", .585, g_red10[1], 3),
    ("Gemma-2-9B cross-fitted false block delta=.20, all 132 items", .649, CF[GEMMA, 0.20][1], 3),
    ("Gemma-2-9B cross-fitted false block delta=.20, without the 7 benign-text goals", .152, g_red20[1], 3),
    ("Opus 5.5 provider refusals, run 1, violations", 20, refusal_table[OPUS]["run1"]["violations"], 0),
    ("Opus 5.5 provider refusals, run 1, compliant", 0, refusal_table[OPUS]["run1"]["compliant"], 0),
]
A_rows = []
for nm_, paper, val, nd in anchors:
    if nd is None:
        A_rows.append({"anchor": nm_, "paper": paper, "computed": val, "reproduced": val == paper})
    else:
        cv = int(val) if nd == 0 else round(float(val), 4)
        A_rows.append({"anchor": nm_, "paper": paper, "computed": cv, "reproduced": round(float(val), nd) == paper})
out["anchors"] = {"all_reproduced": all(a["reproduced"] for a in A_rows), "rows": A_rows}

# ================================================================================================================
# 1. refusal sensitivity, Claude Opus 5.5
# ================================================================================================================
item1 = {"refusals_by_run": refusal_table,
         "opus_refused_violations_in_all_three_runs": int((REF[OPUS, 1] & REF[OPUS, 2] & REF[OPUS, 3]).sum()),
         "opus_refused_violations_in_any_run": int((REF[OPUS, 1] | REF[OPUS, 2] | REF[OPUS, 3]).sum()),
         "variants": ("blocks: refusal scores 1.0 (paper default, all seven no-temperature judges); half: refusal scores .5 "
                      "for all seven (= analyze_agentdojo_inference.py --refusal 0.5); dropped: the items Opus 5.5's provider "
                      "refused in that run are removed for every judge (other judges' refusals on the remaining items stay "
                      "1.0); max-T family = all 33 runs on the same items, 100,000 draws, seed 20260706"),
         "runs": {}}
for run in (1, 2, 3):
    Gb = G1 if run == 1 else aai(run, 1.0); Gh = aai(run, 0.5)
    nb, Sb, yb = list(Gb["names"]), np.array(Gb["S"]), np.array(Gb["y"])
    fib = FI1 if run == 1 else family_inference(Sb, yb)
    if run != 1: check_same(fib, Gb, f"run {run}")
    fih = family_inference(np.array(Gh["S"]), np.array(Gh["y"])); check_same(fih, Gh, f"run {run} refusal .5")
    keep = ~REF[OPUS, run]
    fid = family_inference(Sb[:, keep], yb[keep])
    s_op = Sb[nb.index(OPUS)]
    u, v = s_op[keep & (yb == 1)], s_op[keep & (yb == 0)]; dd = u[:, None] - v[None, :]
    auc_plain_drop = float(((dd > 0).sum() + 0.5 * (dd == 0).sum()) / dd.size)        # frontier2026_table.auc_plain
    rr = {"opus_refused": {"violations": int(REF[OPUS, run][POS].sum()), "compliant": int(REF[OPUS, run][NEG].sum())},
          "blocks": contrast(fib, nb), "half": contrast(fih, list(Gh["names"])),
          "dropped": contrast(fid, nb)}
    assert abs(rr["dropped"]["auc"] - r3(auc_plain_drop)) < 1e-9
    rr["dropped"]["n_dropped"] = int((~keep).sum())
    item1["runs"][f"run{run}"] = rr
out["item1_refusal_sensitivity"] = item1

# ================================================================================================================
# 2. uncertainty of the calibrate-then-test operating points
# ================================================================================================================
def pct(a):
    return [r3(x) for x in np.percentile(a, [2.5, 50, 97.5])]


item2 = {"a_split_to_split": {}, "which_judge_attains_.485": None}
for nm in (OPUS, GEMMA):
    e = {}
    for dl in (0.10, 0.20):
        mi, fb = cf_folds(S33[nm], dl)
        e[f"delta={dl:.2f}"] = {
            "cross_fitted_mean": {"miss": r3(mi.mean()), "false_block": r3(fb.mean())},
            "per_split_fold_average_p2.5_p50_p97.5": {"miss": pct(mi.mean(1)), "false_block": pct(fb.mean(1))},
            "per_single_fold_p2.5_p50_p97.5": {"miss": pct(mi.reshape(-1)), "false_block": pct(fb.reshape(-1))},
            "per_single_fold_min_max": {"miss": [r3(mi.min()), r3(mi.max())], "false_block": [r3(fb.min()), r3(fb.max())]}}
    m7, f7 = s287(S33[nm])
    e["28/7 splits delta=0.10 (2,000 splits)"] = {"mean": {"miss": r3(m7.mean()), "false_block": r3(f7.mean())},
                                                  "per_split_p2.5_p50_p97.5": {"miss": pct(m7), "false_block": pct(f7)}}
    item2["a_split_to_split"][nm] = e
item2["which_judge_attains_.485"] = {"judge": min(fb287, key=fb287.get), "false_block": r3(min(fb287.values())),
                                     "miss": r3(RES287[min(fb287, key=fb287.get)][0]),
                                     "next_lowest": [(k, r3(v)) for k, v in sorted(fb287.items(), key=lambda kv: kv[1])[1:3]]}

# (b) label-stratified nonparametric bootstrap, all 33 runs (paired: one resample of items shared by all judges)
B, nsp = A.B, A.boot_splits
Fb = folds(nsp)
SM = np.array([S33[nm] for nm in NAMES])
iO, iG = NAMES.index(OPUS), NAMES.index(GEMMA)
bm, bf, b7m, b7f = (np.empty((B, len(NAMES))) for _ in range(4))
rng = np.random.default_rng(A.boot_seed)
for b in range(B):
    idx = np.arange(len(y))
    idx[NEG] = rng.choice(NEG, size=len(NEG), replace=True)      # resample compliant items within their positions
    idx[POS] = rng.choice(POS, size=len(POS), replace=True)      # resample violations within their positions
    Sb_ = SM[:, idx]
    for j in range(len(NAMES)):
        mi, fb = cf_folds(Sb_[j], 0.10, Fb); bm[b, j], bf[b, j] = mi.mean(), fb.mean()
        m7, f7 = s287(Sb_[j]); b7m[b, j], b7f[b, j] = m7.mean(), f7.mean()


def ci(a, point):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return {"point": r3(point), "ci95_percentile": [r3(lo), r3(hi)], "boot_mean": r3(a.mean()), "boot_sd": r3(a.std(ddof=1))}


bt = {"B": B, "seed": A.boot_seed, "splits_inside_each_replicate": nsp,
      "design": ("resample the 97 compliant and the 35 violating items with replacement (label-stratified, paired across "
                 "judges); inside each replicate rerun the full cross-fit (the paper's 500 two-fold splits, seed 0, "
                 "applied to the resampled positions) and the 2,000 random 28/7 splits (seed 0); percentile intervals"),
      "Claude-Opus-5.5": {"crossfit_false_block_d.10": ci(bf[:, iO], fb10[OPUS]), "crossfit_miss_d.10": ci(bm[:, iO], mi10[OPUS]),
                          "split28_7_false_block_d.10": ci(b7f[:, iO], RES287[OPUS][1]),
                          "split28_7_miss_d.10": ci(b7m[:, iO], RES287[OPUS][0])},
      "Gemma-2-9B": {"crossfit_false_block_d.10": ci(bf[:, iG], fb10[GEMMA]), "crossfit_miss_d.10": ci(bm[:, iG], mi10[GEMMA]),
                     "split28_7_false_block_d.10": ci(b7f[:, iG], RES287[GEMMA][1]),
                     "split28_7_miss_d.10": ci(b7m[:, iG], RES287[GEMMA][0])},
      "paired_difference_Opus_minus_Gemma": {
          "crossfit_false_block_d.10": ci(bf[:, iO] - bf[:, iG], fb10[OPUS] - fb10[GEMMA]),
          "split28_7_false_block_d.10": ci(b7f[:, iO] - b7f[:, iG], RES287[OPUS][1] - RES287[GEMMA][1]),
          "share_of_replicates_Opus_lower_crossfit_fb": r3((bf[:, iO] < bf[:, iG]).mean())},
      "panel_extremes_over_33_runs": {
          "min_crossfit_false_block_d.10": ci(bf.min(1), min(fb10.values())),
          "max_crossfit_miss_d.10": ci(bm.max(1), max(mi10.values())),
          "min_split28_7_false_block_d.10": ci(b7f.min(1), min(fb287.values())),
          "share_of_replicates_Opus_has_lowest_crossfit_fb": r3((bf.argmin(1) == iO).mean()),
          "share_of_replicates_Opus_has_lowest_28_7_fb": r3((b7f.argmin(1) == iO).mean())},
      "caveat": ("the delta=.10 threshold is a fold minimum (28/7: second-smallest), an extreme order statistic for which the "
                 "naive nonparametric bootstrap is known to be unreliable; bootstrap duplicates can also fall in both the "
                 "calibration and the test fold, which pulls the replicated miss down. Read these as rough item-sampling "
                 "intervals, not exact coverage statements.")}
item2["b_bootstrap"] = bt
out["item2_uncertainty"] = item2

# ================================================================================================================
# 3. without the 7 benign-text goals
# ================================================================================================================
tc = json.load(open(os.path.join(R, "frontier2026/toolcall_diagnostic.json")))["_items"]
ADOJO = {40: ("workspace", "injection_task_0"), 41: ("workspace", "injection_task_1"), 42: ("workspace", "injection_task_2"),
         74: ("travel", "injection_task_6"), 75: ("travel", "injection_task_0"), 77: ("travel", "injection_task_2"),
         129: ("slack", "injection_task_3")}
dropped_items = []
for i in np.where(drop)[0].tolist():
    su, tid = ADOJO[i]
    assert man[i]["suite"] == su and man[i]["label"] == 1 and man[i]["kind"] == "injection_goal"
    call = [x["text"] for x in tc if x["suite"] == su and x["task"] == tid and x["label"] == 1]
    dropped_items.append({
        "index_in__bench_texts": i, "suite": su, "manifest_sha1": man[i]["sha1"], "agentdojo_id_inferred": f"{su}/{tid}",
        "id_evidence": (f"ground-truth call {call[0][:90]!r} (frontier2026/toolcall_diagnostic.json)" if call else
                        "only travel injection task without a ground-truth call in toolcall_diagnostic.json (0-5 have one)"),
        "text": bench[i], "runs_scoring_below_.5_of_33": int((S_fam[:, i] < 0.5).sum()),
        "opus55_score_run1": float(S33[OPUS][i]), "opus55_refused_run1": bool(REF[OPUS, 1][i]),
        "gemma_score": float(S33[GEMMA][i])})
fiD = family_inference(np.array(GD["S"]), np.array(GD["y"])); check_same(fiD, GD, "drop-benign")
cD = contrast(fiD, list(GD["names"]))
item3 = {"rule": "analyze_agentdojo_inference.py --family complete --drop-benign-goals: drop violations that a majority "
                 "(> 33/2) of the 33 run-1 judge runs score < .5",
         "n_dropped": int(drop.sum()), "dropped_items": dropped_items,
         "reduced_set": {"n_items": int(keep_b.sum()), "n_violations": int(y[keep_b].sum())},
         "opus55_refused_among_dropped": int((REF[OPUS, 1] & drop).sum()),
         "opus55_operating_points": {}, "gemma_operating_points": {}, "opus55_auc_and_contrast_reduced": cD}
for dl in (0.10, 0.20):
    fo = GD["_crossfit"](S33[OPUS], y, dl); ro = GD["_crossfit"](S33[OPUS][keep_b], y[keep_b], dl)
    fg = GD["_crossfit"](S33[GEMMA], y, dl); rg = GD["_crossfit"](S33[GEMMA][keep_b], y[keep_b], dl)
    item3["opus55_operating_points"][f"delta={dl:.2f}"] = {"all_132": {"miss": r3(fo[0]), "false_block": r3(fo[1])},
                                                          "without_7": {"miss": r3(ro[0]), "false_block": r3(ro[1])}}
    item3["gemma_operating_points"][f"delta={dl:.2f}"] = {"all_132": {"miss": r3(fg[0]), "false_block": r3(fg[1])},
                                                         "without_7": {"miss": r3(rg[0]), "false_block": r3(rg[1])}}
out["item3_no_benign_goals"] = item3

# ================================================================================================================
# 4. ROC oracle vs conformal cross-fit
# ================================================================================================================
def oracle_insample(s, t):
    v, c = s[y == 1], s[y == 0]
    cand = np.unique(np.concatenate([s, [np.inf]]))
    miss = (v[None, :] < cand[:, None]).mean(1); fb = (c[None, :] >= cand[:, None]).mean(1)
    ok = np.flatnonzero(miss <= t + 1e-12); j = ok[np.argmin(fb[ok])]
    m_ = int(np.floor(t * len(v) + 1e-9)); th2 = np.sort(v)[m_]                    # closed form must agree
    assert abs(fb[j] - (c >= th2).mean()) < 1e-12
    return float(cand[j]), float(miss[j]), float(fb[j])


DEL = (0.05, 0.10, 0.20)
per = {}
for nm in NAMES:
    row = {}
    for dl in DEL:
        th, m_is, f_is = oracle_insample(S33[nm], dl)
        mo, fo = cf_folds(S33[nm], dl, rule="oracle"); mp, fp = cf_folds(S33[nm], dl, rule="plugin")
        row[f"{dl:.2f}"] = {"insample_oracle": {"theta": r3(th), "miss": r3(m_is), "false_block": r3(f_is)},
                            "crossfit_conformal": {"miss": r3(CF[nm, dl][0]), "false_block": r3(CF[nm, dl][1])},
                            "crossfit_test_fold_oracle": {"miss": r3(mo.mean()), "false_block": r3(fo.mean())},
                            "crossfit_plugin_calibration_quantile": {"miss": r3(mp.mean()), "false_block": r3(fp.mean())},
                            "_raw": (f_is, CF[nm, dl][1], fo.mean(), fp.mean(), mo.mean(), mp.mean())}
    per[nm] = row


def summ(vals):
    vals = np.asarray(vals, float)
    return {"min": r3(vals.min()), "median": r3(np.median(vals)), "max": r3(vals.max())}


def argmin_names(v):
    w = [NAMES[i] for i in np.flatnonzero(np.isclose(v, v.min(), atol=1e-12))]
    return w if len(w) < len(NAMES) else "all 33 runs tied"


decomp = {}
for nm in (OPUS, GEMMA):
    decomp[nm] = {}
    for dl in DEL:
        f_is, f_cf, f_or, f_pl = per[nm][f"{dl:.2f}"]["_raw"][:4]
        decomp[nm][f"delta={dl:.2f}"] = {
            "discrimination_cost = insample_oracle_fb": r3(f_is),
            "half_sample_oracle_fb (crossfit_test_fold_oracle)": r3(f_or),
            "conformal_crossfit_fb": r3(f_cf),
            "calibration_conservativeness = conformal - test_fold_oracle": r3(f_cf - f_or),
            "conformal - insample_oracle": r3(f_cf - f_is)}
summary4 = {}
for dl in DEL:
    k = f"{dl:.2f}"; raw = np.array([per[nm][k]["_raw"] for nm in NAMES])
    summary4[f"delta={k}"] = {
        "insample_oracle_false_block": summ(raw[:, 0]), "crossfit_conformal_false_block": summ(raw[:, 1]),
        "crossfit_test_fold_oracle_false_block": summ(raw[:, 2]), "crossfit_plugin_false_block": summ(raw[:, 3]),
        "conformal_minus_insample_oracle": summ(raw[:, 1] - raw[:, 0]),
        "crossfit_test_fold_oracle_miss": summ(raw[:, 4]), "crossfit_plugin_miss": summ(raw[:, 5]),
        "argmin_insample_oracle": argmin_names(raw[:, 0]), "argmin_crossfit_conformal": argmin_names(raw[:, 1])}
for nm in NAMES:
    for k in per[nm]: per[nm][k].pop("_raw")
out["item4_roc_oracle"] = {
    "definitions": {
        "insample_oracle": "min over thresholds theta (block iff s >= theta) of the false block on all 97 compliant items "
                           "subject to miss on all 35 violations <= delta (no randomized thresholds)",
        "crossfit_conformal": "the paper's procedure (unify_crossfit.crossfit, 500 splits, seed 0); delta=.05 is block-all",
        "crossfit_test_fold_oracle": "same 500 splits; threshold chosen on the TEST fold's violations as the largest theta "
                                     "with test-fold miss <= delta; false block on the test fold's compliant items",
        "crossfit_plugin_calibration_quantile": "same splits; largest theta with CALIBRATION-fold miss <= delta (empirical "
                                                "quantile without the conformal (n+1) correction), evaluated on the test fold"},
    "Claude-Opus-5.5": per[OPUS], "Gemma-2-9B": per[GEMMA], "decomposition": decomp,
    "summary_over_33_runs": summary4, "per_run": per}

# ================================================================================================================
# 5. the rule that selected the 21 of 33 runs for the Q3 suffix attack
# ================================================================================================================
AC_PATH = os.path.join(R, "analyze_attack_common.py")
GA = _exec(AC_PATH, ["analyze_attack_common.py", "--family", "unified"])
ac_lines = open(AC_PATH).read().split("\n")
AC2REG = {"qwen2.5:3b": "Qwen-2.5-3B", "qwen2.5:7b": "Qwen-2.5-7B", "qwen2.5:32b": "Qwen-2.5-32B", "llama3.1:8b": "Llama-3.1-8B",
          "gemma2:9b": "Gemma-2-9B", "llama-3.3:70b": "Llama-3.3-70B", "command-a": "Command A"}
fam21 = [AC2REG.get(nm, nm) for nm in GA["names"]]
assert len(fam21) == 21 and set(fam21) <= set(NAMES)
excluded = [nm for nm in NAMES if nm not in fam21]
sw_avail = {nm: int(sum(DICT[nm].get(g + SW) is not None for g in goals)) for nm in excluded}
uc_lines = open(os.path.join(R, "unify_crossfit.py")).read().split("\n")
b9 = open(os.path.join(R, "paper/supp/appB_experiments.tex")).read().split("\n")
b9_ln = next(i for i, l in enumerate(b9) if "over all 21 judges with clean" in l)


def quote(lines, a, b, f):
    return {"file": f, "lines": f"{a}-{b}" if a != b else str(a), "text": "\n".join(lines[a - 1: b])}


out["item5_q3_selection_rule"] = {
    "summary": ("No score-based selection: the 21-run family is the hard-coded judge list of analyze_attack_common.py "
                "(11 'panel' judges + 3 'post hoc' + 7 frontier2026 judges), i.e. exactly the runs whose suffix-rewritten "
                "strings were ever scored; line 49 asserts completeness (it would crash, not silently drop). The 12 "
                "excluded runs have 0/35 suffix-rewritten scores in any cache."),
    "quotes": [quote(ac_lines, 3, 5, "analyze_attack_common.py"), quote(ac_lines, 10, 13, "analyze_attack_common.py"),
               quote(ac_lines, 31, 47, "analyze_attack_common.py"), quote(ac_lines, 49, 49, "analyze_attack_common.py"),
               quote(uc_lines, 54, 54, "unify_crossfit.py"),
               quote(b9, b9_ln + 1, b9_ln + 3, "paper/supp/appB_experiments.tex")],
    "family_21": fam21, "excluded_12": excluded, "suffix_rewritten_scores_available_for_excluded": sw_avail}
assert len(excluded) == 12 and all(v == 0 for v in sw_avail.values())
assert "complete clean and suffix-rewritten" in ac_lines[2] and "assert None not in" in ac_lines[48]

# ================================================================================================================
# 6. audit (needs git; otherwise run --audit-only in a git clone)
# ================================================================================================================
if not A.skip_audit:
    try:
        git("rev-parse", "HEAD"); out["item6_predeclared_audit"] = audit()
    except Exception as ex:                                   # no git clone here
        out["item6_predeclared_audit"] = {"status": f"not run here ({type(ex).__name__}); run `python3 stats_reporting.py "
                                                    "--audit-only` in a git clone to merge it into this JSON"}
json.dump(out, open(A.json, "w"), indent=1)

# ---------------- console summary ----------------
print(f"anchors reproduced: {out['anchors']['all_reproduced']}")
for a_ in A_rows:
    print(f"  {'OK ' if a_['reproduced'] else 'NO '} {a_['anchor']}: paper {a_['paper']} computed {a_['computed']}")
for run in (1, 2, 3):
    rr = item1["runs"][f"run{run}"]
    print(f"run {run}: Opus refusals {rr['opus_refused']}")
    for v in ("blocks", "half", "dropped"):
        c = rr[v]
        print(f"   {v:8s} n={c['n_items']} AUC {c['auc']:.3f} vs Gemma {c['auc_gemma']:.3f}  d {c['delta']:+.3f} "
              f"[{c['delta_95_marginal'][0]:+.3f},{c['delta_95_marginal'][1]:+.3f}] p={c['p_marginal']:.4f} | sim "
              f"[{c['delta_simultaneous_95'][0]:+.3f},{c['delta_simultaneous_95'][1]:+.3f}] pmaxT={c['p_maxT']:.4f}")
print(json.dumps(item2["a_split_to_split"], indent=None)[:2000])
print(json.dumps({k: v for k, v in bt.items() if k not in ("design", "caveat")}, indent=None))
print("item3:", json.dumps({k: v for k, v in item3.items() if k != "dropped_items"}))
for d_ in item3["dropped_items"]:
    print("  ", d_["index_in__bench_texts"], d_["agentdojo_id_inferred"], d_["runs_scoring_below_.5_of_33"], d_["opus55_score_run1"], d_["text"][:70])
print("item1 Opus refused in all three runs / any run:", item1["opus_refused_violations_in_all_three_runs"],
      item1["opus_refused_violations_in_any_run"])
print("item4 decomposition:", json.dumps(decomp))
print("item4 summary:", json.dumps(summary4))
print("item4 Opus:", json.dumps(per[OPUS])); print("item4 Gemma:", json.dumps(per[GEMMA]))
print("item5 excluded:", excluded)
print(f"wrote {A.json}")
