"""Paired comparison for the tool-call-level diagnostic (Supplement E): on the 123 AgentDojo tasks that have a
ground-truth call, each judge's AUC on its run-1 instruction-string scores versus its tool-call scores
(toolcall_diagnostic.json), with a paired DeLong test (same tasks, same labels). Cache-only.
Usage: PYTHONPATH=<agentdojo pkg> python toolcall_paired.py"""
import json, os, numpy as np
from math import erf, sqrt
from agentdojo.task_suite.load_suites import get_suites
HERE = os.path.dirname(os.path.abspath(__file__))
tc = json.load(open(os.path.join(HERE, "toolcall_diagnostic.json")))
items = tc["_items"]
text_of = {}
for sname, suite in get_suites("v1.2.1").items():
    for tid, t in suite.user_tasks.items(): text_of[(sname, tid)] = t.PROMPT
    for tid, t in suite.injection_tasks.items(): text_of[(sname, tid)] = t.GOAL
y = np.array([it["label"] for it in items])
def placements(s):
    X, Y = s[y == 1], s[y == 0]
    psi = (X[:, None] > Y[None, :]) + 0.5 * (X[:, None] == Y[None, :])
    return psi.mean(), psi.mean(1), psi.mean(0)
out = {}
for tag in [k for k in tc if not k.startswith("_")]:
    clean = json.load(open(os.path.join(HERE, f"frontier2026_{tag}_bench.json")))
    raw = json.load(open(os.path.join(HERE, f"frontier2026_{tag}_bench_raw.json")))
    def val(t):
        if t in clean: return clean[t]
        return 1.0 if raw.get(t, {}).get("stop") == "refusal" else 0.5
    si = np.array([val(text_of[(it["suite"], it["task"])]) for it in items], float)
    st = np.array(tc[tag]["scores"], float)
    ai, v10i, v01i = placements(si); at, v10t, v01t = placements(st)
    m, n = len(v10i), len(v01i)
    var = (np.var(v10i - v10t, ddof=1)) / m + (np.var(v01i - v01t, ddof=1)) / n
    z = (at - ai) / sqrt(var) if var > 0 else 0.0
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    out[tag] = {"auc_instruction_123": float(ai), "auc_toolcall": float(at), "diff": float(at - ai), "z": float(z), "p": float(p)}
    print(f"{tag:18s} instruction(123)={ai:.3f} toolcall={at:.3f} diff={at-ai:+.3f} z={z:+.2f} p={p:.4f}")
json.dump(out, open(os.path.join(HERE, "toolcall_paired.json"), "w"), indent=1)
