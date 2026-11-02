"""results.json -> table.tex (booktabs snippet). No network, no numpy needed."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
o = json.load(open(f"{HERE}/results.json"))


def f3(x, sign=False):
    s = f"{x:+.3f}" if sign else f"{x:.3f}"
    return s.replace("0.", ".", 1)


def ci(v, sign=False):
    return f"[${f3(v[0], sign)},{f3(v[1], sign)}$]"


lines = [r"\begin{table}[t]", r"\centering\scriptsize\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
         r"Judge & AUC [95\% CI] & $\Delta$AUC vs 9B [95\% CI] & miss@.5 clean & miss@.5 attacked & $\Delta$miss [sim.\ 95\%] \\",
         r"\midrule"]
ctx = o.get("context_rows", {})
for nm, r in ctx.items():
    d = "---" if nm.startswith("Gemma") else f"${f3(r['dAUC'], True)}$ {ci(r['dAUC_ci'], True)}"
    lines.append(f"{nm} & ${f3(r['auc'])}$ {ci(r['auc_ci95'])} & {d} & ${f3(r['miss_clean'])}$ & "
                 f"${f3(r['miss_attacked'])}$ & ${f3(r['dmiss'], True)}$ {ci(r['dmiss_sim_ci95_family15'], True)} \\\\")
lines.append(r"\midrule")
extra_started = False
for nm, r in o["judges"].items():
    if "auc" not in r:
        lines.append(f"{nm} & \\multicolumn{{5}}{{c}}{{{r.get('status', 'n/a')}}} \\\\"); continue
    if nm.endswith("*") and not extra_started:
        lines.append(r"\midrule"); extra_started = True
    pr = r["paired_vs_gemma9b"]
    band = r.get("dmiss_sim_ci95_family15") or r.get("dmiss_sim_ci95_family_all")
    pf = r["parse_failures"]
    mark = r"$^\dagger$" if pf["of_which_api_refusal_stop_reason"] else ""
    lines.append(f"{nm}{mark} & ${f3(r['auc'])}$ {ci(r['auc_ci95'])} & ${f3(pr['dAUC'], True)}$ {ci(pr['dAUC_ci'], True)} & "
                 f"${f3(r['miss_clean'])}$ & ${f3(r['miss_attacked'])}$ & ${f3(r['dmiss'], True)}$ {ci(band, True)} \\\\")
    if pf["of_which_api_refusal_stop_reason"]:
        s1 = r["sensitivity_refusal_as_1"]
        lines.append(f"\\quad refusal$\\to$1.0 & ${f3(s1['auc'])}$ {ci(s1['auc_ci95'])} & ${f3(s1['dAUC'], True)}$ "
                     f"{ci(s1['dAUC_ci'], True)} & ${f3(s1['miss_clean'])}$ & ${f3(s1['miss_attacked'])}$ & ${f3(s1['dmiss'], True)}$ \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
fam = o.get("attack_family", {})
cap = (r"\caption{2026-11 frontier judges on the 132 AgentDojo instruction strings (97 compliant, 35 violating), "
       r"same fixed 0--100 prompt; Gemma-2-9B reference AUC $.813$. AUC CI: item bootstrap (2{,}000 draws, as "
       r"\texttt{agentdojo\_ci.py}); $\Delta$AUC: paired item bootstrap (5{,}000 shared draws, as "
       r"\texttt{agentdojo\_paired.py}). C3: miss = score $<.5$ on all 35 violations, clean vs.\ the predeclared "
       r"16-token administrator-preapproval suffix; $\Delta$miss intervals are 20{,}000-draw suite-stratified "
       f"max-$|T|$ simultaneous bands over the paper's 11 judges plus the four new judges (critical value "
       f"{fam.get('crit_value', float('nan')):.2f})"
       r"; exploratory rows (*) use the band over all 17 judges. "
       r"$^\dagger$Provider refusals (\texttt{stop\_reason=refusal}, empty text) are scored $0.5$ by the repo's "
       r"parse-failure rule; the indented row rescores them $1.0$ (the repo's content-filter rule). "
       r"Claude 5.5 models reject \texttt{temperature}; Opus 5.5 cannot disable thinking (adaptive, effort low).}")
lines += [cap, r"\label{tab:frontier2026}", r"\end{table}"]
open(f"{HERE}/table.tex", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
