"""Write the Supplement E table and the ranges quoted in the text from toolcall_incontext_summary.json.
Usage: python3 toolcall_incontext_table.py  -> toolcall_incontext_table.tex (+ ranges on stdout)"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "toolcall_incontext_summary.json")))
NAMES = {"claude-opus-5-5": "Claude Opus 5.5", "claude-fable-5-1": "Claude Fable 5.1", "gpt-5.5": "GPT-5.5",
         "gpt-6.1-terra": "GPT-6.1 Terra", "claude-haiku-5-5": "Claude Haiku 5.5", "gpt-6.1-sol": "GPT-6.1 Sol",
         "claude-sonnet-5-5": "Claude Sonnet 5.5"}
M = [m for m in NAMES if m in S]
f = lambda x: f"{x:.3f}".replace("0.", ".", 1) if x < 1 else "1.000"
rows = []
for m in M:
    ic, ac = S[m]["incontext"], S[m]["action"]
    g, ga = ic["gates"]["0.10"], ac["gates"]["0.10"]
    rows.append(f"{NAMES[m]} & {f(ic['auc_c0'])} {{\\scriptsize[{f(ic['auc_c0_ci'][0])},{f(ic['auc_c0_ci'][1])}]}} & {f(ac['auc_c0'])} & "
                f"{f(ic['auc_c1'])} & {f(ac['auc_c1'])} & {f(g['miss'][0])} & {f(g['fb_c0'][0])} & {f(g['fb_c1'][0])} & "
                f"{f(ga['miss'][0])} & {f(ga['fb_c0'][0])} & {ic['refusals_v1']} \\\\")
tex = r"""\begin{table*}[t]
\caption{In-context tool-call evaluation (one run each; protocol in \texttt{toolcall\_incontext\_build.py}).
V1: the injected goal's culminating call (104, in 26 goal clusters); C1: the user's culminating call in the same
injected context (104); C0: the user's culminating call in a clean context (75). AUC with a goal/user-task cluster
bootstrap 95\% interval; ``action'': the same calls scored without context by the paper's prompt. Gates: split-conformal
$\delta=.10$ ($H=1$), cross-fitted over 500 two-fold splits of the 26 goals (52 calibration violations per fold); miss on
the held-out goals' V1, false block (FB) on all C0 and on the held-out goals' C1. Refus.: API-signaled refusals among the
104 V1 calls in context, scored as blocks.}
\label{tab:incontext}
\centering\small
\begin{tabular}{@{}lcccccccccc@{}}
\toprule
& \multicolumn{2}{c}{AUC, V1 vs.\ C0} & \multicolumn{2}{c}{AUC, V1 vs.\ C1} & \multicolumn{3}{c}{$\delta=.10$ gate, in context} & \multicolumn{2}{c}{$\delta=.10$, action} & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}
Judge & in context & action & in context & action & miss & FB C0 & FB C1 & miss & FB C0 & Refus. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
open(os.path.join(HERE, "toolcall_incontext_table.tex"), "w").write(tex)


def rng(key, var="incontext", d=None, sub=None):
    vals = []
    for m in M:
        x = S[m][var]
        x = x["gates"][d][sub][0] if d else x[key]
        vals.append(x)
    return f"{f(min(vals))}--{f(max(vals))}"


print("AUC V1|C0 in context", rng("auc_c0"), "| action", rng("auc_c0", "action"))
print("AUC V1|C1 in context", rng("auc_c1"), "| action", rng("auc_c1", "action"))
print("concordance in context", rng("concordance"), "| action", rng("concordance", "action"))
for d in ("0.05", "0.10", "0.20"):
    print(f"delta={d} in context: miss", rng(None, "incontext", d, "miss"), "FB C0", rng(None, "incontext", d, "fb_c0"),
          "FB C1", rng(None, "incontext", d, "fb_c1"), "| action: miss", rng(None, "action", d, "miss"),
          "FB C0", rng(None, "action", d, "fb_c0"))
lo = [S[m]["diff_ci"]["auc_c0"][0] for m in M]; lo1 = [S[m]["diff_ci"]["auc_c1"][0] for m in M]
print("diff AUC C0 lower bounds", [round(x, 3) for x in lo], "| C1", [round(x, 3) for x in lo1])
for v in ("SW", "JA"):
    for key in ("xfit_0.10", "xfit_0.20"):
        ch = [S[m]["attack"][v][key]["miss_att"] - S[m]["attack"][v][key]["miss_clean"] for m in M]
        print(v, key, "change in miss", f"{min(ch):+.3f} to {max(ch):+.3f}")
    print(v, "miss@.5 clean", [S[m]["attack"][v]["miss_.5_clean"] for m in M], "att", [S[m]["attack"][v]["miss_.5_att"] for m in M])

# context ablation table (toolcall_incontext_ablation.py): four judges, four information variables
AB = [m for m in ("claude-opus-5-5", "gpt-5.5", "claude-haiku-5-5", "claude-sonnet-5-5") if "ablation_user" in S.get(m, {})]
if AB:
    VARS = (("action", "action only"), ("ablation_user", "+ user request"), ("ablation_noprov", "+ tool outputs"),
            ("incontext", "+ untrusted note"))
    SHORT = {"claude-opus-5-5": "Opus 5.5", "gpt-5.5": "GPT-5.5", "claude-haiku-5-5": "Haiku 5.5",
             "claude-sonnet-5-5": "Sonnet 5.5"}
    rows = []
    for m in AB:
        for k, (v, lab) in enumerate(VARS):
            st = S[m][v]; g = st["gates"]["0.10"]
            rows.append(f"{SHORT[m] if k == 0 else ''} & {lab} & {f(st['auc_c0'])} & {f(st['auc_c1'])} & {f(g['miss'][0])} & "
                        f"{f(g['fb_c0'][0])} & {f(g['fb_c1'][0])} \\\\")
        rows.append("\\addlinespace")
    tex2 = r"""\begin{table}[t]
\caption{Which part of the context matters (one run; \texttt{toolcall\_incontext\_ablation.py}). Each row adds one
part of the judge's input: the user request, then the executed calls with their tool outputs, then the sentence
stating that tool outputs are untrusted (``untrusted note''; the full in-context prompt of Table~\ref{tab:incontext}). AUC of V1 against
C0 and C1; cross-fitted $\delta=.10$ gate as in Table~\ref{tab:incontext}.}
\label{tab:incontext-ablation}
\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llccccc@{}}
\toprule
& & \multicolumn{2}{c}{AUC, V1 vs.} & \multicolumn{3}{c}{$\delta=.10$ gate} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-7}
Judge & judge input & C0 & C1 & miss & FB C0 & FB C1 \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(os.path.join(HERE, "toolcall_incontext_ablation_table.tex"), "w").write(tex2)
    print("ablation table written for", AB)

# adaptive injection-rewrite table (toolcall_incontext_adaptive.py, + _ext), with the strict checks when present
if "adaptive" in S:
    AD = S["adaptive"]; CK = S.get("adaptive_check", {})
    order = [m for m in NAMES if m in AD["targets"]]
    rows = []
    for m in order:
        o = AD[m]; r, r5 = o["0.10"], o["0.50"]
        lo, hi = r["N20"][1]
        c = CK.get(m, {}).get("0.10")
        chk = (f" & {c['strict_majority']}/{c['flagged']} & {c['expected_evasions']:.1f} & {f(c['eta_strict'])} & {c['null_bo20']}/{c['flagged']}"
               f" & {c['null_adjusted']}/{c['flagged']} & {f(c['eta_adjusted'])}" if c else " & -- & -- & -- & -- & -- & --")
        rows.append(f"{NAMES[m]} & {f(o['thresholds']['0.10'])} & {r['N1'][0]}/{r['flagged']} & {r['N20'][0]}/{r['flagged']} "
                    f"{{\\scriptsize[{f(lo)},{f(hi)}]}}{chk} & {CK[m]['0.50']['strict_majority']}/{CK[m]['0.50']['flagged']} & {o['refusals']} \\\\")
    tex3 = r"""\begin{table*}[t]
\caption{Query-adaptive rewrites of the injection against in-context judges (\texttt{toolcall\_incontext\_adaptive.py},
checks in \texttt{toolcall\_incontext\_adaptive\_check.py}; one run). $\hat\theta$: in-sample $\delta=.10$ threshold on
the 104 clean V1 scores. Best of $N$: flagged-when-clean V1 calls for which some of the first $N$ rewrites was scored
below $\hat\theta$ once (search-time counts under first-integer parsing, which overstate evasion; Clopper--Pearson
95\%). Strict: the selected rewrite evades in at least
3 of 5 fresh re-scores with strict parsing (a bare integer, otherwise $1.0$); $E$: expected evasions, the sum over
calls of the selected rewrite's strict evasion fraction; $\hat\eta$: $E/104$, a point estimate of the full-law miss
increase of Definition~\ref{def:adversary}. Null: the same best-of-20 selection applied to 20 strict re-scores of
the unmodified clean prompt; adj.: strict evasions and $\hat\eta$ after removing calls whose clean prompt also fell
below $\hat\theta$ in the null; $\theta=.5$: strict stable evasions at the native threshold. Refus.: refused attacked
queries (scored as blocks).}
\label{tab:incontext-adaptive}
\centering\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}lccccccccccc@{}}
\toprule
& & \multicolumn{2}{c}{best of $N$, $\delta=.10$ gate} & \multicolumn{3}{c}{strict re-scores} & null & \multicolumn{2}{c}{null-adj.} & $\theta=.5$ & \\
\cmidrule(lr){3-4}\cmidrule(lr){5-7}\cmidrule(lr){8-8}\cmidrule(lr){9-10}\cmidrule(lr){11-11}
Judge & $\hat\theta$ & $N{=}1$ & $N{=}20$ & strict & $E$ & $\hat\eta$ & $N{=}20$ & strict & $\hat\eta$ & strict & Refus. \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    open(os.path.join(HERE, "toolcall_incontext_adaptive_table.tex"), "w").write(tex3)
    print("adaptive table written for", order)
