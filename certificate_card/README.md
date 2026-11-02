# Guardrail certificate card (template)

A runtime-guardrail evaluation supports a security claim only if it states the five fields
below (Table 2 of the paper; each field is required by one of its results). Copy this file,
fill every field, and keep the red flags in mind. A machine-readable version is
`card.schema.json` (JSON Schema, draft 2020-12); `example_claude_opus_5_5.json` fills it in for
the best calibrated gate in the paper.

| # | Field | State | Required by | Red flag |
|---|-------|-------|-------------|----------|
| 1 | Policy, oracle interface, judge input | policy language; oracle predicates Π the gate may query; trusted metadata; what the judge sees (instruction, serialized call, call in context) | Thm. 1, Prop. 2 | a policy called "enforced" although a judge decides one of its predicates |
| 2 | Operating point | miss and false block (with intervals) at the deployed threshold; target δ; horizon H | Prop. 2 | AUC without an operating point |
| 3 | Calibration | source and number n of violating calibration examples; exchangeability claim; training-conditional miss | Thm. 3 | n < H/δ − 1: the certified gate blocks everything |
| 4 | Adversarial margin | a verified margin η̄(ρ) with its budget ρ, or "uncertified under attack"; attacks only lower-bound it | Thm. 4, Cor. 3 | "robust" backed only by failed attacks |
| 5 | Feedback | whether blocks are visible to the agent; paired closed-loop reruns (attack success, task success, gating fraction) with A/A and block-rate-matched random-drop controls; exploratory logs (sandboxed) for claims about other gates | Prop. 3, Prop. 4 | closed-loop claims from static scores or ungated logs |

## One-sentence form

> Relative to oracle predicates Π = {…} and trusted metadata {…}, the gate at threshold θ has
> miss β̂ [CI] and false block α̂ [CI] (target δ, horizon H, judge input: …), calibrated on n
> violating examples from {population} under {exchangeability claim}; under attack it is
> {uncertified | certified for margin η̄(ρ) at budget ρ}; feedback is {absent | modeled by … |
> evaluated with A/A and random-drop controls}.

## Quick checks

- Split-conformal calibration at per-step level δ/H is non-vacuous only with n ≥ H/δ − 1
  violating examples (19 at δ = .05, H = 1; 199 at H = 10). Below that, report block-all.
- With k = ⌊(n+1)δ/H⌋ = 1 the deployed gate's true miss is Beta(1, n): for n = 17 it exceeds
  δ = .10 with probability .9^17 ≈ .17. State the training-conditional miss, not only the
  marginal bound.
- A measured miss increase L under attack refutes margin-aware recalibration whenever
  δ/H ≤ L; it can never certify a margin.
