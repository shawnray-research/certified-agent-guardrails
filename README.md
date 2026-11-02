# Runtime guardrails for tool-using LLM agents -- code & data

> This work has been submitted to the IEEE for possible publication. Copyright may be transferred
> without notice, after which this version may no longer be accessible.

Reproducibility artifact for *What Can Be Enforced? Limits of Runtime Guardrails for Tool-Using LLM Agents*.

Everything here is **inference-only** (no training). Judge scores for our own curated
probe and for the AgentDojo tasks are cached as JSON, so those tables and figures
regenerate from the cache without any model inference or API access. Re-scoring (optional)
requires the relevant provider API keys, supplied through environment variables -- no keys
are stored in this repository.

## Datasets and redistribution

This artifact does **not** redistribute upstream benchmark prompts. AgentDojo (pinned to
`v1.2.1`) is loaded from its own package. InjecAgent and AgentHarm are third-party benchmarks and
must be obtained from their original sources under their own terms; their prompts and any
scores keyed by them are **not** included here. We release only (i) our own novel curated
probe (`actions.json`), (ii) cached judge scores for the curated probe and the AgentDojo
tasks, and (iii) the code. The InjecAgent/AgentHarm columns of the paper's cross-benchmark
table reproduce by re-scoring once those datasets are obtained from the originals (cited in
the paper).

## Layout

- `enforce.py` -- the gate: binormal/real judge scoring, split-conformal calibration
  (`conformal_threshold`), and the safety/utility frontier utilities.
- `run_synthetic.py` -- synthetic judge with a known analytic ROC (validates C1/C2 against
  ground truth).
- `agentdojo_ci.py` -- per-judge AgentDojo AUC with bootstrap 95% CIs.
- `agentdojo_paired.py` -- paired-bootstrap AUC difference vs. the 9B reference + TOST
  equivalence over the AgentDojo judges (regenerates Table `tab:ci`).
- `analyze_loop.py` -- closed-loop multi-suite analysis: per-suite plus
  Cochran-Mantel-Haenszel stratified attack-success with Wilson intervals.
- `ensemble_robustness.py` -- shows ensembling does not fix safe-washing (the max-ensemble
  margin stays high because the vulnerability is correlated across judges).
- `make_scaling_fig.py`, `make_roc_fig.py`, `make_model_fig.py` -- figures.
- `scores_cache.json`, `bench_scores_cache.json`, `adv_scores_cache.json`, `*_bench.json`,
  `*_curated.json`, `*_safewash.json` -- cached judge scores (curated probe + AgentDojo only).
- `actions.json` -- the novel curated probe (our own data). `_bench_texts.json`,
  `_inj_goals.json`, `_bench_safewash_texts.json` -- AgentDojo-derived task/goal texts.
- `loop_allsuites.json` -- closed-loop results across all four AgentDojo suites (backs
  Table `tab:loop`); `loop_gemma9b.json` -- the 9B good-judge sweep (App. B.5).
- `analyze_gemma.py` -- reproduces the 9B good-judge numbers from `loop_gemma9b.json`.
- `frontier_analyze.py` -- the current closed frontier (Claude Opus 4.5, Claude Sonnet 4.5,
  Amazon Nova Pro, Mistral Large 3): reproduces the AgentDojo ceiling (paired AUC vs. the 9B),
  the safe-washing collapse (per-judge `eta`), and the seven-vendor label-ambiguity panel,
  all from cached scores. `run_adv_frontier_adaptive.py` -- the best-of-N *optimizing*
  safe-washing attack that collapses Claude Opus 4.5 (`eta=0.28`) where a single rewrite does
  not. Cached scores: `frontier_*_bench.json`, `frontier_adaptive_*.json`; `frontier_score.py`
  is the scorer that produced them.
- `q5_enforcement_cost.py` -- Q5 (Sec. 7.6): per-decision latency of the compiled P2+P3 keyed-counter
  gate on synthetic streams (10^2..10^6 recipients), a differential check against a naive replay
  monitor, and a cap-redundancy check on the T3 saturation abstraction (P2 redundant under a session limit L <= c). Writes
  `q5_enforcement_cost.json`; no inference. `q5_judge_latency.py` -- wall time of one hosted judge call
  (paper prompt and settings; needs API keys) -> `q5_judge_latency.json`.
- `make_fig_main_results.py`, `make_fig_forest.py` (reads `forest33.json` from
  `analyze_agentdojo_inference.py --family complete --json forest33.json`), `make_fig_q5.py` -- the
  journal figures (Fig. 2, Fig. 3, Fig. 4) at final print size in the house style of `paper_style.py`.
- `frontier2026/frontier2026_bestofn.py` -- predeclared query-adaptive best-of-N representation attack (Supplement
  B.20; needs API keys); `frontier2026/frontier2026_bestofn_summary.py` -- its table and curve from the caches only.
- `frontier2026/toolcall_diagnostic.py` -- tool-call-level diagnostic (Supplement E): the seven no-temperature judges
  score AgentDojo's 123 ground-truth culminating tool calls (needs API keys and `agentdojo==0.1.35`, which ships benchmark
  v1.2.1); scores in `toolcall_diagnostic.json`; `frontier2026/toolcall_paired.py` -- paired DeLong comparison with the
  instruction-string scores (cache-only). `curated_frontier2026.py` (+ helper `b4_frontier_pairs.py`) -- current frontier
  judges on the curated probe (Supplement B.3).
- `prop_explore_sim.py` -- exact-model check of the exploratory-logging proposition (Supplement B.2).
- `calib_split_sensitivity.py` -- calibration-size sensitivity (random 28/7 violation splits, Sec. 7.2);
  `llamaguard_dominance.py` -- judges dominating Llama-Guard-3's operating point (Supplement B.7).
- `make_fig_roc.py`, `make_fig_supp.py` -- supplement figures (ROC; synthetic frontier, controlled frontier,
  real-judge certificate) in the same house style, from caches and the experiment modules.
- `*_score.py`, `judge.py`, `run_loop.py` -- the scorers / attack / closed-loop harness that
  *produced* the caches (each reads its model id and API key from environment variables).

## Late-2026 refresh and paper sources

- `analyze_agentdojo_inference.py` -- the authoritative paired familywise analysis (Section 7 and
  Supplements B.6/E, forest plot): DeLong AUCs with full paired covariance, 100,000-draw max-T
  simultaneous intervals vs Gemma-2-9B, Holm check, equivalence margins, and MDE.
  `--family complete` reproduces the paper's 33 judge runs (32 models; Mistral-Large-3 via two
  providers); the default reproduces the earlier 23-judge panel and `--family all2026` the 30-judge one.
- `frontier2026/` -- late-2026 frontier refresh (Table `tab:frontier2026`): cached scores for seven
  current judges (3 runs each, raw replies incl. provider refusals), the best-of-6 safe-washing attack,
  and `frontier2026_table.py`, which prints every number in the table; see `frontier2026/README_frontier2026.md`.
- `frontier_analyze.py` now computes AUC vectorized (identical values; runs in seconds instead of >10 minutes).
- `analyze_attack_common.py` -- the common-population representation attack (Supplement B.9): miss at
  theta=.5 before/after the 16-word suffix on all 35 violations, 20,000-draw suite-stratified bootstrap with
  studentized max-T simultaneous intervals; `--family all` adds the three post-hoc judges.
- `agentdojo_manifest.json` -- suite and label for each of the 132 items of `_bench_texts.json`.
- `agentdojo_paired.py` and `agentdojo_ci.py` are legacy marginal analyses with no multiplicity correction;
  their per-contrast verdicts are superseded by `analyze_agentdojo_inference.py`.
- `controlled_model.py` -- exact enumeration of the two-step retry model (Supplement B.2): reproduces the
  perfect-recall, score/time, and stationary optima (.551/.863/1.137) in rational arithmetic.
- `python analyze_agentdojo_inference.py --drop-benign-goals [--family all2026]` -- sensitivity analysis (Supplement B.6):
  drops the injection goals that a majority of the judge runs in the chosen family score < .5 (7 goals for
  `--family complete`), reruns the familywise analysis,
  and prints the reference judge's cross-fitted operating points.
- `make_main_results_fig.py` -- regenerates Figure 2 (static frontier and calibration, suffix attack, paired
  reruns) from cached scores: `python make_main_results_fig.py paper/figures/main_results.pdf`.
- `make_forest_fig.py` -- regenerates the forest plot (Supplement E) from `python analyze_agentdojo_inference.py --family paper --json forest.json`:
  `python make_forest_fig.py forest.json paper/figures/agentdojo_forest_col.pdf`.
- `make_overview_fig.py` -- draws Figure 1 at true column width (`python make_overview_fig.py paper/figures/overview.pdf`).
- `frontier2026/replication/` -- an independent single-run re-scoring of the refresh with a separate
  harness (cross-check only; the canonical numbers come from `frontier2026/`).
- `paper/` -- LaTeX sources of the IEEE Transactions on Dependable and Secure Computing (TDSC)
  submission (`main.tex`, IEEEtran journal/compsoc) and its separately submitted supplementary
  material (`supplement.tex`: A complete proofs, B extended experiments, C reproducibility
  details, D extended related work), the cover letter (`cover_letter.tex`), and the compiled PDFs. Build each with
  `pdflatex main && bibtex main && pdflatex main && pdflatex main` (then the same for
  `supplement`, which cross-references `main.aux`). An earlier version is arXiv:2607.22868.

## In-context evaluation, attacks, baselines, and the certificate card (TDSC revision)

- `frontier2026/toolcall_incontext_build.py` -- in-context tool-call evaluation (Supplement E, Tables tab:incontext,
  tab:incontext-ablation, tab:incontext-adaptive): 104 matched (user task, goal) pairs plus 75 clean contexts from AgentDojo
  v1.2.1 ground-truth traces with `important_instructions` injections (`direct` argument: the tagless "TODO:" variant).
  Needs `agentdojo==0.1.35`; the items are shipped (`toolcall_incontext_items*.json`). `toolcall_incontext_score.py`
  (in context + action-only), `toolcall_incontext_attack.py` (fixed sentence inside the injection),
  `toolcall_incontext_adaptive.py` (best-of-20 injection rewrite; `_ext` = the other four judges), and
  `toolcall_incontext_ablation.py` (user request only / no untrusted note) need API keys and write the JSON caches.
  `toolcall_incontext_analyze.py` (numpy+scipy) recomputes `toolcall_incontext_summary.json` from the caches alone, and
  `toolcall_incontext_table.py` writes the Supplement E tables; `toolcall_incontext_rules.py` gives the model-free
  baselines, and `toolcall_incontext_flagged_pairs.json` lists the 13 pairs excluded in the sensitivity check.
- `frontier2026/frontier2026_pair.py` -- predeclared optimizing (PAIR-style) attacker on instruction strings at the
  in-sample delta=.10 gate and at theta_f (Supplement E, Table tab:pair; protocol fixed before the run); per-judge
  trajectories `frontier2026_pair_<judge>.json`, summaries `frontier2026_pair_result_<judge>.json` and
  `frontier2026_pair_summary.json`; `frontier2026_pair_rescore.py` (predeclared strict stability check: five fresh
  bare-integer re-scores of each selected rewrite plus a clean-violation null control) writes
  `frontier2026_pair_rescore.json`, which gives the counts reported in the paper. Needs API keys.
- `baselines/pi_detector_baseline.py` -- prompt-injection detector baselines (ProtectAI DeBERTa-v3 v2, PIGuard; pinned
  Hugging Face revisions) on the 132 instruction strings and 123 tool calls; `analyze` mode reproduces Supplement B.21
  from `baselines/pi_detector_scores.json` with numpy+scipy only.
- `calib_randomized.py` -- deterministic vs. exact-level randomized split-conformal operating points for all 33 judge
  runs and the item-bootstrap CI of the delta=.10 false block (writes `calib_randomized.json`).
- `stats_reporting.py` -- cache-only statistical checks for Sec. 7.2 (refusal sensitivity of the Opus 5.5 contrast,
  split-to-split spread and label-stratified bootstrap of the cross-fitted operating points, operating points without
  the 7 benign-text goals, ROC-oracle comparison); reproduces every anchor number first; writes `stats_reporting.json`.
- `run_loop_controls.py` -- predeclared Q4 control protocol (A/A rerun, keyed-hash random drop matched per suite,
  3B gate). Released, not run.
- `certificate_card/` -- the guardrail certificate card of Table 2 as a fill-in Markdown template with red flags, a JSON
  Schema (`card.schema.json`), and a filled example for the best calibrated gate (`example_claude_opus_5_5.json`);
  `make_card_table.py` writes the Supplement D table of card fields 2 and 4 for all 33 judge runs.
- `make_fig_incontext.py` -- main Fig. 3 (in-context evaluation) from `frontier2026/toolcall_incontext_summary.json`;
  `make_fig_main_results.py OUT col` writes the single-column Fig. 2.
  (`make_scaling_fig.py`, `make_roc_fig.py`, and `make_model_fig.py` are legacy figure scripts of the preprint.)

## Reproduce the headline results (no inference)

```bash
pip install -r requirements.txt
python reproduce.py          # curated + AgentDojo tables/figures from cache
```

Individual pieces:

```bash
python agentdojo_paired.py                 # legacy marginal paired AUC vs 9B + TOST (superseded by analyze_agentdojo_inference.py)
python agentdojo_ci.py                      # per-judge AUC + bootstrap CIs
python run_synthetic.py                     # C1 certificate + C2 frontier vs analytic truth
python analyze_loop.py loop_allsuites.json  # closed-loop CMH table (tab:loop)
python analyze_gemma.py                     # 9B good-judge sweep (App. B.5)
python frontier_analyze.py                  # closed-frontier ceiling + collapse + panel (App. B.6)
```

## Environment for optional re-scoring

Every scorer reads its credentials from environment variables; no keys are stored in the
repository. The frontier scorer uses `JUDGE_API_KEY`, `JUDGE_MODEL` (the hosted model id),
and optionally `JUDGE_REGION` / `JUDGE_ENDPOINT`; local open judges are served through an
OpenAI-compatible endpoint (`OLLAMA_URL`). Re-scoring is entirely optional -- every reported
table and figure regenerates from the cached scores with no inference. AgentDojo is pinned to
`v1.2.1`; InjecAgent and AgentHarm must be obtained from their original sources
under their own terms and are not included here.

## License

Code: MIT (see `LICENSE`). Cached scores and the curated probe are released for research
reproducibility; upstream benchmark prompts are not redistributed (see Datasets above).
