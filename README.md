# Certified Runtime Safety for Tool-Using Agents -- code & data

Reproducibility artifact for *What Can Be Enforced? A Theory of Certified Runtime Safety
for Tool-Using Agents*.

Everything here is **inference-only** (no training). Judge scores for our own curated
probe and for the AgentDojo tasks are cached as JSON, so those tables and figures
regenerate from the cache without any model inference or API access. Re-scoring (optional)
requires the relevant provider API keys, supplied through environment variables -- no keys
are stored in this repository.

## Datasets and redistribution

This artifact does **not** redistribute upstream benchmark prompts. AgentDojo (pinned to
`v1.2.1`) is loaded from its own package. InjecAgent and AgentHarm are access-gated and
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
- `*_score.py`, `judge.py`, `cerebras_judge.py`, `samba70b_attack.py`, `run_loop.py` -- the
  scorers / attack / closed-loop harness that *produced* the caches (read API keys from env).

## Reproduce the headline results (no inference)

```bash
pip install -r requirements.txt
python reproduce.py          # curated + AgentDojo tables/figures from cache
```

Individual pieces:

```bash
python agentdojo_paired.py                 # paired AUC vs 9B + TOST (no judge beats the 9B)
python agentdojo_ci.py                      # per-judge AUC + bootstrap CIs
python run_synthetic.py                     # C1 certificate + C2 frontier vs analytic truth
python analyze_loop.py loop_allsuites.json  # closed-loop CMH table (tab:loop)
python analyze_gemma.py                     # 9B good-judge sweep (App. B.5)
```

## Environment for optional re-scoring

Scorers read keys from environment variables (e.g. `JUDGE_API_KEY`, `AZURE_API_KEY`,
`AZURE_URL`, `AZURE_DEPLOYMENT`); local open judges are served via an OpenAI-compatible
endpoint (`OLLAMA_URL`). AgentDojo is pinned to `v1.2.1`; InjecAgent and AgentHarm must be
obtained from their original (access-gated) sources under their own terms and are not
included here.

## License

Code: MIT (see `LICENSE`). Cached scores and the curated probe are released for research
reproducibility; upstream benchmark prompts are not redistributed (see Datasets above).
