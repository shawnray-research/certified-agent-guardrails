# Certified Runtime Safety for Tool-Using Agents â code & data

Reproducibility artifact for *What Can Be Enforced? A Theory of Certified Runtime Safety
for Tool-Using Agents*.

Everything here is **inference-only** (no training). All judge scores used in the paper
are cached as JSON in this repository, so **every table and figure regenerates from the
cache without any model inference or API access**. Re-scoring (optional) requires the
relevant provider API keys, supplied through environment variables â no keys are stored
in this repository.

## Layout

- `enforce.py` â the gate: binormal/real judge scoring, split-conformal calibration
  (`conformal_threshold`), and the safety/utility frontier utilities.
- `run_synthetic.py` â synthetic judge with a known analytic ROC (validates C1/C2 against
  ground truth).
- `agentdojo_ci.py` â per-judge AgentDojo AUC with bootstrap 95% CIs.
- `agentdojo_paired.py` â **paired-bootstrap ÎAUC vs. the 9B reference + TOST equivalence**
- `ensemble_robustness.py` — shows ensembling does not fix safe-washing (max-ensemble margin eta stays high because the vulnerability is correlated across judges).
  over all 19 AgentDojo judges (regenerates Table `tab:ci`).
- `make_scaling_fig.py`, `make_roc_fig.py`, `make_model_fig.py` â figures.
- `*_bench.json`, `*_curated.json`, `*_ia.json`, `*_ah.json`, `*_safewash.json`,
  `bench_scores_cache.json`, `scores_cache.json` â cached judge scores.
- `actions.json`, `agentharm.json`, `injecagent.json`, `_bench_texts.json`,
  `_inj_goals.json`, `_bench_safewash_texts.json` â labeled action sets / benchmark texts.
- `*_score.py`, `judge.py`, `cerebras_judge.py`, `samba70b_attack.py`, `run_loop.py` â the
  scorers/attack/closed-loop harness used to *produce* the caches (read API keys from env).

## Reproduce the headline results (no inference)

```bash
pip install -r requirements.txt
python reproduce.py          # regenerates the AUC / paired-significance / synthetic tables
```

Individual pieces:

```bash
python agentdojo_paired.py   # Table: paired ÎAUC vs 9B + TOST (no judge beats the 9B)
python agentdojo_ci.py       # per-judge AUC + bootstrap CIs
python run_synthetic.py      # C1 certificate + C2 frontier vs analytic ground truth
```

## Environment for optional re-scoring

Scorers read keys from environment variables (e.g. `JUDGE_API_KEY`, `AZURE_API_KEY`,
`AZURE_URL`, `AZURE_DEPLOYMENT`); local open judges are served via an OpenAI-compatible
endpoint (`OLLAMA_URL`). Datasets: AgentDojo is pinned to `v1.2.1`; InjecAgent and
AgentHarm use their public test splits.

## License

MIT (see `LICENSE`).
