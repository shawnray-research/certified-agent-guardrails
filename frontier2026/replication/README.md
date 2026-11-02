# Independent replication of the late-2026 refresh (cross-check only)

A second, independently written harness re-scored the 132 AgentDojo instruction strings and the
35 suffix-rewritten violations once with Claude Opus 5.5, Claude Sonnet 5.5, Claude Haiku 5.5,
GPT-5.5, and GPT-6.1 Sol. It used the same fixed 0-100 prompt and scored provider refusals at
both 0.5 and 1.0. The canonical numbers in the paper (Table `tab:frontier2026`) come from
`../frontier2026_*.py`; this folder only checks that they are not an artifact of one harness
(Supplement B.19).

- `score_proxy.py` -- scorer (Anthropic Messages / OpenAI Responses APIs; credentials and base
  URLs from environment variables only).
- `analyze.py` -- AUC with bootstrap CIs, paired difference from Gemma-2-9B, and the fixed-suffix
  miss change at theta = .5; writes `results.json`.
- `make_table.py` -- renders `table.tex` from `results.json`.
- `caches/` -- cached scores (`*_bench.json`) and raw replies with stop reasons (`*.raw.json`).

Re-run the analysis with no API access: `python analyze.py && python make_table.py`.
