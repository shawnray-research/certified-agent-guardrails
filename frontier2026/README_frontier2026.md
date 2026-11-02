# Late-2026 frontier refresh (Table `tab:frontier2026`)

Seven current frontier judges (`claude-opus-5-5`, `claude-fable-5-1`, `claude-sonnet-5-5`,
`claude-haiku-5-5`, `gpt-6.1-sol`, `gpt-6.1-terra`, `gpt-5.5`) scored on 1 Nov 2026 with the
same 0-100 prompt as every other judge, on the 132 AgentDojo instructions plus the 35
suffix-rewritten violations (`_frontier_input.json`). The APIs accept no temperature, so
decoding uses the provider default at the lowest reasoning setting (Claude Opus 5.5 / Fable 5.1:
effort `low`, thinking cannot be disabled; Sonnet 5.5: thinking off; Haiku 5.5: thinking
disabled; GPT: `reasoning_effort=low`). Each item was scored in three independent runs
(`frontier2026_<model>_bench.json`, `rep_r2_...`, `rep_r3_...`); `*_raw.json` keeps the raw
reply and stop reason, so provider refusals are auditable.

Everything regenerates from these caches with no API access:

```bash
python analyze_agentdojo_inference.py                       # paper's 23-judge family (Fig. agentdojo_forest)
python analyze_agentdojo_inference.py --family all2026 --json frontier2026/fam_all2026_ref1.0.json
python analyze_agentdojo_inference.py --family all2026 --refusal 0.5 --json frontier2026/fam_all2026_ref0.5.json
python analyze_agentdojo_inference.py --family all2026 --run 2   # (or --run 3, --avg) run-to-run check
python analyze_agentdojo_inference.py --family complete     # + DeepSeek-V3.2, GLM-5, Kimi-K2.5
python frontier2026/frontier2026_table.py                   # every number in tab:frontier2026
python frontier2026/frontier2026_adaptive.py claude-opus-5-5   # best-of-6 attack result (cached)
```

Refusals are scored fail-closed (1.0) by default; `--refusal 0.5` uses the protocol's
parse-failure default. Re-scoring (optional) needs `pip install anthropic openai` and
`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` in the environment:
`python frontier2026/frontier2026_score.py MODEL _frontier_input.json OUT.json`.
