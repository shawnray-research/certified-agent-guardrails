"""Curated tool-call probe (Supplement B.3) scored by current frontier judges.

Scores the 60 serialized tool calls of actions.json with the SAME judge prompt, decoding
settings and scoring rule as frontier2026/frontier2026_score.py (refusal = 1.0, no integer = .5;
helpers imported from b4_frontier_pairs.py) and reports AUC with a 20,000-draw label-stratified
bootstrap 95% interval. Usage: python3 curated_frontier2026.py OUT.json MODEL [MODEL ...]
"""
import sys, json, random
from concurrent.futures import ThreadPoolExecutor
from b4_frontier_pairs import score_all, value

def auc(pos, neg):
    return sum((p > n) + .5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))

if __name__ == "__main__":
    out_path, models = sys.argv[1], sys.argv[2:]
    acts = json.load(open("actions.json"))["actions"]
    texts = [a["text"] for a in acts]
    with ThreadPoolExecutor(len(models)) as ex:
        raws = dict(zip(models, ex.map(lambda m: score_all(m, texts, out_path.replace(".json", f"_{m}_raw.json"), 3), models)))
    res = {}
    for m in models:
        pos = [value(raws[m][a["text"]]) for a in acts if a["label"] == 1]
        neg = [value(raws[m][a["text"]]) for a in acts if a["label"] == 0]
        rng = random.Random(20000)
        bs = sorted(auc([rng.choice(pos) for _ in pos], [rng.choice(neg) for _ in neg]) for _ in range(20000))
        res[m] = {"auc": auc(pos, neg), "ci95": [bs[500], bs[19499]], "n_viol": len(pos), "n_comp": len(neg),
                  "refusals": sum(raws[m][t]["stop"] == "refusal" for t in texts),
                  "errors": sum(raws[m][t]["stop"] == "error" for t in texts)}
        print(f"{m}: AUC {res[m]['auc']:.3f} [{res[m]['ci95'][0]:.3f},{res[m]['ci95'][1]:.3f}] "
              f"(viol {len(pos)}, comp {len(neg)}) refusals={res[m]['refusals']} errors={res[m]['errors']}")
    json.dump({"set": "actions.json (60 curated tool calls)", "results": res}, open(out_path, "w"), indent=1)
