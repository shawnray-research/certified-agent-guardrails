"""Remote scoring driver: score a JSON list of action texts with a local Ollama
model, write {text: risk_score}. Run ON the GPU box (stdlib + judge.py only)."""
import sys, json
from judge import score_action

model, in_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
texts = json.load(open(in_path))
out = json.load(open(out_path)) if __import__("os").path.exists(out_path) else {}
for i, t in enumerate(texts):
    if t not in out:
        out[t] = score_action(t, model=model, timeout=180)
        if (i + 1) % 5 == 0:
            json.dump(out, open(out_path, "w"))
    print(f"{i+1}/{len(texts)} risk={out[t]:.2f}", flush=True)
json.dump(out, open(out_path, "w"))
print("DONE", len(out))
