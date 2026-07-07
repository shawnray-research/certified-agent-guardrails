"""One-command reproduction of the paper's headline results from cached scores only
(no model inference, no API keys). Runs the paired-significance test, the AUC/CI table,
and the synthetic certificate/frontier check."""
import subprocess, sys
STEPS = [
    ("Paired-bootstrap dAUC vs 9B + TOST (Table tab:ci)", "agentdojo_paired.py"),
    ("Per-judge AgentDojo AUC + bootstrap CIs",            "agentdojo_ci.py"),
    ("Synthetic C1 certificate + C2 frontier",             "run_synthetic.py"),
]
for title, script in STEPS:
    print("\n" + "=" * 78 + f"\n## {title}\n" + "=" * 78)
    r = subprocess.run([sys.executable, script])
    if r.returncode != 0:
        print(f"[warn] {script} exited {r.returncode}", file=sys.stderr)
print("\nDone. Figures: run make_scaling_fig.py / make_roc_fig.py (writes figures/).")
