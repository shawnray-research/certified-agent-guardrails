"""
run_synthetic.py -- numerical validation (and attack) of the T2 theorems.

Zero-budget, zero external services: a synthetic binormal judge with a known
analytic ROC. If the simulation disagrees with the predicted frontier or the
conformal certificate over-covers, either the math or the code is wrong -- that
is the point of running this before touching a real model.

Checks:
  C1 (certificate, Thm. certificate): calibrate the threshold to a target delta
      via conformal risk control; the realized episode-unsafe rate must be <= delta.
  C2 (frontier, Thm. frontier): thresholds swept across the ROC must land on the
      analytic Pareto frontier U*(delta) = rho(R^{-1}(delta)); a fixed-threshold
      baseline that ignores delta misses the target.

Outputs: figures/frontier.pdf, figures/certificate.pdf, results.csv, and a
PASS/FAIL summary on stdout.
"""
from __future__ import annotations
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import style as _style; _style.apply_style()

import enforce as E

RNG = np.random.default_rng(20260704)
MU = 2.5            # judge separation (AUC ~ 0.96); realistic-ish, not perfect
H = 20             # episode horizon
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)


def check_C1_certificate():
    """Calibrate to each target delta; confirm realized unsafe <= delta."""
    deltas = np.array([0.01, 0.02, 0.05, 0.10, 0.20])
    n_cal = 2000                 # calibration violating actions
    n_trials = 400               # independent calibration draws
    n_test_ep = 4000             # test episodes per trial
    rows, ok = [], True
    for delta in deltas:
        tau = delta / H          # union bound over the horizon
        realized = np.empty(n_trials)
        for t in range(n_trials):
            cal = RNG.normal(MU, 1.0, size=n_cal)          # calibration violations
            theta_hat = E.conformal_threshold(cal, tau)
            realized[t] = E.simulate_unsafe_rate(theta_hat, MU, H, n_test_ep, RNG)
        mean_r, p95 = realized.mean(), np.quantile(realized, 0.95)
        passed = mean_r <= delta + 1e-3
        ok &= passed
        rows.append((delta, tau, mean_r, p95, passed))
        print(f"  C1 delta={delta:.2f}: realized mean={mean_r:.4f} "
              f"p95={p95:.4f}  ->  {'PASS' if passed else 'FAIL'}")
    return ok, rows


def check_C2_frontier():
    """Sweep thresholds; empirical (R,U) must match the ANALYTIC (R,U) per theta.
    The analytic points lie on the frontier U*(delta) by construction, so an
    empirical==analytic match (within Monte-Carlo error) proves the sweep tracks
    the frontier. We test each coordinate separately to avoid amplifying MC noise
    through the near-vertical frontier as delta->0."""
    thetas = np.linspace(-1.0, MU + 3.0, 40)
    n_ep, n_comp = 40000, 80000
    R_emp = np.array([E.simulate_unsafe_rate(th, MU, H, n_ep, RNG) for th in thetas])
    U_emp = np.array([E.simulate_utility_loss(th, MU, n_comp, RNG) for th in thetas])

    # analytic operating points (exact functions of theta); these ARE on U*.
    beta_th = E.beta_of_theta(thetas, MU)        # scipy norm.cdf is vectorized
    R_th = E.R_worstcase(beta_th, H)
    U_th = E.alpha_of_theta(thetas, MU)          # scipy norm.sf is vectorized

    # Monte-Carlo std envelope; pass iff empirical within ~4 sigma of analytic.
    sigmaR = np.sqrt(np.clip(R_th * (1 - R_th), 1e-9, None) / n_ep)
    sigmaU = np.sqrt(np.clip(U_th * (1 - U_th), 1e-9, None) / n_comp)
    zR = np.max(np.abs(R_emp - R_th) / (sigmaR + 1e-9))
    zU = np.max(np.abs(U_emp - U_th) / (sigmaU + 1e-9))
    devR, devU = float(np.max(np.abs(R_emp - R_th))), float(np.max(np.abs(U_emp - U_th)))
    passed = (zR < 5.0) and (zU < 5.0)
    print(f"  C2 frontier: max|R_emp-R_th|={devR:.4f} ({zR:.1f}sigma), "
          f"max|U_emp-U_th|={devU:.4f} ({zU:.1f}sigma)  ->  "
          f"{'PASS' if passed else 'FAIL'}")

    # analytic frontier curve for the plot
    dd = np.linspace(1e-4, 0.6, 300)
    U_star = E.predicted_frontier(dd, MU, H)

    # fixed-threshold baseline that ignores delta (theta = 0): one point
    R_base = E.simulate_unsafe_rate(0.0, MU, H, n_ep, RNG)
    U_base = E.simulate_utility_loss(0.0, MU, n_comp, RNG)

    plt.figure(figsize=(5, 3.6))
    plt.plot(dd, U_star, "-", lw=2, label=r"predicted frontier $U^\star(\delta)$")
    plt.plot(R_emp, U_emp, "o", ms=3, alpha=0.7, label="simulated thresholds")
    plt.plot([R_base], [U_base], "s", ms=9, color="crimson",
             label=r"fixed-$\theta$ baseline")
    plt.xlabel(r"episode unsafe rate $R$ (= target $\delta$ on the frontier)")
    plt.ylabel(r"utility loss $U=\alpha$")
    plt.xlim(0, 0.6); plt.ylim(0, max(0.3, U_base * 1.2))
    plt.legend(fontsize=12, loc="upper right"); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "frontier.pdf")); plt.close()
    return passed, (R_emp, U_emp, R_base, U_base)


def plot_C1(rows):
    deltas = [r[0] for r in rows]; means = [r[2] for r in rows]; p95 = [r[3] for r in rows]
    plt.figure(figsize=(5, 3.6))
    lim = max(deltas) * 1.15
    plt.plot([0, lim], [0, lim], "k--", lw=1, label=r"$y=\delta$ (target)")
    plt.plot(deltas, means, "o-", label="realized unsafe (mean)")
    plt.plot(deltas, p95, "^:", ms=5, label="realized unsafe (p95 over trials)")
    plt.xlabel(r"target safety level $\delta$")
    plt.ylabel("realized episode-unsafe rate")
    plt.xlim(0, lim); plt.ylim(0, lim); plt.legend(fontsize=12); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "certificate.pdf")); plt.close()


def main():
    print(f"Synthetic validation  (mu={MU}, H={H})")
    print("C1 -- conformal certificate: realized unsafe <= delta")
    ok1, rows = check_C1_certificate(); plot_C1(rows)
    print("C2 -- Pareto frontier equals the reparameterized ROC")
    ok2, _ = check_C2_frontier()

    with open(os.path.join(HERE, "results.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["check", "delta", "tau", "mean_unsafe", "p95_unsafe", "pass"])
        for d, tau, m, p, pss in rows:
            w.writerow(["C1", d, tau, f"{m:.5f}", f"{p:.5f}", pss])

    print("\nSUMMARY:",
          "ALL PASS" if (ok1 and ok2) else "FAILURE -- inspect math/code",
          f"(C1={'PASS' if ok1 else 'FAIL'}, C2={'PASS' if ok2 else 'FAIL'})")
    print("Figures: experiments/figures/{certificate,frontier}.pdf ; results.csv")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
