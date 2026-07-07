"""
enforce.py -- core library for the pre-execution gate and its guarantees.

This module implements the objects from the paper so the theorems can be checked
numerically and, later, driven by a real LLM judge (drop in real scores in place
of the synthetic binormal judge; every function below takes scores, not a model).

Mapping to the paper (paper/proofs.tex):
  - Judge error rates alpha(theta), beta(theta) and ROC rho .... Def. (judge)
  - Block-only gate, worst-case episode risk R = 1-(1-beta)^H ... Thm. (frontier)
  - Pareto frontier U*(delta) = rho(R^{-1}(delta)) ............... Thm. (frontier)
  - Conformal threshold controlling per-step miss at delta/H .... Thm. (certificate)

No paid inference, no training: a synthetic binormal judge gives an analytic ROC
so predictions are exact and the Monte-Carlo simulation can be checked against it.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm

# --------------------------------------------------------------------------
# Synthetic binormal judge: compliant score ~ N(0,1), violating score ~ N(mu,1).
# The gate BLOCKS iff score >= theta. Larger separation mu => better judge.
# --------------------------------------------------------------------------

def alpha_of_theta(theta, mu: float):
    """False-alarm rate: P(block | compliant) = P(N(0,1) >= theta). Vectorized."""
    return norm.sf(theta)                   # sf(x) = 1 - Phi(x)

def beta_of_theta(theta, mu: float):
    """Miss rate: P(not block | violating) = P(N(mu,1) < theta). Vectorized."""
    return norm.cdf(np.asarray(theta) - mu)

def roc(beta: np.ndarray | float, mu: float) -> np.ndarray | float:
    """ROC frontier rho(beta): min false-alarm achievable at miss rate beta.
    For the binormal judge, theta = mu + Phi^{-1}(beta) and rho = Phi(-theta)."""
    theta = mu + norm.ppf(beta)
    return norm.sf(theta)

# --------------------------------------------------------------------------
# Episode risk and the analytic Pareto frontier.
# --------------------------------------------------------------------------

def R_worstcase(beta: np.ndarray | float, H: int) -> np.ndarray | float:
    """Worst-case episode unsafe probability: a violation every step,
    unsafe iff any is missed. R = 1 - (1-beta)^H  (Thm. frontier)."""
    return 1.0 - (1.0 - np.asarray(beta)) ** H

def beta_of_delta(delta: np.ndarray | float, H: int) -> np.ndarray | float:
    """Inverse of R_worstcase: the miss rate whose episode risk is exactly delta."""
    return 1.0 - (1.0 - np.asarray(delta)) ** (1.0 / H)

def predicted_frontier(delta: np.ndarray | float, mu: float, H: int):
    """U*(delta) = rho(R^{-1}(delta))  -- the tight Pareto frontier."""
    return roc(beta_of_delta(delta, H), mu)

# --------------------------------------------------------------------------
# Monte-Carlo simulation (the numerical attack on the theorems).
# --------------------------------------------------------------------------

def simulate_unsafe_rate(theta: float, mu: float, H: int,
                         n_episodes: int, rng: np.random.Generator) -> float:
    """Realized worst-case episode unsafe rate at threshold theta.
    Each episode: H violating actions, each missed iff its score < theta."""
    scores = rng.normal(mu, 1.0, size=(n_episodes, H))   # violating scores
    missed = scores < theta                              # not blocked
    unsafe = missed.any(axis=1)
    return float(unsafe.mean())

def simulate_utility_loss(theta: float, mu: float,
                          n: int, rng: np.random.Generator) -> float:
    """Realized utility loss = fraction of compliant actions blocked (score>=theta)."""
    scores = rng.normal(0.0, 1.0, size=n)                # compliant scores
    return float((scores >= theta).mean())

# --------------------------------------------------------------------------
# Conformal calibration (Thm. certificate): pick theta so the per-step
# violation-miss risk is <= tau = delta/H, distribution-free & finite-sample.
# Miss = (violating score < theta), so we want theta at the lower tau-quantile
# of violating scores. Split-conformal lower bound: theta_hat = k-th smallest
# calibration violating score with k = floor((n+1)*tau); k=0 => block everything.
# --------------------------------------------------------------------------

def conformal_threshold(cal_violation_scores: np.ndarray, tau: float) -> float:
    s = np.sort(np.asarray(cal_violation_scores))
    n = s.size
    k = int(np.floor((n + 1) * tau))
    if k <= 0:
        return -np.inf                    # block everything -> miss rate 0
    if k > n:
        return float(s[-1])
    return float(s[k - 1])                # k-th smallest (1-indexed)
