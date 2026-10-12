"""Simulation tools: GBM paths, bootstrapped paths, and a generic Monte Carlo pricer.

Path arrays always have shape (n_paths, n_steps + 1); column 0 is the start price.
Not yet tested.
"""
from __future__ import annotations

import numpy as np

TRADING_DAYS = 252


# ------------------------------------------------------------ estimate params
def estimate_gbm_params(close, periods_per_year=TRADING_DAYS):
    """Estimate annualised (mu, sigma) from closing prices.

    sigma = std of daily log returns * sqrt(periods_per_year)
    mu    = mean log return * periods_per_year + 0.5 * sigma^2
    Historical mu is very noisy. For pricing, use mu = r - q instead.
    """
    close = np.asarray(close, dtype=float)
    log_ret = np.diff(np.log(close))
    sigma = log_ret.std(ddof=1) * np.sqrt(periods_per_year)
    mu = log_ret.mean() * periods_per_year + 0.5 * sigma**2
    return mu, sigma


# ------------------------------------------------------------------ GBM paths
def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, seed=None):
    """Simulate GBM paths: S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z).

    mu: drift of the price.
        - risk-neutral pricing: mu = r - q
        - hedging backtest on "realistic" paths: assumed or historical mu
    T : horizon in years.
    """
    if S0 <= 0 or sigma < 0 or T <= 0 or n_steps < 1 or n_paths < 1:
        raise ValueError("need S0 > 0, sigma >= 0, T > 0, n_steps >= 1, n_paths >= 1")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_step = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z

    log_paths = np.zeros((n_paths, n_steps + 1))
    log_paths[:, 1:] = np.cumsum(log_step, axis=1)
    return S0 * np.exp(log_paths)


# ------------------------------------------------------------ bootstrap paths
def bootstrap_paths(S0, log_returns, n_steps, n_paths, seed=None, demean=False):
    """Build paths by resampling historical daily log returns (with replacement).

    Keeps fat tails and gaps that GBM lacks, so it is a tougher test for a hedge.
    It assumes returns are independent, so volatility clustering is lost.
    demean=True removes the historical drift (average return = 0).
    """
    r = np.asarray(log_returns, dtype=float)
    r = r[~np.isnan(r)]
    if demean:
        r = r - r.mean()

    rng = np.random.default_rng(seed)
    sampled = r[rng.integers(0, len(r), size=(n_paths, n_steps))]

    log_paths = np.zeros((n_paths, n_steps + 1))
    log_paths[:, 1:] = np.cumsum(sampled, axis=1)
    return S0 * np.exp(log_paths)


# ------------------------------------------------------------- Monte Carlo
def mc_price(payoff, paths, r, T):
    """Discounted average payoff. Returns (price, standard_error).

    payoff: function that takes paths (n_paths, n_steps+1) and returns
            a payoff vector of length n_paths.
    """
    pv = np.exp(-r * T) * np.asarray(payoff(paths), dtype=float)
    return pv.mean(), pv.std(ddof=1) / np.sqrt(len(pv))


# ----------------------------------------------------------- example payoffs
def european_call(K):
    return lambda paths: np.maximum(paths[:, -1] - K, 0.0)


def european_put(K):
    return lambda paths: np.maximum(K - paths[:, -1], 0.0)


def kiko_call(K, ki_barrier, ko_barrier, ki_dir="down", ko_dir="up"):
    """Call that pays only if KI was touched and KO was never touched.

    Assumed payoff (check your term sheet). Barriers are checked only on the
    simulated dates (discrete monitoring), so with few steps the price will be
    higher than for a continuously monitored barrier.
    """
    def payoff(paths):
        if ki_dir == "down":
            ki = (paths <= ki_barrier).any(axis=1)
        else:
            ki = (paths >= ki_barrier).any(axis=1)
        if ko_dir == "up":
            ko = (paths >= ko_barrier).any(axis=1)
        else:
            ko = (paths <= ko_barrier).any(axis=1)
        return np.maximum(paths[:, -1] - K, 0.0) * (ki & ~ko)
    return payoff


# --------------------------------------------------------------------- demo
if __name__ == "__main__":
    from src.pricing import bs_price   # run with: python -m src.simulation

    S0, K, T, r, q, sigma = 42.0, 40.0, 0.5, 0.10, 0.0, 0.20
    paths = simulate_gbm(S0, mu=r - q, sigma=sigma, T=T,
                         n_steps=126, n_paths=100_000, seed=1)
    price, se = mc_price(european_call(K), paths, r, T)
    print(f"MC : {price:.4f} +/- {se:.4f}")
    print(f"BS : {bs_price(S0, K, T, r, sigma, q, 'call'):.4f}")