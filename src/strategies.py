"""Hedging strategies.

Every strategy answers one question:
    "At this time step, how many shares of the underlying should I hold?"

Convention (so backtest.py can use it):
  - The number returned is per ONE option on ONE share of the underlying
    (so a call delta of 0.6 returns 0.6). backtest.py multiplies by
    number of options and conversion ratio.
  - Positive = long the underlying. Hedging a SHORT call means holding +delta.
  - tau = time to maturity in years (same unit as pricing.py).

Not yet tested.
"""
from __future__ import annotations

from math import erf, exp, log, sqrt


# ------------------------------------------------------------ delta function
def _norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_delta(S, K, tau, r, sigma, q=0.0, kind="call"):
    """Black-Scholes delta (kept here so this file stands alone)."""
    if tau <= 0:  # at maturity the delta is 0 or 1 (step function)
        itm = S > K if kind == "call" else S < K
        if kind == "call":
            return 1.0 if itm else 0.0
        return -1.0 if itm else 0.0
    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt(tau))
    call_delta = exp(-q * tau) * _norm_cdf(d1)
    return call_delta if kind == "call" else call_delta - exp(-q * tau)


def make_delta_fn(K, r, sigma, q=0.0, kind="call"):
    """Return delta_fn(S, tau) -> delta, with the contract terms fixed.

    The sigma here is the vol your HEDGER believes. It does not have to equal
    the vol of the simulated paths. Using a different one is a useful experiment.
    You can replace this with any function of (S, tau), for example a bumped
    price from pricing.py for a KIKO.
    """
    def delta_fn(S, tau):
        return bs_delta(S, K, tau, r, sigma, q, kind)
    return delta_fn


# --------------------------------------------------------------- strategies
class NoHedge:
    """Benchmark: never hold any underlying (a 'naked' short option)."""
    name = "no_hedge"

    def target_position(self, step, S, tau, current):
        return 0.0


class StaticHedge:
    """Buy the initial delta at step 0 and never change it."""
    name = "static"

    def __init__(self, delta_fn):
        self.delta_fn = delta_fn

    def target_position(self, step, S, tau, current):
        if step == 0:
            return self.delta_fn(S, tau)
        return current


class DeltaHedge:
    """Rebalance to the model delta every `every` steps (1 = every step)."""

    def __init__(self, delta_fn, every=1):
        self.delta_fn = delta_fn
        self.every = every
        self.name = f"delta_every_{every}"

    def target_position(self, step, S, tau, current):
        if step % self.every == 0:  # step 0 always qualifies
            return self.delta_fn(S, tau)
        return current


class BandHedge:
    """Rebalance only when the hedge is off by more than `band` shares.

    Trades less often than DeltaHedge, so it pays lower costs but tracks the
    option less tightly.
    """

    def __init__(self, delta_fn, band=0.05):
        self.delta_fn = delta_fn
        self.band = band
        self.name = f"band_{band}"

    def target_position(self, step, S, tau, current):
        target = self.delta_fn(S, tau)
        if abs(target - current) > self.band:
            return target
        return current


class StopLoss:
    """Classic stop-loss hedge for a short CALL: hold 1 share when S >= K,
    hold 0 when S < K. Needs no model, but trades a lot when S hovers near K
    and loses at every crossing because of costs and gaps."""

    def __init__(self, K):
        self.K = K
        self.name = "stop_loss"

    def target_position(self, step, S, tau, current):
        return 1.0 if S >= self.K else 0.0


# ------------------------------------------------------------------ factory
def make_strategy(cfg, K, r, sigma, q=0.0, kind="call"):
    """Build a strategy from a config dict, e.g. from configs/exp01.yaml:

        {"type": "delta", "every": 1}
        {"type": "band", "band": 0.05}
        {"type": "static"}
        {"type": "stop_loss"}
        {"type": "none"}
    """
    t = cfg["type"]
    delta_fn = make_delta_fn(K, r, sigma, q, kind)

    if t == "none":
        return NoHedge()
    if t == "static":
        return StaticHedge(delta_fn)
    if t == "delta":
        return DeltaHedge(delta_fn, every=cfg.get("every", 1))
    if t == "band":
        return BandHedge(delta_fn, band=cfg.get("band", 0.05))
    if t == "stop_loss":
        return StopLoss(K)
    raise ValueError(f"Unknown strategy type: {t}")


# --------------------------------------------------------------------- demo
if __name__ == "__main__":
    K, r, sigma = 40.0, 0.10, 0.20
    strat = make_strategy({"type": "band", "band": 0.05}, K, r, sigma)
    pos = 0.0
    for step, S in enumerate([42, 42.5, 41, 39, 38, 40.5]):
        tau = 0.5 - step * 0.01
        new = strat.target_position(step, S, tau, pos)
        print(f"step {step}  S={S:5.1f}  hold {pos:.3f} -> {new:.3f}")
        pos = new