"""Transaction cost models.

Every model has the same method:
    cost(shares, price) -> cost in currency units (always >= 0)

  shares: shares traded (sign ignored: a buy and a sell cost the same here).
          Pass the TOTAL shares actually traded (delta change x number of
          options x conversion ratio), not "per option". Minimum fees and
          market impact only make sense on real order sizes.
  price : price per share at the time of the trade.

Works with floats or numpy arrays (so a vectorised backtest can use it).
Not yet tested. All default numbers are PLACEHOLDERS: check your broker's
fee schedule and the exchange fees before using results.
"""
from __future__ import annotations

import numpy as np


def _notional(shares, price):
    return np.abs(np.asarray(shares, dtype=float)) * np.asarray(price, dtype=float)


def _traded(shares):
    return np.abs(np.asarray(shares, dtype=float)) > 0


# ------------------------------------------------------------------ models
class ZeroCost:
    """Benchmark: no costs. Use it to see the pure hedging error."""
    name = "zero"

    def cost(self, shares, price):
        return 0.0 * _notional(shares, price)


class CommissionCost:
    """Commission as a fraction of trade value, with optional minimum and fixed fee.

    commission = max(rate_bps/10000 * notional, min_fee) + fixed_fee   (only if shares != 0)
    total      = commission * (1 + vat)

    vat: tax charged on the commission (0.07 = 7%). Set 0 if your rate is already VAT-inclusive.
    """

    def __init__(self, rate_bps=10.0, vat=0.0, min_fee=0.0, fixed_fee=0.0):
        self.rate = rate_bps / 1e4
        self.vat = vat
        self.min_fee = min_fee
        self.fixed_fee = fixed_fee
        self.name = f"commission_{rate_bps}bps"

    def cost(self, shares, price):
        fee = np.maximum(self.rate * _notional(shares, price), self.min_fee) + self.fixed_fee
        return np.where(_traded(shares), fee * (1.0 + self.vat), 0.0)


class SpreadCost:
    """Half the bid-ask spread paid on every trade (you buy at the ask, sell at the bid).

    Give half_spread_bps (fraction of price), half_spread_abs (currency per share,
    e.g. half a price tick), or both (they add up).
    """

    def __init__(self, half_spread_bps=0.0, half_spread_abs=0.0):
        self.bps = half_spread_bps / 1e4
        self.abs = half_spread_abs
        self.name = "spread"

    def cost(self, shares, price):
        sh = np.abs(np.asarray(shares, dtype=float))
        return sh * (self.abs + self.bps * np.asarray(price, dtype=float))


class ImpactCost:
    """Square-root market impact: bigger orders move the price more.

    cost = coef * notional * sqrt(|shares| / adv)
    adv : average daily volume in shares for the underlying.
    coef: roughly the stock's daily volatility times a constant near 1 is a
          common rule of thumb, but this is an assumption to calibrate. It
          matters only when you trade a large part of daily volume.
    """

    def __init__(self, adv, coef=0.1):
        if adv <= 0:
            raise ValueError("adv must be > 0")
        self.adv = adv
        self.coef = coef
        self.name = "impact"

    def cost(self, shares, price):
        sh = np.abs(np.asarray(shares, dtype=float))
        return self.coef * _notional(shares, price) * np.sqrt(sh / self.adv)


class CompositeCost:
    """Sum of several cost models (commission + spread + impact ...)."""

    def __init__(self, models):
        self.models = list(models)
        self.name = "+".join(m.name for m in self.models) or "zero"

    def cost(self, shares, price):
        total = 0.0 * _notional(shares, price)
        for m in self.models:
            total = total + m.cost(shares, price)
        return total


# ------------------------------------------------------------------ factory
def make_cost_model(cfg):
    """Build a cost model from a config dict (the `costs:` block of exp01.yaml).

    Recognised keys (all optional):
        commission_bps, vat, min_fee, fixed_fee   -> CommissionCost
        half_spread_bps, half_spread_abs          -> SpreadCost
        adv, impact_coef                          -> ImpactCost
    An empty or missing config gives ZeroCost.
    """
    cfg = cfg or {}
    models = []

    if any(k in cfg for k in ("commission_bps", "min_fee", "fixed_fee")):
        models.append(CommissionCost(cfg.get("commission_bps", 0.0), cfg.get("vat", 0.0),
                                     cfg.get("min_fee", 0.0), cfg.get("fixed_fee", 0.0)))
    if any(k in cfg for k in ("half_spread_bps", "half_spread_abs")):
        models.append(SpreadCost(cfg.get("half_spread_bps", 0.0),
                                 cfg.get("half_spread_abs", 0.0)))
    if "adv" in cfg:
        models.append(ImpactCost(cfg["adv"], cfg.get("impact_coef", 0.1)))

    if not models:
        return ZeroCost()
    return models[0] if len(models) == 1 else CompositeCost(models)


# --------------------------------------------------------------------- demo
if __name__ == "__main__":
    cfg = {"commission_bps": 10, "vat": 0.07, "min_fee": 0,
           "half_spread_abs": 0.01, "adv": 5_000_000, "impact_coef": 0.1}
    model = make_cost_model(cfg)
    print("model:", model.name)
    for shares in (0, 100, 10_000, -10_000):
        print(f"trade {shares:>8} sh @ 40.0 -> cost {float(model.cost(shares, 40.0)):.2f}")