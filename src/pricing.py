"""Pricing models.

- DW  : European call/put on a stock/index, Black-Scholes closed form,
        divided by the conversion ratio.
- KIKO: option that is alive only if the knock-in barrier has been touched AND
        the knock-out barrier has never been touched (see kiko_price docstring).
        Priced by Monte Carlo with a Brownian-bridge correction for barriers.

NOTE: not yet tested. Verify against the product term sheet / prospectus.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

TRADING_DAYS = 252


# ---------------------------------------------------------------- helpers
def year_fraction(start, end, basis=365.0):
    """Year fraction between two date/datetime objects (calendar days / basis)."""
    return (end - start).days / basis


def _check_option(option):
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")


def _d1_d2(S, K, T, r, sigma, q):
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def _vanilla_payoff(ST, K, option):
    return np.maximum(ST - K, 0.0) if option == "call" else np.maximum(K - ST, 0.0)


# ---------------------------------------------------------- Black-Scholes
def bs_price(S, K, T, r, sigma, q=0.0, option="call"):
    """European option price. q = continuous dividend yield. T in years."""
    _check_option(option)
    if T <= 0:
        return float(_vanilla_payoff(S, K, option))
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, q=0.0, option="call"):
    """Greeks per 1 option on 1 share.

    vega is per 1.00 change in vol (divide by 100 for per 1 vol point);
    theta is per year (divide by 365 for per day).
    """
    _check_option(option)
    if T <= 0:
        itm = (S > K) if option == "call" else (S < K)
        delta = (1.0 if option == "call" else -1.0) if itm else 0.0
        return {"delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    sqrt_t = np.sqrt(T)
    disc_q, disc_r = np.exp(-q * T), np.exp(-r * T)
    pdf_d1 = norm.pdf(d1)

    gamma = disc_q * pdf_d1 / (S * sigma * sqrt_t)
    vega = S * disc_q * pdf_d1 * sqrt_t
    if option == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (-S * disc_q * pdf_d1 * sigma / (2 * sqrt_t)
                 - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1))
        rho = K * T * disc_r * norm.cdf(d2)
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta = (-S * disc_q * pdf_d1 * sigma / (2 * sqrt_t)
                 + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1))
        rho = -K * T * disc_r * norm.cdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


# --------------------------------------------------------------------- DW
def dw_price(S, K, T, r, sigma, q=0.0, option="call", conv_ratio=1.0):
    """Price of ONE DW = BS price / conversion ratio
    (conv_ratio = number of DW needed to cover 1 underlying share)."""
    return bs_price(S, K, T, r, sigma, q, option) / conv_ratio


def dw_greeks(S, K, T, r, sigma, q=0.0, option="call", conv_ratio=1.0):
    """Greeks per ONE DW. delta is in shares per DW.

    Hedge for a short position of N DW: hold N * delta shares (delta already / ratio).
    """
    g = bs_greeks(S, K, T, r, sigma, q, option)
    return {k: v / conv_ratio for k, v in g.items()}


def implied_vol(price, S, K, T, r, q=0.0, option="call", conv_ratio=1.0,
                lo=1e-4, hi=5.0):
    """Implied vol from a DW market price. Returns nan if no solution
    (price outside no-arbitrage bounds, T <= 0, ...)."""
    if T <= 0 or price <= 0:
        return float("nan")
    target = price * conv_ratio

    def f(sig):
        return bs_price(S, K, T, r, sig, q, option) - target

    try:
        if f(lo) * f(hi) > 0:
            return float("nan")
        return brentq(f, lo, hi, xtol=1e-8)
    except ValueError:
        return float("nan")


# ------------------------------------------------------------------- KIKO
def _touched(logS, barrier, up, sigma, dt, u):
    """True for each path that touched `barrier` at any time.

    logS: (n_paths, n_steps+1) log prices at monitoring dates.
    u   : uniform randoms (n_paths, n_steps) for the Brownian-bridge correction,
          or None for discrete monitoring only (daily-close style).
    The bridge term estimates crossings *between* grid points (continuous monitoring).
    """
    lb = np.log(barrier)
    if up:
        grid_hit = (logS >= lb).any(axis=1)
        a, b = lb - logS[:, :-1], lb - logS[:, 1:]
    else:
        grid_hit = (logS <= lb).any(axis=1)
        a, b = logS[:, :-1] - lb, logS[:, 1:] - lb
    if u is None:
        return grid_hit
    p_cross = np.exp(-2.0 * a * b / (sigma**2 * dt))
    p_cross = np.where((a > 0) & (b > 0), p_cross, 1.0)
    return grid_hit | (u < p_cross).any(axis=1)


def kiko_price(S0, K, ki_barrier, ko_barrier, T, r, sigma, q=0.0, option="call",
               *, ki_hit=False, ko_hit=False, n_paths=100_000,
               steps_per_year=TRADING_DAYS, continuous=True, seed=0, batch=20_000):
    """Monte Carlo price of a KIKO option. Returns (price, standard_error).

    ASSUMED payoff (check against your term sheet):
        payoff = vanilla(S_T, K)  if  KI barrier was touched
                                  and KO barrier was NEVER touched
               = 0                otherwise            (no rebate)
    Barrier direction (up/down) is inferred from the barrier vs the spot.

    ki_hit / ko_hit: barrier history so far. KIKO is path-dependent, so when
    re-pricing mid-life in a backtest you MUST pass what has already happened.
      ko_hit=True -> product is dead, price 0.
      ki_hit=True -> only the KO condition remains.
    continuous=True : barrier monitored continuously (Brownian-bridge correction).
    continuous=False: barrier checked only on the simulated dates (e.g. daily close).
    Same `seed` => same random numbers, so bump-and-revalue Greeks are less noisy.
    """
    _check_option(option)
    if ko_hit:
        return 0.0, 0.0
    if T <= 0:
        pay = float(_vanilla_payoff(S0, K, option)) if ki_hit else 0.0
        return pay, 0.0
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    n_steps = max(1, int(round(T * steps_per_year)))
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    ki_up, ko_up = ki_barrier > S0, ko_barrier > S0
    disc = np.exp(-r * T)

    rng = np.random.default_rng(seed)
    total = total_sq = 0.0
    done = 0
    while done < n_paths:
        m = min(batch, n_paths - done)
        z = rng.standard_normal((m, n_steps))
        incr = np.cumsum(drift + vol * z, axis=1)
        logS = np.log(S0) + np.concatenate([np.zeros((m, 1)), incr], axis=1)
        ST = np.exp(logS[:, -1])

        u_ko = rng.random((m, n_steps)) if continuous else None
        u_ki = rng.random((m, n_steps)) if continuous else None

        ko_touch = _touched(logS, ko_barrier, ko_up, sigma, dt, u_ko)
        if ki_hit:
            alive = ~ko_touch
        else:
            alive = _touched(logS, ki_barrier, ki_up, sigma, dt, u_ki) & ~ko_touch

        pv = disc * _vanilla_payoff(ST, K, option) * alive
        total += pv.sum()
        total_sq += (pv**2).sum()
        done += m

    mean = total / n_paths
    var = max(total_sq / n_paths - mean**2, 0.0)
    return mean, float(np.sqrt(var / n_paths))


def kiko_greeks(S0, K, ki_barrier, ko_barrier, T, r, sigma, q=0.0, option="call",
                *, bump=0.01, vol_bump=0.01, **mc_kwargs):
    """Delta, gamma, vega by bump-and-revalue (central differences, common random numbers).

    bump     : relative spot bump (0.01 = 1%).
    vol_bump : absolute vol bump (0.01 = 1 vol point); vega is per 1.00 vol.
    Expect noisy gamma, and large/unstable delta near the barriers - that is a
    property of the product, and is what makes hedging it hard.
    """
    def p(S, sig):
        return kiko_price(S, K, ki_barrier, ko_barrier, T, r, sig, q, option,
                          **mc_kwargs)[0]

    h = S0 * bump
    base, up, dn = p(S0, sigma), p(S0 + h, sigma), p(S0 - h, sigma)
    return {
        "price": base,
        "delta": (up - dn) / (2 * h),
        "gamma": (up - 2 * base + dn) / h**2,
        "vega": (p(S0, sigma + vol_bump) - p(S0, sigma - vol_bump)) / (2 * vol_bump),
    }