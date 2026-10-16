"""
OptiHedge-RF: Black-Scholes DW Pricer Engine (Layer 2)
File: src/products/dw/dw_pricer.py

Description:
    Implements Black-Scholes-Merton European Option valuation and analytical
    Greeks calculation for DWs, featuring numerical stability safeguards for
    near-expiry states (T -> 0).
"""

import numpy as np
from scipy.stats import norm
from typing import Dict

from src.products.base_product import BasePricer
from src.products.dw.dw_product import DWProduct
from src.products.product_factory import ProductFactory


@ProductFactory.register_pricer("black_scholes_dw")
class DWBSPricer(BasePricer):
    """Black-Scholes Pricer for Derivative Warrants with near-expiry safeguards."""

    def __init__(self, product: DWProduct):
        super().__init__(product)
        self.dw = product

    def calculate_price(
        self, spot: float, ttm: float, vol: float, r: float, q: float = 0.0
    ) -> float:
        """Calculate theoretical Black-Scholes DW price adjusted by conversion ratio."""
        if ttm <= 1e-6:  # Near-expiry condition (T -> 0)
            return self.dw.calculate_payoff(spot)

        d1 = (np.log(spot / self.dw.strike) + (r - q + 0.5 * vol**2) * ttm) / (vol * np.sqrt(ttm))
        d2 = d1 - vol * np.sqrt(ttm)

        if self.dw.option_type == "call":
            price = spot * np.exp(-q * ttm) * norm.cdf(d1) - self.dw.strike * np.exp(-r * ttm) * norm.cdf(d2)
        else:
            price = self.dw.strike * np.exp(-r * ttm) * norm.cdf(-d2) - spot * np.exp(-q * ttm) * norm.cdf(-d1)

        return price * self.dw.conversion_ratio

    def calculate_greeks(
        self, spot: float, ttm: float, vol: float, r: float, q: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate analytical Black-Scholes Greeks: Delta, Gamma, Vega, Theta.
        Returns effective Delta adjusted for the conversion ratio.
        """
        if ttm <= 1e-6:
            if self.dw.option_type == "call":
                raw_delta = 1.0 if spot > self.dw.strike else 0.0
            else:
                raw_delta = -1.0 if spot < self.dw.strike else 0.0

            return {
                "delta": raw_delta,
                "effective_delta": raw_delta * self.dw.conversion_ratio,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0
            }

        sqrt_t = np.sqrt(ttm)
        d1 = (np.log(spot / self.dw.strike) + (r - q + 0.5 * vol**2) * ttm) / (vol * sqrt_t)
        d2 = d1 - vol * sqrt_t

        pdf_d1 = norm.pdf(d1)

        # Delta Calculation
        if self.dw.option_type == "call":
            raw_delta = np.exp(-q * ttm) * norm.cdf(d1)
        else:
            raw_delta = -np.exp(-q * ttm) * norm.cdf(-d1)

        # Gamma (same for Call and Put)
        gamma = (np.exp(-q * ttm) * pdf_d1) / (spot * vol * sqrt_t)

        # Vega (converted to 1% vol change)
        vega = (spot * np.exp(-q * ttm) * pdf_d1 * sqrt_t) / 100.0

        # Theta (converted to 1-day decay)
        term1 = -(spot * vol * np.exp(-q * ttm) * pdf_d1) / (2.0 * sqrt_t)
        if self.dw.option_type == "call":
            term2 = q * spot * np.exp(-q * ttm) * norm.cdf(d1)
            term3 = -r * self.dw.strike * np.exp(-r * ttm) * norm.cdf(d2)
            theta = (term1 + term2 + term3) / 365.0
        else:
            term2 = -q * spot * np.exp(-q * ttm) * norm.cdf(-d1)
            term3 = r * self.dw.strike * np.exp(-r * ttm) * norm.cdf(-d2)
            theta = (term1 + term2 + term3) / 365.0

        return {
            "delta": raw_delta,
            "effective_delta": raw_delta * self.dw.conversion_ratio,
            "gamma": gamma * self.dw.conversion_ratio,
            "vega": vega * self.dw.conversion_ratio,
            "theta": theta * self.dw.conversion_ratio,
        }