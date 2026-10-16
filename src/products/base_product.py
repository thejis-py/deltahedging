"""
OptiHedge-RF: Quant Core Engine Base Classes (Layer 2)
File: src/products/base_product.py

Description:
    Defines Abstract Base Classes (ABCs) for Financial Products and Pricers.
    Forces uniform API design for Greeks calculation and payoff evaluation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict


@dataclass
class ValuationState:
    """Dataclass holding theoretical price and computed Greeks for a product."""
    timestamp: str
    spot_price: float
    theoretical_price: float
    greeks: Dict[str, float]  # e.g., {'delta': 0.45, 'gamma': 0.02, 'vega': 0.12, 'theta': -0.01}
    effective_delta: float    # Delta * Conversion Ratio


class BaseProduct(ABC):
    """Abstract Base Class for all derivative product specifications."""

    def __init__(self, symbol: str, underlying_symbol: str, strike: float, expiry_date: str):
        self.symbol = symbol
        self.underlying_symbol = underlying_symbol
        self.strike = strike
        self.expiry_date = expiry_date

    @abstractmethod
    def calculate_payoff(self, spot_price: float) -> float:
        """Calculate product payoff at expiration given underlying spot price."""
        pass


class BasePricer(ABC):
    """Abstract Base Class for all valuation and Greeks engines."""

    def __init__(self, product: BaseProduct):
        self.product = product

    @abstractmethod
    def calculate_price(
        self, spot: float, ttm: float, vol: float, r: float, q: float = 0.0
    ) -> float:
        """Calculate theoretical product price."""
        pass

    @abstractmethod
    def calculate_greeks(
        self, spot: float, ttm: float, vol: float, r: float, q: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate analytical or numerical Greeks.
        Returns dictionary containing: {'delta', 'effective_delta', 'gamma', 'vega', 'theta'}.
        """
        pass