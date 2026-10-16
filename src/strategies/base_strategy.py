"""
OptiHedge-RF: Base Strategy Interface (Layer 3)
File: src/strategies/base_strategy.py

Description:
    Defines abstract base class for hedging strategies, tracking target vs current delta
    and issuing rebalance signals.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class RebalanceSignal:
    """Dataclass holding trade instruction for rebalancing."""
    timestamp: str
    symbol: str
    target_hedge_qty: float  # Required total position size in underlying
    trade_qty: float         # Net trade required (Target - Current)
    reason: str              # e.g., 'Threshold Exceeded (0.05)', 'Periodic Rebalance'


class BaseStrategy(ABC):
    """Abstract Base Class for all delta-hedging strategies."""

    def __init__(self, target_delta: float = 0.0):
        self.target_delta = target_delta
        self.current_hedge_qty: float = 0.0  # Current shares held in underlying

    @abstractmethod
    def evaluate(
        self,
        timestamp: str,
        spot_price: float,
        effective_delta: float,
        dw_position_qty: float,
    ) -> Optional[RebalanceSignal]:
        """
        Evaluate market state and determine if rebalancing is required.

        Parameters:
            timestamp: Current bar time.
            spot_price: Current underlying spot price.
            effective_delta: Delta * Conversion Ratio for 1 DW.
            dw_position_qty: Number of DW units held (e.g., -10,000 for short position).

        Returns:
            RebalanceSignal if trade is needed, else None.
        """
        pass

    def calculate_required_hedge_qty(self, effective_delta: float, dw_position_qty: float) -> float:
        """
        Calculate total underlying shares needed for complete delta neutrality.
        Portfolio Delta = (DW_Qty * Effective_Delta) + Underlying_Shares
        For Portfolio Delta = 0: Underlying_Shares = -(DW_Qty * Effective_Delta)
        """
        return -(dw_position_qty * effective_delta)