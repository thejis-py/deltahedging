"""
OptiHedge-RF: Threshold-Based Delta Hedging (Layer 3)
File: src/strategies/threshold_based.py

Description:
    Rebalances portfolio only when unhedged portfolio delta exceeds a set band threshold,
    reducing transaction costs and market impact.
"""

from typing import Optional
from src.strategies.base_strategy import BaseStrategy, RebalanceSignal


class ThresholdDeltaStrategy(BaseStrategy):
    """Rebalances when delta imbalance exceeds a defined threshold."""

    def __init__(self, threshold: float = 0.05, target_delta: float = 0.0):
        super().__init__(target_delta=target_delta)
        self.threshold = threshold  # Delta tolerance band (e.g., 0.05)

    def evaluate(
        self,
        timestamp: str,
        spot_price: float,
        effective_delta: float,
        dw_position_qty: float,
    ) -> Optional[RebalanceSignal]:
        # Required underlying position for complete neutrality
        target_hedge_qty = self.calculate_required_hedge_qty(effective_delta, dw_position_qty)

        # Calculate current unhedged delta drift
        trade_qty = target_hedge_qty - self.current_hedge_qty

        # Normalize drift per unit of DW position to check threshold
        delta_drift = abs(trade_qty) / abs(dw_position_qty) if dw_position_qty != 0 else 0.0

        if delta_drift >= self.threshold:
            signal = RebalanceSignal(
                timestamp=timestamp,
                symbol="UNDERLYING",
                target_hedge_qty=target_hedge_qty,
                trade_qty=trade_qty,
                reason=f"Delta Band Exceeded ({delta_drift:.4f} >= {self.threshold})",
            )
            self.current_hedge_qty = target_hedge_qty  # Update internal state
            return signal

        return None