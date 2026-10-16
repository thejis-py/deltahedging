"""
OptiHedge-RF: Time-Based Periodic Delta Hedging (Layer 3)
File: src/strategies/time_based.py

Description:
    Rebalances portfolio at fixed time intervals (e.g., every N bars/minutes).
"""

from typing import Optional
from src.strategies.base_strategy import BaseStrategy, RebalanceSignal


class TimeBasedDeltaStrategy(BaseStrategy):
    """Rebalances periodically after a fixed number of timesteps."""

    def __init__(self, rebalance_interval_bars: int = 15, target_delta: float = 0.0):
        super().__init__(target_delta=target_delta)
        self.rebalance_interval = rebalance_interval_bars
        self.bar_counter = 0

    def evaluate(
        self,
        timestamp: str,
        spot_price: float,
        effective_delta: float,
        dw_position_qty: float,
    ) -> Optional[RebalanceSignal]:
        self.bar_counter += 1

        if self.bar_counter % self.rebalance_interval == 0:
            target_hedge_qty = self.calculate_required_hedge_qty(effective_delta, dw_position_qty)
            trade_qty = target_hedge_qty - self.current_hedge_qty

            if abs(trade_qty) > 1e-5:  # Avoid micro-orders
                signal = RebalanceSignal(
                    timestamp=timestamp,
                    symbol="UNDERLYING",
                    target_hedge_qty=target_hedge_qty,
                    trade_qty=trade_qty,
                    reason=f"Periodic Rebalance (Every {self.rebalance_interval} bars)",
                )
                self.current_hedge_qty = target_hedge_qty
                return signal

        return None