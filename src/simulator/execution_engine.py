"""
OptiHedge-RF: Execution Engine & Cost Model (Layer 4)
File: src/simulator/execution_engine.py

Description:
    Simulates trade execution with SET market transaction costs (brokerage fee + VAT)
    and optional bid-ask spread slippage.
"""

from dataclasses import dataclass
from src.strategies.base_strategy import RebalanceSignal
from src.simulator.portfolio import Portfolio


@dataclass
class TradeFill:
    timestamp: str
    symbol: str
    side: str            # 'BUY' or 'SELL'
    quantity: float
    fill_price: float
    commission: float
    slippage: float


class ExecutionEngine:
    """Fills rebalance orders and applies trading fee structures."""

    def __init__(self, commission_rate: float = 0.001, vat_rate: float = 0.07, slippage_bps: float = 0.0):
        """
        Parameters:
            commission_rate: Brokerage commission (default: 0.10% for online equity trade).
            vat_rate: Value Added Tax on commission (default: 7%).
            slippage_bps: Half-spread or market impact in basis points (1 bp = 0.0001).
        """
        self.effective_fee_rate = commission_rate * (1.0 + vat_rate)
        self.slippage_bps = slippage_bps

    def execute_signal(self, signal: RebalanceSignal, spot_price: float, portfolio: Portfolio) -> TradeFill:
        """Executes a rebalance signal, updates portfolio state, and returns a TradeFill record."""
        trade_qty = signal.trade_qty
        side = "BUY" if trade_qty > 0 else "SELL"

        # Apply slippage to execution price
        slippage_per_share = spot_price * (self.slippage_bps / 10000.0)
        fill_price = spot_price + slippage_per_share if side == "BUY" else spot_price - slippage_per_share

        trade_value = abs(trade_qty) * fill_price
        commission = trade_value * self.effective_fee_rate
        total_slippage = abs(trade_qty) * slippage_per_share

        # Update portfolio positions and cash balance
        portfolio.update_position(
            symbol=signal.symbol,
            quantity_change=trade_qty,
            execution_price=fill_price,
            commission=commission,
            slippage=total_slippage,
        )

        return TradeFill(
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            side=side,
            quantity=abs(trade_qty),
            fill_price=fill_price,
            commission=commission,
            slippage=total_slippage,
        )

        