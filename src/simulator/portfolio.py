"""
OptiHedge-RF: Portfolio State & Accounting (Layer 4)
File: src/simulator/portfolio.py

Description:
    Tracks cash balances, stock shares, DW short positions, transaction costs,
    and total portfolio Mark-to-Market (MtM) value.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_entry_price: float


class Portfolio:
    """Tracks portfolio cash, holdings, transaction costs, and MtM valuation."""

    def __init__(self, initial_cash: float = 1_000_000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.total_commission_paid: float = 0.0
        self.total_slippage_paid: float = 0.0

    def update_position(self, symbol: str, quantity_change: float, execution_price: float, commission: float, slippage: float = 0.0):
        """Update position quantity and adjust cash balance after a trade fill."""
        cost = (quantity_change * execution_price) + commission
        self.cash -= cost
        self.total_commission_paid += commission
        self.total_slippage_paid += slippage

        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol, quantity=quantity_change, avg_entry_price=execution_price)
        else:
            pos = self.positions[symbol]
            new_qty = pos.quantity + quantity_change
            if new_qty == 0:
                pos.avg_entry_price = 0.0
            elif (pos.quantity > 0 and quantity_change > 0) or (pos.quantity < 0 and quantity_change < 0):
                # Weighted average entry price for adding to position
                total_cost = (pos.quantity * pos.avg_entry_price) + (quantity_change * execution_price)
                pos.avg_entry_price = total_cost / new_qty
            pos.quantity = new_qty

    def get_position_qty(self, symbol: str) -> float:
        """Returns quantity held for a given symbol."""
        return self.positions[symbol].quantity if symbol in self.positions else 0.0

    def calculate_total_equity(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total Mark-to-Market (MtM) equity.
        Equity = Cash + Sum(Quantity * Current Market Price)
        """
        holdings_value = 0.0
        for symbol, pos in self.positions.items():
            price = current_prices.get(symbol, pos.avg_entry_price)
            holdings_value += pos.quantity * price
        return self.cash + holdings_value
    