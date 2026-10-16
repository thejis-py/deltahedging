"""
OptiHedge-RF: Layer 4 Simulator Smoke Test
File: scripts/test_layer4.py
"""

from src.simulator.portfolio import Portfolio
from src.simulator.execution_engine import ExecutionEngine
from src.strategies.base_strategy import RebalanceSignal


def test_layer4():
    print("==========================================")
    print("   OptiHedge-RF: Layer 4 Simulator Test   ")
    print("==========================================")

    portfolio = Portfolio(initial_cash=1_000_000.0)
    engine = ExecutionEngine(commission_rate=0.001, vat_rate=0.07, slippage_bps=2.0)

    # 1. Setup short DW position (-10,000 units sold @ 1.00 THB)
    portfolio.update_position(symbol="ADVA01C2612A", quantity_change=-10000, execution_price=1.00, commission=10.7)

    print("\n[Initial State]")
    print(f"  Cash Balance:        {portfolio.cash:,.2f} THB")
    print(f"  DW Quantity Held:    {portfolio.get_position_qty('ADVA01C2612A'):,.0f} units")

    # 2. Issue Rebalance Signal to buy 500 underlying shares @ 220.00 THB
    signal = RebalanceSignal(
        timestamp="2026-10-01 10:00:00",
        symbol="ADVANC",
        target_hedge_qty=500.0,
        trade_qty=500.0,
        reason="Threshold Exceeded",
    )

    fill = engine.execute_signal(signal=signal, spot_price=220.00, portfolio=portfolio)

    print("\n[Trade Fill Executed]")
    print(f"  Side & Qty:          {fill.side} {fill.quantity:,.0f} {fill.symbol}")
    print(f"  Fill Price:          {fill.fill_price:.4f} THB")
    print(f"  Commission Paid:     {fill.commission:.2f} THB")

    # 3. Calculate Portfolio MtM Equity
    current_prices = {"ADVANC": 222.00, "ADVA01C2612A": 1.10}
    total_equity = portfolio.calculate_total_equity(current_prices)

    print("\n[Updated State]")
    print(f"  Cash Balance:        {portfolio.cash:,.2f} THB")
    print(f"  Stock Quantity Held: {portfolio.get_position_qty('ADVANC'):,.0f} shares")
    print(f"  Total MtM Equity:    {total_equity:,.2f} THB")

    assert portfolio.get_position_qty("ADVANC") == 500.0, "Stock position not updated correctly!"
    assert fill.commission > 0, "Commission was not charged!"

    print("\n✅ Layer 4 smoke test passed successfully!")


if __name__ == "__main__":
    test_layer4()