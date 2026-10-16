"""
OptiHedge-RF: Layer 3 Strategy Engine Test
File: scripts/test_layer3.py
"""

from src.strategies.threshold_based import ThresholdDeltaStrategy
from src.strategies.time_based import TimeBasedDeltaStrategy


def test_strategies():
    print("==========================================")
    print("   OptiHedge-RF: Layer 3 Strategy Test    ")
    print("==========================================")

    # Short 10,000 DWs with Effective Delta = 0.05 (Raw Delta 0.5 * 0.1 ratio)
    dw_qty = -10000.0
    effective_delta = 0.05  # Target hedge = -(-10000 * 0.05) = +500 shares

    # 1. Test Threshold Strategy
    thresh_strat = ThresholdDeltaStrategy(threshold=0.02)
    sig1 = thresh_strat.evaluate("2026-10-01 10:00:00", spot_price=220.0, effective_delta=effective_delta, dw_position_qty=dw_qty)

    print("\n[Threshold Strategy Test]")
    print(f"  Signal Issued: {sig1 is not None}")
    if sig1:
        print(f"  Trade Qty:     {sig1.trade_qty} shares")
        print(f"  Reason:        {sig1.reason}")

    # Second evaluation with minor delta move (should NOT trigger)
    sig2 = thresh_strat.evaluate("2026-10-01 10:01:00", spot_price=220.1, effective_delta=0.051, dw_position_qty=dw_qty)
    print(f"  Small Drift Signal Issued: {sig2 is not None} (Expected: False)")

    # 2. Test Time-Based Strategy
    time_strat = TimeBasedDeltaStrategy(rebalance_interval_bars=3)
    print("\n[Time-Based Strategy Test]")
    for bar in range(1, 4):
        sig = time_strat.evaluate(f"2026-10-01 10:0{bar}:00", spot_price=220.0, effective_delta=effective_delta, dw_position_qty=dw_qty)
        print(f"  Bar {bar} Triggered: {sig is not None}")

    assert sig1 is not None, "Threshold strategy failed to trigger!"
    assert sig2 is None, "Threshold strategy triggered incorrectly on small drift!"
    assert sig is not None, "Time strategy failed to trigger on interval bar!"

    print("\n✅ Layer 3 smoke test passed successfully!")


if __name__ == "__main__":
    test_strategies()