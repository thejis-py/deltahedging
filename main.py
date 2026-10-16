"""
OptiHedge-RF: Main Backtest Execution Entry Point
File: main.py

Description:
    Runs an end-to-end backtest of the Delta Hedging framework using mock data.
"""

from src.backtest.engine import BacktestEngine
from src.strategies.threshold_based import ThresholdDeltaStrategy
from src.simulator.execution_engine import ExecutionEngine


def main():
    print("==========================================================")
    print("     OptiHedge-RF: Algorithmic Delta-Hedging Backtest     ")
    print("==========================================================")

    # 1. Initialize Backtest Engine
    engine = BacktestEngine(
        data_file="market_data_ADVANC.parquet",
        dw_symbol="ADVA01C2612A",
        underlying_symbol="ADVANC",
        strike=225.0,
        conversion_ratio=0.1,
        initial_cash=1_000_000.0,
        short_dw_qty=10_000.0,  # Market maker short 10,000 DWs
    )

    # 2. Define Strategy (0.02 Delta Band Threshold)
    strategy = ThresholdDeltaStrategy(threshold=0.02)

    # 3. Define Execution Engine (0.10% commission + 7% VAT, 1 bp slippage)
    execution_engine = ExecutionEngine(
        commission_rate=0.001,
        vat_rate=0.07,
        slippage_bps=1.0,
    )

    # 4. Run Backtest
    print("\nRunning Backtest Pipeline...")
    results = engine.run(strategy=strategy, execution_engine=execution_engine)

    # 5. Output Summary Metrics
    print("\n==========================================================")
    print("                 BACKTEST PERFORMANCE SUMMARY             ")
    print("==========================================================")
    metrics = results["metrics"]
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:<25}: {value:,.4f}")
        else:
            print(f"  {key:<25}: {value}")

    print("\nFirst 5 History Rows:")
    print(results["history"][["timestamp", "spot_price", "effective_delta", "underlying_qty", "total_equity"]].head())


if __name__ == "__main__":
    main()