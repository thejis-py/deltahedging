"""
OptiHedge-RF: Strategy Comparison Benchmark (Layer 5)
File: scripts/compare_strategies.py

Description:
    Runs a side-by-side backtest comparison between ThresholdDeltaStrategy
    (Delta Band) and TimeBasedDeltaStrategy (Periodic Rebalancing) to analyze
    turnover, transaction costs, execution efficiency, and hedging risk.
"""

import pandas as pd
from src.backtest.engine import BacktestEngine
from src.strategies.threshold_based import ThresholdDeltaStrategy
from src.strategies.time_based import TimeBasedDeltaStrategy
from src.simulator.execution_engine import ExecutionEngine


def run_strategy_comparison():
    print("=======================================================================")
    print("   OptiHedge-RF: Strategy Benchmark (Threshold vs. Time-Based)         ")
    print("=======================================================================")

    # 1. Initialize Backtest Engine
    data_file = "market_data_ADVANC.parquet"
    initial_cash = 1_000_000.0
    short_dw_qty = 10_000.0

    engine = BacktestEngine(
        data_file=data_file,
        dw_symbol="ADVA01C2612A",
        underlying_symbol="ADVANC",
        strike=225.0,
        conversion_ratio=0.1,
        initial_cash=initial_cash,
        short_dw_qty=short_dw_qty,
    )

    # 2. Define Execution Engine (0.10% commission + 7% VAT, 1 bp slippage)
    execution_engine = ExecutionEngine(
        commission_rate=0.001,
        vat_rate=0.07,
        slippage_bps=1.0,
    )

    # 3. Define Strategies to Benchmark
    strategies = {
        "Threshold (Band = 0.02)": ThresholdDeltaStrategy(threshold=0.02),
        "Time-Based (15 Bars)": TimeBasedDeltaStrategy(rebalance_interval_bars=15),
    }

    results_list = []

    for name, strategy in strategies.items():
        print(f"\nRunning Backtest for: {name}...")
        res = engine.run(strategy=strategy, execution_engine=execution_engine)
        metrics = res["metrics"]
        trades_df = res["trades"]

        # Calculate Total Share Turnover (Total volume of shares bought/sold)
        total_turnover = trades_df["quantity"].sum() if not trades_df.empty else 0.0
        total_slippage = trades_df["slippage"].sum() if not trades_df.empty else 0.0

        results_list.append({
            "Strategy": name,
            "Total PnL (THB)": metrics["Total PnL (THB)"],
            "Return (%)": metrics["Return (%)"],
            "Hedging Error RMS": metrics["Hedging Error RMS"],
            "Total Trades": metrics["Total Trades"],
            "Share Turnover (Qty)": total_turnover,
            "Commission Paid (THB)": metrics["Total Commission (THB)"],
            "Slippage Cost (THB)": total_slippage,
            "Total Execution Cost": metrics["Total Commission (THB)"] + total_slippage,
            "Max Drawdown (%)": metrics["Max Drawdown (%)"],
        })

    # 4. Format and Display Results DataFrame
    comp_df = pd.DataFrame(results_list)

    print("\n=======================================================================")
    print("                     SIDE-BY-SIDE BENCHMARK COMPARISON                  ")
    print("=======================================================================")

    # Display formatted table
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(comp_df.to_string(index=False))

    # 5. Comparative Analytics Insights
    thresh_row = comp_df[comp_df["Strategy"].str.contains("Threshold")].iloc[0]
    time_row = comp_df[comp_df["Strategy"].str.contains("Time")].iloc[0]

    cost_diff = time_row["Total Execution Cost"] - thresh_row["Total Execution Cost"]
    trade_diff = time_row["Total Trades"] - thresh_row["Total Trades"]

    print("\n=======================================================================")
    print("                          KEY TAKEAWAYS & ANALYSIS                      ")
    print("=======================================================================")
    if cost_diff > 0:
        print(f"• Transaction Cost Efficiency: Threshold Strategy saved {cost_diff:,.2f} THB in execution costs.")
        print(f"• Trade Reduction: Threshold Strategy reduced total trades by {trade_diff} executions.")
    else:
        print(f"• Transaction Cost Efficiency: Time-based strategy had lower execution costs by {abs(cost_diff):,.2f} THB.")

    print(f"• Risk Control (Hedging Error RMS):")
    print(f"    - Threshold Strategy : {thresh_row['Hedging Error RMS']:.4f}")
    print(f"    - Time-Based Strategy: {time_row['Hedging Error RMS']:.4f}")
    print("=======================================================================\n")


if __name__ == "__main__":
    run_strategy_comparison()
    