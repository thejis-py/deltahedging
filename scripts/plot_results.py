"""
OptiHedge-RF: Backtest Result Visualization (Layer 5)
File: scripts/plot_results.py

Description:
    Generates Matplotlib/Seaborn plots showing Underlying Spot Price with Trade Markers,
    Portfolio Delta Drift, and Cumulative Portfolio Equity over time.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.backtest.engine import BacktestEngine
from src.simulator.execution_engine import ExecutionEngine
from src.strategies.threshold_based import ThresholdDeltaStrategy


def generate_backtest_plots():
    # 1. Run Backtest
    data_file = "market_data_ADVANC.parquet"
    engine = BacktestEngine(
        data_file=data_file,
        dw_symbol="ADVA01C2612A",
        underlying_symbol="ADVANC",
        strike=225.0,
        conversion_ratio=0.1,
        initial_cash=1_000_000.0,
        short_dw_qty=10_000.0,
    )

    strategy = ThresholdDeltaStrategy(threshold=0.02)
    execution_engine = ExecutionEngine(
        commission_rate=0.001,
        vat_rate=0.07,
        slippage_bps=1.0,
    )

    print("Running backtest for plot generation...")
    results = engine.run(strategy=strategy, execution_engine=execution_engine)

    history_df = results["history"]
    trades_df = results["trades"]

    # Convert timestamps to datetime for proper plotting
    history_df["datetime"] = pd.to_datetime(history_df["timestamp"])
    if not trades_df.empty:
        trades_df["datetime"] = pd.to_datetime(trades_df["timestamp"])

    # 2. Set Style
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("OptiHedge-RF: Delta Hedging Performance Dashboard", fontsize=16, fontweight="bold")

    # --- PANEL 1: Spot Price & Rebalance Trade Markers ---
    ax1 = axes[0]
    ax1.plot(history_df["datetime"], history_df["spot_price"], color="#1f77b4", label="ADVANC Spot Price", linewidth=1.5)

    if not trades_df.empty:
        buy_trades = trades_df[trades_df["side"] == "BUY"]
        sell_trades = trades_df[trades_df["side"] == "SELL"]

        if not buy_trades.empty:
            ax1.scatter(
                buy_trades["datetime"],
                buy_trades["fill_price"],
                color="green",
                marker="^",
                s=70,
                label="Buy Hedge (Rebalance)",
                zorder=5,
            )
        if not sell_trades.empty:
            ax1.scatter(
                sell_trades["datetime"],
                sell_trades["fill_price"],
                color="red",
                marker="v",
                s=70,
                label="Sell Hedge (Rebalance)",
                zorder=5,
            )

    ax1.set_ylabel("Spot Price (THB)", fontweight="bold")
    ax1.set_title("Underlying Spot Price & Rebalance Executions", fontsize=12)
    ax1.legend(loc="upper left")

    # --- PANEL 2: Portfolio Unhedged Delta Drift ---
    ax2 = axes[1]
    ax2.plot(
        history_df["datetime"],
        history_df["portfolio_delta"],
        color="#ff7f0e",
        label="Net Portfolio Delta",
        linewidth=1.2,
    )
    ax2.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7, label="Neutral Delta (0.0)")
    ax2.axhline(0.02 * 10000, color="red", linestyle=":", linewidth=1.0, alpha=0.5, label="Upper Band (+2%)")
    ax2.axhline(-0.02 * 10000, color="red", linestyle=":", linewidth=1.0, alpha=0.5, label="Lower Band (-2%)")

    ax2.set_ylabel("Portfolio Delta", fontweight="bold")
    ax2.set_title("Net Portfolio Unhedged Delta Drift", fontsize=12)
    ax2.legend(loc="upper left")

    # --- PANEL 3: Cumulative Portfolio Equity (PnL) ---
    ax3 = axes[2]
    ax3.plot(
        history_df["datetime"],
        history_df["total_equity"],
        color="#2ca02c",
        label="Portfolio Equity (MtM)",
        linewidth=1.5,
    )
    ax3.axhline(1_000_000.0, color="gray", linestyle="--", linewidth=1.0, label="Initial Capital (1M THB)")

    ax3.set_xlabel("Timestamp", fontweight="bold")
    ax3.set_ylabel("Equity (THB)", fontweight="bold")
    ax3.set_title("Cumulative Portfolio Equity & Mark-to-Market PnL", fontsize=12)
    ax3.legend(loc="upper left")

    # Format Layout and Save Figure
    plt.xticks(rotation=30)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "backtest_performance.png")
    plt.savefig(output_path, dpi=300)
    print(f"\n✅ Plot successfully saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    generate_backtest_plots()
    