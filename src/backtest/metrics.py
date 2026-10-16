"""
OptiHedge-RF: Backtest Performance Metrics (Layer 5)
File: src/backtest/metrics.py

Description:
    Computes key performance metrics for delta-hedging strategies,
    including PnL, Max Drawdown, Sharpe Ratio, and Hedging Error RMS.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


class PerformanceMetrics:
    """Calculates summary statistics and risk metrics for backtest results."""

    @staticmethod
    def calculate_summary(
        history_df: pd.DataFrame,
        initial_cash: float,
        trade_count: int,
        total_commission: float,
    ) -> Dict[str, Any]:
        """
        Computes summary statistics from the backtest history DataFrame.
        """
        equities = history_df["total_equity"].values
        returns = pd.Series(equities).pct_change().dropna()

        total_pnl = equities[-1] - initial_cash
        return_pct = (total_pnl / initial_cash) * 100.0

        # Maximum Drawdown
        peak = np.maximum.accumulate(equities)
        drawdown = (equities - peak) / peak
        max_drawdown = np.min(drawdown) * 100.0

        # Sharpe Ratio (annualized for 1-min data, assuming 252 days * 390 mins = 98,280 mins)
        annual_factor = np.sqrt(98280)
        sharpe_ratio = (returns.mean() / returns.std() * annual_factor) if returns.std() > 0 else 0.0

        # Hedging Error Root Mean Square (RMS of Portfolio Unhedged Delta)
        unhedged_deltas = history_df["portfolio_delta"].values
        hedging_error_rms = np.sqrt(np.mean(unhedged_deltas ** 2))

        return {
            "Initial Cash (THB)": initial_cash,
            "Final Equity (THB)": equities[-1],
            "Total PnL (THB)": total_pnl,
            "Return (%)": return_pct,
            "Max Drawdown (%)": max_drawdown,
            "Sharpe Ratio": sharpe_ratio,
            "Hedging Error RMS": hedging_error_rms,
            "Total Trades": trade_count,
            "Total Commission (THB)": total_commission,
        }