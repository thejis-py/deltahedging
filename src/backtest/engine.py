"""
OptiHedge-RF: Backtest Orchestration Engine (Layer 5)
File: src/backtest/engine.py

Description:
    Main backtesting engine that orchestrates Data Loading (Layer 1),
    Greeks Engine (Layer 2), Strategy Evaluation (Layer 3),
    and Execution Simulation (Layer 4).
"""

import pandas as pd
from typing import Dict, Any, List

from data.data_loader import DataLoader
from src.products.product_factory import ProductFactory
from src.strategies.base_strategy import BaseStrategy
from src.simulator.portfolio import Portfolio
from src.simulator.execution_engine import ExecutionEngine
from src.backtest.metrics import PerformanceMetrics

# Import DW submodules to trigger ProductFactory decorators
import src.products.dw.dw_product
import src.products.dw.dw_pricer


class BacktestEngine:
    """Orchestrates the full event-driven backtest pipeline."""

    def __init__(
        self,
        data_file: str,
        dw_symbol: str = "ADVA01C2612A",
        underlying_symbol: str = "ADVANC",
        strike: float = 225.0,
        conversion_ratio: float = 0.1,
        initial_cash: float = 1_000_000.0,
        short_dw_qty: float = 10_000.0,  # Number of DW units sold short to clients
    ):
        self.data_file = data_file
        self.dw_symbol = dw_symbol
        self.underlying_symbol = underlying_symbol
        self.short_dw_qty = short_dw_qty
        self.initial_cash = initial_cash

        # 1. Instantiate DW Product & Pricer
        self.dw_product = ProductFactory.create_product(
            product_type="dw",
            symbol=dw_symbol,
            underlying_symbol=underlying_symbol,
            strike=strike,
            expiry_date="2026-12-30",
            option_type="call",
            conversion_ratio=conversion_ratio,
        )

        self.pricer = ProductFactory.create_pricer(
            pricer_type="black_scholes_dw",
            product=self.dw_product,
        )

    def run(self, strategy: BaseStrategy, execution_engine: ExecutionEngine) -> Dict[str, Any]:
        """Runs the event-driven backtest loop over time-series market data."""
        # Load raw/processed data via Layer 1
        loader = DataLoader(data_dir="data/raw")
        df = loader.load_parquet(self.data_file)

        portfolio = Portfolio(initial_cash=self.initial_cash)

        # Setup initial short DW position (-short_dw_qty units)
        initial_dw_price = df.iloc[0]["dw_price"]
        portfolio.update_position(
            symbol=self.dw_symbol,
            quantity_change=-self.short_dw_qty,
            execution_price=initial_dw_price,
            commission=0.0,
        )

        history: List[Dict[str, Any]] = []
        trade_records: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            timestamp = str(row["timestamp"])
            spot = float(row["spot_price"])
            dw_price = float(row["dw_price"])
            ttm = float(row["time_to_maturity"])
            r = float(row["risk_free_rate"])
            vol = float(row["implied_volatility"])

            # Layer 2: Compute theoretical Greeks
            greeks = self.pricer.calculate_greeks(spot=spot, ttm=ttm, vol=vol, r=r)
            eff_delta = greeks["effective_delta"]

            # Current position holdings
            dw_qty = portfolio.get_position_qty(self.dw_symbol)
            underlying_qty = portfolio.get_position_qty(self.underlying_symbol)

            # Net Portfolio Delta = (DW_Qty * Effective_Delta) + Underlying_Shares
            net_portfolio_delta = (dw_qty * eff_delta) + underlying_qty

            # Layer 3: Evaluate strategy for rebalance signal
            signal = strategy.evaluate(
                timestamp=timestamp,
                spot_price=spot,
                effective_delta=eff_delta,
                dw_position_qty=dw_qty,
            )

            # Layer 4: Execute trade if signal generated
            if signal:
                fill = execution_engine.execute_signal(signal=signal, spot_price=spot, portfolio=portfolio)
                trade_records.append({
                    "timestamp": fill.timestamp,
                    "symbol": fill.symbol,
                    "side": fill.side,
                    "quantity": fill.quantity,
                    "fill_price": fill.fill_price,
                    "commission": fill.commission,
                    "slippage": fill.slippage,
                })

            # Calculate updated total equity
            current_prices = {self.underlying_symbol: spot, self.dw_symbol: dw_price}
            total_equity = portfolio.calculate_total_equity(current_prices)

            history.append({
                "timestamp": timestamp,
                "spot_price": spot,
                "dw_price": dw_price,
                "effective_delta": eff_delta,
                "portfolio_delta": net_portfolio_delta,
                "underlying_qty": portfolio.get_position_qty(self.underlying_symbol),
                "cash": portfolio.cash,
                "total_equity": total_equity,
            })

        history_df = pd.DataFrame(history)
        trades_df = pd.DataFrame(trade_records)

        summary_metrics = PerformanceMetrics.calculate_summary(
            history_df=history_df,
            initial_cash=self.initial_cash,
            trade_count=len(trades_df),
            total_commission=portfolio.total_commission_paid,
        )

        return {
            "metrics": summary_metrics,
            "history": history_df,
            "trades": trades_df,
        }