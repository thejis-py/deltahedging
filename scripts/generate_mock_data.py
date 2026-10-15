
#pip install pyarrow first, if not compatible downgrade it
"""
OptiHedge-RF: Mock Data Generator
File: scripts/generate_mock_data.py

Description:
    Generates synthetic intraday market data for ADVANC and ADVA01C2612A DW
    and saves it as a Parquet file in data/raw/ for offline testing.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_mock_market_data(
    underlying_symbol: str = "ADVANC",
    dw_symbol: str = "ADVA01C2612A",
    start_spot: float = 220.0,
    strike: float = 225.0,
    conversion_ratio: float = 0.1,
    start_date: str = "2026-10-01 10:00:00",
    expiry_date: str = "2026-12-30 16:30:00",
    num_bars: int = 1000,
    output_filename: str = "market_data_ADVANC.parquet",
) -> None:
    """Generates synthetic intraday time-series market data and saves to parquet."""

    np.random.seed(42)  # For reproducible mock data
    timestamps = pd.date_range(start=start_date, periods=num_bars, freq="1min")

    # 1. Simulate Geometric Brownian Motion (GBM) for Underlying Spot Mid-Price
    dt = 1.0 / (365.0 * 24.0 * 60.0)  # 1-minute time step in years
    volatility = 0.25  # 25% annual volatility
    drift = 0.05       # 5% annual drift

    shocks = np.random.normal(0, 1, num_bars)
    price_paths = np.zeros(num_bars)
    price_paths[0] = start_spot

    for t in range(1, num_bars):
        price_paths[t] = price_paths[t - 1] * np.exp(
            (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shocks[t]
        )

    # 2. Add Bid-Ask Spread dynamics to Spot
    spot_mid = price_paths
    half_spread_spot = 0.25  # 0.50 THB spread
    spot_bid = np.round(spot_mid - half_spread_spot, 2)
    spot_ask = np.round(spot_mid + half_spread_spot, 2)

    # 3. Derive Theoretical/Mock DW Price & Spreads
    # Intrinsic + Time Value approximation
    intrinsic = np.maximum(spot_mid - strike, 0.0) * conversion_ratio
    time_value = 0.05 + 0.02 * np.random.normal(0, 0.1, num_bars)
    dw_mid = np.maximum(0.01, np.round(intrinsic + time_value, 2))

    half_spread_dw = 0.01  # 0.02 THB spread
    dw_bid = np.maximum(0.01, np.round(dw_mid - half_spread_dw, 2))
    dw_ask = np.round(dw_mid + half_spread_dw, 2)

    # 4. Generate Volume & Volatility dynamics
    volume = np.random.randint(1000, 50000, size=num_bars)
    implied_vol = 0.25 + 0.05 * np.sin(np.linspace(0, 10, num_bars))  # Fluctuating IV
    risk_free_rate = 0.025  # 2.5%

    # 5. Assemble DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "underlying_symbol": underlying_symbol,
        "spot_bid": spot_bid,
        "spot_ask": spot_ask,
        "spot_last": spot_mid,
        "dw_symbol": dw_symbol,
        "dw_bid": dw_bid,
        "dw_ask": dw_ask,
        "dw_last": dw_mid,
        "expiry_date": pd.to_datetime(expiry_date),
        "volume": volume,
        "risk_free_rate": risk_free_rate,
        "implied_volatility": np.round(implied_vol, 4),
    })

    # 6. Save to data/raw/
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename

    df.to_parquet(output_path, index=False)
    print(f"✅ Successfully generated mock market data!")
    print(f"📁 Saved to: {output_path.resolve()}")
    print(f"📊 Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df[["timestamp", "spot_bid", "spot_ask", "dw_bid", "dw_ask", "volume"]].head())
    #print(df[["timestamp", "spot_bid", "spot_ask", "dw_bid", "dw_ask", "volume"]])


if __name__ == "__main__":
    generate_mock_market_data()