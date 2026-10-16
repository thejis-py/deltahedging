"""
OptiHedge-RF: Layer 2 Smoke Test
File: scripts/test_layer2.py

Description:
    Verifies that ProductFactory, DWProduct, and DWBSPricer function correctly,
    computing theoretical prices, raw Delta, and conversion-ratio adjusted Effective Delta.
"""

from src.products.product_factory import ProductFactory
# Import DW modules to trigger @ProductFactory decorators
import src.products.dw.dw_product
import src.products.dw.dw_pricer


def run_layer2_test():
    print("==========================================")
    print("   OptiHedge-RF: Layer 2 Quant Core Test  ")
    print("==========================================")

    # 1. Instantiate DW via ProductFactory
    dw_call = ProductFactory.create_product(
        product_type="dw",
        symbol="ADVA01C2612A",
        underlying_symbol="ADVANC",
        strike=225.0,
        expiry_date="2026-12-30",
        option_type="call",
        conversion_ratio=0.1,  # 10 DWs = 1 Underlying Share
    )

    # 2. Instantiate Pricer via ProductFactory
    pricer = ProductFactory.create_pricer(
        pricer_type="black_scholes_dw",
        product=dw_call,
    )

    # 3. Market Parameters for Test Case
    spot = 230.0  # In-the-money (Spot > Strike)
    ttm = 0.25    # 0.25 years (~90 days to expiry)
    vol = 0.20    # 20% annual volatility
    r = 0.025     # 2.5% risk-free rate

    # 4. Perform Valuation and Greeks Calculations
    theo_price = pricer.calculate_price(spot=spot, ttm=ttm, vol=vol, r=r)
    greeks = pricer.calculate_greeks(spot=spot, ttm=ttm, vol=vol, r=r)

    # 5. Display Valuation Outputs
    print(f"\n[Product Config]")
    print(f"  Symbol:            {dw_call.symbol}")
    print(f"  Underlying:        {dw_call.underlying_symbol}")
    print(f"  Strike:            {dw_call.strike}")
    print(f"  Conversion Ratio:  {dw_call.conversion_ratio}")

    print(f"\n[Valuation Inputs]")
    print(f"  Spot Price:        {spot} THB")
    print(f"  Time to Maturity:  {ttm} Years")
    print(f"  Implied Vol:       {vol * 100}%")
    print(f"  Risk-Free Rate:    {r * 100}%")

    print(f"\n[Valuation Outputs]")
    print(f"  Theoretical Price: {theo_price:.4f} THB")
    print(f"  Raw Option Delta:  {greeks['delta']:.4f}")
    print(f"  Effective Delta:   {greeks['effective_delta']:.4f} (Raw Delta * Conversion Ratio)")
    print(f"  Gamma:             {greeks['gamma']:.6f}")
    print(f"  Vega (1% vol):     {greeks['vega']:.6f}")
    print(f"  Theta (1 day):     {greeks['theta']:.6f}")

    # 6. Basic Sanity Assertions
    assert theo_price > 0, "Theoretical price must be positive!"
    assert 0.0 <= greeks["delta"] <= 1.0, "Call Delta must be between 0 and 1!"
    assert (
        abs(greeks["effective_delta"] - (greeks["delta"] * dw_call.conversion_ratio)) < 1e-6
    ), "Effective Delta calculation mismatch!"

    print("\n✅ Layer 2 smoke test passed successfully!")


if __name__ == "__main__":
    run_layer2_test()