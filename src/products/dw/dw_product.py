#An empty __init__.py file tells Python that a folder should be treated as a package (a collection of modules).

"""
OptiHedge-RF: Derivative Warrant (DW) Product Definition (Layer 2)
File: src/products/dw/dw_product.py

Description:
    Defines the DWProduct class representing European-style Derivative Warrants
    traded on SET/TFEX, including Call and Put options with conversion ratios.
"""

from src.products.base_product import BaseProduct
from src.products.product_factory import ProductFactory


@ProductFactory.register_product("dw")
class DWProduct(BaseProduct):
    """Product specification for Derivative Warrants (DWs)."""

    def __init__(
        self,
        symbol: str,
        underlying_symbol: str,
        strike: float,
        expiry_date: str,
        option_type: str = "call",  # 'call' or 'put'
        conversion_ratio: float = 0.1,  # Shares per 1 DW
    ):
        super().__init__(symbol, underlying_symbol, strike, expiry_date)
        self.option_type = option_type.lower()
        if self.option_type not in ["call", "put"]:
            raise ValueError(f"Option type must be 'call' or 'put', got '{option_type}'")
        self.conversion_ratio = conversion_ratio

    def calculate_payoff(self, spot_price: float) -> float:
        """
        Calculate DW intrinsic payoff per warrant unit at expiration.
        Payoff = Max(0, Spot - Strike) * Conversion_Ratio (for Call)
        """
        if self.option_type == "call":
            intrinsic = max(spot_price - self.strike, 0.0)
        else:
            intrinsic = max(self.strike - spot_price, 0.0)

        return intrinsic * self.conversion_ratio