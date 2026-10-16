"""
OptiHedge-RF: Product Factory & Registry (Layer 2)
File: src/products/product_factory.py

Description:
    Implements Factory & Registry patterns to dynamically instantiate Product
    and Pricer objects based on configuration strings.
"""

import logging
from typing import Dict, Type
from src.products.base_product import BaseProduct, BasePricer

logger = logging.getLogger(__name__)


class ProductFactory:
    """Registry and Factory for Product and Pricer instances."""

    _product_registry: Dict[str, Type[BaseProduct]] = {}
    _pricer_registry: Dict[str, Type[BasePricer]] = {}

    @classmethod
    def register_product(cls, product_type: str):
        """Decorator to register a BaseProduct subclass."""
        def decorator(subclass: Type[BaseProduct]):
            cls._product_registry[product_type.lower()] = subclass
            logger.info(f"Registered Product type: '{product_type.lower()}' -> {subclass.__name__}")
            return subclass
        return decorator

    @classmethod
    def register_pricer(cls, pricer_type: str):
        """Decorator to register a BasePricer subclass."""
        def decorator(subclass: Type[BasePricer]):
            cls._pricer_registry[pricer_type.lower()] = subclass
            logger.info(f"Registered Pricer type: '{pricer_type.lower()}' -> {subclass.__name__}")
            return subclass
        return decorator

    @classmethod
    def create_product(cls, product_type: str, **kwargs) -> BaseProduct:
        """Instantiate a registered product."""
        key = product_type.lower()
        if key not in cls._product_registry:
            raise KeyError(
                f"Product type '{product_type}' not found in registry. "
                f"Available types: {list(cls._product_registry.keys())}"
            )
        return cls._product_registry[key](**kwargs)

    @classmethod
    def create_pricer(cls, pricer_type: str, product: BaseProduct, **kwargs) -> BasePricer:
        """Instantiate a registered pricer for a given product."""
        key = pricer_type.lower()
        if key not in cls._pricer_registry:
            raise KeyError(
                f"Pricer type '{pricer_type}' not found in registry. "
                f"Available types: {list(cls._pricer_registry.keys())}"
            )
        return cls._pricer_registry[key](product=product, **kwargs)