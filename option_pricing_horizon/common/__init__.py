"""
option_pricing_horizon.common
=============================
Shared primitives used by all pricing modules:

- ``market_params``  – :class:`MarketParams` dataclass
- ``math_utils``     – high-precision normal CDF/PDF and log-normal statistics
- ``simulation``     – Numba-JIT GBM path generator
"""

from .market_params import MarketParams
from .math_utils import norm_cdf, norm_pdf, lognorm_mean, lognorm_variance
from .simulation import simulate_gbm_paths

__all__ = [
    "MarketParams",
    "norm_cdf",
    "norm_pdf",
    "lognorm_mean",
    "lognorm_variance",
    "simulate_gbm_paths",
]
