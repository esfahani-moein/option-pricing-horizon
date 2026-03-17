"""
Option Pricing Horizon
======================
A research-grade library for pricing European and Asian options under the
risk-neutral GBM framework.

Modules
-------
option_pricing_horizon.common
    Shared market parameter dataclass, high-precision math utilities, and
    Numba-accelerated GBM path simulation.

option_pricing_horizon.european
    Exact Black-Scholes closed-form pricing and all five Greeks for vanilla
    European call and put options.

option_pricing_horizon.asian
    Geometric Asian option (closed-form, Kemna-Vorst), arithmetic Asian option
    (Monte Carlo with antithetic variates and geometric control variate),
    Greeks estimation, and convergence analysis.
"""

__version__ = "0.1.0"
__author__ = "Option Pricing Horizon"
