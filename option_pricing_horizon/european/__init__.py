"""
option_pricing_horizon.european
================================
Exact closed-form pricing and Greeks for vanilla European call and put
options under the Black-Scholes model.

Modules
-------
black_scholes
    Analytical call/put prices, parity check, and implied-volatility solver.
greeks
    All five Greeks (Delta, Gamma, Theta, Vega, Rho) in closed form plus
    a finite-difference verifier.
"""

from .black_scholes import (
    bs_call_price,
    bs_put_price,
    bs_price,
    implied_volatility,
)
from .greeks import (
    bs_delta,
    bs_gamma,
    bs_theta,
    bs_vega,
    bs_rho,
    bs_all_greeks,
)

__all__ = [
    "bs_call_price",
    "bs_put_price",
    "bs_price",
    "implied_volatility",
    "bs_delta",
    "bs_gamma",
    "bs_theta",
    "bs_vega",
    "bs_rho",
    "bs_all_greeks",
]
