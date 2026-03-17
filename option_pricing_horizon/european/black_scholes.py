"""
option_pricing_horizon.european.black_scholes
=============================================
Exact closed-form Black-Scholes pricing for vanilla European options.

Theory
------
Under the risk-neutral measure Q, the stock follows GBM:

    dS_t = (r - q) S_t dt + σ S_t dW_t

where r is the risk-free rate, q the continuous dividend yield, and σ the
volatility.  The time-0 price of a European call is:

    C = S₀ e^{-qT} Φ(d₁)  −  K e^{-rT} Φ(d₂)

    d₁ = [ln(S₀/K) + (r - q + σ²/2) T] / (σ √T)
    d₂ = d₁ − σ √T

and a European put satisfies put-call parity:

    P = C − S₀ e^{-qT} + K e^{-rT}

Alternatively directly:

    P = K e^{-rT} Φ(−d₂)  −  S₀ e^{-qT} Φ(−d₁)

All normal CDF evaluations use ``scipy.special.ndtr`` (machine precision).

Functions
---------
bs_call_price(S, K, r, sigma, T, q) -> float
bs_put_price(S, K, r, sigma, T, q) -> float
bs_price(S, K, r, sigma, T, option_type, q) -> float
implied_volatility(market_price, S, K, r, T, option_type, q) -> float
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from option_pricing_horizon.common.math_utils import norm_cdf, bs_d1, bs_d2


# ---------------------------------------------------------------------------
# Core pricing functions
# ---------------------------------------------------------------------------

def bs_call_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    q: float = 0.0,
) -> float:
    """Black-Scholes price of a European call option.

    Formula
    -------
        C = S·e^{-qT}·Φ(d₁) − K·e^{-rT}·Φ(d₂)

    Parameters
    ----------
    S : float
        Current spot price S₀.
    K : float
        Strike price.
    r : float
        Annualised continuously-compounded risk-free rate.
    sigma : float
        Annualised volatility σ > 0.
    T : float
        Time to expiry in years T > 0.
    q : float
        Continuous dividend yield (default 0.0).

    Returns
    -------
    float
        European call price.

    Examples
    --------
    >>> from option_pricing_horizon.european.black_scholes import bs_call_price
    >>> round(bs_call_price(100, 100, 0.05, 0.20, 1.0), 4)
    10.4506
    """
    d1 = bs_d1(S, K, r, sigma, T, q)
    d2 = bs_d2(S, K, r, sigma, T, q)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    return float(S * disc_q * norm_cdf(d1) - K * disc_r * norm_cdf(d2))


def bs_put_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    q: float = 0.0,
) -> float:
    """Black-Scholes price of a European put option.

    Formula
    -------
        P = K·e^{-rT}·Φ(−d₂) − S·e^{-qT}·Φ(−d₁)

    Parameters
    ----------
    S : float
        Current spot price S₀.
    K : float
        Strike price.
    r : float
        Annualised risk-free rate.
    sigma : float
        Annualised volatility.
    T : float
        Time to expiry.
    q : float
        Continuous dividend yield (default 0.0).

    Returns
    -------
    float
        European put price.

    Examples
    --------
    >>> from option_pricing_horizon.european.black_scholes import bs_put_price
    >>> round(bs_put_price(100, 100, 0.05, 0.20, 1.0), 4)
    5.5735
    """
    d1 = bs_d1(S, K, r, sigma, T, q)
    d2 = bs_d2(S, K, r, sigma, T, q)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    return float(K * disc_r * norm_cdf(-d2) - S * disc_q * norm_cdf(-d1))


def bs_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """Dispatch to call or put price based on ``option_type``.

    Parameters
    ----------
    S, K, r, sigma, T, q : float
        Market and contract parameters (see :func:`bs_call_price`).
    option_type : {'call', 'put'}
        Option type string (case-insensitive).

    Returns
    -------
    float
        Option price.

    Raises
    ------
    ValueError
        If ``option_type`` is not ``'call'`` or ``'put'``.
    """
    ot = option_type.lower().strip()
    if ot == "call":
        return bs_call_price(S, K, r, sigma, T, q)
    elif ot == "put":
        return bs_put_price(S, K, r, sigma, T, q)
    else:
        raise ValueError(
            f"option_type must be 'call' or 'put'; got '{option_type}'"
        )


# ---------------------------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------------------------

def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    option_type: str = "call",
    q: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """Solve for the Black-Scholes implied volatility via Brent's method.

    Uses ``scipy.optimize.brentq`` which is guaranteed to converge if a root
    exists within the search bracket [σ_lo, σ_hi] = [1e-6, 10.0].

    Parameters
    ----------
    market_price : float
        Observed market price of the option.
    S : float
        Spot price.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    T : float
        Time to expiry.
    option_type : {'call', 'put'}
        Option type.
    q : float
        Dividend yield (default 0.0).
    tol : float
        Tolerance for Brent's method convergence (default 1e-8).
    max_iter : int
        Maximum Brent iterations (default 200).

    Returns
    -------
    float
        Implied volatility σ_IV such that BS_price(σ_IV) = market_price.

    Raises
    ------
    ValueError
        If no solution is found within the bracket [1e-6, 10].

    Examples
    --------
    >>> from option_pricing_horizon.european.black_scholes import (
    ...     bs_call_price, implied_volatility)
    >>> price = bs_call_price(100, 100, 0.05, 0.25, 1.0)
    >>> abs(implied_volatility(price, 100, 100, 0.05, 1.0) - 0.25) < 1e-7
    True
    """
    def objective(sigma: float) -> float:
        return bs_price(S, K, r, sigma, T, option_type, q) - market_price

    sigma_lo, sigma_hi = 1e-6, 10.0
    f_lo = objective(sigma_lo)
    f_hi = objective(sigma_hi)

    if f_lo * f_hi > 0:
        raise ValueError(
            f"No IV solution in [1e-6, 10]: f({sigma_lo})={f_lo:.6f}, "
            f"f({sigma_hi})={f_hi:.6f}.  Check market_price is in-model range."
        )

    return float(
        brentq(objective, sigma_lo, sigma_hi, xtol=tol, maxiter=max_iter)
    )
