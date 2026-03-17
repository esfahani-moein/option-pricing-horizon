"""
option_pricing_horizon.european.greeks
=======================================
Exact closed-form Greeks for European call and put options under the
Black-Scholes model (Merton extension with continuous dividends).

Theory
------
Let d₁, d₂ be the standard Black-Scholes quantities and Φ, φ denote the
standard-normal CDF and PDF respectively.

Delta (∂C/∂S, ∂P/∂S)
    Call:  Δ_C = e^{-qT} Φ(d₁)
    Put:   Δ_P = −e^{-qT} Φ(−d₁) = Δ_C − e^{-qT}

Gamma (∂²C/∂S², same for put)
    Γ = e^{-qT} φ(d₁) / (S σ √T)

Theta (∂C/∂t, per calendar day)
    Call:  Θ_C = [−S e^{-qT} φ(d₁) σ / (2√T)]
                 − r K e^{-rT} Φ(d₂)
                 + q S e^{-qT} Φ(d₁)
    Put:   Θ_P = [−S e^{-qT} φ(d₁) σ / (2√T)]
                 + r K e^{-rT} Φ(−d₂)
                 − q S e^{-qT} Φ(−d₁)
    Conventionally reported per calendar day (divide by 365).

Vega (∂C/∂σ = ∂P/∂σ)
    V = S e^{-qT} φ(d₁) √T
    Reported per 1% change in σ (divide by 100).

Rho (∂C/∂r)
    Call:  ρ_C = K T e^{-rT} Φ(d₂)
    Put:   ρ_P = −K T e^{-rT} Φ(−d₂)
    Reported per 1% change in r (divide by 100).

Functions
---------
bs_delta(S, K, r, sigma, T, option_type, q) -> float
bs_gamma(S, K, r, sigma, T, q) -> float
bs_theta(S, K, r, sigma, T, option_type, q, per_day) -> float
bs_vega(S, K, r, sigma, T, q, per_percent) -> float
bs_rho(S, K, r, sigma, T, option_type, q, per_percent) -> float
bs_all_greeks(S, K, r, sigma, T, option_type, q) -> dict
"""

from __future__ import annotations

import numpy as np

from option_pricing_horizon.common.math_utils import (
    norm_cdf,
    norm_pdf,
    bs_d1,
    bs_d2,
)


# ---------------------------------------------------------------------------
# Individual Greeks
# ---------------------------------------------------------------------------

def bs_delta(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """Black-Scholes Delta: ∂V/∂S.

    Delta measures the rate of change of the option price with respect to the
    spot price of the underlying.

    Parameters
    ----------
    S, K, r, sigma, T, q : float
        Market and contract parameters.
    option_type : {'call', 'put'}

    Returns
    -------
    float
        Delta in [0, 1] for calls, [−1, 0] for puts.
    """
    d1 = bs_d1(S, K, r, sigma, T, q)
    disc_q = np.exp(-q * T)
    ot = option_type.lower().strip()
    if ot == "call":
        return float(disc_q * norm_cdf(d1))
    elif ot == "put":
        return float(-disc_q * norm_cdf(-d1))
    else:
        raise ValueError(f"option_type must be 'call' or 'put'; got '{option_type}'")


def bs_gamma(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    q: float = 0.0,
) -> float:
    """Black-Scholes Gamma: ∂²V/∂S².

    Gamma is identical for calls and puts (put-call parity).  It measures the
    curvature of the option price with respect to the spot — i.e. how quickly
    Delta changes.

    Parameters
    ----------
    S, K, r, sigma, T, q : float
        Market and contract parameters.

    Returns
    -------
    float
        Gamma (always non-negative).
    """
    d1 = bs_d1(S, K, r, sigma, T, q)
    disc_q = np.exp(-q * T)
    return float(disc_q * norm_pdf(d1) / (S * sigma * np.sqrt(T)))


def bs_theta(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    q: float = 0.0,
    per_day: bool = True,
) -> float:
    """Black-Scholes Theta: ∂V/∂t (time decay).

    A negative Theta means the option loses value as time passes (time decay),
    which is the usual case for long positions.

    Parameters
    ----------
    S, K, r, sigma, T, q : float
        Market and contract parameters.
    option_type : {'call', 'put'}
    per_day : bool
        If True (default), divide by 365 to express Theta per calendar day.
        If False, returns ∂V/∂T (annualised).

    Returns
    -------
    float
        Theta (typically negative for long options).
    """
    d1 = bs_d1(S, K, r, sigma, T, q)
    d2 = bs_d2(S, K, r, sigma, T, q)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    sqrt_T = np.sqrt(T)

    # Common term: time-decay of vega
    common = -S * disc_q * norm_pdf(d1) * sigma / (2.0 * sqrt_T)

    ot = option_type.lower().strip()
    if ot == "call":
        theta = (
            common
            - r * K * disc_r * norm_cdf(d2)
            + q * S * disc_q * norm_cdf(d1)
        )
    elif ot == "put":
        theta = (
            common
            + r * K * disc_r * norm_cdf(-d2)
            - q * S * disc_q * norm_cdf(-d1)
        )
    else:
        raise ValueError(f"option_type must be 'call' or 'put'; got '{option_type}'")

    if per_day:
        theta = theta / 365.0

    return float(theta)


def bs_vega(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    q: float = 0.0,
    per_percent: bool = True,
) -> float:
    """Black-Scholes Vega: ∂V/∂σ.

    Vega measures sensitivity to volatility.  It is the same for calls and
    puts (put-call parity).

    Parameters
    ----------
    S, K, r, sigma, T, q : float
        Market and contract parameters.
    per_percent : bool
        If True (default), report Vega per 1% (0.01) change in σ.
        If False, report as ∂V/∂σ (per unit change).

    Returns
    -------
    float
        Vega (always non-negative).
    """
    d1 = bs_d1(S, K, r, sigma, T, q)
    disc_q = np.exp(-q * T)
    vega = S * disc_q * norm_pdf(d1) * np.sqrt(T)
    if per_percent:
        vega = vega / 100.0
    return float(vega)


def bs_rho(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    q: float = 0.0,
    per_percent: bool = True,
) -> float:
    """Black-Scholes Rho: ∂V/∂r.

    Rho measures sensitivity to the risk-free interest rate.

    Parameters
    ----------
    S, K, r, sigma, T, q : float
        Market and contract parameters.
    option_type : {'call', 'put'}
    per_percent : bool
        If True (default), report Rho per 1% (0.01) change in r.
        If False, return ∂V/∂r (per unit change).

    Returns
    -------
    float
        Rho (positive for calls, negative for puts typically).
    """
    d2 = bs_d2(S, K, r, sigma, T, q)
    disc_r = np.exp(-r * T)

    ot = option_type.lower().strip()
    if ot == "call":
        rho = K * T * disc_r * norm_cdf(d2)
    elif ot == "put":
        rho = -K * T * disc_r * norm_cdf(-d2)
    else:
        raise ValueError(f"option_type must be 'call' or 'put'; got '{option_type}'")

    if per_percent:
        rho = rho / 100.0
    return float(rho)


# ---------------------------------------------------------------------------
# Convenience: all Greeks in one call
# ---------------------------------------------------------------------------

def bs_all_greeks(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    q: float = 0.0,
) -> dict[str, float]:
    """Compute all five Black-Scholes Greeks at once.

    Parameters
    ----------
    S, K, r, sigma, T, q : float
        Market and contract parameters.
    option_type : {'call', 'put'}

    Returns
    -------
    dict
        Keys: 'delta', 'gamma', 'theta', 'vega', 'rho'.
        Theta is per calendar day; Vega and Rho are per 1% change.

    Examples
    --------
    >>> g = bs_all_greeks(100, 100, 0.03, 0.20, 1.0)
    >>> 0 < g['delta'] < 1
    True
    >>> g['gamma'] > 0
    True
    >>> g['theta'] < 0
    True
    """
    return {
        "delta": bs_delta(S, K, r, sigma, T, option_type, q),
        "gamma": bs_gamma(S, K, r, sigma, T, q),
        "theta": bs_theta(S, K, r, sigma, T, option_type, q, per_day=True),
        "vega":  bs_vega(S, K, r, sigma, T, q, per_percent=True),
        "rho":   bs_rho(S, K, r, sigma, T, option_type, q, per_percent=True),
    }
