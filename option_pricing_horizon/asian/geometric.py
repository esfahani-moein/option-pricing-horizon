"""
option_pricing_horizon.asian.geometric
=======================================
Exact closed-form pricing of discrete geometric Asian options.

Theory
------
Under risk-neutral GBM  dS_t = r S_t dt + σ S_t dW_t  the stock price at
monitoring date tᵢ = i·Δt is:

    S_{tᵢ} = S₀ · exp( (r - σ²/2) tᵢ  +  σ Wᵢ )

The discrete geometric average of N monitoring dates is:

    A_geo = ( ∏ᵢ₌₁ᴺ S_{tᵢ} )^{1/N}
           = S₀ · exp( (1/N) Σᵢ [ (r - σ²/2) tᵢ + σ Wᵢ ] )

Key observation: (1/N) Σᵢ tᵢ is deterministic and (1/N) Σᵢ Wᵢ is
normally distributed, so ln(A_geo) is also normally distributed.

Step 1 – Derive the mean μ_G and variance σ_G² of ln(A_geo / S₀)
------------------------------------------------------------------
Mean of ln(A_geo / S₀):

    μ_G = (r - σ²/2) · (1/N) Σᵢ₌₁ᴺ tᵢ
        = (r - σ²/2) · Δt · (N+1)/2              [arithmetic series]
        = (r - σ²/2) · T · (N+1) / (2N)

Variance of ln(A_geo / S₀)  (using Cov[Wᵢ, Wⱼ] = min(tᵢ, tⱼ)):

    σ_G² = σ² · (1/N²) Σᵢ Σⱼ min(tᵢ, tⱼ)

For equally-spaced dates tᵢ = i·Δt:

    Σᵢ Σⱼ min(i,j) = N(N+1)(2N+1)/6   (combinatorial identity)

    ⟹  σ_G² = σ² · Δt² · (N+1)(2N+1) / (6N)
             = σ² · T² / N² · (N+1)(2N+1) / 6
             = σ² · T · (N+1)(2N+1) / (6N²)

Step 2 – Black-Scholes-style formula
--------------------------------------
Since A_geo is log-normal with parameters (ln S₀ + μ_G, σ_G²), we apply
the Black-Scholes formula with adjusted parameters:

    F_G  = S₀ · exp(μ_G)           ← adjusted forward of A_geo
    σ̂_G  = σ_G / √T                ← adjusted volatility (annualised)

    d₁_G = [ ln(F_G / K) + σ_G²/2 ] / σ_G
    d₂_G = d₁_G − σ_G

    C_geo = e^{-rT} [ F_G · Φ(d₁_G) − K · Φ(d₂_G) ]
    P_geo = e^{-rT} [ K · Φ(−d₂_G) − F_G · Φ(−d₁_G) ]

Reference: Kemna & Vorst (1990), "A Pricing Method for Options Based on
Average Asset Values", Journal of Banking & Finance.

Functions
---------
geometric_asian_call(S, K, r, sigma, T, N) -> float
geometric_asian_put(S, K, r, sigma, T, N) -> float
geometric_asian_price(S, K, r, sigma, T, N, option_type) -> float
geo_asian_adjusted_params(S, K, r, sigma, T, N) -> dict
"""

from __future__ import annotations

import numpy as np

from option_pricing_horizon.common.math_utils import norm_cdf


# ---------------------------------------------------------------------------
# Internal helper: compute adjusted GBM parameters for the geometric average
# ---------------------------------------------------------------------------

def geo_asian_adjusted_params(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    N: int,
) -> dict[str, float]:
    """Compute the log-normal parameters of the discrete geometric average.

    Returns the adjusted forward price ``F_G`` and total log-volatility
    ``sigma_G`` such that:

        ln(A_geo) ~ N( ln(F_G) − σ_G²/2,  σ_G² )

    Parameters
    ----------
    S : float
        Spot price S₀.
    K : float
        Strike (used only for documentation; not part of σ_G).
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Time to expiry.
    N : int
        Number of discrete monitoring dates.

    Returns
    -------
    dict with keys:
        'mu_G'    : float  — mean of ln(A_geo / S₀)
        'sigma_G' : float  — total log-volatility of A_geo (= σ_G, NOT annualised)
        'F_G'     : float  — adjusted forward: S₀ · exp(μ_G + r·T)
                            (= discounted expectation of A_geo under risk-neutral Q,
                             corrected for the e^{-rT} discount in the option price)
        'sigma_G_ann' : float — σ_G / √T (annualised adjusted volatility for display)
    """
    dt = T / N

    # Mean of ln(A_geo / S₀)   [see theory in module docstring]
    mu_G = (r - 0.5 * sigma**2) * dt * (N + 1) / 2.0

    # Variance of ln(A_geo / S₀)
    sigma_G_sq = sigma**2 * dt**2 * (N + 1) * (2 * N + 1) / 6.0
    sigma_G = np.sqrt(sigma_G_sq)

    # Risk-neutral expectation of A_geo = S₀ · exp(μ_G + σ_G²/2)
    # (this is E_Q[A_geo], NOT discounted)
    F_G = S * np.exp(mu_G + 0.5 * sigma_G_sq)

    return {
        "mu_G": float(mu_G),
        "sigma_G": float(sigma_G),
        "sigma_G_sq": float(sigma_G_sq),
        "F_G": float(F_G),
        "sigma_G_ann": float(sigma_G / np.sqrt(T)),
    }


# ---------------------------------------------------------------------------
# Closed-form geometric Asian option prices
# ---------------------------------------------------------------------------

def geometric_asian_call(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    N: int = 252,
) -> float:
    """Exact closed-form price of a discrete geometric Asian call option.

    Formula (Kemna & Vorst 1990, discrete-monitoring extension):

        d₁_G = [ ln(F_G / K) + σ_G² / 2 ] / σ_G
        d₂_G = d₁_G − σ_G

        C_geo = e^{-rT} [ F_G · Φ(d₁_G) − K · Φ(d₂_G) ]

    where F_G and σ_G are the adjusted forward and total log-volatility of
    the discrete geometric average (see :func:`geo_asian_adjusted_params`).

    Parameters
    ----------
    S : float
        Spot price S₀.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility.
    T : float
        Time to expiry (years).
    N : int
        Number of discrete monitoring dates (default 252).

    Returns
    -------
    float
        Geometric Asian call price.

    Examples
    --------
    >>> from option_pricing_horizon.asian.geometric import geometric_asian_call
    >>> price = geometric_asian_call(100, 100, 0.03, 0.20, 1.0, 252)
    >>> 0 < price < 10   # should be cheaper than ATM European call
    True
    """
    params = geo_asian_adjusted_params(S, K, r, sigma, T, N)
    F_G = params["F_G"]
    sigma_G = params["sigma_G"]
    disc = np.exp(-r * T)

    if sigma_G < 1e-14:
        # Zero volatility: intrinsic value only
        return float(disc * max(F_G - K, 0.0))

    d1 = (np.log(F_G / K) + 0.5 * sigma_G**2) / sigma_G
    d2 = d1 - sigma_G

    return float(disc * (F_G * norm_cdf(d1) - K * norm_cdf(d2)))


def geometric_asian_put(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    N: int = 252,
) -> float:
    """Exact closed-form price of a discrete geometric Asian put option.

    Formula:

        P_geo = e^{-rT} [ K · Φ(−d₂_G) − F_G · Φ(−d₁_G) ]

    Parameters
    ----------
    S : float
        Spot price S₀.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Time to expiry (years).
    N : int
        Number of discrete monitoring dates (default 252).

    Returns
    -------
    float
        Geometric Asian put price.
    """
    params = geo_asian_adjusted_params(S, K, r, sigma, T, N)
    F_G = params["F_G"]
    sigma_G = params["sigma_G"]
    disc = np.exp(-r * T)

    if sigma_G < 1e-14:
        return float(disc * max(K - F_G, 0.0))

    d1 = (np.log(F_G / K) + 0.5 * sigma_G**2) / sigma_G
    d2 = d1 - sigma_G

    return float(disc * (K * norm_cdf(-d2) - F_G * norm_cdf(-d1)))


def geometric_asian_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    N: int = 252,
    option_type: str = "call",
) -> float:
    """Dispatch to geometric Asian call or put.

    Parameters
    ----------
    S, K, r, sigma, T, N : see :func:`geometric_asian_call`.
    option_type : {'call', 'put'}

    Returns
    -------
    float
        Option price.

    Raises
    ------
    ValueError
        If option_type is not 'call' or 'put'.
    """
    ot = option_type.lower().strip()
    if ot == "call":
        return geometric_asian_call(S, K, r, sigma, T, N)
    elif ot == "put":
        return geometric_asian_put(S, K, r, sigma, T, N)
    else:
        raise ValueError(
            f"option_type must be 'call' or 'put'; got '{option_type}'"
        )
