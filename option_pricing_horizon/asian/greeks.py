"""
option_pricing_horizon.asian.greeks
=====================================
Greeks estimation for Asian options via finite-difference bump-and-reprice.

Theory
------
No closed-form Greeks exist for arithmetic Asian options under GBM.
We estimate them via the **finite-difference (bump-and-reprice)** method:

Delta (∂C/∂S)
    Central-difference estimator:
        Δ ≈ [ C(S₀+h) − C(S₀−h) ] / (2h)
    where h = ε · S₀  (typically ε = 0.01, i.e. 1% bump).
    We fix the random seed across both evaluations so that the same Brownian
    paths are used — this is the **common random numbers (CRN)** technique,
    which dramatically reduces estimator variance.

Vega (∂C/∂σ)
    Central-difference estimator:
        V ≈ [ C(σ+h) − C(σ−h) ] / (2h)
    with h = ε · σ  (typically ε = 0.01).
    Reported per 1 % (0.01) change in σ.

Rho (∂C/∂r)
    Central-difference estimator (optional):
        ρ ≈ [ C(r+h) − C(r−h) ] / (2h)
    with h = 0.001 (10 bps).

Note on CRN
-----------
We pass the same ``seed`` to all three MC calls inside each Greek estimator.
The path simulator ensures Z and −Z are generated from a single common seed,
so numerically the paths differ only because of the parameter bump, not
random noise.  This gives variance of the Greek estimator ∝ h² rather than
independent MC noise.

Functions
---------
asian_delta(S, K, r, sigma, T, N, n_paths, seed, option_type, eps) -> float
asian_vega(S, K, r, sigma, T, N, n_paths, seed, option_type, eps) -> float
asian_rho(S, K, r, sigma, T, N, n_paths, seed, option_type, h) -> float
asian_all_greeks(S, K, r, sigma, T, N, n_paths, seed, option_type) -> dict
"""

from __future__ import annotations

import numpy as np

from option_pricing_horizon.asian.monte_carlo import arithmetic_asian_mc


# ---------------------------------------------------------------------------
# Helper: extract price from MC result
# ---------------------------------------------------------------------------

def _mc_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    N: int,
    n_paths: int,
    seed: int,
    option_type: str,
) -> float:
    """Run MC and return the control-variate corrected price."""
    result = arithmetic_asian_mc(
        S=S, K=K, r=r, sigma=sigma, T=T, N=N,
        n_paths=n_paths, seed=seed,
        option_type=option_type,
        use_cv=True,
        use_antithetic=True,
    )
    return result.price


# ---------------------------------------------------------------------------
# Delta
# ---------------------------------------------------------------------------

def asian_delta(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    N: int = 252,
    n_paths: int = 100_000,
    seed: int = 42,
    option_type: str = "call",
    eps: float = 0.01,
) -> float:
    """Estimate Delta of an arithmetic Asian option via central finite difference.

    Uses Common Random Numbers (CRN) — same seed for all three MC evaluations —
    so that the estimator noise comes from the bump, not Monte Carlo variance.

    Formula
    -------
        Δ ≈ [ C(S₀(1+ε)) − C(S₀(1−ε)) ] / (2 · ε · S₀)

    Parameters
    ----------
    S : float
        Spot price S₀.
    K, r, sigma, T, N : float / int
        Market and contract parameters.
    n_paths : int
        Number of MC paths per evaluation (default 100,000).
    seed : int
        Random seed — same seed used for up, base, and down bumps.
    option_type : {'call', 'put'}
    eps : float
        Fractional bump size (default 0.01 = 1 %).

    Returns
    -------
    float
        Estimated Delta.
    """
    h = eps * S
    price_up   = _mc_price(S + h, K, r, sigma, T, N, n_paths, seed, option_type)
    price_down = _mc_price(S - h, K, r, sigma, T, N, n_paths, seed, option_type)
    return float((price_up - price_down) / (2.0 * h))


# ---------------------------------------------------------------------------
# Vega
# ---------------------------------------------------------------------------

def asian_vega(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    N: int = 252,
    n_paths: int = 100_000,
    seed: int = 42,
    option_type: str = "call",
    eps: float = 0.01,
) -> float:
    """Estimate Vega of an arithmetic Asian option via central finite difference.

    Formula
    -------
        V ≈ [ C(σ(1+ε)) − C(σ(1−ε)) ] / (2 · ε · σ)

    Reported per 1 % absolute change in σ (i.e. divided by 100 so a 1 vol-point
    increase corresponds to +1 unit of the returned value).

    Parameters
    ----------
    S, K, r, sigma, T, N : float / int
        Market and contract parameters.
    n_paths : int
        Number of MC paths per evaluation.
    seed : int
        Random seed (CRN applied).
    option_type : {'call', 'put'}
    eps : float
        Fractional bump on σ (default 0.01 = 1 %).

    Returns
    -------
    float
        Estimated Vega per 1 % change in σ.
    """
    h = eps * sigma
    price_up   = _mc_price(S, K, r, sigma + h, T, N, n_paths, seed, option_type)
    price_down = _mc_price(S, K, r, sigma - h, T, N, n_paths, seed, option_type)
    # Central difference, then normalise to "per 1% of sigma"
    raw_vega = (price_up - price_down) / (2.0 * h)
    return float(raw_vega / 100.0)


# ---------------------------------------------------------------------------
# Rho
# ---------------------------------------------------------------------------

def asian_rho(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    N: int = 252,
    n_paths: int = 100_000,
    seed: int = 42,
    option_type: str = "call",
    h: float = 1e-3,
) -> float:
    """Estimate Rho of an arithmetic Asian option via central finite difference.

    Formula
    -------
        ρ ≈ [ C(r+h) − C(r−h) ] / (2h)

    Reported per 1 % change in r (divided by 100).

    Parameters
    ----------
    S, K, r, sigma, T, N : float / int
        Market and contract parameters.
    n_paths : int
        MC paths per evaluation.
    seed : int
        Random seed (CRN applied).
    option_type : {'call', 'put'}
    h : float
        Absolute bump on r (default 0.001 = 10 bps).

    Returns
    -------
    float
        Estimated Rho per 1 % change in r.
    """
    price_up   = _mc_price(S, K, r + h, sigma, T, N, n_paths, seed, option_type)
    price_down = _mc_price(S, K, r - h, sigma, T, N, n_paths, seed, option_type)
    raw_rho = (price_up - price_down) / (2.0 * h)
    return float(raw_rho / 100.0)


# ---------------------------------------------------------------------------
# All Greeks convenience function
# ---------------------------------------------------------------------------

def asian_all_greeks(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    N: int = 252,
    n_paths: int = 100_000,
    seed: int = 42,
    option_type: str = "call",
) -> dict[str, float]:
    """Estimate Delta, Vega, and Rho for an arithmetic Asian option.

    Uses central finite differences with Common Random Numbers (CRN) and
    the geometric control-variate MC pricer for high accuracy.

    Parameters
    ----------
    S, K, r, sigma, T, N : float / int
        Market and contract parameters.
    n_paths : int
        Number of MC paths per pricing call (default 100,000).
    seed : int
        Random seed.
    option_type : {'call', 'put'}

    Returns
    -------
    dict
        Keys: 'delta', 'vega', 'rho'.
        Vega and Rho are per 1 % change in σ and r respectively.

    Examples
    --------
    >>> from option_pricing_horizon.asian.greeks import asian_all_greeks
    >>> g = asian_all_greeks(100, 100, 0.03, 0.20, 1.0, N=252, n_paths=50_000)
    >>> 0 < g['delta'] < 1
    True
    """
    return {
        "delta": asian_delta(S, K, r, sigma, T, N, n_paths, seed, option_type),
        "vega":  asian_vega(S, K, r, sigma, T, N, n_paths, seed, option_type),
        "rho":   asian_rho(S, K, r, sigma, T, N, n_paths, seed, option_type),
    }
