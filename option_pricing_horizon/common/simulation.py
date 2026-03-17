"""
option_pricing_horizon.common.simulation
=========================================
Numba-JIT accelerated GBM path generator used by all Monte Carlo pricers.

Theory
------
Under the risk-neutral measure Q the stock price follows:

    dS_t = r S_t dt + σ S_t dW_t

The exact discrete solution over a step Δt is:

    S_{t+Δt} = S_t * exp( (r - σ²/2) Δt  +  σ √Δt  Z )

where Z ~ N(0,1).  This Euler-exact (log-Euler) scheme has **no
discretisation bias** for GBM — it is the exact solution, not an
approximation.

Variance-Reduction Techniques
------------------------------
Antithetic Variates
    For every standard-normal draw Z we also simulate a path with −Z.
    The pair (V(Z), V(-Z)) is negatively correlated, so their average has
    variance ≤ half the crude MC variance.  This is applied by default.

Functions
---------
simulate_gbm_paths(S0, r, sigma, T, N, n_paths, seed, antithetic)
    Core path generator.  Returns an array of shape (n_paths, N).
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


# ---------------------------------------------------------------------------
# Numba-JIT core: generates log-returns matrix (no Python overhead per path)
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _gbm_log_returns(
    r: float,
    sigma: float,
    dt: float,
    n_paths: int,
    N: int,
    randn: np.ndarray,
) -> np.ndarray:
    """Compute cumulative log-price increments for every path.

    Parameters
    ----------
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    dt : float
        Time step T/N.
    n_paths : int
        Number of paths.
    N : int
        Number of time steps.
    randn : ndarray, shape (n_paths, N)
        Pre-drawn standard-normal innovations.

    Returns
    -------
    log_prices : ndarray, shape (n_paths, N)
        Cumulative log-price relative to S0 at each monitoring date.
        S_t = S0 * exp(log_prices[path, t]).
    """
    drift = (r - 0.5 * sigma * sigma) * dt
    diffusion = sigma * np.sqrt(dt)
    log_prices = np.empty((n_paths, N), dtype=np.float64)
    for i in prange(n_paths):
        cumsum = 0.0
        for j in range(N):
            cumsum += drift + diffusion * randn[i, j]
            log_prices[i, j] = cumsum
    return log_prices


def simulate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    N: int,
    n_paths: int,
    seed: int = 42,
    antithetic: bool = True,
) -> np.ndarray:
    """Simulate risk-neutral GBM paths using the exact log-Euler discretisation.

    The discretisation is bias-free for GBM:

        S_{t+Δt} = S_t * exp( (r - σ²/2) Δt  +  σ √Δt  Z ),  Z ~ N(0,1)

    Parameters
    ----------
    S0 : float
        Initial stock price.
    r : float
        Risk-free rate (annualised, continuously compounded).
    sigma : float
        Volatility (annualised).
    T : float
        Time to expiry (years).
    N : int
        Number of discrete monitoring steps.
    n_paths : int
        Number of Monte Carlo paths to simulate.
        If ``antithetic=True`` the *effective* number of paths generated is
        ``n_paths`` (antithetic pairs are accounted for internally so the
        returned array always has exactly ``n_paths`` rows).
    seed : int
        NumPy random seed for reproducibility.  Default: 42.
    antithetic : bool
        If ``True`` (default), generate n_paths // 2 standard normal draws
        and stack them with their negatives to halve variance.
        n_paths must be even when antithetic=True (odd values are rounded down).

    Returns
    -------
    paths : ndarray, shape (n_paths, N)
        Simulated stock prices at each monitoring date.
        ``paths[i, j]`` = S at date j+1 on path i.

    Raises
    ------
    ValueError
        If n_paths < 2 when antithetic=True.

    Examples
    --------
    >>> paths = simulate_gbm_paths(100, 0.03, 0.20, 1.0, 252, 10_000, seed=0)
    >>> paths.shape
    (10000, 252)
    >>> (paths > 0).all()
    True
    """
    dt = T / N
    rng = np.random.default_rng(seed)

    if antithetic:
        half = n_paths // 2
        if half < 1:
            raise ValueError("n_paths must be >= 2 when antithetic=True")
        Z_half = rng.standard_normal((half, N))
        Z = np.concatenate([Z_half, -Z_half], axis=0)
    else:
        Z = rng.standard_normal((n_paths, N))

    log_prices = _gbm_log_returns(r, sigma, dt, Z.shape[0], N, Z)
    paths = S0 * np.exp(log_prices)
    return paths
