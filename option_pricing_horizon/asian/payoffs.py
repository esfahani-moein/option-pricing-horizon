"""
option_pricing_horizon.asian.payoffs
=====================================
Discrete-monitoring payoff functions for Asian (average-price) options.

Product Specification
---------------------
An Asian option's payoff depends on the **average** of the underlying asset
price observed at N discrete monitoring dates t₁ < t₂ < … < t_N = T.

Two averaging conventions are standard:

Arithmetic Average
    A_arith = (1/N) Σᵢ S_{tᵢ}

    Payoff of arithmetic Asian call: max(A_arith − K, 0)
    Payoff of arithmetic Asian put:  max(K − A_arith, 0)

    No closed-form exists under GBM ⟶ numerical method required.

Geometric Average
    A_geo = (∏ᵢ S_{tᵢ})^{1/N} = exp( (1/N) Σᵢ ln S_{tᵢ} )

    Payoff of geometric Asian call: max(A_geo − K, 0)
    Payoff of geometric Asian put:  max(K − A_geo, 0)

    Under GBM the log of A_geo is normally distributed ⟶ closed-form
    exists (see geometric.py).

Why Asian Options?
------------------
The averaging feature makes Asian options:
  - Cheaper than vanilla Europeans (averaging smooths out extremes)
  - Less susceptible to price manipulation at expiry
  - More suitable for commodities / FX where average prices matter
    (e.g. monthly oil revenue, quarterly fuel cost)

Functions
---------
arithmetic_average_payoff(paths, K, option_type) -> ndarray
    Discounted payoff for each simulated path using arithmetic average.
geometric_average_payoff(paths, K, option_type) -> ndarray
    Discounted payoff for each simulated path using geometric average.
"""

from __future__ import annotations

import numpy as np


def arithmetic_average_payoff(
    paths: np.ndarray,
    K: float,
    option_type: str = "call",
) -> np.ndarray:
    """Compute the arithmetic-average Asian payoff for each simulated path.

    The payoff is computed **before discounting** (undiscounted):

        Payoff_call[i] = max( A_arith[i] − K,  0 )
        Payoff_put[i]  = max( K − A_arith[i],  0 )

    where A_arith[i] = (1/N) Σⱼ paths[i, j].

    Parameters
    ----------
    paths : ndarray, shape (n_paths, N)
        Simulated underlying price paths from :func:`~common.simulation.simulate_gbm_paths`.
        Each row is one path; columns are monitoring dates.
    K : float
        Strike price.
    option_type : {'call', 'put'}
        Option type (case-insensitive).

    Returns
    -------
    payoffs : ndarray, shape (n_paths,)
        Undiscounted payoff for each path.

    Raises
    ------
    ValueError
        If ``option_type`` is not 'call' or 'put'.
    """
    A = paths.mean(axis=1)   # arithmetic average, shape (n_paths,)
    ot = option_type.lower().strip()
    if ot == "call":
        return np.maximum(A - K, 0.0)
    elif ot == "put":
        return np.maximum(K - A, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put'; got '{option_type}'")


def geometric_average_payoff(
    paths: np.ndarray,
    K: float,
    option_type: str = "call",
) -> np.ndarray:
    """Compute the geometric-average Asian payoff for each simulated path.

    The geometric average is computed in log-space to avoid numerical overflow:

        A_geo[i] = exp( (1/N) Σⱼ ln paths[i, j] )

    Payoffs (before discounting):

        Payoff_call[i] = max( A_geo[i] − K,  0 )
        Payoff_put[i]  = max( K − A_geo[i],  0 )

    Parameters
    ----------
    paths : ndarray, shape (n_paths, N)
        Simulated underlying price paths.
    K : float
        Strike price.
    option_type : {'call', 'put'}
        Option type (case-insensitive).

    Returns
    -------
    payoffs : ndarray, shape (n_paths,)
        Undiscounted payoff for each path.

    Notes
    -----
    Computing in log-space (``np.log`` → mean → ``np.exp``) is numerically
    superior to computing the product directly (which can underflow/overflow
    for large N or extreme σ).
    """
    log_paths = np.log(paths)
    A_geo = np.exp(log_paths.mean(axis=1))   # geometric average, shape (n_paths,)
    ot = option_type.lower().strip()
    if ot == "call":
        return np.maximum(A_geo - K, 0.0)
    elif ot == "put":
        return np.maximum(K - A_geo, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put'; got '{option_type}'")
