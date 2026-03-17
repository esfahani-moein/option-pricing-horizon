"""
option_pricing_horizon.common.math_utils
=========================================
High-precision mathematical utilities shared by all pricing modules.

All normal CDF / PDF evaluations use ``scipy.special.ndtr`` and
``scipy.special.ndtri``, which are implemented in C and match the precision
of the Abramowitz & Stegun approximations to machine epsilon (~1e-15).

Functions
---------
norm_cdf(x)
    Cumulative distribution function of N(0,1).
norm_pdf(x)
    Probability density function of N(0,1).
lognorm_mean(mu, sigma_sq)
    Mean of a log-normal variable: exp(mu + sigma_sq / 2).
lognorm_variance(mu, sigma_sq)
    Variance of a log-normal variable.
lognorm_second_moment(mu, sigma_sq)
    E[X^2] for a log-normal variable.
bs_d1(S, K, r, sigma, T, q=0.0)
    Black-Scholes d1 quantity.
bs_d2(S, K, r, sigma, T, q=0.0)
    Black-Scholes d2 quantity.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr as _ndtr   # machine-precision normal CDF


# ---------------------------------------------------------------------------
# Normal distribution helpers
# ---------------------------------------------------------------------------

def norm_cdf(x: float | np.ndarray) -> float | np.ndarray:
    """Cumulative distribution function of the standard normal N(0,1).

    Uses ``scipy.special.ndtr`` for machine-precision accuracy (~1e-15).

    Parameters
    ----------
    x : float or ndarray
        Argument(s).

    Returns
    -------
    float or ndarray
        Φ(x) = P(Z ≤ x) for Z ~ N(0,1).

    Examples
    --------
    >>> from option_pricing_horizon.common.math_utils import norm_cdf
    >>> abs(norm_cdf(0.0) - 0.5) < 1e-15
    True
    >>> abs(norm_cdf(1.959964) - 0.975) < 1e-6
    True
    """
    return _ndtr(x)


def norm_pdf(x: float | np.ndarray) -> float | np.ndarray:
    """Probability density function of the standard normal N(0,1).

    φ(x) = (1 / sqrt(2π)) * exp(-x² / 2)

    Parameters
    ----------
    x : float or ndarray
        Argument(s).

    Returns
    -------
    float or ndarray
        φ(x).
    """
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


# ---------------------------------------------------------------------------
# Log-normal moment utilities
# ---------------------------------------------------------------------------

def lognorm_mean(mu: float, sigma_sq: float) -> float:
    """Mean of a log-normal random variable X = exp(Y), Y ~ N(mu, sigma_sq).

    E[X] = exp(mu + sigma_sq / 2)

    Parameters
    ----------
    mu : float
        Mean of the underlying normal (log-space).
    sigma_sq : float
        Variance of the underlying normal (log-space).

    Returns
    -------
    float
        E[X].
    """
    return float(np.exp(mu + 0.5 * sigma_sq))


def lognorm_variance(mu: float, sigma_sq: float) -> float:
    """Variance of a log-normal random variable X = exp(Y), Y ~ N(mu, sigma_sq).

    Var[X] = (exp(sigma_sq) - 1) * exp(2*mu + sigma_sq)

    Parameters
    ----------
    mu : float
        Mean of the underlying normal (log-space).
    sigma_sq : float
        Variance of the underlying normal (log-space).

    Returns
    -------
    float
        Var[X].
    """
    return float((np.exp(sigma_sq) - 1.0) * np.exp(2.0 * mu + sigma_sq))


def lognorm_second_moment(mu: float, sigma_sq: float) -> float:
    """Second moment E[X²] of a log-normal X = exp(Y), Y ~ N(mu, sigma_sq).

    E[X²] = exp(2*mu + 2*sigma_sq)

    Parameters
    ----------
    mu : float
        Mean of the underlying normal.
    sigma_sq : float
        Variance of the underlying normal.

    Returns
    -------
    float
        E[X²].
    """
    return float(np.exp(2.0 * mu + 2.0 * sigma_sq))


# ---------------------------------------------------------------------------
# Black-Scholes d1 / d2
# ---------------------------------------------------------------------------

def bs_d1(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    q: float = 0.0,
) -> float:
    """Compute Black-Scholes d1.

    d1 = [ln(S/K) + (r - q + σ²/2) * T] / (σ * √T)

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Time to expiry.
    q : float
        Continuous dividend yield (default 0).

    Returns
    -------
    float
        d1 value.
    """
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_d2(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    q: float = 0.0,
) -> float:
    """Compute Black-Scholes d2.

    d2 = d1 - σ * √T

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Time to expiry.
    q : float
        Continuous dividend yield (default 0).

    Returns
    -------
    float
        d2 value.
    """
    return bs_d1(S, K, r, sigma, T, q) - sigma * np.sqrt(T)
