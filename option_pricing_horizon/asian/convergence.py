"""
option_pricing_horizon.asian.convergence
=========================================
Convergence and variance-reduction analysis for Asian option MC pricing.

Purpose
-------
Demonstrates:
1. How the MC price estimate converges as n_paths increases.
2. The standard-error reduction from antithetic variates and the
   geometric control variate compared to crude MC.
3. The effect of monitoring frequency N on the option price.

Convergence Analysis
--------------------
For a Monte Carlo estimator X̄_M = (1/M) Σᵢ Xᵢ:

    Standard Error  =  σ_X / √M
    95% CI width    =  2 × 1.96 × σ_X / √M  ∝  M^{-1/2}

Empirically we run for a geometric grid of M values and check that
SE ∝ M^{-0.5}, i.e. the log-log slope of (SE vs M) is −0.5.

Variance Reduction Summary
---------------------------
Crude MC:                  Var_crude
+ Antithetic variates:     Var_AV  ≤  Var_crude / 2
+ Geometric control var.:  Var_CV  «  Var_AV    (typ. 10–30× reduction total)

Functions
---------
convergence_study(S, K, r, sigma, T, N, path_grid, seed, option_type)
    Run MC at each M in ``path_grid`` and collect price, SE, CI width.
    Returns a list of result dicts for plotting.

monitoring_freq_study(S, K, r, sigma, T, N_grid, n_paths, seed, option_type)
    Price arithmetic and geometric Asian options across a range of N values.
    Shows how discrete monitoring frequency affects price.

variance_reduction_comparison(S, K, r, sigma, T, N, n_paths, seed, option_type)
    Compare SE of crude MC, antithetic-only, and full CV estimators.
    Returns a summary dict.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from option_pricing_horizon.asian.monte_carlo import arithmetic_asian_mc, ArithmeticMCResult
from option_pricing_horizon.asian.geometric import geometric_asian_price


# ---------------------------------------------------------------------------
# Convergence study: price vs number of paths
# ---------------------------------------------------------------------------

def convergence_study(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    N: int = 252,
    path_grid: list[int] | None = None,
    seed: int = 42,
    option_type: str = "call",
) -> list[dict[str, Any]]:
    """Run MC pricer at increasing path counts and collect convergence metrics.

    For each M in ``path_grid`` we run the full control-variate MC pricer
    and record: price estimate, standard error, CI bounds.

    Parameters
    ----------
    S, K, r, sigma, T, N : float / int
        Market and contract parameters.
    path_grid : list[int], optional
        List of path counts to evaluate.  Defaults to a log-spaced grid:
        [500, 1000, 2000, 5000, 10000, 25000, 50000, 100000].
    seed : int
        NumPy random seed (same seed used at every M for fair comparison).
    option_type : {'call', 'put'}

    Returns
    -------
    list of dict
        Each dict has keys:
        - 'n_paths'    : int
        - 'price'      : float   (control-variate price)
        - 'price_crude': float   (crude MC price)
        - 'std_err'    : float   (SE of CV estimator)
        - 'ci_lower'   : float
        - 'ci_upper'   : float
        - 'ci_width'   : float
        - 'vr_ratio'   : float   (variance reduction ratio)

    Examples
    --------
    >>> from option_pricing_horizon.asian.convergence import convergence_study
    >>> results = convergence_study(100, 100, 0.03, 0.20, 1.0, N=252,
    ...                             path_grid=[1000, 5000, 20000])
    >>> len(results) == 3
    True
    >>> results[-1]['std_err'] < results[0]['std_err']
    True
    """
    if path_grid is None:
        path_grid = [500, 1_000, 2_000, 5_000, 10_000, 25_000, 50_000, 100_000]

    records = []
    for M in path_grid:
        res: ArithmeticMCResult = arithmetic_asian_mc(
            S=S, K=K, r=r, sigma=sigma, T=T, N=N,
            n_paths=M, seed=seed,
            option_type=option_type,
            use_cv=True,
            use_antithetic=True,
        )
        records.append({
            "n_paths":     res.n_paths,
            "price":       res.price,
            "price_crude": res.price_crude,
            "std_err":     res.std_err,
            "ci_lower":    res.ci_lower,
            "ci_upper":    res.ci_upper,
            "ci_width":    res.ci_upper - res.ci_lower,
            "vr_ratio":    res.variance_reduction_ratio,
        })
    return records


# ---------------------------------------------------------------------------
# Monitoring-frequency study: price vs N
# ---------------------------------------------------------------------------

def monitoring_freq_study(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    N_grid: list[int] | None = None,
    n_paths: int = 100_000,
    seed: int = 42,
    option_type: str = "call",
) -> list[dict[str, Any]]:
    """Study how monitoring frequency N affects arithmetic and geometric Asian prices.

    As N → ∞ the discrete average converges to the continuous average, and
    the gap between arithmetic and geometric Asian prices narrows.

    Parameters
    ----------
    S, K, r, sigma, T : float
        Market and contract parameters.
    N_grid : list[int], optional
        Monitoring date counts to evaluate.
        Defaults to [1, 2, 4, 12, 52, 126, 252].
    n_paths : int
        MC paths for the arithmetic price (default 100,000).
    seed : int
        Random seed.
    option_type : {'call', 'put'}

    Returns
    -------
    list of dict
        Each dict has keys:
        - 'N'           : int
        - 'arith_price' : float  (arithmetic Asian MC price)
        - 'geo_price'   : float  (geometric Asian exact price)
        - 'vanilla_price': float (European BS price, for reference)
        - 'arith_se'    : float
    """
    if N_grid is None:
        N_grid = [1, 2, 4, 12, 52, 126, 252]

    from option_pricing_horizon.european.black_scholes import bs_price

    records = []
    for N in N_grid:
        arith = arithmetic_asian_mc(
            S=S, K=K, r=r, sigma=sigma, T=T, N=N,
            n_paths=n_paths, seed=seed,
            option_type=option_type,
            use_cv=True, use_antithetic=True,
        )
        geo_price = geometric_asian_price(S, K, r, sigma, T, N, option_type)
        vanilla   = bs_price(S, K, r, sigma, T, option_type)

        records.append({
            "N":            N,
            "arith_price":  arith.price,
            "geo_price":    geo_price,
            "vanilla_price": vanilla,
            "arith_se":     arith.std_err,
        })
    return records


# ---------------------------------------------------------------------------
# Variance reduction comparison
# ---------------------------------------------------------------------------

def variance_reduction_comparison(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    N: int = 252,
    n_paths: int = 100_000,
    seed: int = 42,
    option_type: str = "call",
) -> dict[str, Any]:
    """Compare standard errors across three MC estimator configurations.

    Configurations compared
    -----------------------
    1. Crude MC:             use_cv=False, use_antithetic=False
    2. Antithetic only:      use_cv=False, use_antithetic=True
    3. Antithetic + CV:      use_cv=True,  use_antithetic=True   (recommended)

    Parameters
    ----------
    S, K, r, sigma, T, N : float / int
        Market and contract parameters.
    n_paths : int
        Number of MC paths for all three runs.
    seed : int
        Same seed used for all three runs.
    option_type : {'call', 'put'}

    Returns
    -------
    dict
        Keys:
        - 'crude'          : ArithmeticMCResult
        - 'antithetic'     : ArithmeticMCResult
        - 'cv'             : ArithmeticMCResult
        - 'se_crude'       : float
        - 'se_antithetic'  : float
        - 'se_cv'          : float
        - 'speedup_av_vs_crude'  : float  (variance ratio)
        - 'speedup_cv_vs_crude'  : float  (variance ratio)
        - 'speedup_cv_vs_av'     : float  (variance ratio)

    Examples
    --------
    >>> from option_pricing_horizon.asian.convergence import variance_reduction_comparison
    >>> vrc = variance_reduction_comparison(100, 100, 0.03, 0.20, 1.0, N=252, n_paths=50_000)
    >>> vrc['se_cv'] < vrc['se_antithetic'] < vrc['se_crude']
    True
    """
    crude = arithmetic_asian_mc(
        S=S, K=K, r=r, sigma=sigma, T=T, N=N,
        n_paths=n_paths, seed=seed, option_type=option_type,
        use_cv=False, use_antithetic=False,
    )
    antithetic = arithmetic_asian_mc(
        S=S, K=K, r=r, sigma=sigma, T=T, N=N,
        n_paths=n_paths, seed=seed, option_type=option_type,
        use_cv=False, use_antithetic=True,
    )
    cv = arithmetic_asian_mc(
        S=S, K=K, r=r, sigma=sigma, T=T, N=N,
        n_paths=n_paths, seed=seed, option_type=option_type,
        use_cv=True, use_antithetic=True,
    )

    se_c  = crude.std_err
    se_av = antithetic.std_err
    se_cv = cv.std_err

    return {
        "crude":       crude,
        "antithetic":  antithetic,
        "cv":          cv,
        "se_crude":    se_c,
        "se_antithetic": se_av,
        "se_cv":       se_cv,
        "speedup_av_vs_crude": float((se_c / se_av) ** 2) if se_av > 0 else np.nan,
        "speedup_cv_vs_crude": float((se_c / se_cv) ** 2) if se_cv > 0 else np.nan,
        "speedup_cv_vs_av":    float((se_av / se_cv) ** 2) if se_cv > 0 else np.nan,
    }
