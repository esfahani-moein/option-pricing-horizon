"""
option_pricing_horizon.asian.monte_carlo
=========================================
Monte Carlo pricing of arithmetic average Asian options with variance
reduction via antithetic variates and a geometric average control variate.

Theory
------
No closed-form solution exists for the arithmetic Asian option under GBM.
We therefore estimate the price as a risk-neutral expectation:

    C_arith = e^{-rT} · E_Q[ max(A_arith − K, 0) ]

    where A_arith = (1/N) Σᵢ S_{tᵢ}

We approximate this expectation by averaging over M simulated paths:

    Ĉ_crude = e^{-rT} · (1/M) Σₘ max(A_arith[m] − K, 0)

Variance Reduction Techniques
-------------------------------

1. Antithetic Variates
~~~~~~~~~~~~~~~~~~~~~~
For every standard-normal draw Z we also simulate −Z.  The arithmetic
average payoffs of the two paths are negatively correlated so their
mean has variance ≤ half that of crude MC.

    Corr[V(Z), V(−Z)] < 0   ⟹   Var[(V(Z)+V(−Z))/2] ≤ Var[V(Z)]/2

This is applied transparently by the path simulator.

2. Geometric Control Variate (Kemna & Vorst 1990)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Let G[m] = geometric average payoff for path m and Ĝ_exact be the
exact analytical geometric Asian price from the closed-form formula.

The control-variate estimator is:

    Ĉ_CV = Ĉ_crude + β̂ · (Ĝ_exact − Ĝ_mc)

where
    Ĝ_mc  = e^{-rT} · (1/M) Σₘ max(A_geo[m] − K, 0)
    β̂     = Cov[arith payoff, geo payoff] / Var[geo payoff]   (OLS)
    Ĝ_exact = closed-form geometric Asian price

Intuition: arithmetic and geometric averages are highly correlated.
Any MC over-estimate in the geometric payoff is likely mirrored in
the arithmetic payoff, so we correct for it.

Typical variance reduction: 10x–30x (i.e. equivalent to 100–900x more paths).

Algorithm Steps
---------------
1. Draw M / 2 standard-normal matrices Z of shape (M/2, N).
2. Stack [Z; −Z] for antithetic pairs → shape (M, N).
3. Apply log-Euler discretisation to get price paths.
4. Compute arithmetic payoffs  a[m] = max(A_arith[m] − K, 0).
5. Compute geometric payoffs   g[m] = max(A_geo[m]   − K, 0).
6. Discount:  payoff → e^{-rT} × payoff.
7. Compute OLS β̂ using sample covariance and variance.
8. Look up exact geometric price Ĝ_exact from closed-form.
9. Output: Ĉ_CV = mean(a) + β̂ · (Ĝ_exact − mean(g))
10. Compute 95% CI on Ĉ_CV using se_CV = std(a − β̂ g) / √M.

Computational Complexity
------------------------
Path simulation: O(M × N) with Numba parallelism ⟶ scales linearly.
Payoff computation: O(M × N) for mean; O(M) for payoff.
OLS: O(M) (single-pass covariance).

Functions
---------
arithmetic_asian_mc(S, K, r, sigma, T, N, n_paths, seed, use_cv, use_antithetic)
    Main pricing function.  Returns :class:`ArithmeticMCResult`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from option_pricing_horizon.common.simulation import simulate_gbm_paths
from option_pricing_horizon.asian.payoffs import (
    arithmetic_average_payoff,
    geometric_average_payoff,
)
from option_pricing_horizon.asian.geometric import geometric_asian_call, geometric_asian_put


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ArithmeticMCResult:
    """Container for arithmetic Asian MC pricing output.

    Attributes
    ----------
    price : float
        Estimated option price (control-variate corrected if ``use_cv=True``).
    price_crude : float
        Crude MC price (no control variate correction).
    std_err : float
        Standard error of the price estimate.
    ci_lower : float
        Lower bound of the 95% confidence interval.
    ci_upper : float
        Upper bound of the 95% confidence interval.
    beta_cv : float
        Fitted OLS control-variate coefficient β̂.
        Only meaningful when ``use_cv=True``; set to NaN otherwise.
    n_paths : int
        Number of simulated paths used.
    variance_reduction_ratio : float
        Estimated variance reduction vs crude MC (VR = Var_crude / Var_CV).
        Only computed when ``use_cv=True``; set to NaN otherwise.
    """

    price: float
    price_crude: float
    std_err: float
    ci_lower: float
    ci_upper: float
    beta_cv: float
    n_paths: int
    variance_reduction_ratio: float

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"ArithmeticMCResult(\n"
            f"  price            = {self.price:.6f}\n"
            f"  price_crude      = {self.price_crude:.6f}\n"
            f"  std_err          = {self.std_err:.6f}\n"
            f"  95% CI           = [{self.ci_lower:.6f}, {self.ci_upper:.6f}]\n"
            f"  beta_cv          = {self.beta_cv:.6f}\n"
            f"  n_paths          = {self.n_paths}\n"
            f"  VR ratio         = {self.variance_reduction_ratio:.2f}x\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Main MC pricer
# ---------------------------------------------------------------------------

def arithmetic_asian_mc(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    N: int = 252,
    n_paths: int = 100_000,
    seed: int = 42,
    option_type: str = "call",
    use_cv: bool = True,
    use_antithetic: bool = True,
) -> ArithmeticMCResult:
    """Price a discrete arithmetic Asian option via Monte Carlo.

    Combines antithetic variates (built into path generation) and a
    geometric average control variate (Kemna & Vorst 1990) for dramatically
    reduced variance.

    Parameters
    ----------
    S : float
        Spot price S₀.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Annualised volatility.
    T : float
        Time to expiry (years).
    N : int
        Number of discrete monitoring dates (default 252).
    n_paths : int
        Number of Monte Carlo paths (default 100,000).
        When ``use_antithetic=True``, n_paths must be even.
    seed : int
        NumPy random seed for reproducibility (default 42).
    option_type : {'call', 'put'}
        Option type (case-insensitive).
    use_cv : bool
        If True (default), apply the geometric control-variate correction.
    use_antithetic : bool
        If True (default), use antithetic variates in path generation.

    Returns
    -------
    ArithmeticMCResult
        Pricing results including price, standard error, and 95% CI.

    Examples
    --------
    >>> from option_pricing_horizon.asian.monte_carlo import arithmetic_asian_mc
    >>> res = arithmetic_asian_mc(100, 100, 0.03, 0.20, 1.0, N=252,
    ...                           n_paths=50_000, seed=42)
    >>> 0 < res.price < 10
    True
    >>> res.ci_lower < res.price < res.ci_upper
    True
    """
    disc = np.exp(-r * T)

    # ------------------------------------------------------------------
    # 1. Simulate GBM paths: shape (n_paths, N)
    # ------------------------------------------------------------------
    paths = simulate_gbm_paths(
        S0=S,
        r=r,
        sigma=sigma,
        T=T,
        N=N,
        n_paths=n_paths,
        seed=seed,
        antithetic=use_antithetic,
    )
    actual_paths = paths.shape[0]   # may differ from n_paths if antithetic rounded

    # ------------------------------------------------------------------
    # 2. Compute undiscounted payoffs
    # ------------------------------------------------------------------
    arith_payoffs = arithmetic_average_payoff(paths, K, option_type)   # shape (M,)
    geo_payoffs   = geometric_average_payoff(paths, K, option_type)    # shape (M,)

    # ------------------------------------------------------------------
    # 3. Discount
    # ------------------------------------------------------------------
    arith_disc = disc * arith_payoffs
    geo_disc   = disc * geo_payoffs

    # ------------------------------------------------------------------
    # 4. Crude MC estimate
    # ------------------------------------------------------------------
    price_crude = float(arith_disc.mean())
    std_crude   = float(arith_disc.std(ddof=1) / np.sqrt(actual_paths))

    # ------------------------------------------------------------------
    # 5. Control-variate correction (optional)
    # ------------------------------------------------------------------
    beta_cv = np.nan
    var_reduction = np.nan

    if use_cv:
        # Exact geometric price (analytic)
        if option_type.lower().strip() == "call":
            geo_exact = geometric_asian_call(S, K, r, sigma, T, N)
        else:
            geo_exact = geometric_asian_put(S, K, r, sigma, T, N)

        # OLS coefficient: β̂ = Cov(arith, geo) / Var(geo)
        cov_matrix = np.cov(arith_disc, geo_disc, ddof=1)
        cov_ag = cov_matrix[0, 1]
        var_g  = cov_matrix[1, 1]

        if var_g > 1e-20:
            beta_cv = float(cov_ag / var_g)
        else:
            beta_cv = 0.0

        # Corrected residuals for SE computation
        residuals = arith_disc - beta_cv * geo_disc

        price_cv  = price_crude + beta_cv * (geo_exact - float(geo_disc.mean()))
        std_cv    = float(residuals.std(ddof=1) / np.sqrt(actual_paths))

        var_reduction = float((std_crude**2) / (std_cv**2)) if std_cv > 0 else np.nan

        price_final = price_cv
        std_final   = std_cv
    else:
        price_final = price_crude
        std_final   = std_crude

    # ------------------------------------------------------------------
    # 6. 95% confidence interval
    # ------------------------------------------------------------------
    z95 = 1.959963985   # Φ^{-1}(0.975), exact to 9 decimal places
    ci_lower = price_final - z95 * std_final
    ci_upper = price_final + z95 * std_final

    return ArithmeticMCResult(
        price=float(price_final),
        price_crude=float(price_crude),
        std_err=float(std_final),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        beta_cv=float(beta_cv),
        n_paths=actual_paths,
        variance_reduction_ratio=float(var_reduction),
    )
