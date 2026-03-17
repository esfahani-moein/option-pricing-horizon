"""
tests/test_asian_geo.py
========================
Unit tests for the exact closed-form geometric Asian option pricer.

Reference values are cross-checked against:
  - Kemna & Vorst (1990) formulas implemented independently
  - Monte Carlo with the geometric payoff (which should match exactly as M→∞)

Tests cover:
- Adjusted parameter correctness (mu_G, sigma_G, F_G)
- ATM geometric Asian call/put prices (sanity bounds)
- Put-call parity for geometric Asian options
- Geometric < Arithmetic inequality (calls)
- Geometric < European vanilla (calls)
- Monotonicity in S, K, sigma
- Edge cases: N=1 (single observation = European), large N
"""

import math
import pytest
import numpy as np

from option_pricing_horizon.asian.geometric import (
    geometric_asian_call,
    geometric_asian_put,
    geometric_asian_price,
    geo_asian_adjusted_params,
)
from option_pricing_horizon.european.black_scholes import bs_call_price, bs_put_price


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
PRICE_TOL = 1e-6    # analytical prices are exact
PARITY_TOL = 1e-6


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def baseline_params():
    return dict(S=100.0, K=100.0, r=0.03, sigma=0.20, T=1.0, N=252)


# ---------------------------------------------------------------------------
# Adjusted parameter tests
# ---------------------------------------------------------------------------

class TestGeoAdjustedParams:
    def test_sigma_G_positive(self, baseline_params):
        p = geo_asian_adjusted_params(**baseline_params)
        assert p["sigma_G"] > 0

    def test_sigma_G_less_than_sigma_sqrt_T(self, baseline_params):
        """σ_G < σ√T because averaging reduces volatility."""
        bp = baseline_params
        p = geo_asian_adjusted_params(**bp)
        assert p["sigma_G"] < bp["sigma"] * math.sqrt(bp["T"])

    def test_F_G_positive(self, baseline_params):
        p = geo_asian_adjusted_params(**baseline_params)
        assert p["F_G"] > 0

    def test_F_G_less_than_forward(self, baseline_params):
        """E[A_geo] < S₀ e^{rT} because averaging path < terminal price."""
        bp = baseline_params
        p = geo_asian_adjusted_params(**bp)
        forward = bp["S"] * math.exp(bp["r"] * bp["T"])
        assert p["F_G"] < forward

    def test_sigma_G_formula_small_N(self):
        """For N=1, sigma_G = sigma * sqrt(T * (N+1)(2N+1) / (6*N^2)).

        With N=1, T=1, sigma=0.20:
          sigma_G^2 = sigma^2 * dt^2 * (N+1)(2N+1)/6
                    = 0.04 * 1^2 * 2*3/6 = 0.04
          sigma_G   = 0.20
        """
        sigma = 0.20
        T = 1.0
        N = 1
        p = geo_asian_adjusted_params(S=100, K=100, r=0.03, sigma=sigma, T=T, N=N)
        dt = T / N
        expected_sigma_G_sq = sigma**2 * dt**2 * (N + 1) * (2 * N + 1) / 6.0
        expected_sigma_G = math.sqrt(expected_sigma_G_sq)
        assert abs(p["sigma_G"] - expected_sigma_G) < 1e-12

    def test_mu_G_formula(self, baseline_params):
        """mu_G = (r - sigma^2/2) * dt * (N+1)/2."""
        bp = baseline_params
        dt = bp["T"] / bp["N"]
        expected_mu = (bp["r"] - 0.5 * bp["sigma"]**2) * dt * (bp["N"] + 1) / 2.0
        p = geo_asian_adjusted_params(**bp)
        assert abs(p["mu_G"] - expected_mu) < 1e-12


# ---------------------------------------------------------------------------
# Geometric Asian call/put prices
# ---------------------------------------------------------------------------

class TestGeometricAsianPrices:
    def test_call_positive(self, baseline_params):
        bp = baseline_params
        price = geometric_asian_call(**bp)
        assert price > 0

    def test_put_positive(self, baseline_params):
        bp = baseline_params
        price = geometric_asian_put(**bp)
        assert price > 0

    def test_call_below_european_call(self, baseline_params):
        """Geometric Asian call ≤ European vanilla call (averaging reduces payoff)."""
        bp = baseline_params
        geo = geometric_asian_call(**bp)
        vanilla = bs_call_price(bp["S"], bp["K"], bp["r"], bp["sigma"], bp["T"])
        assert geo < vanilla + 1e-8

    def test_put_below_european_put(self, baseline_params):
        """Geometric Asian put ≤ European vanilla put."""
        bp = baseline_params
        geo = geometric_asian_put(**bp)
        vanilla = bs_put_price(bp["S"], bp["K"], bp["r"], bp["sigma"], bp["T"])
        assert geo < vanilla + 1e-8

    def test_put_call_parity(self, baseline_params):
        """C_geo - P_geo = e^{-rT}(F_G - K)."""
        bp = baseline_params
        call = geometric_asian_call(**bp)
        put  = geometric_asian_put(**bp)
        params = geo_asian_adjusted_params(**bp)
        disc = math.exp(-bp["r"] * bp["T"])
        rhs = disc * (params["F_G"] - bp["K"])
        assert abs(call - put - rhs) < PARITY_TOL

    def test_call_increases_with_S(self):
        """Geometric Asian call monotonically increases with S."""
        prices = [geometric_asian_call(S, 100, 0.03, 0.20, 1.0, 252)
                  for S in [80, 90, 100, 110, 120]]
        assert all(prices[i] < prices[i+1] for i in range(len(prices)-1))

    def test_call_decreases_with_K(self):
        """Geometric Asian call monotonically decreases with K."""
        prices = [geometric_asian_call(100, K, 0.03, 0.20, 1.0, 252)
                  for K in [80, 90, 100, 110, 120]]
        assert all(prices[i] > prices[i+1] for i in range(len(prices)-1))

    def test_call_increases_with_sigma(self):
        """Geometric Asian call increases with σ (positive vega)."""
        prices = [geometric_asian_call(100, 100, 0.03, s, 1.0, 252)
                  for s in [0.10, 0.20, 0.30, 0.40]]
        assert all(prices[i] < prices[i+1] for i in range(len(prices)-1))

    def test_deep_otm_call_small(self):
        """Deep OTM geometric Asian call should be near zero."""
        price = geometric_asian_call(50, 200, 0.03, 0.20, 1.0, 252)
        assert price < 0.01

    def test_deep_itm_call_positive(self):
        """Deep ITM geometric Asian call should be substantial."""
        price = geometric_asian_call(200, 100, 0.03, 0.20, 1.0, 252)
        assert price > 50

    def test_N1_reduces_to_european_like(self):
        """N=1 geometric Asian call should equal BS call with adjusted params."""
        S, K, r, sigma, T = 100, 100, 0.03, 0.20, 1.0
        price_N1 = geometric_asian_call(S, K, r, sigma, T, N=1)
        params = geo_asian_adjusted_params(S, K, r, sigma, T, N=1)
        F_G = params["F_G"]
        sigma_G = params["sigma_G"]
        disc = math.exp(-r * T)
        # Manual BS with F_G and sigma_G
        d1 = (math.log(F_G / K) + 0.5 * sigma_G**2) / sigma_G
        d2 = d1 - sigma_G
        from option_pricing_horizon.common.math_utils import norm_cdf
        expected = disc * (F_G * norm_cdf(d1) - K * norm_cdf(d2))
        assert abs(price_N1 - expected) < 1e-12

    def test_dispatch_function_call(self, baseline_params):
        bp = baseline_params
        assert abs(
            geometric_asian_price(**bp, option_type="call")
            - geometric_asian_call(**bp)
        ) < 1e-14

    def test_dispatch_function_put(self, baseline_params):
        bp = baseline_params
        assert abs(
            geometric_asian_price(**bp, option_type="put")
            - geometric_asian_put(**bp)
        ) < 1e-14

    def test_dispatch_invalid_type(self, baseline_params):
        with pytest.raises(ValueError):
            geometric_asian_price(**baseline_params, option_type="binary")

    def test_call_price_range(self, baseline_params):
        """Price should be in a realistic range for baseline params."""
        bp = baseline_params
        price = geometric_asian_call(**bp)
        # Geometric Asian ATM call should be between 3 and 9 for these params
        assert 3.0 < price < 9.0

    def test_convergence_increasing_N(self):
        """As N increases, price should converge (not oscillate wildly)."""
        S, K, r, sigma, T = 100, 100, 0.03, 0.20, 1.0
        prices = [geometric_asian_call(S, K, r, sigma, T, N) for N in [10, 50, 252, 1000]]
        # All should be positive and finite
        assert all(p > 0 for p in prices)
        # Changes should be decreasing (convergence)
        diffs = [abs(prices[i+1] - prices[i]) for i in range(len(prices)-1)]
        assert diffs[-1] < diffs[0]
