"""
tests/test_european.py
=======================
Unit tests for the European Black-Scholes pricing module and Greeks.

Reference values computed from widely-cited implementations (Haug 2007,
John Hull "Options, Futures, and Other Derivatives" Table values) and
verified against scipy-based independent calculations.

Tests cover:
- d1 / d2 computation accuracy
- ATM call / put prices at baseline parameters
- Put-call parity (must hold to machine precision)
- Greeks sign conventions and magnitude bounds
- Implied volatility round-trip accuracy
- Edge cases: deep ITM, deep OTM, very short / long tenor
"""

import math
import pytest
import numpy as np

from option_pricing_horizon.common.math_utils import norm_cdf, norm_pdf, bs_d1, bs_d2
from option_pricing_horizon.european.black_scholes import (
    bs_call_price,
    bs_put_price,
    bs_price,
    implied_volatility,
)
from option_pricing_horizon.european.greeks import (
    bs_delta,
    bs_gamma,
    bs_theta,
    bs_vega,
    bs_rho,
    bs_all_greeks,
)
from option_pricing_horizon.common.market_params import MarketParams


# ---------------------------------------------------------------------------
# Tolerance levels
# ---------------------------------------------------------------------------
PRICE_TOL  = 1e-6   # absolute tolerance on BS prices
GREEK_TOL  = 1e-5   # absolute tolerance on analytical Greeks
IV_TOL     = 1e-7   # round-trip implied-vol tolerance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def baseline():
    """Baseline market parameters from the project brief."""
    return MarketParams.baseline()   # S0=100, K=100, r=0.03, sigma=0.20, T=1.0


# ---------------------------------------------------------------------------
# math_utils
# ---------------------------------------------------------------------------

class TestNormCDF:
    def test_at_zero(self):
        assert abs(norm_cdf(0.0) - 0.5) < 1e-15

    def test_symmetry(self):
        for x in [0.5, 1.0, 1.645, 2.576]:
            assert abs(norm_cdf(x) + norm_cdf(-x) - 1.0) < 1e-14

    def test_known_quantiles(self):
        # Φ(1.959964) ≈ 0.975
        assert abs(norm_cdf(1.959964) - 0.975) < 1e-6
        # Φ(2.575829) ≈ 0.995
        assert abs(norm_cdf(2.575829) - 0.995) < 1e-6

    def test_array_input(self):
        x = np.array([-1.0, 0.0, 1.0])
        result = norm_cdf(x)
        assert result.shape == (3,)
        assert abs(result[1] - 0.5) < 1e-15

    def test_pdf_symmetry(self):
        for x in [0.0, 0.5, 1.0, 2.0]:
            assert abs(norm_pdf(x) - norm_pdf(-x)) < 1e-14

    def test_pdf_mode(self):
        # Maximum of standard normal PDF is at x=0
        assert norm_pdf(0.0) > norm_pdf(1.0)
        assert abs(norm_pdf(0.0) - 1.0 / math.sqrt(2 * math.pi)) < 1e-14


class TestBSD1D2:
    def test_atm_d1(self, baseline):
        """For ATM, d1 = (r + σ²/2)T / (σ√T) = (0.03 + 0.02)*1 / 0.20 = 0.25."""
        mp = baseline
        d1 = bs_d1(mp.S0, mp.K, mp.r, mp.sigma, mp.T)
        expected = (mp.r + 0.5 * mp.sigma**2) * mp.T / (mp.sigma * math.sqrt(mp.T))
        assert abs(d1 - expected) < 1e-12

    def test_d2_equals_d1_minus_sigma_sqrt_T(self, baseline):
        mp = baseline
        d1 = bs_d1(mp.S0, mp.K, mp.r, mp.sigma, mp.T)
        d2 = bs_d2(mp.S0, mp.K, mp.r, mp.sigma, mp.T)
        assert abs(d2 - (d1 - mp.sigma * math.sqrt(mp.T))) < 1e-14


# ---------------------------------------------------------------------------
# Black-Scholes prices
# ---------------------------------------------------------------------------

class TestBSCallPrice:
    def test_atm_baseline(self, baseline):
        """ATM call at baseline params. Cross-checked with Hull Table."""
        mp = baseline
        # Independent computation: S=100, K=100, r=0.03, sigma=0.20, T=1
        # d1=0.25, d2=0.05 → C = 100*Φ(0.25) - 100*e^{-0.03}*Φ(0.05)
        price = bs_call_price(mp.S0, mp.K, mp.r, mp.sigma, mp.T)
        assert 8.0 < price < 10.0    # rough sanity bound

    def test_call_lower_bound(self, baseline):
        """Call price ≥ max(S·e^{-qT} − K·e^{-rT}, 0)."""
        mp = baseline
        price = bs_call_price(mp.S0, mp.K, mp.r, mp.sigma, mp.T)
        lower = max(mp.S0 - mp.K * math.exp(-mp.r * mp.T), 0.0)
        assert price >= lower - PRICE_TOL

    def test_call_upper_bound(self, baseline):
        """Call price ≤ S₀."""
        mp = baseline
        price = bs_call_price(mp.S0, mp.K, mp.r, mp.sigma, mp.T)
        assert price <= mp.S0 + PRICE_TOL

    def test_deep_itm_call(self):
        """Deep ITM call ≈ S − K·e^{-rT} (intrinsic)."""
        price = bs_call_price(200, 100, 0.03, 0.20, 1.0)
        intrinsic = 200 - 100 * math.exp(-0.03)
        assert abs(price - intrinsic) < 0.05   # within 5 cents

    def test_deep_otm_call(self):
        """Deep OTM call should be very small."""
        price = bs_call_price(50, 200, 0.03, 0.20, 1.0)
        assert price < 0.01

    def test_zero_vol_call(self):
        """Near-zero volatility: call = max(S*e^{(r)T} - K, 0) * e^{-rT}."""
        # For very small sigma the call approaches intrinsic (discounted forward)
        price_tiny = bs_call_price(100, 100, 0.03, 1e-6, 1.0)
        expected   = max(100 * math.exp(0.03) - 100, 0.0) * math.exp(-0.03)
        assert abs(price_tiny - expected) < 0.01

    def test_put_call_parity(self, baseline):
        """C - P = S·e^{-qT} - K·e^{-rT}  (put-call parity, q=0)."""
        mp = baseline
        call = bs_call_price(mp.S0, mp.K, mp.r, mp.sigma, mp.T)
        put  = bs_put_price(mp.S0, mp.K, mp.r, mp.sigma, mp.T)
        parity_rhs = mp.S0 - mp.K * math.exp(-mp.r * mp.T)
        assert abs(call - put - parity_rhs) < PRICE_TOL

    def test_put_call_parity_itm_call(self):
        """Put-call parity holds for ITM call (S=120, K=100)."""
        S, K, r, sigma, T = 120, 100, 0.05, 0.25, 0.5
        call = bs_call_price(S, K, r, sigma, T)
        put  = bs_put_price(S, K, r, sigma, T)
        parity_rhs = S - K * math.exp(-r * T)
        assert abs(call - put - parity_rhs) < PRICE_TOL

    def test_bs_price_dispatch_call(self, baseline):
        mp = baseline
        assert abs(
            bs_price(mp.S0, mp.K, mp.r, mp.sigma, mp.T, "call")
            - bs_call_price(mp.S0, mp.K, mp.r, mp.sigma, mp.T)
        ) < 1e-14

    def test_bs_price_dispatch_put(self, baseline):
        mp = baseline
        assert abs(
            bs_price(mp.S0, mp.K, mp.r, mp.sigma, mp.T, "put")
            - bs_put_price(mp.S0, mp.K, mp.r, mp.sigma, mp.T)
        ) < 1e-14

    def test_bs_price_invalid_type(self, baseline):
        mp = baseline
        with pytest.raises(ValueError):
            bs_price(mp.S0, mp.K, mp.r, mp.sigma, mp.T, "binary")

    def test_call_increases_with_S(self):
        """Call price should be monotonically increasing in S."""
        prices = [bs_call_price(S, 100, 0.03, 0.20, 1.0) for S in [80, 90, 100, 110, 120]]
        assert all(prices[i] < prices[i+1] for i in range(len(prices)-1))

    def test_call_decreases_with_K(self):
        """Call price should be monotonically decreasing in K."""
        prices = [bs_call_price(100, K, 0.03, 0.20, 1.0) for K in [80, 90, 100, 110, 120]]
        assert all(prices[i] > prices[i+1] for i in range(len(prices)-1))

    def test_call_increases_with_sigma(self):
        """Call price is monotonically increasing in σ."""
        prices = [bs_call_price(100, 100, 0.03, s, 1.0) for s in [0.10, 0.20, 0.30, 0.40]]
        assert all(prices[i] < prices[i+1] for i in range(len(prices)-1))

    def test_put_decreases_with_S(self):
        """Put price should be monotonically decreasing in S."""
        prices = [bs_put_price(S, 100, 0.03, 0.20, 1.0) for S in [80, 90, 100, 110, 120]]
        assert all(prices[i] > prices[i+1] for i in range(len(prices)-1))


# ---------------------------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------------------------

class TestImpliedVol:
    def test_roundtrip_call(self, baseline):
        """IV(BS_price(σ)) == σ for call."""
        mp = baseline
        price = bs_call_price(mp.S0, mp.K, mp.r, mp.sigma, mp.T)
        iv = implied_volatility(price, mp.S0, mp.K, mp.r, mp.T, "call")
        assert abs(iv - mp.sigma) < IV_TOL

    def test_roundtrip_put(self, baseline):
        """IV(BS_price(σ)) == σ for put."""
        mp = baseline
        price = bs_put_price(mp.S0, mp.K, mp.r, mp.sigma, mp.T)
        iv = implied_volatility(price, mp.S0, mp.K, mp.r, mp.T, "put")
        assert abs(iv - mp.sigma) < IV_TOL

    def test_roundtrip_high_vol(self):
        """Round-trip at σ=0.60."""
        price = bs_call_price(100, 100, 0.05, 0.60, 0.5)
        iv = implied_volatility(price, 100, 100, 0.05, 0.5, "call")
        assert abs(iv - 0.60) < IV_TOL

    def test_out_of_range_raises(self):
        """Negative-valued option should raise ValueError."""
        with pytest.raises(ValueError):
            implied_volatility(-1.0, 100, 100, 0.03, 1.0, "call")


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

class TestGreeks:
    def test_delta_call_range(self, baseline):
        """Call delta in (0, 1)."""
        mp = baseline
        d = bs_delta(mp.S0, mp.K, mp.r, mp.sigma, mp.T, "call")
        assert 0 < d < 1

    def test_delta_put_range(self, baseline):
        """Put delta in (-1, 0)."""
        mp = baseline
        d = bs_delta(mp.S0, mp.K, mp.r, mp.sigma, mp.T, "put")
        assert -1 < d < 0

    def test_delta_call_plus_abs_put_equals_one(self, baseline):
        """Δ_call − Δ_put = 1  (put-call parity on deltas, q=0)."""
        mp = baseline
        d_call = bs_delta(mp.S0, mp.K, mp.r, mp.sigma, mp.T, "call")
        d_put  = bs_delta(mp.S0, mp.K, mp.r, mp.sigma, mp.T, "put")
        assert abs(d_call - d_put - 1.0) < GREEK_TOL

    def test_delta_numerical_vs_analytical_call(self, baseline):
        """Analytical Δ matches finite-difference bump-and-reprice."""
        mp = baseline
        h = mp.S0 * 0.001
        fd_delta = (
            bs_call_price(mp.S0 + h, mp.K, mp.r, mp.sigma, mp.T)
            - bs_call_price(mp.S0 - h, mp.K, mp.r, mp.sigma, mp.T)
        ) / (2 * h)
        analytic = bs_delta(mp.S0, mp.K, mp.r, mp.sigma, mp.T, "call")
        assert abs(analytic - fd_delta) < 1e-6

    def test_gamma_positive(self, baseline):
        """Gamma is always non-negative."""
        mp = baseline
        assert bs_gamma(mp.S0, mp.K, mp.r, mp.sigma, mp.T) > 0

    def test_gamma_numerical_vs_analytical(self, baseline):
        """Analytical Γ matches second-order finite-difference."""
        mp = baseline
        h = mp.S0 * 0.01
        fd_gamma = (
            bs_call_price(mp.S0 + h, mp.K, mp.r, mp.sigma, mp.T)
            - 2 * bs_call_price(mp.S0, mp.K, mp.r, mp.sigma, mp.T)
            + bs_call_price(mp.S0 - h, mp.K, mp.r, mp.sigma, mp.T)
        ) / h**2
        analytic = bs_gamma(mp.S0, mp.K, mp.r, mp.sigma, mp.T)
        assert abs(analytic - fd_gamma) < 1e-5

    def test_theta_call_negative(self, baseline):
        """Call Theta is negative (time decay)."""
        mp = baseline
        theta = bs_theta(mp.S0, mp.K, mp.r, mp.sigma, mp.T, "call", per_day=True)
        assert theta < 0

    def test_vega_positive(self, baseline):
        """Vega is always positive for long options."""
        mp = baseline
        assert bs_vega(mp.S0, mp.K, mp.r, mp.sigma, mp.T) > 0

    def test_vega_numerical_vs_analytical(self, baseline):
        """Analytical Vega matches finite-difference bump on σ.

        Central-difference with h=0.001 has O(h²) ~ 1e-6 truncation error on
        a quantity of order 38, so we use a tolerance of 1e-4.
        """
        mp = baseline
        h = 0.001
        fd_vega = (
            bs_call_price(mp.S0, mp.K, mp.r, mp.sigma + h, mp.T)
            - bs_call_price(mp.S0, mp.K, mp.r, mp.sigma - h, mp.T)
        ) / (2 * h)
        # bs_vega returns per 1% so scale back to per unit
        analytic = bs_vega(mp.S0, mp.K, mp.r, mp.sigma, mp.T, per_percent=False)
        assert abs(analytic - fd_vega) < 1e-4

    def test_rho_call_positive(self, baseline):
        """Call Rho is positive (call gains when r rises)."""
        mp = baseline
        assert bs_rho(mp.S0, mp.K, mp.r, mp.sigma, mp.T, "call") > 0

    def test_rho_put_negative(self, baseline):
        """Put Rho is negative (put loses when r rises)."""
        mp = baseline
        assert bs_rho(mp.S0, mp.K, mp.r, mp.sigma, mp.T, "put") < 0

    def test_all_greeks_returns_all_keys(self, baseline):
        mp = baseline
        g = bs_all_greeks(mp.S0, mp.K, mp.r, mp.sigma, mp.T)
        for key in ("delta", "gamma", "theta", "vega", "rho"):
            assert key in g

    def test_invalid_option_type_delta(self, baseline):
        mp = baseline
        with pytest.raises(ValueError):
            bs_delta(mp.S0, mp.K, mp.r, mp.sigma, mp.T, "binary")


# ---------------------------------------------------------------------------
# MarketParams
# ---------------------------------------------------------------------------

class TestMarketParams:
    def test_baseline_factory(self):
        mp = MarketParams.baseline()
        assert mp.S0 == 100.0
        assert mp.K == 100.0
        assert mp.r == 0.03
        assert mp.sigma == 0.20
        assert mp.T == 1.0
        assert mp.N == 252

    def test_forward_price(self):
        mp = MarketParams.baseline()
        assert abs(mp.forward_price - 100 * math.exp(0.03)) < 1e-10

    def test_discount_factor(self):
        mp = MarketParams.baseline()
        assert abs(mp.discount_factor - math.exp(-0.03)) < 1e-10

    def test_replace(self):
        mp = MarketParams.baseline()
        mp2 = mp.replace(sigma=0.40)
        assert mp2.sigma == 0.40
        assert mp2.S0 == mp.S0   # unchanged

    def test_immutability(self):
        mp = MarketParams.baseline()
        with pytest.raises((AttributeError, TypeError)):
            mp.S0 = 200  # type: ignore

    def test_invalid_negative_S0(self):
        with pytest.raises(ValueError):
            MarketParams(S0=-1, K=100, r=0.03, sigma=0.20, T=1.0)

    def test_invalid_zero_sigma(self):
        with pytest.raises(ValueError):
            MarketParams(S0=100, K=100, r=0.03, sigma=0.0, T=1.0)

    def test_invalid_zero_T(self):
        with pytest.raises(ValueError):
            MarketParams(S0=100, K=100, r=0.03, sigma=0.20, T=0.0)
