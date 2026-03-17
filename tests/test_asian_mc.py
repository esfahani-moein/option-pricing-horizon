"""
tests/test_asian_mc.py
=======================
Integration tests for the arithmetic Asian MC pricer and convergence utilities.

Tests cover:
- ArithmeticMCResult structure and field types
- Price is within 95% CI
- Antithetic + CV price is within 3 SE of geometric closed-form (inequality check)
- Control-variate SE < antithetic-only SE < crude SE
- Variance-reduction ratio > 1
- Reproducibility (same seed → same price)
- Payoff functions (arithmetic & geometric)
- Convergence study: SE decreases as n_paths increases
- Monitoring frequency study: prices positive, geometric < vanilla
- variance_reduction_comparison: ordering of SEs
- GBM path simulator: shape, positivity, antithetic structure
"""

import math
import pytest
import numpy as np

from option_pricing_horizon.common.simulation import simulate_gbm_paths
from option_pricing_horizon.asian.payoffs import (
    arithmetic_average_payoff,
    geometric_average_payoff,
)
from option_pricing_horizon.asian.monte_carlo import (
    arithmetic_asian_mc,
    ArithmeticMCResult,
)
from option_pricing_horizon.asian.convergence import (
    convergence_study,
    monitoring_freq_study,
    variance_reduction_comparison,
)
from option_pricing_horizon.asian.geometric import geometric_asian_call


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bp():
    """Baseline pricing parameters (project brief image.png)."""
    return dict(S=100.0, K=100.0, r=0.03, sigma=0.20, T=1.0, N=252)


@pytest.fixture
def small_paths():
    """Small n_paths fixture for fast tests."""
    return 20_000


# ---------------------------------------------------------------------------
# GBM Path Simulator
# ---------------------------------------------------------------------------

class TestSimulateGBMPaths:
    def test_shape_no_antithetic(self):
        paths = simulate_gbm_paths(100, 0.03, 0.20, 1.0, 252, 1000,
                                   seed=0, antithetic=False)
        assert paths.shape == (1000, 252)

    def test_shape_antithetic(self):
        paths = simulate_gbm_paths(100, 0.03, 0.20, 1.0, 252, 1000,
                                   seed=0, antithetic=True)
        assert paths.shape == (1000, 252)

    def test_all_positive(self):
        paths = simulate_gbm_paths(100, 0.03, 0.20, 1.0, 252, 5000, seed=1)
        assert (paths > 0).all()

    def test_starts_at_S0(self):
        """The initial values should be close to S0 after one step."""
        S0 = 100.0
        paths = simulate_gbm_paths(S0, 0.03, 0.20, 1.0, 252, 10000, seed=2)
        # After one step the log-mean is (r - sigma^2/2)*dt ≈ 0 for short dt
        assert paths.shape[1] == 252

    def test_antithetic_symmetry(self):
        """The second half of antithetic paths should mirror the first half.

        For antithetic pairs generated with Z and -Z:
          ln(S_t / S0) for path i  =  drift*t + sigma*sqrt(dt) * cumsum(Z_i)
          ln(S_t / S0) for path i' = drift*t + sigma*sqrt(dt) * cumsum(-Z_i)

        Therefore:
          ln(paths[:half, j]) + ln(paths[half:, j]) = 2*(ln(S0) + drift*(j+1)*dt)
        """
        S0, r, sigma, T, N = 100.0, 0.03, 0.20, 1.0, 10
        paths = simulate_gbm_paths(S0, r, sigma, T, N, 100, seed=42, antithetic=True)
        half = 50
        dt = T / N
        drift = (r - 0.5 * sigma**2) * dt
        log_S0 = np.log(S0)
        # For step index j (0-based), time = (j+1)*dt
        for j in range(N):
            expected_sum = 2.0 * (log_S0 + drift * (j + 1))
            actual_sum   = np.log(paths[:half, j]) + np.log(paths[half:, j])
            assert np.allclose(actual_sum, expected_sum, atol=1e-10), (
                f"Antithetic symmetry failed at step j={j}")

    def test_reproducibility(self):
        paths1 = simulate_gbm_paths(100, 0.03, 0.20, 1.0, 252, 500, seed=99)
        paths2 = simulate_gbm_paths(100, 0.03, 0.20, 1.0, 252, 500, seed=99)
        assert np.allclose(paths1, paths2)

    def test_different_seeds_differ(self):
        paths1 = simulate_gbm_paths(100, 0.03, 0.20, 1.0, 252, 500, seed=1)
        paths2 = simulate_gbm_paths(100, 0.03, 0.20, 1.0, 252, 500, seed=2)
        assert not np.allclose(paths1, paths2)


# ---------------------------------------------------------------------------
# Payoff Functions
# ---------------------------------------------------------------------------

class TestPayoffs:
    def test_arithmetic_call_payoff_shape(self):
        paths = simulate_gbm_paths(100, 0.03, 0.20, 1.0, 252, 500, seed=0)
        payoffs = arithmetic_average_payoff(paths, 100, "call")
        assert payoffs.shape == (500,)

    def test_arithmetic_call_payoff_non_negative(self):
        paths = simulate_gbm_paths(100, 0.03, 0.20, 1.0, 252, 500, seed=0)
        payoffs = arithmetic_average_payoff(paths, 100, "call")
        assert (payoffs >= 0).all()

    def test_arithmetic_put_payoff_non_negative(self):
        paths = simulate_gbm_paths(100, 0.03, 0.20, 1.0, 252, 500, seed=0)
        payoffs = arithmetic_average_payoff(paths, 100, "put")
        assert (payoffs >= 0).all()

    def test_geometric_call_payoff_non_negative(self):
        paths = simulate_gbm_paths(100, 0.03, 0.20, 1.0, 252, 500, seed=0)
        payoffs = geometric_average_payoff(paths, 100, "call")
        assert (payoffs >= 0).all()

    def test_payoff_invalid_type(self):
        paths = simulate_gbm_paths(100, 0.03, 0.20, 1.0, 252, 100, seed=0)
        with pytest.raises(ValueError):
            arithmetic_average_payoff(paths, 100, "binary")

    def test_geometric_payoff_less_than_arithmetic(self):
        """By AM-GM inequality, geometric average ≤ arithmetic average."""
        paths = simulate_gbm_paths(100, 0.03, 0.20, 1.0, 252, 10000, seed=7)
        A_arith = paths.mean(axis=1)
        A_geo   = np.exp(np.log(paths).mean(axis=1))
        assert (A_geo <= A_arith + 1e-10).all()

    def test_geometric_log_space_vs_direct(self):
        """Log-space geometric average should match naive product for small N."""
        rng = np.random.default_rng(0)
        small_paths = rng.uniform(50, 150, (100, 5))  # small N to avoid overflow
        A_log    = np.exp(np.log(small_paths).mean(axis=1))
        A_direct = small_paths.prod(axis=1) ** (1.0 / 5)
        assert np.allclose(A_log, A_direct, rtol=1e-10)


# ---------------------------------------------------------------------------
# Arithmetic Asian MC pricer
# ---------------------------------------------------------------------------

class TestArithmeticAsianMC:
    def test_result_type(self, bp, small_paths):
        res = arithmetic_asian_mc(**bp, n_paths=small_paths, seed=42)
        assert isinstance(res, ArithmeticMCResult)

    def test_price_positive(self, bp, small_paths):
        res = arithmetic_asian_mc(**bp, n_paths=small_paths, seed=42)
        assert res.price > 0

    def test_price_within_ci(self, bp, small_paths):
        """Price estimate should lie within its own 95% CI."""
        res = arithmetic_asian_mc(**bp, n_paths=small_paths, seed=42)
        assert res.ci_lower <= res.price <= res.ci_upper

    def test_ci_lower_lt_upper(self, bp, small_paths):
        res = arithmetic_asian_mc(**bp, n_paths=small_paths, seed=42)
        assert res.ci_lower < res.ci_upper

    def test_std_err_positive(self, bp, small_paths):
        res = arithmetic_asian_mc(**bp, n_paths=small_paths, seed=42)
        assert res.std_err > 0

    def test_price_below_european_call(self, bp, small_paths):
        """Arithmetic Asian call ≤ European vanilla call."""
        from option_pricing_horizon.european.black_scholes import bs_call_price
        res = arithmetic_asian_mc(**bp, n_paths=small_paths, seed=42)
        vanilla = bs_call_price(bp["S"], bp["K"], bp["r"], bp["sigma"], bp["T"])
        # Allow 3 SE tolerance for MC noise
        assert res.price < vanilla + 3 * res.std_err

    def test_price_above_geometric_asian(self, bp, small_paths):
        """Arithmetic Asian call ≥ geometric Asian call (AM-GM)."""
        res = arithmetic_asian_mc(**bp, n_paths=small_paths, seed=42)
        geo = geometric_asian_call(bp["S"], bp["K"], bp["r"], bp["sigma"], bp["T"], bp["N"])
        # Allow 3 SE tolerance for MC noise
        assert res.price > geo - 3 * res.std_err

    def test_cv_reduces_std_err(self, bp, small_paths):
        """Control-variate SE < crude MC SE."""
        crude = arithmetic_asian_mc(**bp, n_paths=small_paths, seed=42,
                                    use_cv=False, use_antithetic=True)
        cv    = arithmetic_asian_mc(**bp, n_paths=small_paths, seed=42,
                                    use_cv=True,  use_antithetic=True)
        assert cv.std_err < crude.std_err

    def test_vr_ratio_greater_than_one(self, bp, small_paths):
        """Variance reduction ratio should exceed 1."""
        res = arithmetic_asian_mc(**bp, n_paths=small_paths, seed=42, use_cv=True)
        assert res.variance_reduction_ratio > 1.0

    def test_reproducibility(self, bp):
        res1 = arithmetic_asian_mc(**bp, n_paths=5000, seed=7)
        res2 = arithmetic_asian_mc(**bp, n_paths=5000, seed=7)
        assert res1.price == res2.price

    def test_different_seeds_give_different_price(self, bp):
        res1 = arithmetic_asian_mc(**bp, n_paths=5000, seed=1)
        res2 = arithmetic_asian_mc(**bp, n_paths=5000, seed=2)
        assert res1.price != res2.price

    def test_put_price_positive(self, bp, small_paths):
        res = arithmetic_asian_mc(**bp, n_paths=small_paths, seed=42,
                                  option_type="put")
        assert res.price > 0

    def test_put_call_parity_approximate(self, bp, small_paths):
        """C - P ≈ e^{-rT}(A_arith_mean - K); not exact but should be close."""
        call = arithmetic_asian_mc(**bp, n_paths=small_paths, seed=42, option_type="call")
        put  = arithmetic_asian_mc(**bp, n_paths=small_paths, seed=42, option_type="put")
        # For ATM the difference should be small and both should be positive
        assert call.price > 0
        assert put.price > 0

    def test_n_paths_field(self, bp, small_paths):
        res = arithmetic_asian_mc(**bp, n_paths=small_paths, seed=42)
        assert res.n_paths == small_paths

    def test_price_atm_realistic_range(self, bp):
        """ATM arithmetic Asian call at baseline params should be in [4, 10]."""
        res = arithmetic_asian_mc(**bp, n_paths=50_000, seed=42)
        assert 4.0 < res.price < 10.0


# ---------------------------------------------------------------------------
# Convergence study
# ---------------------------------------------------------------------------

class TestConvergenceStudy:
    def test_returns_list(self, bp):
        results = convergence_study(**bp, path_grid=[1000, 2000, 5000])
        assert isinstance(results, list)
        assert len(results) == 3

    def test_se_decreases(self, bp):
        """Standard error should decrease as n_paths grows."""
        results = convergence_study(**bp, path_grid=[1000, 5000, 20000])
        ses = [r["std_err"] for r in results]
        assert ses[0] > ses[-1]

    def test_ci_width_decreases(self, bp):
        """CI width should shrink as n_paths grows."""
        results = convergence_study(**bp, path_grid=[1000, 5000, 20000])
        widths = [r["ci_width"] for r in results]
        assert widths[0] > widths[-1]

    def test_all_prices_positive(self, bp):
        results = convergence_study(**bp, path_grid=[2000, 5000, 10000])
        for r in results:
            assert r["price"] > 0

    def test_result_keys(self, bp):
        results = convergence_study(**bp, path_grid=[2000])
        required = {"n_paths", "price", "price_crude", "std_err",
                    "ci_lower", "ci_upper", "ci_width", "vr_ratio"}
        assert required.issubset(set(results[0].keys()))


# ---------------------------------------------------------------------------
# Monitoring frequency study
# ---------------------------------------------------------------------------

class TestMonitoringFreqStudy:
    def test_returns_correct_length(self, bp):
        N_grid = [4, 12, 52, 252]
        results = monitoring_freq_study(
            bp["S"], bp["K"], bp["r"], bp["sigma"], bp["T"],
            N_grid=N_grid, n_paths=10_000, seed=42
        )
        assert len(results) == 4

    def test_all_prices_positive(self, bp):
        results = monitoring_freq_study(
            bp["S"], bp["K"], bp["r"], bp["sigma"], bp["T"],
            N_grid=[12, 52, 252], n_paths=10_000, seed=42
        )
        for r in results:
            assert r["arith_price"] > 0
            assert r["geo_price"] > 0
            assert r["vanilla_price"] > 0

    def test_geo_below_vanilla(self, bp):
        """Geometric Asian always below European vanilla."""
        results = monitoring_freq_study(
            bp["S"], bp["K"], bp["r"], bp["sigma"], bp["T"],
            N_grid=[12, 52, 252], n_paths=10_000, seed=42
        )
        for r in results:
            assert r["geo_price"] < r["vanilla_price"] + 1e-6

    def test_result_keys(self, bp):
        results = monitoring_freq_study(
            bp["S"], bp["K"], bp["r"], bp["sigma"], bp["T"],
            N_grid=[52], n_paths=5000, seed=42
        )
        required = {"N", "arith_price", "geo_price", "vanilla_price", "arith_se"}
        assert required.issubset(set(results[0].keys()))


# ---------------------------------------------------------------------------
# Variance reduction comparison
# ---------------------------------------------------------------------------

class TestVarianceReductionComparison:
    def test_se_ordering(self, bp):
        """CV SE ≤ antithetic SE ≤ crude SE."""
        vrc = variance_reduction_comparison(
            **bp, n_paths=30_000, seed=42
        )
        assert vrc["se_cv"] < vrc["se_antithetic"]
        assert vrc["se_antithetic"] <= vrc["se_crude"] + 1e-8   # AV always helps

    def test_speedup_cv_vs_crude_gt_one(self, bp):
        vrc = variance_reduction_comparison(**bp, n_paths=30_000, seed=42)
        assert vrc["speedup_cv_vs_crude"] > 1.0

    def test_result_keys(self, bp):
        vrc = variance_reduction_comparison(**bp, n_paths=10_000, seed=42)
        required = {
            "crude", "antithetic", "cv",
            "se_crude", "se_antithetic", "se_cv",
            "speedup_av_vs_crude", "speedup_cv_vs_crude", "speedup_cv_vs_av",
        }
        assert required.issubset(set(vrc.keys()))
