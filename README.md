# Option Pricing Horizon

A high-accuracy, research-grade Python library for pricing **European** and **Asian** options
under the risk-neutral GBM framework, with full Greeks, convergence analysis, and
real-world hedging case studies.

---

## Project Layout

```
option_pricing_horizon/
├── common/              # Shared primitives: market params, normal CDF, stats helpers
│   ├── __init__.py
│   ├── market_params.py # MarketParams dataclass (S0, K, r, sigma, T, N)
│   ├── math_utils.py    # High-precision normal CDF/PDF, log-normal stats
│   └── simulation.py    # Numba-accelerated GBM path generator
│
├── european/            # Black-Scholes closed-form pricing & Greeks
│   ├── __init__.py
│   ├── black_scholes.py # Call/put prices, d1/d2, exact analytical formulas
│   └── greeks.py        # Delta, Gamma, Theta, Vega, Rho (analytical + finite-diff)
│
└── asian/               # Asian option pricing: geometric (exact) + arithmetic (MC)
    ├── __init__.py
    ├── payoffs.py        # Discrete arithmetic & geometric average payoff definitions
    ├── geometric.py      # Closed-form geometric Asian price (Kemna-Vorst adjusted BS)
    ├── monte_carlo.py    # Arithmetic MC with antithetic variates + geometric control variate
    ├── greeks.py         # Delta, Vega estimation via finite differences on MC price
    └── convergence.py    # Convergence and variance-reduction analysis utilities

tests/
├── test_european.py     # Unit tests for Black-Scholes prices and Greeks
├── test_asian_geo.py    # Unit tests for geometric Asian closed-form
└── test_asian_mc.py     # Integration tests for MC pricing and control-variate bias

notebooks/
├── 01_european_black_scholes.ipynb   # Theory + interactive BS pricer + Greeks surface
├── 02_asian_geometric_exact.ipynb    # Geometric Asian derivation and closed-form plots
├── 03_asian_arithmetic_montecarlo.ipynb  # MC simulation, antithetic, control variate
├── 04_greeks_sensitivity.ipynb       # Greeks surfaces, sigma/K/r sensitivity sweeps
└── 05_realworld_airline_hedging.ipynb    # Airline fuel hedging applied case study
```

---

## Quick Start

```bash
conda activate quantenv
pip install -e .           # editable install (no new deps, uses quantenv)
pytest tests/ -v           # run full test suite
```

---

## Baseline Market Parameters (from project brief)

| Parameter | Value        |
|-----------|-------------|
| S₀        | 100          |
| r         | 3 %          |
| σ         | 20 %         |
| T         | 1 year       |
| N         | 252 days     |

Under risk-neutral GBM:  `dS_t = r S_t dt + σ S_t dW_t`

---

## Module Highlights

### European (Black-Scholes)
Exact closed-form call/put prices and all five Greeks using the cumulative normal distribution
evaluated with `scipy.special.ndtr` for machine-precision accuracy.

### Asian – Geometric (Closed-Form)
The discrete geometric average follows a log-normal distribution exactly.
We derive modified drift `μ_G` and volatility `σ_G` and apply a Black-Scholes-style formula
(Kemna & Vorst 1990, extended for discrete monitoring).

### Asian – Arithmetic (Monte Carlo)
- **GBM path simulation** via Numba JIT for speed
- **Antithetic variates**: pairs (Z, −Z) reduce variance ~50 %
- **Geometric control variate** (Kemna & Vorst): exploits known analytical price to
  dramatically reduce MC standard error (10–30× improvement typical)
- **Confidence intervals**, bias analysis, and convergence plots provided

### Greeks & Sensitivity
Delta estimated via finite-difference bump-and-reprice on the MC pricer (same random seed).
Full σ / K / r / N sensitivity sweeps with interactive Plotly charts.

### Real-World Case: Airline Fuel Hedging
An airline buys an arithmetic Asian call on jet-fuel prices to cap average quarterly fuel cost.
Demonstrates risk reduction vs vanilla European call at realistic parameters.

