"""
option_pricing_horizon.asian
=============================
Pricing and analysis of Asian (average-price) options under the
risk-neutral GBM framework.

Both *arithmetic* and *geometric* averaging are covered:

Geometric Asian (exact closed-form)
    The discrete geometric average of GBM increments is itself log-normal,
    allowing an exact Black-Scholes-style formula.  See ``geometric.py``.

Arithmetic Asian (Monte Carlo)
    No closed-form exists for arithmetic averaging.  We use:
      - Crude Monte Carlo (baseline)
      - Antithetic variates (built into the path simulator)
      - Geometric average as a control variate (Kemna & Vorst 1990),
        which dramatically reduces the MC standard error.
    See ``monte_carlo.py``.

Modules
-------
payoffs
    Discrete arithmetic and geometric average payoff definitions.
geometric
    Closed-form geometric Asian call/put prices.
monte_carlo
    Arithmetic Asian MC pricer with variance reduction.
greeks
    Delta and Vega estimation via finite-difference bump-and-reprice.
convergence
    Convergence analysis and variance-reduction comparison utilities.
"""

from .payoffs import (
    arithmetic_average_payoff,
    geometric_average_payoff,
)
from .geometric import (
    geometric_asian_call,
    geometric_asian_put,
    geometric_asian_price,
)
from .monte_carlo import (
    arithmetic_asian_mc,
    ArithmeticMCResult,
)
from .greeks import (
    asian_delta,
    asian_vega,
    asian_all_greeks,
)
from .convergence import convergence_study

__all__ = [
    "arithmetic_average_payoff",
    "geometric_average_payoff",
    "geometric_asian_call",
    "geometric_asian_put",
    "geometric_asian_price",
    "arithmetic_asian_mc",
    "ArithmeticMCResult",
    "asian_delta",
    "asian_vega",
    "asian_all_greeks",
    "convergence_study",
]
