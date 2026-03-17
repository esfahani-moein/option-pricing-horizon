"""
option_pricing_horizon.common.market_params
==========================================
Central dataclass that bundles all market and contract parameters used
throughout the pricing library.

Using a dataclass guarantees immutability (via ``frozen=True``) so that
no pricing routine can accidentally mutate a shared parameter set.

Baseline parameters (from project brief / image.png)
-----------------------------------------------------
S0 = 100,  K = 100,  r = 0.03,  sigma = 0.20,  T = 1.0,  N = 252
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MarketParams:
    """Container for all market and contract parameters.

    Parameters
    ----------
    S0 : float
        Current (spot) price of the underlying asset.
    K : float
        Strike price of the option.
    r : float
        Continuously compounded risk-free interest rate (annualised).
        Example: 0.03 for 3 %.
    sigma : float
        Annualised volatility of the underlying asset.
        Example: 0.20 for 20 %.
    T : float
        Time to expiry in years.  Example: 1.0 for one year.
    N : int
        Number of discrete monitoring dates (relevant for Asian options).
        For European options this is unused.
        Default: 252 (one observation per trading day).
    q : float
        Continuous dividend yield.  Default: 0.0 (no dividends).

    Examples
    --------
    >>> from option_pricing_horizon.common.market_params import MarketParams
    >>> mp = MarketParams(S0=100, K=100, r=0.03, sigma=0.20, T=1.0)
    >>> mp.S0
    100
    >>> mp.forward_price
    103.04545...
    """

    S0: float
    K: float
    r: float
    sigma: float
    T: float
    N: int = 252
    q: float = 0.0

    # ------------------------------------------------------------------
    # Derived quantities (computed once, available as read-only properties)
    # ------------------------------------------------------------------

    @property
    def dt(self) -> float:
        """Length of one time step: T / N (in years)."""
        return self.T / self.N

    @property
    def forward_price(self) -> float:
        """Risk-neutral forward price: S0 * exp((r - q) * T)."""
        import math
        return self.S0 * math.exp((self.r - self.q) * self.T)

    @property
    def discount_factor(self) -> float:
        """Discount factor: exp(-r * T)."""
        import math
        return math.exp(-self.r * self.T)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.S0 <= 0:
            raise ValueError(f"S0 must be positive; got {self.S0}")
        if self.K <= 0:
            raise ValueError(f"K must be positive; got {self.K}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive; got {self.sigma}")
        if self.T <= 0:
            raise ValueError(f"T must be positive; got {self.T}")
        if self.N < 1:
            raise ValueError(f"N must be >= 1; got {self.N}")
        if self.r < 0:
            raise ValueError(f"r must be non-negative; got {self.r}")
        if self.q < 0:
            raise ValueError(f"q must be non-negative; got {self.q}")

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def baseline(cls) -> "MarketParams":
        """Return the baseline parameters from the project brief.

        S0=100, K=100, r=3%, sigma=20%, T=1yr, N=252 trading days.
        """
        return cls(S0=100.0, K=100.0, r=0.03, sigma=0.20, T=1.0, N=252)

    def replace(self, **kwargs) -> "MarketParams":
        """Return a new :class:`MarketParams` with selected fields overridden.

        Parameters
        ----------
        **kwargs
            Any subset of the dataclass fields to override.

        Examples
        --------
        >>> mp = MarketParams.baseline()
        >>> mp_high_vol = mp.replace(sigma=0.40)
        >>> mp_high_vol.sigma
        0.4
        """
        import dataclasses
        return dataclasses.replace(self, **kwargs)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"MarketParams(S0={self.S0}, K={self.K}, r={self.r}, "
            f"sigma={self.sigma}, T={self.T}, N={self.N}, q={self.q})"
        )
