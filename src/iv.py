"""Implied volatility: safeguarded Newton-Raphson with bisection fallback.

The Black-Scholes price is strictly increasing in sigma (vega > 0), so the
implied vol is unique when the price lies inside the no-arbitrage bounds:

    call: max(S e^{-qT} - K e^{-rT}, 0) < C < S e^{-qT}
    put:  max(K e^{-rT} - S e^{-qT}, 0) < P < K e^{-rT}

Newton steps sigma <- sigma - (BS(sigma) - price) / vega converge quadratically
near the root. A bracket [lo, hi] is maintained throughout; if a Newton step
leaves it, or vega is too small, a bisection step is taken instead.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.greeks import vega as bs_vega
from src.models.bs import bs_price, validate_option_type


@dataclass(frozen=True)
class IVResult:
    """Outcome of an implied volatility solve."""

    sigma: float          # implied vol (NaN if no solution)
    iterations: int
    newton_steps: int
    bisection_steps: int
    converged: bool
    message: str


def price_bounds(S: float, K: float, T: float, r: float, option_type: str,
                 q: float = 0.0) -> tuple[float, float]:
    """No-arbitrage (lower, upper) bounds for a European option price."""
    ot = validate_option_type(option_type)
    fs, fk = S * math.exp(-q * T), K * math.exp(-r * T)
    if ot == "call":
        return max(fs - fk, 0.0), fs
    return max(fk - fs, 0.0), fk


def implied_vol_detailed(
    price: float, S: float, K: float, T: float, r: float, option_type: str = "call",
    q: float = 0.0, tol: float = 1e-8, max_iter: int = 100,
    sigma_lo: float = 1e-4, sigma_hi: float = 5.0,
) -> IVResult:
    """Solve for implied volatility and report how the solver got there.

    Args:
        price: Observed option price.
        S, K, T, r, q: Black-Scholes inputs.
        option_type: ``"call"`` or ``"put"``.
        tol: Absolute price tolerance for convergence.
        max_iter: Maximum total iterations.
        sigma_lo, sigma_hi: Initial search bracket for sigma.

    Returns:
        :class:`IVResult`. ``sigma`` is NaN when the price violates
        no-arbitrage bounds or falls outside the bracket.
    """
    ot = validate_option_type(option_type)
    if not (np.isfinite(price) and T > 0 and S > 0 and K > 0):
        return IVResult(math.nan, 0, 0, 0, False, "invalid inputs")
    lower, upper = price_bounds(S, K, T, r, ot, q)
    if price <= lower + 1e-12 or price >= upper:
        return IVResult(math.nan, 0, 0, 0, False, "price outside no-arbitrage bounds")

    def f(sig: float) -> float:
        return float(bs_price(S, K, T, r, sig, ot, q)) - price

    lo, hi = sigma_lo, sigma_hi
    f_lo, f_hi = f(lo), f(hi)
    if f_lo > 0 or f_hi < 0:
        return IVResult(math.nan, 0, 0, 0, False, "root not bracketed in [sigma_lo, sigma_hi]")

    # Brenner-Subrahmanyam ATM approximation as a starting guess.
    sigma = min(max(math.sqrt(2 * math.pi / T) * price / S, 0.05), 2.0)
    n_newton = n_bisect = 0
    for it in range(1, max_iter + 1):
        diff = f(sigma)
        if abs(diff) < tol:
            return IVResult(sigma, it, n_newton, n_bisect, True, "converged")
        if diff > 0:
            hi = sigma
        else:
            lo = sigma
        v = float(bs_vega(S, K, T, r, sigma, ot, q))
        candidate = sigma - diff / v if v > 1e-10 else math.nan
        if np.isfinite(candidate) and lo < candidate < hi:
            sigma, n_newton = candidate, n_newton + 1
        else:
            sigma, n_bisect = 0.5 * (lo + hi), n_bisect + 1
        if hi - lo < 1e-12:
            return IVResult(sigma, it, n_newton, n_bisect, True, "converged (bracket collapsed)")
    return IVResult(sigma, max_iter, n_newton, n_bisect, False, "max iterations reached")


def implied_vol(price: float, S: float, K: float, T: float, r: float,
                option_type: str = "call", q: float = 0.0, **kwargs) -> float:
    """Implied volatility (NaN if no valid solution). See :func:`implied_vol_detailed`."""
    res = implied_vol_detailed(price, S, K, T, r, option_type, q, **kwargs)
    return res.sigma if res.converged else math.nan


def implied_vol_chain(chain: pd.DataFrame, spot: float, r: float, q: float = 0.0,
                      price_col: str = "mid") -> pd.Series:
    """Implied vol for every row of an option chain.

    ``chain`` needs columns ``strike``, ``T`` (years), ``type`` and ``price_col``.
    """
    return pd.Series(
        [implied_vol(p, spot, k, t, r, ot, q)
         for p, k, t, ot in zip(chain[price_col], chain["strike"], chain["T"], chain["type"])],
        index=chain.index, name="iv",
    )
