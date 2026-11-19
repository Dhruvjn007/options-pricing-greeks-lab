"""Cox-Ross-Rubinstein (CRR) binomial tree for European and American options.

Each of ``steps`` periods of length dt = T/steps, the price moves up by
u = e^{sigma sqrt(dt)} or down by d = 1/u. The risk-neutral up probability

    p = (e^{(r-q) dt} - d) / (u - d)

makes the discounted stock a martingale. Option values are found by backward
induction: V = e^{-r dt} [p V_up + (1-p) V_down], and for American options
V = max(continuation value, immediate exercise value) at every node.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.models.bs import validate_option_type


def _payoff(prices: NDArray[np.float64], K: float, option_type: str) -> NDArray[np.float64]:
    """Intrinsic value of the option for an array of underlying prices."""
    if option_type == "call":
        return np.maximum(prices - K, 0.0)
    return np.maximum(K - prices, 0.0)


def crr_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    steps: int = 500,
    option_type: str = "call",
    american: bool = False,
    q: float = 0.0,
) -> float:
    """Price an option on a CRR binomial tree.

    Args:
        S: Spot price.
        K: Strike.
        T: Time to expiry in years.
        r: Continuously-compounded risk-free rate.
        sigma: Annualised volatility.
        steps: Number of time steps in the tree (more steps = more accurate).
        option_type: ``"call"`` or ``"put"``.
        american: If True, allow early exercise at every node.
        q: Continuous dividend yield.

    Returns:
        Option value at the root of the tree.

    Raises:
        ValueError: If inputs are invalid or the risk-neutral probability falls
            outside (0, 1) (too few steps for the given rate/volatility).
    """
    ot = validate_option_type(option_type)
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if T <= 0:
        return float(_payoff(np.array([float(S)]), K, ot)[0])
    if sigma <= 0:
        raise ValueError("sigma must be > 0 for a binomial tree")

    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    growth = np.exp((r - q) * dt)
    p = (growth - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError(f"Risk-neutral probability p={p:.4f} outside (0,1); increase steps.")
    disc = np.exp(-r * dt)

    # Terminal prices: S * u^(steps - j) * d^j = S * u^(steps - 2j), j = number of down moves.
    j = np.arange(steps + 1)
    values = _payoff(S * u ** (steps - 2.0 * j), K, ot)

    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1.0 - p) * values[1:])
        if american:
            ji = np.arange(i + 1)
            values = np.maximum(values, _payoff(S * u ** (i - 2.0 * ji), K, ot))
    return float(values[0])
