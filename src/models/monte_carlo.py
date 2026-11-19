"""Monte Carlo pricing of European options under risk-neutral GBM.

Terminal prices are simulated exactly (no discretisation error):

    S_T = S exp((r - q - sigma^2/2) T + sigma sqrt(T) Z),  Z ~ N(0, 1)

The price is e^{-rT} * mean(payoff). With antithetic variates each draw Z is
paired with -Z; the *pair average* is the i.i.d. sample used for the mean and
the standard error, which is the statistically correct treatment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from src.models.bs import validate_option_type


@dataclass(frozen=True)
class MCResult:
    """Result of a Monte Carlo pricing run.

    Attributes:
        price: Point estimate of the option value.
        std_error: Standard error of the estimate.
        ci_low: Lower bound of the confidence interval.
        ci_high: Upper bound of the confidence interval.
        n_paths: Total number of simulated terminal prices (including antithetic).
        confidence: Confidence level of the interval, e.g. 0.95.
    """

    price: float
    std_error: float
    ci_low: float
    ci_high: float
    n_paths: int
    confidence: float


def mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
    n_paths: int = 100_000,
    antithetic: bool = True,
    seed: int | None = 42,
    confidence: float = 0.95,
) -> MCResult:
    """Price a European option by Monte Carlo simulation.

    Args:
        S, K, T, r, sigma, q: Standard Black-Scholes inputs.
        option_type: ``"call"`` or ``"put"``.
        n_paths: Total number of terminal prices to simulate.
        antithetic: Use antithetic variates (Z and -Z) for variance reduction.
        seed: RNG seed for reproducibility (``None`` for random).
        confidence: Confidence level for the interval.

    Returns:
        An :class:`MCResult` with price, standard error and confidence interval.
    """
    ot = validate_option_type(option_type)
    if n_paths < 2:
        raise ValueError("n_paths must be >= 2")
    rng = np.random.default_rng(seed)
    drift = (r - q - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    disc = np.exp(-r * T)

    def discounted_payoff(z: np.ndarray) -> np.ndarray:
        s_t = S * np.exp(drift + vol * z)
        intrinsic = np.maximum(s_t - K, 0.0) if ot == "call" else np.maximum(K - s_t, 0.0)
        return disc * intrinsic

    if antithetic:
        n_pairs = (n_paths + 1) // 2
        z = rng.standard_normal(n_pairs)
        samples = 0.5 * (discounted_payoff(z) + discounted_payoff(-z))
        total = 2 * n_pairs
    else:
        samples = discounted_payoff(rng.standard_normal(n_paths))
        total = n_paths

    price = float(samples.mean())
    std_error = float(samples.std(ddof=1) / np.sqrt(samples.size))
    z_crit = float(norm.ppf(0.5 + confidence / 2.0))
    return MCResult(
        price=price,
        std_error=std_error,
        ci_low=price - z_crit * std_error,
        ci_high=price + z_crit * std_error,
        n_paths=total,
        confidence=confidence,
    )
