"""Black-Scholes-Merton closed-form pricing for European options.

Assumptions: the underlying follows geometric Brownian motion with constant
volatility ``sigma``, a constant continuously-compounded risk-free rate ``r``
and a continuous dividend yield ``q``; no transaction costs; continuous trading.

    C = S e^{-qT} N(d1) - K e^{-rT} N(d2)
    P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)
    d1 = [ln(S/K) + (r - q + sigma^2 / 2) T] / (sigma sqrt(T)),  d2 = d1 - sigma sqrt(T)

All functions are vectorised: any argument may be a NumPy array and the usual
broadcasting rules apply. Scalar inputs return a Python ``float``.
"""
from __future__ import annotations

from typing import Literal, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr  # standard normal CDF, fast for arrays and scalars

OptionType = Literal["call", "put"]
FloatOrArray = Union[float, NDArray[np.float64]]


def validate_option_type(option_type: str) -> OptionType:
    """Normalise and validate an option type string ('call' or 'put')."""
    ot = option_type.lower().strip()
    if ot not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    return ot  # type: ignore[return-value]


def _as_output(x: NDArray[np.float64]) -> FloatOrArray:
    """Return a float for 0-d arrays, otherwise the array itself."""
    return float(x) if np.ndim(x) == 0 else x


def d1_d2(
    S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike = 0.0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the Black-Scholes d1 and d2 terms.

    Requires T > 0 and sigma > 0 (callers handle the degenerate cases).
    """
    S, K, T, r, sigma, q = (np.asarray(a, dtype=float) for a in (S, K, T, r, sigma, q))
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def bs_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    option_type: str = "call",
    q: ArrayLike = 0.0,
) -> FloatOrArray:
    """Black-Scholes-Merton price of a European call or put.

    Args:
        S: Spot price of the underlying.
        K: Strike price.
        T: Time to expiry in years.
        r: Continuously-compounded risk-free rate (0.05 = 5%).
        sigma: Annualised volatility (0.20 = 20%).
        option_type: ``"call"`` or ``"put"``.
        q: Continuous dividend yield.

    Returns:
        The option price. When ``T <= 0`` or ``sigma <= 0`` the deterministic
        limit is returned (intrinsic value on the discounted forward).
    """
    ot = validate_option_type(option_type)
    S, K, T, r, sigma, q = np.broadcast_arrays(
        *(np.asarray(a, dtype=float) for a in (S, K, T, r, sigma, q))
    )
    T_pos = np.maximum(T, 0.0)
    disc_s = S * np.exp(-q * T_pos)  # PV of the stock delivered at T
    disc_k = K * np.exp(-r * T_pos)  # PV of the strike paid at T

    degenerate = (T <= 0) | (sigma <= 0)
    # Safe placeholders avoid divide-by-zero warnings in the degenerate branch.
    T_safe = np.where(degenerate, 1.0, T)
    sig_safe = np.where(degenerate, 1.0, sigma)
    d1, d2 = d1_d2(S, K, T_safe, r, sig_safe, q)

    if ot == "call":
        price = disc_s * ndtr(d1) - disc_k * ndtr(d2)
        limit = np.maximum(disc_s - disc_k, 0.0)
    else:
        price = disc_k * ndtr(-d2) - disc_s * ndtr(-d1)
        limit = np.maximum(disc_k - disc_s, 0.0)
    return _as_output(np.where(degenerate, limit, price))
