"""Black-Scholes Greeks: analytic formulas and finite-difference verification.

Raw units (returned by the analytic functions):
    delta: dV/dS             gamma: d2V/dS2
    vega:  dV/dsigma (per 1.00 = 100 vol points)
    theta: dV/dt (per year; t is calendar time, so theta = -dV/dT)
    rho:   dV/dr (per 1.00 = 100% rate change)

``Greeks.display()`` converts to trader conventions: vega and rho per 1%,
theta per calendar day.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr
from scipy.stats import norm

from src.models.bs import bs_price, d1_d2, validate_option_type

FloatOrArray = Union[float, NDArray[np.float64]]


def _out(x: NDArray[np.float64]) -> FloatOrArray:
    return float(x) if np.ndim(x) == 0 else x


def _prep(S, K, T, r, sigma, q):
    S, K, T, r, sigma, q = np.broadcast_arrays(
        *(np.asarray(a, dtype=float) for a in (S, K, T, r, sigma, q))
    )
    if np.any(T <= 0) or np.any(sigma <= 0):
        raise ValueError("Analytic Greeks require T > 0 and sigma > 0")
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    return S, K, T, r, sigma, q, d1, d2


def delta(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike,
          option_type: str = "call", q: ArrayLike = 0.0) -> FloatOrArray:
    """Delta: e^{-qT} N(d1) for calls, e^{-qT} (N(d1) - 1) for puts."""
    ot = validate_option_type(option_type)
    S, K, T, r, sigma, q, d1, _ = _prep(S, K, T, r, sigma, q)
    dq = np.exp(-q * T)
    return _out(dq * ndtr(d1) if ot == "call" else dq * (ndtr(d1) - 1.0))


def gamma(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike,
          option_type: str = "call", q: ArrayLike = 0.0) -> FloatOrArray:
    """Gamma (same for calls and puts): e^{-qT} n(d1) / (S sigma sqrt(T))."""
    validate_option_type(option_type)
    S, K, T, r, sigma, q, d1, _ = _prep(S, K, T, r, sigma, q)
    return _out(np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def vega(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike,
         option_type: str = "call", q: ArrayLike = 0.0) -> FloatOrArray:
    """Vega (same for calls and puts): S e^{-qT} n(d1) sqrt(T), per 1.00 of vol."""
    validate_option_type(option_type)
    S, K, T, r, sigma, q, d1, _ = _prep(S, K, T, r, sigma, q)
    return _out(S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T))


def theta(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike,
          option_type: str = "call", q: ArrayLike = 0.0) -> FloatOrArray:
    """Theta per year (dV/dt = -dV/dT).

    Call: -S e^{-qT} n(d1) sigma / (2 sqrt T) - r K e^{-rT} N(d2) + q S e^{-qT} N(d1)
    Put:  -S e^{-qT} n(d1) sigma / (2 sqrt T) + r K e^{-rT} N(-d2) - q S e^{-qT} N(-d1)
    """
    ot = validate_option_type(option_type)
    S, K, T, r, sigma, q, d1, d2 = _prep(S, K, T, r, sigma, q)
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    decay = -S * dq * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    if ot == "call":
        return _out(decay - r * K * dr * ndtr(d2) + q * S * dq * ndtr(d1))
    return _out(decay + r * K * dr * ndtr(-d2) - q * S * dq * ndtr(-d1))


def rho(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike,
        option_type: str = "call", q: ArrayLike = 0.0) -> FloatOrArray:
    """Rho per 1.00 of rate: K T e^{-rT} N(d2) for calls, -K T e^{-rT} N(-d2) for puts."""
    ot = validate_option_type(option_type)
    S, K, T, r, sigma, q, _, d2 = _prep(S, K, T, r, sigma, q)
    kt = K * T * np.exp(-r * T)
    return _out(kt * ndtr(d2) if ot == "call" else -kt * ndtr(-d2))


@dataclass(frozen=True)
class Greeks:
    """A set of option Greeks in raw (per-unit, per-year) units."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def display(self) -> dict[str, float]:
        """Trader conventions: vega/rho per 1 percentage point, theta per calendar day."""
        return {
            "Delta": self.delta,
            "Gamma": self.gamma,
            "Vega (per 1 vol pt)": self.vega / 100.0,
            "Theta (per day)": self.theta / 365.0,
            "Rho (per 1% rate)": self.rho / 100.0,
        }

    def as_dict(self) -> dict[str, float]:
        """Raw units as a plain dict."""
        return asdict(self)


def analytic_greeks(S: float, K: float, T: float, r: float, sigma: float,
                    option_type: str = "call", q: float = 0.0) -> Greeks:
    """All five Black-Scholes Greeks from the closed-form formulas."""
    args = (S, K, T, r, sigma, option_type, q)
    return Greeks(
        delta=float(delta(*args)), gamma=float(gamma(*args)), vega=float(vega(*args)),
        theta=float(theta(*args)), rho=float(rho(*args)),
    )


def finite_difference_greeks(S: float, K: float, T: float, r: float, sigma: float,
                             option_type: str = "call", q: float = 0.0) -> Greeks:
    """Greeks by central finite differences on the Black-Scholes price.

    Central differences have O(h^2) truncation error. Gamma uses a larger bump
    because the second difference divides by h^2 and amplifies round-off.
    """
    def V(s: float = S, t: float = T, rr: float = r, v: float = sigma) -> float:
        return float(bs_price(s, K, t, rr, v, option_type, q))

    h_s, h_g = 1e-4 * S, 1e-3 * S
    h_v, h_r = 1e-4, 1e-5
    h_t = min(1e-5, 0.5 * T)
    return Greeks(
        delta=(V(s=S + h_s) - V(s=S - h_s)) / (2 * h_s),
        gamma=(V(s=S + h_g) - 2 * V() + V(s=S - h_g)) / h_g**2,
        vega=(V(v=sigma + h_v) - V(v=sigma - h_v)) / (2 * h_v),
        theta=-(V(t=T + h_t) - V(t=T - h_t)) / (2 * h_t),
        rho=(V(rr=r + h_r) - V(rr=r - h_r)) / (2 * h_r),
    )
