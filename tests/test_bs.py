"""Black-Scholes against textbook values and no-arbitrage identities."""
import math

import numpy as np
import pytest

from src.models.bs import bs_price


@pytest.mark.parametrize(
    "S,K,T,r,sigma,q,call,put",
    [
        # Hull, Options Futures & Other Derivatives, Example 15.6
        (42, 40, 0.5, 0.10, 0.20, 0.0, 4.7594, 0.8086),
        # Hull Example 17.1 / delta-hedging example (S=49, K=50, ~20 weeks)
        (49, 50, 0.3846, 0.05, 0.20, 0.0, 2.4005, None),
        # Hull Example 17.1 index option with dividend yield (Merton)
        (930, 900, 2 / 12, 0.08, 0.20, 0.03, 51.83, None),
    ],
)
def test_textbook_values(S, K, T, r, sigma, q, call, put):
    assert bs_price(S, K, T, r, sigma, "call", q) == pytest.approx(call, abs=5e-3)
    if put is not None:
        assert bs_price(S, K, T, r, sigma, "put", q) == pytest.approx(put, abs=5e-3)


@pytest.mark.parametrize("S", [60, 100, 140])
@pytest.mark.parametrize("T", [0.05, 1.0, 3.0])
@pytest.mark.parametrize("q", [0.0, 0.02])
def test_put_call_parity(S, T, q):
    K, r, sigma = 100.0, 0.04, 0.3
    c = bs_price(S, K, T, r, sigma, "call", q)
    p = bs_price(S, K, T, r, sigma, "put", q)
    assert c - p == pytest.approx(S * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-10)


def test_vectorised_matches_scalar():
    S = np.array([80.0, 100.0, 120.0])
    vec = bs_price(S, 100, 0.5, 0.03, 0.25, "put")
    assert isinstance(vec, np.ndarray)
    for s, v in zip(S, vec):
        assert bs_price(s, 100, 0.5, 0.03, 0.25, "put") == pytest.approx(v)


def test_expiry_and_zero_vol_limits():
    assert bs_price(110, 100, 0.0, 0.05, 0.2, "call") == pytest.approx(10.0)
    assert bs_price(90, 100, 0.0, 0.05, 0.2, "put") == pytest.approx(10.0)
    # zero vol: deterministic forward, discounted
    assert bs_price(100, 90, 1.0, 0.05, 0.0, "call") == pytest.approx(100 - 90 * math.exp(-0.05))


def test_price_bounds_and_monotonicity():
    S, K, T, r = 100, 100, 1.0, 0.05
    vols = np.linspace(0.05, 1.0, 20)
    prices = bs_price(S, K, T, r, vols, "call")
    assert np.all(np.diff(prices) > 0)  # increasing in vol (vega > 0)
    assert np.all(prices < S) and np.all(prices > S - K * math.exp(-r * T))


def test_invalid_type():
    with pytest.raises(ValueError):
        bs_price(100, 100, 1, 0.05, 0.2, "straddle")
