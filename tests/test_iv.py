"""Implied vol solver: round trips, edge cases and arbitrage bounds."""
import math

import pandas as pd
import pytest

from src.greeks import vega
from src.iv import implied_vol, implied_vol_chain, implied_vol_detailed
from src.models.bs import bs_price


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("K", [60, 90, 100, 110, 150])
@pytest.mark.parametrize("T", [0.02, 0.25, 1.0, 3.0])
@pytest.mark.parametrize("sigma", [0.08, 0.2, 0.6, 1.5])
def test_round_trip(option_type, K, T, sigma):
    S, r, q = 100.0, 0.04, 0.01
    price = bs_price(S, K, T, r, sigma, option_type, q)
    lower = max((S * math.exp(-q * T) - K * math.exp(-r * T)) * (1 if option_type == "call" else -1), 0)
    if price - lower < 1e-6:  # no information left in the price (pure intrinsic)
        pytest.skip("price indistinguishable from intrinsic value")
    iv = implied_vol(price, S, K, T, r, option_type, q)
    if vega(S, K, T, r, sigma, option_type, q) < 1e-2:
        # Ill-conditioned: price barely depends on vol, so check the price round-trips.
        assert bs_price(S, K, T, r, iv, option_type, q) == pytest.approx(price, abs=1e-7)
    else:
        assert iv == pytest.approx(sigma, abs=1e-5)


def test_fallback_used_for_deep_otm():
    price = bs_price(100, 130, 0.05, 0.04, 0.35, "call")
    res = implied_vol_detailed(price, 100, 130, 0.05, 0.04, "call")
    assert res.converged and res.sigma == pytest.approx(0.35, abs=1e-5)
    assert res.bisection_steps >= 1


@pytest.mark.parametrize("price", [-1.0, 0.0, 101.0, float("nan")])
def test_arbitrage_violations_return_nan(price):
    assert math.isnan(implied_vol(price, 100, 100, 1.0, 0.05, "call"))


def test_below_intrinsic_returns_nan():
    # Call with intrinsic ~ 20 * discounting quoted at 15 -> no valid vol.
    assert math.isnan(implied_vol(15.0, 120, 100, 0.5, 0.05, "call"))


def test_chain_helper():
    df = pd.DataFrame({"strike": [90, 100, 110], "T": [0.5] * 3, "type": ["put", "call", "call"]})
    df["mid"] = [bs_price(100, k, 0.5, 0.03, 0.22, t) for k, t in zip(df.strike, df.type)]
    assert implied_vol_chain(df, 100, 0.03).tolist() == pytest.approx([0.22] * 3, abs=1e-6)
