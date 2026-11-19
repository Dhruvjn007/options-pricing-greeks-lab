"""Strategy payoffs, breakevens and max profit/loss."""
import math

import numpy as np
import pytest

from src.strategies import STRATEGY_STRIKES, Leg, Strategy, build_strategy

MKT = dict(S=100.0, T=0.25, r=0.04, sigma=0.2)


def test_long_call_analytics():
    s = build_strategy("Long call", [100], **MKT)
    st = s.analyze()
    prem = s.legs[0].premium
    assert st.breakevens == pytest.approx([100 + prem])
    assert math.isinf(st.max_profit) and st.max_loss == pytest.approx(-prem)


def test_straddle_breakevens():
    s = build_strategy("Straddle", [100], **MKT)
    cost = s.net_cost
    assert s.analyze().breakevens == pytest.approx([100 - cost, 100 + cost])


def test_iron_condor_defined_risk():
    s = build_strategy("Iron condor", [90, 95, 105, 110], **MKT)
    st = s.analyze()
    credit = -s.net_cost
    assert credit > 0
    assert st.max_profit == pytest.approx(credit)
    assert st.max_loss == pytest.approx(credit - 5.0)  # wing width 5
    assert st.breakevens == pytest.approx([95 - credit, 105 + credit])


def test_bull_call_spread():
    s = build_strategy("Bull call spread", [100, 110], **MKT)
    st = s.analyze()
    assert st.max_profit == pytest.approx(10 - s.net_cost)
    assert st.max_loss == pytest.approx(-s.net_cost)


def test_covered_call():
    s = build_strategy("Covered call", [105], **MKT)
    st = s.analyze()
    prem = s.legs[1].premium
    assert st.max_profit == pytest.approx(105 - 100 + prem)
    assert st.breakevens == pytest.approx([100 - prem])
    assert st.max_loss == pytest.approx(-(100 - prem))


def test_naked_short_call_unlimited_loss():
    st = Strategy("Short call", [Leg("call", -1, 100, 3.0)]).analyze()
    assert st.max_loss == float("-inf") and st.max_profit == pytest.approx(3.0)


def test_pnl_now_converges_to_expiry_payoff():
    s = build_strategy("Iron condor", [90, 95, 105, 110], **MKT)
    grid = np.linspace(70, 130, 61)
    assert np.allclose(s.pnl_now(grid, 0.0, 0.04, 0.2), s.pnl_at_expiry(grid), atol=1e-6)


def test_validation():
    with pytest.raises(ValueError):
        build_strategy("Iron condor", [95, 90, 105, 110], **MKT)
    with pytest.raises(ValueError):
        build_strategy("Butterfly", [100], **MKT)
    assert set(STRATEGY_STRIKES) >= {"Long call", "Straddle", "Strangle", "Bull call spread",
                                     "Iron condor", "Covered call"}
