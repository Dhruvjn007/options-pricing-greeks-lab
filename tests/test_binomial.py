"""CRR binomial tree: convergence to Black-Scholes and American option properties."""
import pytest

from src.models.binomial import crr_price
from src.models.bs import bs_price

BASE = dict(S=100.0, K=105.0, T=0.75, r=0.04, sigma=0.25)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_converges_to_bs(option_type):
    bs = bs_price(**BASE, option_type=option_type)
    errors = [abs(crr_price(**BASE, steps=n, option_type=option_type) - bs) for n in (50, 200, 1000)]
    assert errors[-1] < 5e-3
    assert errors[-1] < errors[0]  # error shrinks with steps (roughly O(1/N))


def test_converges_with_dividends():
    bs = bs_price(**BASE, option_type="call", q=0.02)
    assert crr_price(**BASE, steps=2000, option_type="call", q=0.02) == pytest.approx(bs, abs=3e-3)


def test_hull_american_put_five_steps():
    # Hull Figure 21.4: S=K=50, r=10%, sigma=40%, T=5 months, 5 steps -> 4.49
    assert crr_price(50, 50, 5 / 12, 0.10, 0.40, steps=5, option_type="put", american=True) == pytest.approx(4.49, abs=5e-3)


def test_american_put_early_exercise_premium():
    eur = crr_price(**BASE, steps=500, option_type="put")
    amer = crr_price(**BASE, steps=500, option_type="put", american=True)
    assert amer > eur


def test_american_call_no_dividend_equals_european():
    # Never optimal to exercise a call early on a non-dividend-paying stock.
    eur = crr_price(**BASE, steps=500, option_type="call")
    amer = crr_price(**BASE, steps=500, option_type="call", american=True)
    assert amer == pytest.approx(eur, abs=1e-10)


def test_invalid_steps():
    with pytest.raises(ValueError):
        crr_price(**BASE, steps=0)
