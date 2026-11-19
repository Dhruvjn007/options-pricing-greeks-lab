"""Monte Carlo: unbiasedness within the CI and antithetic variance reduction."""
import pytest

from src.models.bs import bs_price
from src.models.monte_carlo import mc_price

BASE = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_bs_inside_confidence_interval(option_type):
    bs = bs_price(**BASE, option_type=option_type)
    res = mc_price(**BASE, option_type=option_type, n_paths=200_000, seed=1, confidence=0.99)
    assert res.ci_low <= bs <= res.ci_high


def test_antithetic_reduces_standard_error():
    plain = mc_price(**BASE, n_paths=100_000, antithetic=False, seed=3)
    anti = mc_price(**BASE, n_paths=100_000, antithetic=True, seed=3)
    assert anti.std_error < plain.std_error


def test_reproducible_with_seed():
    assert mc_price(**BASE, seed=11).price == mc_price(**BASE, seed=11).price


def test_error_scales_like_inverse_sqrt_n():
    small = mc_price(**BASE, n_paths=10_000, seed=5).std_error
    large = mc_price(**BASE, n_paths=1_000_000, seed=5).std_error
    assert large / small == pytest.approx(0.1, rel=0.15)
