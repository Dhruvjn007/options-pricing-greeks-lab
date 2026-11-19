"""Analytic Greeks vs textbook values and finite differences."""
import numpy as np
import pytest

from src.greeks import analytic_greeks, finite_difference_greeks, gamma, theta, delta, vega, rho
from src.models.bs import bs_price


def test_hull_textbook_greeks():
    # Hull Examples 19.1-19.6: S=49, K=50, r=5%, sigma=20%, T=20 weeks
    g = analytic_greeks(49, 50, 0.3846, 0.05, 0.20, "call")
    assert g.delta == pytest.approx(0.522, abs=1e-3)
    assert g.gamma == pytest.approx(0.066, abs=1e-3)
    assert g.vega == pytest.approx(12.1, abs=0.05)
    assert g.theta == pytest.approx(-4.31, abs=0.01)
    assert g.rho == pytest.approx(8.91, abs=0.01)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("S,T,q", [(80, 0.25, 0.0), (100, 1.0, 0.02), (125, 2.0, 0.01), (100, 0.02, 0.0)])
def test_analytic_matches_finite_difference(option_type, S, T, q):
    a = analytic_greeks(S, 100, T, 0.04, 0.3, option_type, q).as_dict()
    f = finite_difference_greeks(S, 100, T, 0.04, 0.3, option_type, q).as_dict()
    for name in a:
        assert a[name] == pytest.approx(f[name], rel=1e-4, abs=1e-6), name


def test_call_put_relationships():
    args = (100, 95, 0.5, 0.03, 0.25)
    q = 0.01
    # delta_call - delta_put = e^{-qT}; gamma and vega identical
    assert delta(*args, "call", q) - delta(*args, "put", q) == pytest.approx(np.exp(-q * 0.5))
    assert gamma(*args, "call", q) == pytest.approx(gamma(*args, "put", q))
    assert vega(*args, "call", q) == pytest.approx(vega(*args, "put", q))


def test_black_scholes_pde():
    # theta + 0.5 sigma^2 S^2 gamma + (r - q) S delta - r V = 0
    S, K, T, r, sig, q = 103.0, 100.0, 0.7, 0.045, 0.22, 0.015
    for ot in ("call", "put"):
        V = bs_price(S, K, T, r, sig, ot, q)
        lhs = (theta(S, K, T, r, sig, ot, q) + 0.5 * sig**2 * S**2 * gamma(S, K, T, r, sig, ot, q)
               + (r - q) * S * delta(S, K, T, r, sig, ot, q) - r * V)
        assert lhs == pytest.approx(0.0, abs=1e-10)


def test_rho_signs_and_vectorised():
    S = np.linspace(50, 150, 11)
    assert np.all(rho(S, 100, 1, 0.05, 0.2, "call") > 0)
    assert np.all(rho(S, 100, 1, 0.05, 0.2, "put") < 0)
