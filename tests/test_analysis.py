"""Smile/skew and parity helpers on the deterministic sample chain."""
import pandas as pd
import pytest

from src.analysis import add_implied_vols, otm_smile, parity_table, skew_summary
from src.data import clean_chain, make_sample_chain


@pytest.fixture(scope="module")
def sample():
    return make_sample_chain(now=pd.Timestamp("2026-01-05 10:00", tz="America/New_York"))


def test_sample_iv_recovers_generating_vols(sample):
    clean, _ = clean_chain(sample.chain, sample.spot, min_volume=0, max_spread_pct=1.0)
    iv = add_implied_vols(clean, sample.spot, r=0.04, q=0.012)
    ok = iv["iv"].notna()
    # mids are fair values +/- symmetric noise-free spread -> IV ~ generating vol
    assert (iv.loc[ok, "iv"] - iv.loc[ok, "yahoo_iv"]).abs().median() < 0.01


def test_skew_is_negative_slope(sample):
    clean, _ = clean_chain(sample.chain, sample.spot)
    smile = otm_smile(add_implied_vols(clean, sample.spot, r=0.04))
    table = skew_summary(smile)
    assert (table["skew"].dropna() > 0).all()  # downside vol > upside vol


def test_parity_holds_within_spread_on_model_prices(sample):
    exp = sample.expiries[2]
    tab = parity_table(sample.chain, sample.spot, r=0.04, q=0.012, expiry=exp)
    assert len(tab) > 10
    assert tab["deviation"].abs().median() < 0.05
    assert tab["within_spread"].mean() > 0.9
