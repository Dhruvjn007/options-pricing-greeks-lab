"""Data layer: sample chain, cleaning filters and implied forward (offline only)."""
import math

import pandas as pd
import pytest

from src.data import clean_chain, implied_forward, make_sample_chain


@pytest.fixture(scope="module")
def sample():
    return make_sample_chain(now=pd.Timestamp("2026-01-05 10:00", tz="America/New_York"))


def test_sample_chain_shape(sample):
    assert sample.source == "sample" and len(sample.chain) > 200
    assert set(sample.chain["type"]) == {"call", "put"}


def test_clean_chain_filters(sample):
    clean, report = clean_chain(sample.chain, sample.spot, min_volume=1, max_spread_pct=0.2)
    assert report.kept == len(clean) < report.total
    assert (clean["volume"] >= 1).all()
    assert (clean["spread_pct"] <= 0.2).all()
    assert ((clean["bid"] > 0) & (clean["ask"] >= clean["bid"])).all()


def test_implied_forward_recovers_dividend_yield(sample):
    clean, _ = clean_chain(sample.chain, sample.spot, min_volume=0, max_spread_pct=1.0)
    fwd = implied_forward(clean, sample.spot, r=0.04)
    # Sample chain was generated with q = 1.2%; mid noise keeps it approximate.
    long_dated = fwd[fwd["T"] > 0.2]
    assert long_dated["q_implied"].median() == pytest.approx(0.012, abs=0.004)
    assert (fwd["forward"] > 0).all() and not math.isnan(fwd["forward"].iloc[0])
