"""Market data: live option chains via yfinance, cleaning, and a synthetic fallback.

Everything here degrades gracefully: any network/API failure returns a
deterministic synthetic chain (clearly labelled ``source="sample"``) so the app
always renders. No secrets and no local file paths are used.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.models.bs import bs_price

NY_TZ = "America/New_York"
DEFAULT_RATE = 0.04
CHAIN_COLUMNS = [
    "expiry", "T", "days", "type", "strike", "bid", "ask", "last",
    "volume", "open_interest", "yahoo_iv",
]


@dataclass
class ChainData:
    """An option chain snapshot.

    Attributes:
        ticker: Underlying symbol.
        spot: Underlying price used for the snapshot.
        asof: Snapshot timestamp (New York time).
        chain: Raw chain with columns ``CHAIN_COLUMNS``.
        source: ``"live"`` (yfinance) or ``"sample"`` (synthetic fallback).
        message: Human-readable note (e.g. why the fallback was used).
    """

    ticker: str
    spot: float
    asof: pd.Timestamp
    chain: pd.DataFrame
    source: str
    message: str = ""
    expiries: list[str] = field(default_factory=list)


def _year_fraction(expiry: str, now: pd.Timestamp) -> float:
    """ACT/365 time from ``now`` to 16:00 New York time on ``expiry``."""
    exp_ts = pd.Timestamp(expiry).tz_localize(NY_TZ) + pd.Timedelta(hours=16)
    return max((exp_ts - now).total_seconds() / (365.0 * 24 * 3600), 1.0 / (365 * 24))


def _pick_expiries(expiries: list[str], now: pd.Timestamp, n: int,
                   min_days: int, max_days: int) -> list[str]:
    """Pick up to ``n`` expiries spread evenly across [min_days, max_days]."""
    eligible = [e for e in expiries
                if min_days <= (pd.Timestamp(e).date() - now.date()).days <= max_days]
    if len(eligible) <= n:
        return eligible
    idx = np.unique(np.linspace(0, len(eligible) - 1, n).round().astype(int))
    return [eligible[i] for i in idx]


def fetch_risk_free_rate() -> tuple[float, str]:
    """13-week T-bill yield (^IRX) as a continuously-compounded rate.

    Returns:
        (rate, source description). Falls back to ``DEFAULT_RATE`` on failure.
    """
    try:
        import yfinance as yf

        hist = yf.Ticker("^IRX").history(period="10d")
        y = float(hist["Close"].dropna().iloc[-1]) / 100.0
        if not 0.0 <= y < 0.25:
            raise ValueError(f"implausible yield {y}")
        return math.log1p(y), "13-week T-bill (^IRX), converted to continuous compounding"
    except Exception as exc:  # noqa: BLE001 - any failure -> default
        return DEFAULT_RATE, f"default {DEFAULT_RATE:.2%} (T-bill fetch failed: {type(exc).__name__})"


def fetch_option_chain(ticker: str = "SPY", max_expiries: int = 6,
                       min_days: int = 7, max_days: int = 400) -> ChainData:
    """Download an option chain from Yahoo Finance, falling back to sample data.

    Args:
        ticker: Underlying symbol, e.g. ``"SPY"``.
        max_expiries: Maximum number of expiries to download (spread across the range).
        min_days, max_days: Expiry window in calendar days.

    Returns:
        :class:`ChainData`; ``source`` is ``"sample"`` if the live fetch failed.
    """
    ticker = ticker.strip().upper()
    now = pd.Timestamp.now(tz=NY_TZ)
    try:
        import yfinance as yf

        tk = yf.Ticker(ticker)
        all_expiries = list(tk.options)
        if not all_expiries:
            raise ValueError(f"no listed options for {ticker}")
        hist = tk.history(period="5d")
        if hist.empty:
            raise ValueError(f"no price history for {ticker}")
        spot = float(hist["Close"].dropna().iloc[-1])

        expiries = _pick_expiries(all_expiries, now, max_expiries, min_days, max_days)
        if not expiries:
            raise ValueError("no expiries in the requested window")

        frames = []
        for exp in expiries:
            oc = tk.option_chain(exp)
            for opt_type, df in (("call", oc.calls), ("put", oc.puts)):
                if df is None or df.empty:
                    continue
                frames.append(pd.DataFrame({
                    "expiry": exp,
                    "T": _year_fraction(exp, now),
                    "days": (pd.Timestamp(exp).date() - now.date()).days,
                    "type": opt_type,
                    "strike": df["strike"].astype(float),
                    "bid": df["bid"].astype(float),
                    "ask": df["ask"].astype(float),
                    "last": df["lastPrice"].astype(float),
                    "volume": df["volume"].fillna(0).astype(float),
                    "open_interest": df["openInterest"].fillna(0).astype(float),
                    "yahoo_iv": df["impliedVolatility"].astype(float),
                }))
        if not frames:
            raise ValueError("option chain download returned no rows")
        chain = pd.concat(frames, ignore_index=True)[CHAIN_COLUMNS]
        return ChainData(ticker, spot, now, chain, "live",
                         f"Live Yahoo Finance data ({len(expiries)} expiries).", expiries)
    except Exception as exc:  # noqa: BLE001 - network/API/parsing failures -> fallback
        sample = make_sample_chain(now=now)
        sample.message = (f"Live data for {ticker} unavailable ({type(exc).__name__}: {exc}). "
                          "Showing a synthetic SPY-like sample chain instead.")
        return sample


def make_sample_chain(spot: float = 600.0, r: float = DEFAULT_RATE, q: float = 0.012,
                      now: pd.Timestamp | None = None, seed: int = 7) -> ChainData:
    """Deterministic synthetic index-option chain with a realistic equity skew.

    Vols follow sigma(k, T) = atm(T) + b(T) k + c k^2 with k = ln(K/F), a negative
    skew slope steepening at short maturities, and an upward-sloping ATM term
    structure. Bid/ask spreads widen and volume falls away from the money; a few
    zero-volume / wide quotes are included so the liquidity filters have work to do.
    """
    rng = np.random.default_rng(seed)
    now = now or pd.Timestamp.now(tz=NY_TZ)
    day_list = [14, 30, 60, 91, 182, 365]
    expiries = [(now.normalize() + pd.Timedelta(days=d)).strftime("%Y-%m-%d") for d in day_list]
    rows = []
    for exp, days in zip(expiries, day_list):
        T = _year_fraction(exp, now)
        F = spot * math.exp((r - q) * T)
        atm = 0.15 + 0.03 * (1 - math.exp(-2 * T))
        slope = -0.12 / math.sqrt(T) - 0.15
        width = 0.12 + 0.25 * math.sqrt(T)
        strikes = np.arange(round(spot * (1 - width) / 5) * 5, spot * (1 + 0.6 * width), 5.0)
        for K in strikes:
            k = math.log(K / F)
            sig = float(np.clip(atm + slope * k + 1.2 * k * k, 0.06, 1.2))
            for opt_type in ("call", "put"):
                fair = float(bs_price(spot, K, T, r, sig, opt_type, q))
                if fair < 0.02:
                    continue
                half = max(0.01, 0.015 * fair + 0.02 * abs(k) * fair) * rng.uniform(0.8, 1.2)
                otm = (opt_type == "put" and K < F) or (opt_type == "call" and K >= F)
                vol_lvl = 4000 * math.exp(-abs(k) * 12) * (1.0 if otm else 0.3)
                volume = float(rng.poisson(vol_lvl)) if rng.uniform() > 0.08 else 0.0
                if rng.uniform() < 0.05:  # occasional stale, very wide quote
                    half *= 8
                bid = max(round(fair - half, 2), 0.0)
                ask = round(fair + half, 2)
                rows.append({
                    "expiry": exp, "T": T, "days": days, "type": opt_type, "strike": float(K),
                    "bid": bid, "ask": ask, "last": round(fair * rng.uniform(0.97, 1.03), 2),
                    "volume": volume, "open_interest": float(rng.poisson(vol_lvl * 5)),
                    "yahoo_iv": sig,
                })
    chain = pd.DataFrame(rows)[CHAIN_COLUMNS]
    return ChainData("SPY (sample)", spot, now, chain, "sample",
                     "Synthetic sample chain (deterministic, for offline use).", expiries)


@dataclass
class CleanReport:
    """How many rows each liquidity filter removed (applied in order)."""

    total: int
    removed: dict[str, int]
    kept: int


def clean_chain(chain: pd.DataFrame, spot: float, min_volume: float = 1,
                max_spread_pct: float = 0.25, min_price: float = 0.05,
                moneyness: tuple[float, float] = (0.7, 1.3),
                use_last_if_no_quote: bool = False) -> tuple[pd.DataFrame, CleanReport]:
    """Filter an option chain down to liquid, usable quotes and add a ``mid`` price.

    Filters (in order): valid two-sided quote, volume, relative bid-ask spread,
    minimum price, and strike/spot moneyness window.

    Args:
        chain: Raw chain (``CHAIN_COLUMNS``).
        spot: Underlying price.
        min_volume: Minimum traded volume today.
        max_spread_pct: Maximum (ask - bid) / mid.
        min_price: Minimum option mid price.
        moneyness: Allowed (low, high) range of strike / spot.
        use_last_if_no_quote: Use last trade price when bid/ask are missing
            (e.g. outside market hours); spread filter is skipped for such rows.

    Returns:
        (filtered DataFrame with ``mid``, ``spread_pct``, ``moneyness`` columns, report).
    """
    df = chain.copy()
    df["has_quote"] = (df["bid"] > 0) & (df["ask"] > 0) & (df["ask"] >= df["bid"])
    df["mid"] = np.where(df["has_quote"], 0.5 * (df["bid"] + df["ask"]), np.nan)
    df["spread_pct"] = np.where(df["has_quote"], (df["ask"] - df["bid"]) / df["mid"], np.nan)
    if use_last_if_no_quote:
        df["mid"] = df["mid"].fillna(df["last"].where(df["last"] > 0))
    df["moneyness"] = df["strike"] / spot

    removed: dict[str, int] = {}
    total = len(df)

    def apply(name: str, mask: pd.Series) -> None:
        nonlocal df
        removed[name] = int((~mask).sum())
        df = df[mask]

    apply("no valid quote", df["mid"].notna())
    apply(f"volume < {min_volume:g}", df["volume"] >= min_volume)
    apply(f"spread > {max_spread_pct:.0%} of mid",
          df["spread_pct"].isna() | (df["spread_pct"] <= max_spread_pct))
    apply(f"price < {min_price:g}", df["mid"] >= min_price)
    apply("outside moneyness window",
          df["moneyness"].between(moneyness[0], moneyness[1]))
    return df.reset_index(drop=True), CleanReport(total, removed, len(df))


def implied_forward(chain: pd.DataFrame, spot: float, r: float, n_strikes: int = 5) -> pd.DataFrame:
    """Implied forward and dividend yield per expiry from put-call parity.

    For each expiry, takes the ``n_strikes`` strikes closest to spot that have both
    a call and a put mid, computes F = K + e^{rT} (C - P), and uses the median.
    The implied continuous yield is q = r - ln(F / S) / T.

    Returns:
        DataFrame indexed by expiry with columns ``T``, ``forward``, ``q_implied``.
    """
    out = []
    for exp, g in chain.groupby("expiry"):
        calls = g[g["type"] == "call"].set_index("strike")["mid"]
        puts = g[g["type"] == "put"].set_index("strike")["mid"]
        both = pd.concat([calls.rename("C"), puts.rename("P")], axis=1, join="inner").dropna()
        if both.empty:
            continue
        T = float(g["T"].iloc[0])
        both = both.iloc[np.argsort(np.abs(both.index.values - spot))[:n_strikes]]
        fwd = float(np.median(both.index.values + math.exp(r * T) * (both["C"] - both["P"])))
        if fwd <= 0:
            continue
        out.append({"expiry": exp, "T": T, "forward": fwd,
                    "q_implied": r - math.log(fwd / spot) / T})
    return pd.DataFrame(out).set_index("expiry") if out else pd.DataFrame(
        columns=["T", "forward", "q_implied"])
