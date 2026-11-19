"""Analysis helpers shared by the app: IV smile/skew tables and put-call parity checks."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from src.data import implied_forward
from src.iv import implied_vol


def add_implied_vols(chain: pd.DataFrame, spot: float, r: float, q: float | None = None) -> pd.DataFrame:
    """Attach forward, dividend yield, IV and an OTM flag to a cleaned chain.

    Args:
        chain: Cleaned chain with a ``mid`` column (see :func:`src.data.clean_chain`).
        spot: Underlying price.
        r: Risk-free rate.
        q: Dividend yield. If ``None``, a per-expiry yield implied from put-call
            parity is used (recommended: it also absorbs stale-spot timing errors).

    Returns:
        Copy of ``chain`` with ``forward``, ``q``, ``iv``, ``log_moneyness``,
        ``fwd_moneyness`` and ``otm`` columns.
    """
    df = chain.copy()
    if q is None:
        fwd = implied_forward(df, spot, r)
        df = df.join(fwd[["q_implied"]], on="expiry").rename(columns={"q_implied": "q"})
        df["q"] = df["q"].fillna(0.0)
    else:
        df["q"] = q
    df["forward"] = spot * np.exp((r - df["q"]) * df["T"])
    df["iv"] = [implied_vol(p, spot, k, t, r, ot, qq)
                for p, k, t, ot, qq in zip(df["mid"], df["strike"], df["T"], df["type"], df["q"])]
    df["fwd_moneyness"] = df["strike"] / df["forward"]
    df["log_moneyness"] = np.log(df["fwd_moneyness"])
    df["otm"] = ((df["type"] == "put") & (df["strike"] < df["forward"])) | (
        (df["type"] == "call") & (df["strike"] >= df["forward"]))
    return df


def otm_smile(iv_chain: pd.DataFrame) -> pd.DataFrame:
    """OTM-only smile: puts below the forward, calls above (the market convention)."""
    return (iv_chain[iv_chain["otm"] & iv_chain["iv"].notna()]
            .sort_values(["T", "strike"]).reset_index(drop=True))


def _interp_inside(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    """Linear interpolation that returns NaN instead of extrapolating."""
    if len(x) < 2 or x0 < x.min() or x0 > x.max():
        return math.nan
    return float(np.interp(x0, x, y))


def skew_summary(smile: pd.DataFrame, low: float = 0.90, high: float = 1.10) -> pd.DataFrame:
    """Per-expiry ATM vol and skew (IV at ``low`` minus IV at ``high`` forward moneyness)."""
    rows = []
    for exp, g in smile.groupby("expiry", sort=False):
        g = g.sort_values("fwd_moneyness")
        x, y = g["fwd_moneyness"].to_numpy(), g["iv"].to_numpy()
        lo, atm, hi = (_interp_inside(x, y, m) for m in (low, 1.0, high))
        rows.append({"expiry": exp, "days": int(g["days"].iloc[0]), "T": float(g["T"].iloc[0]),
                     "atm_iv": atm, f"iv_{low:.0%}": lo, f"iv_{high:.0%}": hi,
                     "skew": lo - hi, "n_quotes": len(g)})
    return pd.DataFrame(rows)


def iv_surface_grid(smile: pd.DataFrame, n_moneyness: int = 40, n_t: int = 30,
                    m_range: tuple[float, float] = (0.8, 1.15)):
    """Linearly interpolate OTM IVs onto a (forward moneyness, days) grid for a surface plot.

    Returns:
        (moneyness grid, days grid, IV matrix) with NaN outside the data's convex hull.
    """
    pts = smile[["fwd_moneyness", "days"]].to_numpy(dtype=float)
    m = np.linspace(max(m_range[0], pts[:, 0].min()), min(m_range[1], pts[:, 0].max()), n_moneyness)
    d = np.linspace(pts[:, 1].min(), pts[:, 1].max(), n_t)
    M, D = np.meshgrid(m, d)
    Z = griddata(pts, smile["iv"].to_numpy(), (M, D), method="linear")
    return m, d, Z


def parity_table(chain: pd.DataFrame, spot: float, r: float, q: float, expiry: str,
                 min_volume: float = 0) -> pd.DataFrame:
    """Put-call parity diagnostics for one expiry.

    European parity: C - P = S e^{-qT} - K e^{-rT}.  Using executable prices:
      * conversion edge  = C_bid - P_ask - (S e^{-qT} - K e^{-rT})   (sell call, buy put, buy stock)
      * reversal edge    = P_bid - C_ask + (S e^{-qT} - K e^{-rT})   (buy call, sell put, short stock)
    A positive edge would be a (pre-cost) arbitrage. Stock spread, borrow fees,
    dividends uncertainty and American early exercise are ignored.

    ``min_volume`` drops strikes where either leg traded less than that today;
    zero-volume legs are usually stale quotes that produce fake "arbitrages".
    """
    g = chain[(chain["expiry"] == expiry) & (chain["bid"] > 0) & (chain["ask"] >= chain["bid"])]
    cols = ["strike", "bid", "ask", "volume"]
    calls = g[g["type"] == "call"][cols].set_index("strike")
    puts = g[g["type"] == "put"][cols].set_index("strike")
    df = calls.join(puts, lsuffix="_c", rsuffix="_p", how="inner").reset_index()
    df = df[(df["volume_c"] >= min_volume) & (df["volume_p"] >= min_volume)].reset_index(drop=True)
    if df.empty:
        return df
    T = float(g["T"].iloc[0])
    df["C_mid"] = 0.5 * (df["bid_c"] + df["ask_c"])
    df["P_mid"] = 0.5 * (df["bid_p"] + df["ask_p"])
    df["C_minus_P"] = df["C_mid"] - df["P_mid"]
    df["theory"] = spot * math.exp(-q * T) - df["strike"] * math.exp(-r * T)
    df["deviation"] = df["C_minus_P"] - df["theory"]
    df["band_low"] = df["bid_c"] - df["ask_p"] - df["theory"]
    df["band_high"] = df["ask_c"] - df["bid_p"] - df["theory"]
    df["conversion_edge"] = df["band_low"]
    df["reversal_edge"] = -df["band_high"]
    df["within_spread"] = (df["band_low"] <= 0) & (df["band_high"] >= 0)
    df["T"] = T
    return df
