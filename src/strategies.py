"""Option strategy builder: legs, payoff at expiry, mark-to-model P&L, and analytics.

A strategy is a list of legs (long/short calls, puts or stock). Payoffs at
expiry are piecewise linear in S_T with kinks only at the strikes, so
breakevens and max profit/loss are computed *exactly* from the kink points and
the slopes beyond them, not read off a plot grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.models.bs import bs_price

Instrument = Literal["call", "put", "stock"]


@dataclass(frozen=True)
class Leg:
    """One position in a strategy.

    Attributes:
        instrument: ``"call"``, ``"put"`` or ``"stock"``.
        quantity: Signed size; +1 = long one unit, -1 = short one unit.
        strike: Strike (ignored for stock).
        premium: Entry price per unit (option premium, or stock purchase price).
    """

    instrument: Instrument
    quantity: float
    strike: float = 0.0
    premium: float = 0.0

    def payoff(self, s_t: ArrayLike) -> NDArray[np.float64]:
        """Value of the leg at expiry (per signed quantity), excluding premium."""
        s_t = np.asarray(s_t, dtype=float)
        if self.instrument == "stock":
            v = s_t
        elif self.instrument == "call":
            v = np.maximum(s_t - self.strike, 0.0)
        else:
            v = np.maximum(self.strike - s_t, 0.0)
        return self.quantity * v

    def value(self, s: ArrayLike, T: float, r: float, sigma: float, q: float = 0.0) -> NDArray[np.float64]:
        """Black-Scholes mark-to-model value of the leg with ``T`` years remaining."""
        s = np.asarray(s, dtype=float)
        if self.instrument == "stock":
            return self.quantity * s
        return self.quantity * np.asarray(bs_price(s, self.strike, T, r, sigma, self.instrument, q))

    @property
    def cost(self) -> float:
        """Cash paid at entry (negative = credit received)."""
        return self.quantity * self.premium

    def describe(self) -> str:
        side = "Long" if self.quantity > 0 else "Short"
        qty = abs(self.quantity)
        qty_s = "" if qty == 1 else f"{qty:g}x "
        if self.instrument == "stock":
            return f"{side} {qty_s}stock @ {self.premium:.2f}"
        return f"{side} {qty_s}{self.strike:g} {self.instrument} @ {self.premium:.2f}"


@dataclass
class Strategy:
    """A named collection of legs."""

    name: str
    legs: list[Leg] = field(default_factory=list)

    @property
    def net_cost(self) -> float:
        """Net debit (positive) or credit (negative) to open the position."""
        return float(sum(leg.cost for leg in self.legs))

    def pnl_at_expiry(self, s_t: ArrayLike) -> NDArray[np.float64]:
        """Profit/loss at expiry for terminal prices ``s_t``."""
        s_t = np.asarray(s_t, dtype=float)
        return sum((leg.payoff(s_t) for leg in self.legs), np.zeros_like(s_t)) - self.net_cost

    def pnl_now(self, s: ArrayLike, T: float, r: float, sigma: float, q: float = 0.0) -> NDArray[np.float64]:
        """Mark-to-model P&L if the underlying were at ``s`` with ``T`` years left."""
        s = np.asarray(s, dtype=float)
        return sum((leg.value(s, T, r, sigma, q) for leg in self.legs), np.zeros_like(s)) - self.net_cost

    def upside_slope(self) -> float:
        """dP&L/dS_T above the highest strike: net long calls plus net stock."""
        return float(sum(l.quantity for l in self.legs if l.instrument in ("call", "stock")))

    def analyze(self) -> "StrategyStats":
        """Exact breakevens and max profit / max loss at expiry."""
        strikes = sorted({l.strike for l in self.legs if l.instrument != "stock"})
        top = (max(strikes) if strikes else max(l.premium for l in self.legs)) * 2 + 1.0
        pts = np.array(sorted({0.0, *strikes, top}))
        vals = self.pnl_at_expiry(pts)

        breakevens: list[float] = []
        for (x0, y0), (x1, y1) in zip(zip(pts[:-1], vals[:-1]), zip(pts[1:], vals[1:])):
            if y0 == 0:
                breakevens.append(float(x0))
            elif y0 * y1 < 0:
                breakevens.append(float(x0 - y0 * (x1 - x0) / (y1 - y0)))
        slope = self.upside_slope()
        if vals[-1] == 0 and slope == 0:
            breakevens.append(float(pts[-1]))
        elif slope != 0 and vals[-1] * slope < 0:  # crosses zero beyond the last point
            breakevens.append(float(pts[-1] - vals[-1] / slope))

        max_profit = float("inf") if slope > 0 else float(vals.max())
        max_loss = float("-inf") if slope < 0 else float(vals.min())
        return StrategyStats(sorted(set(round(b, 6) for b in breakevens)), max_profit,
                             max_loss, self.net_cost)


@dataclass(frozen=True)
class StrategyStats:
    """Summary statistics of a strategy at expiry."""

    breakevens: list[float]
    max_profit: float  # inf = unlimited
    max_loss: float    # -inf = unlimited
    net_cost: float    # >0 debit, <0 credit


#: Strike parameters each strategy needs, with default offsets as a fraction of spot.
STRATEGY_STRIKES: dict[str, list[tuple[str, float]]] = {
    "Long call": [("Strike", 1.00)],
    "Straddle": [("Strike", 1.00)],
    "Strangle": [("Put strike", 0.95), ("Call strike", 1.05)],
    "Bull call spread": [("Long call strike", 1.00), ("Short call strike", 1.05)],
    "Iron condor": [("Long put strike", 0.90), ("Short put strike", 0.95),
                    ("Short call strike", 1.05), ("Long call strike", 1.10)],
    "Covered call": [("Short call strike", 1.05)],
}


def build_strategy(name: str, strikes: list[float], S: float, T: float, r: float,
                   sigma: float, q: float = 0.0, quantity: float = 1.0) -> Strategy:
    """Build a preset strategy with premiums priced by Black-Scholes at entry.

    Args:
        name: One of :data:`STRATEGY_STRIKES`.
        strikes: Strikes in the order listed in :data:`STRATEGY_STRIKES`.
        S, T, r, sigma, q: Market inputs used to price the premiums.
        quantity: Number of units of the whole structure.

    Raises:
        ValueError: Unknown strategy, wrong number of strikes or mis-ordered strikes.
    """
    if name not in STRATEGY_STRIKES:
        raise ValueError(f"Unknown strategy {name!r}")
    if len(strikes) != len(STRATEGY_STRIKES[name]):
        raise ValueError(f"{name} needs {len(STRATEGY_STRIKES[name])} strike(s)")
    if any(b <= a for a, b in zip(strikes, strikes[1:])):
        raise ValueError("Strikes must be strictly increasing in the listed order")

    def opt(kind: Instrument, qty: float, K: float) -> Leg:
        return Leg(kind, qty * quantity, K, float(bs_price(S, K, T, r, sigma, kind, q)))

    n = quantity
    if name == "Long call":
        legs = [opt("call", 1, strikes[0])]
    elif name == "Straddle":
        legs = [opt("call", 1, strikes[0]), opt("put", 1, strikes[0])]
    elif name == "Strangle":
        legs = [opt("put", 1, strikes[0]), opt("call", 1, strikes[1])]
    elif name == "Bull call spread":
        legs = [opt("call", 1, strikes[0]), opt("call", -1, strikes[1])]
    elif name == "Iron condor":
        legs = [opt("put", 1, strikes[0]), opt("put", -1, strikes[1]),
                opt("call", -1, strikes[2]), opt("call", 1, strikes[3])]
    else:  # Covered call
        legs = [Leg("stock", n, 0.0, S), opt("call", -1, strikes[0])]
    return Strategy(name, legs)
