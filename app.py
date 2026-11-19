"""Options Pricing & Greeks Lab - Streamlit app.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.analysis import add_implied_vols, iv_surface_grid, otm_smile, parity_table, skew_summary
from src.data import ChainData, clean_chain, fetch_option_chain, fetch_risk_free_rate, implied_forward
from src.greeks import analytic_greeks, delta, finite_difference_greeks, gamma, rho, theta, vega
from src.iv import implied_vol_detailed
from src.models import bs_price, crr_price, mc_price
from src.strategies import STRATEGY_STRIKES, build_strategy

st.set_page_config(page_title="Options Pricing & Greeks Lab", page_icon="📈", layout="wide")

BLUE, ORANGE, GREEN, RED, GREY, PURPLE = "#2563eb", "#ea580c", "#16a34a", "#dc2626", "#6b7280", "#7c3aed"
PLOT_LAYOUT = dict(template="plotly_white", margin=dict(l=10, r=10, t=50, b=10),
                   legend=dict(orientation="h", yanchor="top", y=-0.18, x=0))


def style(fig: go.Figure, title: str, height: int = 420, **kw) -> go.Figure:
    fig.update_layout(title=title, height=height, **PLOT_LAYOUT, **kw)
    return fig


def money(x: float) -> str:
    if math.isinf(x):
        return "Unlimited" if x > 0 else "-Unlimited"
    return f"{x:,.2f}"


# ---------------------------------------------------------------- cached computations
@st.cache_data(ttl=900, show_spinner="Downloading option chain from Yahoo Finance...")
def load_chain(ticker: str) -> ChainData:
    return fetch_option_chain(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def load_rate() -> tuple[float, str]:
    return fetch_risk_free_rate()


@st.cache_data(show_spinner=False)
def binomial_convergence(S, K, T, r, sigma, ot, q, max_steps) -> pd.DataFrame:
    steps = np.arange(10, max_steps + 1)
    eur = [crr_price(S, K, T, r, sigma, int(n), ot, False, q) for n in steps]
    amer = [crr_price(S, K, T, r, sigma, int(n), ot, True, q) for n in steps]
    return pd.DataFrame({"steps": steps, "european": eur, "american": amer})


@st.cache_data(show_spinner=False)
def mc_convergence(S, K, T, r, sigma, ot, q, seed) -> pd.DataFrame:
    rows = []
    for n in np.unique(np.logspace(2, 6, 17).astype(int)):
        for anti in (True, False):
            res = mc_price(S, K, T, r, sigma, ot, q, int(n), anti, seed)
            rows.append({"paths": int(n), "antithetic": anti, "price": res.price,
                         "lo": res.ci_low, "hi": res.ci_high, "se": res.std_error})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Solving implied vols...")
def chain_ivs(chain: pd.DataFrame, spot: float, r: float, q: float | None) -> pd.DataFrame:
    return add_implied_vols(chain, spot, r, q)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("📈 Options Lab")
st.sidebar.caption("Model inputs (Pricing, Greeks and Strategy tabs)")
S = st.sidebar.number_input("Spot price S", 0.01, 1e6, 100.0, 1.0)
K = st.sidebar.number_input("Strike K", 0.01, 1e6, 100.0, 1.0)
days = st.sidebar.number_input("Days to expiry", 1, 3650, 90, 1)
T = days / 365.0
r = st.sidebar.number_input("Risk-free rate r (%)", -5.0, 30.0, 4.0, 0.1) / 100
sigma = st.sidebar.number_input("Volatility σ (%)", 1.0, 300.0, 20.0, 0.5) / 100
q = st.sidebar.number_input("Dividend yield q (%)", 0.0, 20.0, 0.0, 0.1) / 100
opt_type = st.sidebar.radio("Option type", ["call", "put"], horizontal=True)

st.sidebar.divider()
st.sidebar.caption("Market data (Implied Vol and Parity tabs)")
ticker = st.sidebar.text_input("Ticker", "SPY").strip().upper() or "SPY"
if st.sidebar.button("Refresh market data"):
    load_chain.clear()
    load_rate.clear()

st.title("Options Pricing & Greeks Lab")
st.caption("Black-Scholes · CRR binomial · Monte Carlo · Greeks · Implied volatility · Strategies · Put-call parity")

tabs = st.tabs(["💰 Pricing", "🔺 Greeks", "🌊 Implied Vol", "🧩 Strategy Builder",
                "⚖️ Put-Call Parity", "📚 Theory & Limitations"])

# ================================================================ 1. PRICING
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    n_steps = c1.slider("Binomial steps", 10, 2000, 500, 10)
    n_paths = c2.select_slider("Monte Carlo paths", [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000], 100_000)
    seed = c3.number_input("MC random seed", 0, 10_000, 42)

    bs = float(bs_price(S, K, T, r, sigma, opt_type, q))
    crr_e = crr_price(S, K, T, r, sigma, n_steps, opt_type, False, q)
    crr_a = crr_price(S, K, T, r, sigma, n_steps, opt_type, True, q)
    mc = mc_price(S, K, T, r, sigma, opt_type, q, n_paths, True, int(seed))

    m = st.columns(4)
    m[0].metric("Black-Scholes (European)", f"{bs:.4f}")
    m[1].metric(f"CRR European ({n_steps} steps)", f"{crr_e:.4f}", f"{crr_e - bs:+.4f} vs BS", delta_color="off")
    m[2].metric(f"CRR American ({n_steps} steps)", f"{crr_a:.4f}", f"{crr_a - crr_e:+.4f} early-exercise premium",
                delta_color="off")
    m[3].metric(f"Monte Carlo ({mc.n_paths:,} paths, antithetic)", f"{mc.price:.4f}",
                f"95% CI [{mc.ci_low:.4f}, {mc.ci_high:.4f}]", delta_color="off")
    inside = mc.ci_low <= bs <= mc.ci_high
    st.caption(f"MC standard error = {mc.std_error:.5f}. The BS price is "
               f"{'inside' if inside else 'outside'} the 95% confidence interval "
               f"({'as expected' if inside else 'happens ~5% of the time by chance'}).")

    g1, g2 = st.columns(2)
    with g1:
        conv = binomial_convergence(S, K, T, r, sigma, opt_type, q, 300)
        fig = go.Figure()
        fig.add_scatter(x=conv["steps"], y=conv["european"], name="CRR European", line=dict(color=BLUE, width=1.5))
        if opt_type == "put" or q > 0:
            fig.add_scatter(x=conv["steps"], y=conv["american"], name="CRR American",
                            line=dict(color=ORANGE, width=1.5))
        fig.add_hline(y=bs, line=dict(color=RED, dash="dash"), annotation_text=f"BS {bs:.4f}")
        style(fig, "Binomial convergence to Black-Scholes", xaxis_title="Tree steps N", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The odd/even zig-zag comes from where the strike sits relative to the terminal nodes; "
                   "the error shrinks roughly like 1/N.")
    with g2:
        mcc = mc_convergence(S, K, T, r, sigma, opt_type, q, int(seed))
        fig = go.Figure()
        for anti, color, name in ((False, GREY, "Plain MC"), (True, GREEN, "Antithetic MC")):
            d = mcc[mcc["antithetic"] == anti]
            fig.add_scatter(x=d["paths"], y=d["hi"], mode="lines", line=dict(width=0), showlegend=False,
                            hoverinfo="skip")
            fig.add_scatter(x=d["paths"], y=d["lo"], mode="lines", fill="tonexty", line=dict(width=0),
                            hoverinfo="skip",
                            fillcolor=f"rgba{tuple(int(color[i:i + 2], 16) for i in (1, 3, 5)) + (0.18,)}",
                            name=f"{name} 95% CI")
            fig.add_scatter(x=d["paths"], y=d["price"], name=name, mode="lines+markers",
                            line=dict(color=color), marker=dict(size=5))
        fig.add_hline(y=bs, line=dict(color=RED, dash="dash"), annotation_text=f"BS {bs:.4f}")
        style(fig, "Monte Carlo convergence (CI shrinks like 1/√n)", xaxis_title="Paths (log scale)",
              yaxis_title="Price", xaxis_type="log")
        st.plotly_chart(fig, use_container_width=True)
        last = mcc[mcc["paths"] == mcc["paths"].max()].set_index("antithetic")["se"]
        st.caption(f"At {mcc['paths'].max():,} paths the antithetic standard error is "
                   f"{last[True] / last[False]:.0%} of plain MC, i.e. a "
                   f"{(last[False] / last[True]) ** 2:.1f}x variance reduction for free.")

    with st.expander("How the three models work"):
        st.markdown(r"""
**Black-Scholes** — closed form under geometric Brownian motion with constant σ.
$C = Se^{-qT}N(d_1) - Ke^{-rT}N(d_2)$, $d_{1,2} = \frac{\ln(S/K) + (r-q\pm\sigma^2/2)T}{\sigma\sqrt T}$.
$N(d_2)$ is the risk-neutral probability of finishing in the money.

**CRR binomial tree** — in each step $\Delta t$ the price goes up $u=e^{\sigma\sqrt{\Delta t}}$ or down $d=1/u$
with risk-neutral probability $p=\frac{e^{(r-q)\Delta t}-d}{u-d}$. Work backwards from the payoffs:
$V=e^{-r\Delta t}[pV_u+(1-p)V_d]$. For American options take $\max(V, \text{exercise value})$ at each node.

**Monte Carlo** — simulate $S_T = S\exp((r-q-\tfrac12\sigma^2)T + \sigma\sqrt T Z)$, average the discounted payoffs.
Antithetic variates pair each $Z$ with $-Z$; the pair averages are negatively correlated so variance drops.
Standard error $=s/\sqrt{n}$, 95% CI $=\hat V \pm 1.96\,\text{SE}$.
""")

# ================================================================ 2. GREEKS
GREEK_INFO = {
    "Delta": (delta, 1.0, "Change in option value per $1 move in the underlying - the hedge ratio "
              "(shares of stock that replicate the option)."),
    "Gamma": (gamma, 1.0, "Change in delta per $1 move. Highest at-the-money and near expiry - "
              "long gamma profits from large moves in either direction."),
    "Vega": (vega, 0.01, "Change in value per 1 volatility point. Largest ATM and for long-dated options."),
    "Theta": (theta, 1 / 365, "Change in value per calendar day that passes - the 'rent' a long option "
              "holder pays for gamma."),
    "Rho": (rho, 0.01, "Change in value per 1 percentage-point change in the interest rate."),
}
with tabs[1]:
    a = analytic_greeks(S, K, T, r, sigma, opt_type, q).display()
    f = finite_difference_greeks(S, K, T, r, sigma, opt_type, q).display()
    table = pd.DataFrame({"Analytic (BS)": a, "Finite difference": f})
    table["Abs. difference"] = (table["Analytic (BS)"] - table["Finite difference"]).abs()
    left, right = st.columns([1.3, 1])
    with left:
        st.subheader(f"{opt_type.title()} Greeks")
        st.dataframe(table.style.format({"Analytic (BS)": "{:.6f}", "Finite difference": "{:.6f}",
                                         "Abs. difference": "{:.2e}"}), use_container_width=True)
        st.caption("Finite differences use central bumps on the BS price (O(h²) error). "
                   "Vega/rho are per 1 percentage point; theta per calendar day.")
    with right:
        greek = st.radio("Greek to chart", list(GREEK_INFO), horizontal=True)
        fn, scale, blurb = GREEK_INFO[greek]
        st.info(f"**{greek}:** {blurb}")

    c1, c2 = st.columns(2)
    with c1:
        s_grid = np.linspace(0.5 * K, 1.5 * K, 301)
        fig = go.Figure()
        for t_days, dash in ((days, "solid"), (max(days // 3, 1), "dash"), (max(days // 12, 1), "dot")):
            for ot, color in (("call", BLUE), ("put", ORANGE)):
                y = fn(s_grid, K, t_days / 365, r, sigma, ot, q) * scale
                fig.add_scatter(x=s_grid, y=y, name=f"{ot} · {t_days}d", line=dict(color=color, dash=dash))
        fig.add_vline(x=K, line=dict(color=GREY, dash="dot"), annotation_text="K")
        style(fig, f"{greek} vs underlying price", xaxis_title="Underlying price S", yaxis_title=greek)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        t_grid = np.linspace(1 / 365, max(T, 30 / 365) * 1.5, 300)
        fig = go.Figure()
        for mny, color, label in ((0.9, PURPLE, "S = 0.9K"), (1.0, GREEN, "S = K (ATM)"), (1.1, RED, "S = 1.1K")):
            y = fn(mny * K, K, t_grid, r, sigma, opt_type, q) * scale
            fig.add_scatter(x=t_grid * 365, y=y, name=f"{label}", line=dict(color=color))
        style(fig, f"{opt_type.title()} {greek.lower()} vs days to expiry", xaxis_title="Days to expiry",
              yaxis_title=greek, xaxis_autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("x-axis is reversed so time flows left → right towards expiry.")

# ================================================================ 3. IMPLIED VOL
with tabs[2]:
    data = load_chain(ticker)
    mkt_rate, rate_src = load_rate()
    if data.source == "live":
        st.success(f"{data.message} {data.ticker} spot = {data.spot:,.2f} "
                   f"(as of {data.asof:%Y-%m-%d %H:%M} New York).")
    else:
        st.warning(data.message)

    with st.expander("Filters & assumptions", expanded=False):
        f1, f2, f3, f4 = st.columns(4)
        min_vol = f1.number_input("Min volume", 0, 100_000, 1)
        max_spread = f2.slider("Max bid-ask spread (% of mid)", 5, 100, 25) / 100
        mny = f3.slider("Strike / spot window", 0.5, 1.5, (0.7, 1.3), 0.05)
        use_last = f4.checkbox("Use last price if no quote", False,
                               help="Useful outside market hours when bid/ask are zero.")
        g1, g2 = st.columns(2)
        mkt_r = g1.number_input("Risk-free rate for market data (%)", -5.0, 30.0,
                                round(mkt_rate * 100, 3), 0.05, help=rate_src) / 100
        q_mode = g2.radio("Dividend yield", ["Implied from put-call parity (recommended)", "Manual"],
                          horizontal=True)
        q_manual = g2.number_input("Manual dividend yield (%)", 0.0, 20.0, 1.2, 0.1) / 100 \
            if q_mode == "Manual" else None
        st.caption(f"Rate source: {rate_src}.")

    clean, report = clean_chain(data.chain, data.spot, min_vol, max_spread, 0.05, mny, use_last)
    st.caption(f"Liquidity filter kept **{report.kept:,} / {report.total:,}** quotes. Removed: "
               + ", ".join(f"{k}: {v}" for k, v in report.removed.items()))
    if clean.empty:
        st.error("No quotes survive the filters (market may be closed). Relax the filters or tick "
                 "'Use last price if no quote'.")
    else:
        ivc = chain_ivs(clean, data.spot, mkt_r, q_manual)
        smile = otm_smile(ivc)
        expiries = list(dict.fromkeys(smile["expiry"]))
        palette = ["#1d4ed8", "#0891b2", "#16a34a", "#ca8a04", "#ea580c", "#dc2626", "#7c3aed", "#db2777"]

        c1, c2 = st.columns([1.6, 1])
        with c1:
            x_axis = st.radio("Smile x-axis", ["Strike / forward", "Strike"], horizontal=True)
            xcol = "fwd_moneyness" if x_axis == "Strike / forward" else "strike"
            fig = go.Figure()
            for i, exp in enumerate(expiries):
                g = smile[smile["expiry"] == exp]
                fig.add_scatter(x=g[xcol], y=g["iv"] * 100, mode="lines+markers",
                                name=f"{exp} ({int(g['days'].iloc[0])}d)",
                                line=dict(color=palette[i % len(palette)], width=1.5), marker=dict(size=4),
                                customdata=np.c_[g["strike"], g["type"], g["mid"]],
                                hovertemplate="K=%{customdata[0]} %{customdata[1]}<br>mid=%{customdata[2]:.2f}"
                                              "<br>IV=%{y:.2f}%<extra></extra>")
            fig.add_vline(x=1.0 if xcol == "fwd_moneyness" else data.spot, line=dict(color=GREY, dash="dot"))
            style(fig, f"{data.ticker} volatility smile / skew (OTM options)", 460,
                  xaxis_title=x_axis, yaxis_title="Implied volatility (%)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            sk = skew_summary(smile)
            fig = go.Figure()
            fig.add_scatter(x=sk["days"], y=sk["atm_iv"] * 100, mode="lines+markers", name="ATM IV",
                            line=dict(color=BLUE))
            fig.add_scatter(x=sk["days"], y=sk["skew"] * 100, mode="lines+markers", name="Skew (90% − 110%)",
                            line=dict(color=ORANGE), yaxis="y2")
            style(fig, "Term structure: ATM vol and skew", 460, xaxis_title="Days to expiry",
                  yaxis_title="ATM IV (%)",
                  yaxis2=dict(title="Skew (vol pts)", overlaying="y", side="right", showgrid=False))
            st.plotly_chart(fig, use_container_width=True)

        show = sk.drop(columns="T").copy()
        for col in show.columns:
            if col.startswith(("atm", "iv_", "skew")):
                show[col] = show[col] * 100
        st.dataframe(show.style.format({c: "{:.2f}" for c in show.columns if c not in ("expiry", "days", "n_quotes")},
                                       na_rep="—"), hide_index=True, use_container_width=True)
        st.caption("IVs in %. Skew = IV at 90% of the forward minus IV at 110%; '—' means no quotes that far out "
                   "(no extrapolation).")

        c1, c2 = st.columns([1.4, 1])
        with c1:
            if smile["expiry"].nunique() >= 2:
                mg, dg, Z = iv_surface_grid(smile)
                fig = go.Figure(go.Surface(x=mg, y=dg, z=Z * 100, colorscale="Viridis",
                                           colorbar=dict(title="IV %")))
                fig.update_layout(scene=dict(xaxis_title="K / F", yaxis_title="Days", zaxis_title="IV %",
                                             camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8))))
                style(fig, "Implied volatility surface (linear interpolation)", 520)
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            v = ivc[ivc["otm"] & ivc["iv"].notna() & ivc["yahoo_iv"].between(0.01, 3)]
            fig = go.Figure()
            fig.add_scatter(x=v["yahoo_iv"] * 100, y=v["iv"] * 100, mode="markers",
                            marker=dict(size=4, color=BLUE, opacity=0.5), name="OTM quotes")
            lim = [v[["iv", "yahoo_iv"]].min().min() * 100, v[["iv", "yahoo_iv"]].max().max() * 100] if len(v) else [0, 1]
            fig.add_scatter(x=lim, y=lim, mode="lines", line=dict(color=RED, dash="dash"), name="y = x")
            style(fig, "Our IV vs Yahoo's IV (sanity check)", 520, xaxis_title="Yahoo IV (%)",
                  yaxis_title="Our IV (%)")
            st.plotly_chart(fig, use_container_width=True)
            if len(v):
                st.caption(f"Median absolute difference: {(v['iv'] - v['yahoo_iv']).abs().median() * 100:.2f} vol pts. "
                           "Differences come from Yahoo's own rate/dividend/timing assumptions.")

        with st.expander("What the skew means"):
            st.markdown("""
- **Equity index skew:** OTM puts trade at much higher IV than OTM calls. Investors pay up for crash
  protection, dealers who sell it demand a premium, and returns are negatively skewed with fat left
  tails (prices fall faster than they rise, and vol rises when prices fall: the *leverage effect*).
- **Black-Scholes assumes one flat σ for all strikes.** The smile shows that assumption is wrong. Traders
  still quote in BS-implied vol, but they use the smile as a map of the market's risk-neutral distribution
  (a steep put skew means a fat left tail).
- **Skew is steeper at short expiries.** A sudden gap down matters more relative to the small amount of
  diffusion over a few days. Over longer horizons the distribution looks more normal.
- **Term structure:** ATM IV rising with maturity (contango) is the calm-market norm. An inverted curve
  (short-dated above long-dated) signals near-term stress or event risk.
""")

        st.subheader("Single-option IV calculator")
        k1, k2, k3, k4 = st.columns(4)
        px_in = k1.number_input("Option price", 0.0001, 1e6, round(float(bs_price(S, K, T, r, sigma, opt_type, q)), 4))
        res = implied_vol_detailed(px_in, S, K, T, r, opt_type, q)
        if res.converged:
            k2.metric("Implied vol", f"{res.sigma:.4%}")
            k3.metric("Iterations", res.iterations)
            k4.metric("Newton / bisection steps", f"{res.newton_steps} / {res.bisection_steps}")
        else:
            k2.error(f"No solution: {res.message}")
        st.caption("Uses the sidebar S, K, T, r, q and option type.")

# ================================================================ 4. STRATEGY BUILDER
with tabs[3]:
    c1, c2, c3 = st.columns([1.2, 1, 1])
    strat_name = c1.selectbox("Strategy", list(STRATEGY_STRIKES))
    qty = c2.number_input("Number of units", 1, 1000, 1)
    days_left = c3.slider("Days remaining for 'current P&L' curve", 0, int(days), int(days),
                          help="Set below today's days-to-expiry to see the P&L after time decay.")
    spec = STRATEGY_STRIKES[strat_name]
    cols = st.columns(len(spec))
    step = 1.0 if S < 200 else 5.0
    strikes = [cols[i].number_input(label, 0.01, 1e6, float(round(S * off / step) * step), step, key=f"{strat_name}{i}")
               for i, (label, off) in enumerate(spec)]
    try:
        strat = build_strategy(strat_name, strikes, S, T, r, sigma, q, qty)
    except ValueError as exc:
        st.error(str(exc))
        strat = None
    if strat is not None:
        stats = strat.analyze()

        m = st.columns(4)
        m[0].metric("Net debit" if stats.net_cost >= 0 else "Net credit", money(abs(stats.net_cost)))
        m[1].metric("Max profit", money(stats.max_profit))
        m[2].metric("Max loss", money(stats.max_loss))
        m[3].metric("Breakeven(s)", ", ".join(f"{b:,.2f}" for b in stats.breakevens) or "None")

        lo_k = min([l.strike for l in strat.legs if l.instrument != "stock"] + [S])
        hi_k = max([l.strike for l in strat.legs if l.instrument != "stock"] + [S])
        grid = np.linspace(lo_k * 0.75, hi_k * 1.25, 600)
        expiry_pnl = strat.pnl_at_expiry(grid)
        now_pnl = strat.pnl_now(grid, days_left / 365, r, sigma, q)
        fig = go.Figure()
        fig.add_scatter(x=grid, y=np.where(expiry_pnl >= 0, expiry_pnl, np.nan), fill="tozeroy", mode="none",
                        fillcolor="rgba(22,163,74,0.12)", showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=grid, y=np.where(expiry_pnl < 0, expiry_pnl, np.nan), fill="tozeroy", mode="none",
                        fillcolor="rgba(220,38,38,0.12)", showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=grid, y=expiry_pnl, name="P&L at expiry", line=dict(color="#111827", width=2.5))
        fig.add_scatter(x=grid, y=now_pnl, name=f"P&L now ({days_left}d left, BS)", line=dict(color=BLUE, dash="dash"))
        fig.add_hline(y=0, line=dict(color=GREY, width=1))
        fig.add_vline(x=S, line=dict(color=PURPLE, dash="dot"), annotation_text=f"Spot {S:g}")
        for b in stats.breakevens:
            fig.add_vline(x=b, line=dict(color=ORANGE, dash="dot"), annotation_text=f"BE {b:.2f}",
                          annotation_position="bottom right")
        style(fig, f"{strat_name}: payoff diagram", 500, xaxis_title="Underlying price at expiry", yaxis_title="P&L")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Legs** (premiums priced with Black-Scholes at the sidebar inputs)")
            st.dataframe(pd.DataFrame({"Leg": [l.describe() for l in strat.legs],
                                       "Cash flow": [-l.cost for l in strat.legs]}).style.format({"Cash flow": "{:+.2f}"}),
                         hide_index=True, use_container_width=True)
        with c2:
            st.markdown("**Position Greeks** (sum over legs; stock has delta 1)")
            pg = {"Delta": 0.0, "Gamma": 0.0, "Vega (per 1 vol pt)": 0.0, "Theta (per day)": 0.0, "Rho (per 1% rate)": 0.0}
            for leg in strat.legs:
                if leg.instrument == "stock":
                    pg["Delta"] += leg.quantity
                    continue
                gd = analytic_greeks(S, leg.strike, T, r, sigma, leg.instrument, q).display()
                for k_ in pg:
                    pg[k_] += leg.quantity * gd[k_]
            st.dataframe(pd.DataFrame(pg, index=["Position"]).T.style.format("{:+.4f}"), use_container_width=True)

# ================================================================ 5. PUT-CALL PARITY
with tabs[4]:
    data = load_chain(ticker)
    mkt_rate, rate_src = load_rate()
    if data.source != "live":
        st.warning(data.message)
    st.markdown(r"European put-call parity: $\;C - P = S e^{-qT} - K e^{-rT}$. Deviations are measured with "
                "mid prices; the shaded band is what you could actually trade at the bid/ask.")
    p1, p2, p3, p4 = st.columns(4)
    exp_list = list(dict.fromkeys(data.chain.sort_values("T")["expiry"]))
    exp_sel = p1.selectbox("Expiry", exp_list, index=min(1, len(exp_list) - 1))
    par_r = p2.number_input("Rate r (%)", -5.0, 30.0, round(mkt_rate * 100, 3), 0.05, key="par_r") / 100
    par_q = p3.number_input("Dividend yield q (%)", 0.0, 20.0, 1.2 if data.ticker.startswith("SPY") else 0.0,
                            0.05, key="par_q") / 100
    par_vol = p4.number_input("Min volume on both legs", 0, 100_000, 10, key="par_vol",
                              help="Zero-volume legs are often stale quotes that create fake arbitrages.")
    tab = parity_table(data.chain, data.spot, par_r, par_q, exp_sel, par_vol)
    tab = tab[tab["strike"].between(0.8 * data.spot, 1.2 * data.spot)]
    if tab.empty:
        st.error("No strikes with two-sided quotes for both calls and puts in this expiry.")
    else:
        fig = go.Figure()
        fig.add_scatter(x=tab["strike"], y=tab["band_high"], mode="lines", line=dict(width=0), showlegend=False,
                        hoverinfo="skip")
        fig.add_scatter(x=tab["strike"], y=tab["band_low"], mode="lines", fill="tonexty", line=dict(width=0),
                        fillcolor="rgba(37,99,235,0.15)", name="Bid-ask no-arbitrage band")
        fig.add_scatter(x=tab["strike"], y=tab["deviation"], mode="markers+lines", name="Mid deviation",
                        line=dict(color=BLUE), marker=dict(size=5))
        fig.add_hline(y=0, line=dict(color=RED, dash="dash"))
        fig.add_vline(x=data.spot, line=dict(color=GREY, dash="dot"), annotation_text="Spot")
        style(fig, f"Parity deviation (C − P) − (Se^(−qT) − Ke^(−rT)), expiry {exp_sel}", 460,
              xaxis_title="Strike", yaxis_title="Deviation ($)")
        st.plotly_chart(fig, use_container_width=True)

        m = st.columns(4)
        m[0].metric("Strikes checked", len(tab))
        m[1].metric("Median |deviation|", f"${tab['deviation'].abs().median():.3f}")
        m[2].metric("Zero inside bid-ask band", f"{tab['within_spread'].mean():.0%}")
        m[3].metric("Apparent arbitrages (pre-cost)",
                    int(((tab["conversion_edge"] > 0) | (tab["reversal_edge"] > 0)).sum()))

        fwd = implied_forward(clean_chain(data.chain, data.spot, 0, 1.0)[0], data.spot, par_r)
        if not fwd.empty:
            fwd_show = fwd.assign(q_implied=fwd["q_implied"] * 100).rename(
                columns={"forward": "Implied forward", "q_implied": "Implied div. yield (%)"})
            st.markdown("**Implied forward & dividend yield by expiry** (from near-ATM parity)")
            st.dataframe(fwd_show.style.format({"T": "{:.3f}", "Implied forward": "{:,.2f}",
                                                "Implied div. yield (%)": "{:.2f}"}), use_container_width=True)
        with st.expander("Strike-level detail"):
            st.dataframe(tab[["strike", "C_mid", "P_mid", "C_minus_P", "theory", "deviation",
                              "conversion_edge", "reversal_edge", "within_spread"]].round(4),
                         hide_index=True, use_container_width=True)
        with st.expander("Why deviations appear (and why they are not free money)"):
            st.markdown("""
- **American exercise:** US equity/ETF options (SPY, single stocks) are American. Deep ITM puts carry an
  early-exercise premium, so $C-P$ sits *below* the European formula at high strikes. Parity becomes an
  inequality: $S e^{-qT} - K \\le C - P \\le S - K e^{-rT}$.
- **Dividends:** discrete, uncertain dividends rather than a constant yield.
- **Funding:** the options market's implied financing rate (seen in box spreads) usually sits above T-bills.
- **Non-synchronous quotes:** the spot and option quotes are snapshots taken at slightly different times.
  A stale spot shifts *every* strike's deviation by the same amount - look for a flat offset.
- **Stale quotes:** a leg that has not traded today can show a quote far from fair value.
- **Costs:** the stock's bid-ask spread, borrow fees for shorting, margin and commissions eat any small edge.
""")

# ================================================================ 6. THEORY
with tabs[5]:
    st.markdown(r"""
### Model assumptions (Black-Scholes-Merton)
Lognormal prices (GBM) with **constant** volatility and rates, continuous dividend yield, frictionless
continuous trading, no arbitrage. Under the risk-neutral measure every asset grows at $r$, so
$V_0 = e^{-rT}\,\mathbb{E}^{\mathbb Q}[\text{payoff}]$.

### Greeks (with dividend yield q)
| Greek | Call | Put |
|---|---|---|
| Δ | $e^{-qT}N(d_1)$ | $e^{-qT}(N(d_1)-1)$ |
| Γ | $\dfrac{e^{-qT}n(d_1)}{S\sigma\sqrt T}$ | same |
| ν | $Se^{-qT}n(d_1)\sqrt T$ | same |
| Θ | $-\dfrac{Se^{-qT}n(d_1)\sigma}{2\sqrt T} - rKe^{-rT}N(d_2) + qSe^{-qT}N(d_1)$ | $-\dfrac{Se^{-qT}n(d_1)\sigma}{2\sqrt T} + rKe^{-rT}N(-d_2) - qSe^{-qT}N(-d_1)$ |
| ρ | $KTe^{-rT}N(d_2)$ | $-KTe^{-rT}N(-d_2)$ |

They are tied together by the BS PDE: $\Theta + \tfrac12\sigma^2S^2\Gamma + (r-q)S\Delta - rV = 0$.
A delta-hedged long option earns $\tfrac12\Gamma S^2(\sigma_{\text{realised}}^2 - \sigma_{\text{implied}}^2)\,dt$ per period.

### Implied volatility solver
Newton-Raphson $\sigma_{n+1} = \sigma_n - \frac{BS(\sigma_n)-P}{\nu(\sigma_n)}$, kept inside a bracket $[\sigma_{lo},\sigma_{hi}]$
that is tightened every iteration; when a Newton step leaves the bracket or vega is tiny, a bisection step is taken.
Prices outside the no-arbitrage bounds return no vol.

### Limitations
- Constant-σ models cannot reproduce the smile; local/stochastic vol (Dupire, Heston, SABR) are the next step.
- The IVs treat American options (SPY) as European; near the money the difference is small, deep ITM it is not.
- Yahoo Finance data is delayed, sometimes stale, and has no guaranteed quality; quotes and spot are not synchronous.
- Continuous dividend yield and a flat T-bill rate are approximations; there is no discrete dividend schedule.
- The surface is a linear interpolation of raw quotes and is not arbitrage-free (no SVI/SSVI fitting).
- Monte Carlo here prices European payoffs only; it is included to show the method and variance reduction.
""")
