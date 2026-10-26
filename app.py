"""
app.py - Streamlit app: rank Nifty stocks with a walk-forward ML model.
Run:  streamlit run app.py
Educational project. Not investment advice.
"""
import io

import pandas as pd
import streamlit as st

import engine as E

st.set_page_config(page_title="Nifty Stock Ranker", page_icon="📈", layout="wide")

st.title("📈 Nifty Stock Ranker")
st.caption(
    "Educational ML project. It ranks stocks by the probability of beating the "
    "median stock over the next day or week. It is not investment advice, and "
    "short-term returns are mostly noise - read the accuracy tab before trusting any pick."
)

OPTIONAL_GROUPS = [g for g in E.GROUP_NAMES if g != "Technical"]
FII_TEMPLATE = "date,fii_net,dii_net\n2026-10-23,-1500.5,2100.3\n2026-10-26,320.0,-410.7\n"

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    horizon_label = st.radio("Prediction horizon", ["Next day", "Next week (5 trading days)"])
    horizon = 1 if horizon_label == "Next day" else 5
    target_label = st.radio("What to predict", ["Beat the median stock", "Close higher (up/down)"])
    target = "outperform" if target_label.startswith("Beat") else "up"
    model_name = st.selectbox("Model", E.MODELS)
    top_k = st.slider("Picks per period (top-k)", 1, 10, 5)
    cost_pct = st.slider("Round-trip trading cost (%)", 0.0, 1.0, 0.20, 0.05,
                         help="Brokerage + STT + slippage, charged on every rebalance in the backtest.")
    years = st.slider("Years of history", 4, 12, 8)

    with st.expander("Features and outliers", expanded=True):
        st.caption("Technical indicators are always used. Switch other groups on/off and "
                   "compare accuracy to see what actually helps.")
        groups = st.multiselect("Extra feature groups", OPTIONAL_GROUPS, default=OPTIONAL_GROUPS)
        clean = st.checkbox("Remove bad data points", value=True,
                            help="Deletes one-day price spikes that instantly reverse (data glitches).")
        winsor = st.checkbox("Winsorize extreme feature values", value=True,
                             help="Clips each feature to its 1st-99th percentile. Limits are learned "
                                  "from training data only, so nothing leaks.")
        want_earn = st.checkbox("Fetch actual earnings dates (slower)", value=False,
                                help="Uses Yahoo Finance; coverage for Indian stocks is patchy. "
                                     "Without it, a volume-spike + gap proxy is used.")
        fii_file = st.file_uploader("FII/DII flows CSV (optional)", type="csv",
                                    help="Columns: date, fii_net, dii_net (Rs crore). Free long history is "
                                         "hard to automate, so you provide it.")
        st.download_button("Download CSV template", FII_TEMPLATE, "fii_dii_template.csv", "text/csv")

    demo = st.checkbox("Demo mode (simulated data, works offline)", value=False)
    tickers_text = st.text_area("Tickers (Yahoo format, comma separated)",
                                ", ".join(E.NIFTY50_NS), height=140, disabled=demo)
    run = st.button("Run analysis", type="primary", width="stretch")


# ------------------------------------------------------------------ data
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(tickers, years, demo, clean, want_earn, fii_bytes):
    if demo:
        prices, market = E.make_synthetic(n_stocks=30, n_days=1800, phi=0.0, seed=7)
        extras, smap = E.make_synthetic_extras(prices, seed=7)
        return dict(prices=prices, failed=[], cleaned={}, market=market, vix=None,
                    extras=extras, sector_map=smap, notes=[])
    notes = []
    prices, failed, cleaned = E.download_prices(list(tickers), years, clean=clean)
    market, vix = E.download_market(years)
    extras = E.download_extras(years)
    if not extras["sector"]:
        notes.append("Sector indices could not be downloaded; sector features skipped.")
    if not extras["global"]:
        notes.append("Global cues could not be downloaded; global features skipped.")
    if vix is None:
        notes.append("India VIX unavailable; VIX features skipped.")
    if want_earn:
        extras["earnings"] = E.download_earnings(list(prices))
        if not extras["earnings"]:
            notes.append("No earnings dates returned by Yahoo; using the volume-spike proxy.")
    if fii_bytes:
        try:
            extras["fii_dii"] = E.parse_fii_dii(pd.read_csv(io.BytesIO(fii_bytes)))
        except Exception as e:
            notes.append(f"FII/DII file ignored: {e}")
    return dict(prices=prices, failed=failed, cleaned=cleaned, market=market, vix=vix,
                extras=extras, sector_map=None, notes=notes)


if run:
    tickers = tuple(t.strip().upper() for t in tickers_text.replace("\n", ",").split(",") if t.strip())
    if not demo and len(tickers) < 10:
        st.error("Please provide at least 10 tickers so a daily ranking is meaningful.")
        st.stop()

    with st.spinner("Loading data (sectors, global cues, VIX)..."):
        try:
            D = load_data(tickers, years, demo, clean, want_earn,
                          None if (demo or fii_file is None) else fii_file.getvalue())
        except Exception as e:
            st.error(f"Could not load data: {e}")
            st.stop()
    if len(D["prices"]) < 10:
        st.error("Fewer than 10 tickers returned usable data. Check your internet connection or ticker names.")
        st.stop()

    bar = st.progress(0.0, text="Training walk-forward models...")
    try:
        res = E.run_pipeline(
            D["prices"], market=D["market"], vix=D["vix"], horizon=horizon, target=target,
            model_name=model_name, top_k=top_k, cost_pct=cost_pct,
            extras=D["extras"], sector_map=D["sector_map"], groups=groups, winsorize=winsor,
            progress=lambda x: bar.progress(min(x, 1.0), text="Training walk-forward models..."),
        )
    except Exception as e:
        bar.empty()
        st.error(f"Analysis failed: {e}")
        st.stop()
    bar.empty()
    st.session_state["res"] = dict(res=res, D=D, demo=demo, n=len(D["prices"]), horizon=horizon,
                                   top_k=top_k, cost=cost_pct, label=horizon_label,
                                   winsor=winsor, clean=clean, groups=groups)

if "res" not in st.session_state:
    st.info("Choose your settings in the sidebar and press **Run analysis**. "
            "Tick *Demo mode* first if you just want to see how the app works.")
    st.stop()

S = st.session_state["res"]
res, D = S["res"], S["D"]
m, bs = res["metrics"], res["backtest_stats"]

if S["demo"]:
    st.warning("Demo mode: these are SIMULATED random prices and random extra data, not real stocks. "
               "Notice that accuracy stays near 50% - that is what 'no edge' looks like.")
for note in D["notes"]:
    st.caption(f"Note: {note}")
if D["failed"]:
    st.caption(f"Skipped (no data): {', '.join(D['failed'])}")
st.caption("Feature groups used: " + ", ".join(res["feature_groups"]) + f" ({res['n_features']} features)")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Top picks", "How accurate is it?", "Backtest", "What drives it?", "About"])

# ------------------------------------------------------------------ picks
with tab1:
    st.subheader(f"Ranking for {S['label'].lower()} (data as of {res['as_of'].date()})")
    ms = res["market_state"]
    if ms["mkt_shock"]:
        st.warning("Market-wide shock today: the Nifty moved far more than usual. "
                   "Model scores are less reliable right after outlier days.")
    if ms["vix_z"] is not None and ms["vix_z"] > 2:
        st.warning(f"India VIX is unusually high (z-score {ms['vix_z']:.1f}). Expect larger, noisier moves.")
    picks = res["picks"].copy()
    top = picks[picks["pick"]].drop(columns="pick")
    show = top.assign(
        score=(top["score"] * 100).round(1),
        last_close=top["last_close"].round(2),
        ret_1d=(top["ret_1d"] * 100).round(2),
        ret_5d=(top["ret_5d"] * 100).round(2),
        ret_20d=(top["ret_20d"] * 100).round(2),
        rsi_14=top["rsi_14"].round(0),
    ).rename(columns={"score": "Score (%)", "last_close": "Last close (Rs)", "ret_1d": "1d %",
                      "ret_5d": "5d %", "ret_20d": "20d %", "rsi_14": "RSI", "alerts": "Alerts"})
    show.index.name = "Stock"
    st.dataframe(show, width="stretch")
    st.caption("Score = the model's estimated probability (%) of beating the median stock. "
               "A coin flip would be 50, and scores rarely stray far from it - that is normal. "
               "Alerts flag outlier events (big move, volume spike, large gap, earnings soon) that make a pick riskier.")
    with st.expander("Full ranking of all stocks"):
        full = picks.drop(columns="pick")
        full.index.name = "Stock"
        st.dataframe(full.assign(score=(full["score"] * 100).round(1)).round(3), width="stretch")

# ------------------------------------------------------------------ accuracy
with tab2:
    st.subheader("Out-of-sample accuracy (walk-forward, no peeking)")
    st.write(f"Tested on {m['n_days']:,} trading days "
             f"({res['test_start'].date()} to {res['test_end'].date()}), "
             f"{m['n_predictions']:,} predictions made by models that had never seen those days.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m['accuracy']*100:.1f}%",
              f"{(m['accuracy']-m['baseline_accuracy'])*100:+.1f} pts vs naive")
    c2.metric("Naive baseline", f"{m['baseline_accuracy']*100:.1f}%", help="Always guess the more common outcome.")
    c3.metric("AUC", f"{m['auc']:.3f}", help="0.50 = coin flip, 1.00 = perfect. Anything above ~0.55 is rare for stocks.")
    c4.metric(f"Top-{S['top_k']} hit rate", f"{m['topk_hit_rate']*100:.1f}%",
              help="How often the model's top picks actually did better than the median stock.")
    c5, c6, c7 = st.columns(3)
    c5.metric("Rank correlation (IC)", f"{m['ic_mean']:+.3f}", f"t-stat {m['ic_tstat']:+.1f}",
              help="Daily correlation between predicted rank and actual rank. >0.03 with |t|>2 is a meaningful edge.")
    c6.metric(f"Avg return of top-{S['top_k']}", f"{m['topk_avg_ret']*100:+.3f}%",
              f"{(m['topk_avg_ret']-m['all_avg_ret'])*100:+.3f}% vs all stocks",
              help="Per holding period, before costs. Can be pulled around by a few outlier winners.")
    c7.metric(f"Median return of top-{S['top_k']}", f"{m['topk_median_ret']*100:+.3f}%",
              help="Typical pick, ignoring outliers. If this is far below the average, a few lucky "
                   "outliers are carrying the result.")

    edge = m["auc"] - 0.5
    if m["ic_tstat"] > 2 and edge > 0.01:
        st.success("There is a small but statistically visible edge in this test. "
                   "Check the Backtest tab to see if it survives trading costs.")
    else:
        st.error("No reliable edge detected: accuracy is close to a coin flip. "
                 "This is the normal outcome for short-term stock prediction.")
    st.caption("Realistic expectation: 50-55% direction accuracy at best. If you ever see 65%+ on stock "
               "prediction, suspect data leakage (training on future information).")

# ------------------------------------------------------------------ backtest
with tab3:
    bt = res["backtest"]
    st.subheader(f"Backtest: hold the top {S['top_k']} picks for {S['horizon']} day(s), repeat")
    cols = st.columns(3)
    for col, key, title in ((cols[0], "strategy_net", f"Strategy (after {S['cost']:.2f}% cost)"),
                            (cols[1], "strategy_gross", "Strategy (before costs)"),
                            (cols[2], "benchmark", "Equal-weight all stocks")):
        s = bs[key]
        col.metric(title, f"{s['cagr']*100:+.1f}% / yr", f"Sharpe {s['sharpe']:.2f}")
        col.caption(f"Total {s['total_return']*100:+.0f}% | worst drawdown {s['max_drawdown']*100:.0f}%")
    chart = bt[["equity_net", "equity_gross", "equity_bench"]].rename(columns={
        "equity_net": "Strategy after costs", "equity_gross": "Strategy before costs",
        "equity_bench": "Equal-weight benchmark"})
    st.line_chart(chart)
    if bs["strategy_net"]["cagr"] <= bs["benchmark"]["cagr"]:
        st.warning("After costs the strategy did not beat simply holding all the stocks equally. "
                   "This is common: small prediction edges get eaten by brokerage, STT and slippage.")
    st.caption("Assumes full turnover every period (conservative), equal weights, no leverage. "
               "Past results do not predict future results.")

# ------------------------------------------------------------------ drivers
with tab4:
    st.subheader("Which feature groups does the model actually use?")
    imp = res["importance"]
    st.write("For each group, its values were shuffled on unseen test data and the drop in AUC was measured. "
             "A bigger drop means the model relies on that group. Around zero means it adds nothing.")
    st.bar_chart(pd.DataFrame({"AUC drop (points)": (imp * 100).round(3)}))
    for g, v in imp.items():
        verdict = "helps" if v > 0.003 else "no measurable help"
        st.write(f"- **{g}**: {v*100:+.2f} AUC points - {verdict}")
    st.caption("Tip: turn a group off in the sidebar and re-run. If AUC does not fall, the group was not helping.")

    st.subheader("Outliers and data quality")
    o1, o2 = st.columns(2)
    with o1:
        st.write("How common outlier events are (share of stock-days):")
        st.dataframe(pd.DataFrame({"Share of days": {k: f"{v*100:.2f}%" for k, v in res["outliers"].items()}}),
                     width="stretch")
    with o2:
        if S["clean"] and D["cleaned"]:
            st.write("Bad data points removed:")
            st.dataframe(pd.DataFrame({"Rows removed": D["cleaned"]}), width="stretch")
        elif S["clean"]:
            st.write("Bad data points removed: none found." if not S["demo"]
                     else "Bad data points removed: none (simulated data is clean).")
        else:
            st.write("Bad-data cleaning was switched off.")
        st.write("Winsorizing of extreme feature values: " + ("on (1st-99th percentile, learned on training data)."
                                                           if S["winsor"] else "off."))

# ------------------------------------------------------------------ about
with tab5:
    st.markdown(f"""
**How it works**
1. Downloads daily prices (Yahoo Finance), the Nifty 50, India VIX, sector indices and global cues
   (S&P 500, USD/INR, Brent crude, US 10-year yield).
2. Builds features known at the close of each day:
   - *Technical*: returns, volatility, RSI, MACD, moving-average distance, volume surge, gaps, rank vs other stocks.
   - *Market & VIX*: Nifty moves, volatility regime.
   - *Sector*: the stock's own sector index and how the stock compares with it.
   - *Global cues*: lagged one day, because US markets close after India does.
   - *Events & outliers*: big-move, volume-spike and gap flags, days since the last one, a quarterly results-cycle
     proxy (or real earnings dates), and a market-wide shock flag.
   - *FII/DII*: institutional flows from your CSV, lagged one day because NSE publishes them after the close.
3. **Outlier handling**: bad data spikes are removed, extreme feature values are clipped using limits learned only
   on training data, and genuine outliers are kept as features because a big move or volume spike carries information.
4. Target: will the stock beat the median stock over the next {S['horizon']} trading day(s)?
5. **Walk-forward validation**: the model is retrained every ~3 months using only past data, then tested on the
   following unseen months. A purge gap stops overlapping labels from leaking the future.
6. Backtest holds the top picks, charges costs, and compares with an equal-weight benchmark.

**Not included**: delivery percentage and news sentiment need paid or scraped data sources, so they are not automated here.

**Why accuracy looks modest**: markets are very noisy and widely watched, so easy patterns get competed away.
A durable 52-54% is already a real finding.

**Limits**: the ticker list is approximate (index membership changes), Yahoo data can have gaps, the backtest
ignores taxes, liquidity and price impact. Never risk money you cannot afford to lose.
This is not SEBI-registered investment advice.
""")
