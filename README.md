# Nifty Stock Ranker

A Streamlit app that ranks Nifty 50 stocks by the probability of beating the median stock over the
**next day** or **next week**, using walk-forward machine learning, and shows honest out-of-sample
accuracy plus a cost-aware backtest.

Educational project. Not investment advice.

## Run it (needs internet for live data)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the link Streamlit prints (usually http://localhost:8501), choose settings, press **Run analysis**.
Tick **Demo mode** to try the app on simulated data without internet.

## Feature groups (switch on/off in the sidebar)

| Group | What it contains | Source |
|-------|------------------|--------|
| Technical (always on) | returns, volatility, RSI, MACD, moving averages, volume, gaps, cross-sectional ranks | Yahoo Finance |
| Market & VIX | Nifty 50 moves, India VIX level and z-score | Yahoo Finance |
| Sector | the stock's sector index (IT, Banks, Auto, Pharma, FMCG, Metals, Energy, Financials) and relative strength | Yahoo Finance |
| Global cues | S&P 500, USD/INR, Brent crude, US 10y yield, lagged 1 day | Yahoo Finance |
| Events & outliers | big-move, volume-spike, large-gap flags, days since last event, results-cycle proxy, real earnings dates (optional), market-wide shock flag | derived / Yahoo |
| FII/DII | institutional net flows, 5d/20d sums, z-score, lagged 1 day | **your CSV** (`date,fii_net,dii_net`) |

Not automated: delivery percentage and news sentiment (they need paid or scraped sources).

## Outlier handling

1. **Bad-data removal**: one-day price spikes over 25% that reverse next day are deleted (data glitches).
   Real crashes and persistent jumps are kept.
2. **Winsorizing**: each feature is clipped to its 1st-99th percentile, with limits learned on training data only.
3. **Outliers as signals**: genuine outliers become features (big move, volume spike, gap, market shock),
   and picks with such events get an **Alerts** label.
4. The accuracy tab shows both mean and median return of the picks, so you can see if a few outlier winners
   are carrying the result.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit interface |
| `engine.py` | Data download, cleaning, features, walk-forward training, metrics, backtest |
| `selftest.py` | Leakage and sanity tests (random data must give ~50%, planted signal must be found, lag alignment checked) |

Run `python selftest.py` any time after changing the code.

## Reading the results

- **Accuracy / AUC**: 50% / 0.50 is a coin flip. 52-55% is a realistic ceiling for stocks.
- **IC (rank correlation)** with |t-stat| > 2: a statistically visible edge.
- **What drives it?** tab: which feature groups the model really uses. Near zero = no help.
- **Backtest after costs** vs the equal-weight benchmark: the real test of usefulness.
