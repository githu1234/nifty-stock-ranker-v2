"""
engine.py - data, features, outlier handling, leak-free walk-forward ML,
evaluation and backtest for ranking Indian (NSE) stocks.

Core idea: each day, score every stock with the probability that it will beat
the median stock over the next `horizon` trading days, then pick the top-k.
All validation is walk-forward (train on the past, test on the future) with a
purge gap so overlapping labels cannot leak future information.

Feature groups
  Technical          returns, volatility, RSI, MACD, moving averages, volume, ranks
  Market & VIX       Nifty index moves, India VIX
  Sector             the stock's sector index (Nifty IT, Bank, Auto, ...)
  Global cues        S&P 500, USD/INR, Brent crude, US 10y yield (lagged 1 day)
  Events & outliers  big-move / volume-spike / gap flags, earnings proximity,
                     market-shock flag
  FII/DII            institutional flows (needs a CSV from the user), lagged 1 day

Outlier handling
  1. clean_prices(): removes one-day bad ticks that instantly reverse
  2. Winsorizer:     clips extreme feature values using limits learned on the
                     TRAINING data only
  3. event flags:    outliers are also turned into features, since a big move
                     or a volume spike carries information

Educational code. Not investment advice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Approximate Nifty 50 list (constituents change; edit freely in the app).
NIFTY50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL",
    "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL",
    "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
    "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY",
    "ITC", "JIOFIN", "JSWSTEEL", "KOTAKBANK", "LT",
    "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN",
    "SUNPHARMA", "TATACONSUM", "TATASTEEL", "TCS", "TECHM",
    "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
]
NIFTY50_NS = [f"{t}.NS" for t in NIFTY50]

MODELS = ["Gradient boosting", "Logistic regression", "Random forest"]

# Sector indices on Yahoo Finance (any that fail to download are skipped).
SECTOR_INDEX = {
    "IT": "^CNXIT", "Banks": "^NSEBANK", "Auto": "^CNXAUTO",
    "Pharma": "^CNXPHARMA", "FMCG": "^CNXFMCG", "Metals": "^CNXMETAL",
    "Energy": "^CNXENERGY", "Financials": "^CNXFIN",
}
STOCK_SECTOR = {
    "INFY": "IT", "TCS": "IT", "HCLTECH": "IT", "WIPRO": "IT", "TECHM": "IT",
    "HDFCBANK": "Banks", "ICICIBANK": "Banks", "AXISBANK": "Banks",
    "KOTAKBANK": "Banks", "SBIN": "Banks", "INDUSINDBK": "Banks",
    "MARUTI": "Auto", "M&M": "Auto", "BAJAJ-AUTO": "Auto",
    "EICHERMOT": "Auto", "HEROMOTOCO": "Auto",
    "SUNPHARMA": "Pharma", "CIPLA": "Pharma", "DRREDDY": "Pharma",
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG", "TATACONSUM": "FMCG",
    "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "HINDALCO": "Metals",
    "RELIANCE": "Energy", "ONGC": "Energy", "NTPC": "Energy",
    "POWERGRID": "Energy", "COALINDIA": "Energy",
    "BAJFINANCE": "Financials", "BAJAJFINSV": "Financials", "HDFCLIFE": "Financials",
    "SBILIFE": "Financials", "SHRIRAMFIN": "Financials", "JIOFIN": "Financials",
}
# US/FX/commodity closes happen after the Indian close, so they are lagged 1 day.
GLOBAL_SYMBOLS = {"sp500": "^GSPC", "usdinr": "INR=X", "brent": "BZ=F", "us10y": "^TNX"}

GROUP_NAMES = ["Technical", "Market & VIX", "Sector", "Global cues",
               "Events & outliers", "FII/DII"]
_EVENT_COLS = {
    "ret_z", "big_move", "vol_spike", "gap_big", "days_since_big_move",
    "days_since_vol_spike", "cal_days_since_event", "mkt_shock",
    "days_to_earnings", "days_since_earnings", "earn_in_window",
}


def group_of(col):
    if col in _EVENT_COLS:
        return "Events & outliers"
    if col.startswith(("sector_", "rel_sector_")):
        return "Sector"
    if col.startswith(("sp500_", "usdinr_", "brent_", "us10y_")):
        return "Global cues"
    if col.startswith(("fii_", "dii_")):
        return "FII/DII"
    if col.startswith(("mkt_", "vix")):
        return "Market & VIX"
    return "Technical"


# --------------------------------------------------------------------------
# Data download and cleaning
# --------------------------------------------------------------------------
def _clean_index(obj):
    obj = obj.copy()
    idx = pd.to_datetime(obj.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    obj.index = idx.normalize()
    obj = obj[~obj.index.duplicated(keep="last")]
    return obj.sort_index()


def clean_prices(df, spike=0.25):
    """
    Remove obviously bad data: non-positive prices, and one-day spikes larger
    than `spike` that reverse almost completely the next day (typical Yahoo
    glitch). Real crashes and rallies do not reverse like that, so they stay.
    Returns (cleaned_df, n_rows_removed).
    """
    n0 = len(df)
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    r = df["Close"].pct_change()
    nxt = r.shift(-1)
    bad = (r.abs() > spike) & (nxt * r < 0) & (nxt.abs() >= 0.6 * r.abs())
    df = df[~bad].copy()
    df["High"] = df[["High", "Open", "Close"]].max(axis=1)
    df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)
    return df, n0 - len(df)


def download_prices(tickers, years=8, clean=True):
    """Adjusted daily OHLCV from Yahoo Finance. Returns (prices, failed, cleaned)."""
    import yfinance as yf

    tickers = list(tickers)
    raw = yf.download(
        tickers, period=f"{years}y", auto_adjust=True, progress=False,
        group_by="ticker", threads=True,
    )
    prices, failed, cleaned = {}, [], {}
    for t in tickers:
        try:
            df = raw[t][["Open", "High", "Low", "Close", "Volume"]]
            df = df.dropna(subset=["Close"])
            df = _clean_index(df[df["Close"] > 0])
            if clean:
                df, removed = clean_prices(df)
                if removed:
                    cleaned[t] = removed
            if len(df) < 300:
                raise ValueError("too little history")
            prices[t] = df
        except Exception:
            failed.append(t)
    return prices, failed, cleaned


def _dl_close(sym, years):
    import yfinance as yf

    try:
        d = yf.download(sym, period=f"{years}y", auto_adjust=True, progress=False)
        s = d["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s = s.dropna()
        return _clean_index(s) if len(s) else None
    except Exception:
        return None


def download_market(years=8):
    """Nifty 50 index and India VIX (either may be None if unavailable)."""
    return _dl_close("^NSEI", years), _dl_close("^INDIAVIX", years)


def download_extras(years=8, sectors=True, global_cues=True):
    """Sector indices and global cues. Anything that fails is skipped."""
    out = {"sector": {}, "global": {}, "fii_dii": None, "earnings": None}
    if sectors:
        for name, sym in SECTOR_INDEX.items():
            s = _dl_close(sym, years)
            if s is not None and len(s) > 300:
                out["sector"][name] = s
    if global_cues:
        for name, sym in GLOBAL_SYMBOLS.items():
            s = _dl_close(sym, years)
            if s is not None and len(s) > 300:
                out["global"][name] = s
    return out


def download_earnings(tickers):
    """Actual earnings dates from Yahoo (slow, patchy for Indian stocks)."""
    import yfinance as yf

    out = {}
    for t in tickers:
        try:
            ed = yf.Ticker(t).get_earnings_dates(limit=40)
            if ed is not None and len(ed):
                idx = pd.to_datetime(ed.index)
                if getattr(idx, "tz", None) is not None:
                    idx = idx.tz_localize(None)
                out[t] = pd.DatetimeIndex(sorted(set(idx.normalize())))
        except Exception:
            pass
    return out


def parse_fii_dii(df):
    """
    CSV with a date column and FII and DII net-flow columns (Rs crore).
    Column names are matched loosely (anything containing 'date', 'fii'/'fpi', 'dii').
    """
    cols = {str(c).lower().strip(): c for c in df.columns}
    dcol = next((cols[c] for c in cols if "date" in c), None)
    fcol = next((cols[c] for c in cols if "fii" in c or "fpi" in c), None)
    icol = next((cols[c] for c in cols if "dii" in c), None)
    if not (dcol and fcol and icol):
        raise ValueError("CSV needs columns containing: date, FII, DII")

    def num(s):
        return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")

    out = pd.DataFrame({"fii_net": num(df[fcol]).values, "dii_net": num(df[icol]).values})
    out.index = pd.DatetimeIndex(pd.to_datetime(df[dcol].values, dayfirst=True, errors="coerce"))
    out = out[out.index.notna()]
    return _clean_index(out).dropna(how="all")


def make_synthetic(n_stocks=30, n_days=1800, phi=0.0, seed=0):
    """
    Simulated prices for testing. phi=0 -> pure random walk (nothing to learn).
    phi>0 -> stock-specific returns have weak day-to-day autocorrelation, i.e.
    a small genuine, learnable signal.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    mkt = rng.normal(0.0004, 0.010, n_days)
    prices = {}
    for i in range(n_stocks):
        beta = rng.uniform(0.6, 1.4)
        sigma = rng.uniform(0.008, 0.020)
        eps = rng.normal(0, sigma, n_days)
        idio = np.zeros(n_days)
        for t in range(1, n_days):
            idio[t] = phi * idio[t - 1] + eps[t]
        r = beta * mkt + idio
        close = 100 * np.exp(np.cumsum(r))
        prev = np.concatenate([[100.0], close[:-1]])
        opn = prev * (1 + rng.normal(0, 0.003, n_days))
        high = np.maximum(opn, close) * (1 + np.abs(rng.normal(0, 0.004, n_days)))
        low = np.minimum(opn, close) * (1 - np.abs(rng.normal(0, 0.004, n_days)))
        vol = rng.lognormal(13, 0.4, n_days)
        prices[f"SIM{i:02d}.NS"] = pd.DataFrame(
            {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol},
            index=dates,
        )
    market = pd.Series(100 * np.exp(np.cumsum(mkt)), index=dates)
    return prices, market


def make_synthetic_extras(prices, seed=0):
    """Random (signal-free) sector / global / FII-DII / earnings data for demos and tests."""
    rng = np.random.default_rng(seed)
    dates = next(iter(prices.values())).index

    def walk():
        return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))), index=dates)

    extras = {
        "sector": {f"S{i}": walk() for i in range(4)},
        "global": {k: walk() for k in GLOBAL_SYMBOLS},
        "fii_dii": pd.DataFrame({"fii_net": rng.normal(0, 1500, len(dates)),
                                 "dii_net": rng.normal(0, 1500, len(dates))}, index=dates),
        "earnings": {t: pd.DatetimeIndex(dates[::63] + pd.Timedelta(days=int(rng.integers(0, 20))))
                     for t in prices},
    }
    sector_map = {t: f"S{i % 4}" for i, t in enumerate(prices)}
    return extras, sector_map


# --------------------------------------------------------------------------
# Features and targets (every feature uses only information available at the
# close of day t)
# --------------------------------------------------------------------------
def _days_since(flag):
    pos = np.arange(len(flag), dtype=float)
    last = pd.Series(np.where(flag.values > 0, pos, np.nan), index=flag.index).ffill()
    return (pos - last).clip(upper=250)


def _features_one(df):
    c, h, l, o, v = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"]
    r1 = c.pct_change()
    f = pd.DataFrame(index=df.index)
    for n in (1, 3, 5, 10, 20, 60):
        f[f"ret_{n}"] = c.pct_change(n)
    for n in (10, 20, 60):
        f[f"vol_{n}"] = r1.rolling(n).std()

    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi_14"] = 100 - 100 / (1 + up / (dn + 1e-12))

    macd = (c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()) / c
    f["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()

    f["dist_ma20"] = c / c.rolling(20).mean() - 1
    f["dist_ma50"] = c / c.rolling(50).mean() - 1
    f["dist_high_60"] = c / c.rolling(60).max() - 1
    f["vol_ratio"] = np.log((v + 1) / (v.rolling(20).mean() + 1))
    f["gap"] = o / c.shift(1) - 1
    f["range"] = (h - l) / c
    f["close_pos"] = (c - l) / (h - l + 1e-12)

    # ---- event / outlier features -------------------------------------
    prior_vol = r1.rolling(20).std().shift(1)          # volatility BEFORE today
    f["ret_z"] = r1 / (prior_vol + 1e-9)                # today's move in "sigmas"
    f["big_move"] = (f["ret_z"].abs() > 3).astype(float)
    f["vol_spike"] = (f["vol_ratio"] > np.log(2.5)).astype(float)
    f["gap_big"] = (f["gap"].abs() > 2 * prior_vol).astype(float)
    f["days_since_big_move"] = _days_since(f["big_move"])
    f["days_since_vol_spike"] = _days_since(f["vol_spike"])
    # volume spike + gap together usually means results/news; quarterly cycle proxy
    ev = (f["vol_spike"] > 0) & (f["gap_big"] > 0)
    last_ev = pd.Series(df.index.where(ev.values), index=df.index).ffill()
    f["cal_days_since_event"] = (df.index.to_series() - last_ev).dt.days.clip(upper=400)
    return f


def _earn_feats(ed, index, horizon):
    """Days to next / since last known earnings date (dates are announced in advance)."""
    e = np.asarray(ed.values, dtype="datetime64[ns]")
    d = np.asarray(index.values, dtype="datetime64[ns]")
    nat = np.datetime64("NaT")
    pos = np.searchsorted(e, d, side="left")
    nxt = np.where(pos < len(e), e[np.minimum(pos, len(e) - 1)], nat)
    prv_pos = np.searchsorted(e, d, side="right") - 1
    prv = np.where(prv_pos >= 0, e[np.maximum(prv_pos, 0)], nat)
    to_ = ((nxt - d) / np.timedelta64(1, "D")).astype(float)
    since = ((d - prv) / np.timedelta64(1, "D")).astype(float)
    out = pd.DataFrame(index=index)
    out["days_to_earnings"] = np.clip(to_, None, 120)
    out["days_since_earnings"] = np.clip(since, None, 120)
    win = (to_ <= horizon * 7 / 5 + 1).astype(float)
    win[np.isnan(to_)] = np.nan
    out["earn_in_window"] = win
    return out


def _align(df, dates, lag):
    """Put a frame on the Indian trading calendar (forward-fill), then lag it."""
    out = df.reindex(df.index.union(dates)).ffill().reindex(dates)
    return out.shift(lag) if lag else out


def _fii_features(fii, dates):
    d = fii[["fii_net", "dii_net"]].reindex(dates)
    out = pd.DataFrame(index=dates)
    for k in ("fii", "dii"):
        s = d[f"{k}_net"]
        out[f"{k}_net"] = s
        out[f"{k}_5d"] = s.rolling(5, min_periods=3).sum()
        out[f"{k}_20d"] = s.rolling(20, min_periods=10).sum()
        out[f"{k}_z"] = (s - s.rolling(60, min_periods=20).mean()) / (
            s.rolling(60, min_periods=20).std() + 1e-9)
    return out.shift(1)          # NSE publishes flows after the close -> lag 1 day


def build_panel(prices, market=None, vix=None, horizon=5, target="outperform",
                extras=None, sector_map=None):
    """
    Long table indexed by (date, ticker) with features, `fwd_ret` (the return
    over the next `horizon` days) and label `y`.
    target = "outperform": y=1 if the stock beats that day's median stock.
    target = "up":         y=1 if the stock closes higher.
    """
    extras = extras or {}
    all_dates = pd.DatetimeIndex(sorted(set().union(*[df.index for df in prices.values()])))

    sector_frames = {}
    for name, s in (extras.get("sector") or {}).items():
        sf = pd.DataFrame({f"sector_ret_{n}": s.pct_change(n) for n in (1, 5, 20)})
        sector_frames[name] = _align(sf, all_dates, 0)
    earn = extras.get("earnings") or {}

    frames = []
    for t, df in prices.items():
        f = _features_one(df)
        f["fwd_ret"] = df["Close"].shift(-horizon) / df["Close"] - 1
        f["close"] = df["Close"]
        if sector_frames:
            sname = (sector_map or {}).get(t) or STOCK_SECTOR.get(t.replace(".NS", ""))
            sf = sector_frames.get(sname)
            for n in (1, 5, 20):
                f[f"sector_ret_{n}"] = (sf[f"sector_ret_{n}"].reindex(f.index).values
                                        if sf is not None else np.nan)
        if earn:
            if t in earn:
                f = f.join(_earn_feats(earn[t], f.index, horizon))
            else:
                for col in ("days_to_earnings", "days_since_earnings", "earn_in_window"):
                    f[col] = np.nan
        f["ticker"] = t
        frames.append(f)
    panel = pd.concat(frames)
    panel.index.name = "date"
    panel = panel.set_index("ticker", append=True).sort_index()
    dates = panel.index.get_level_values("date")

    if market is None:
        closes = pd.DataFrame({t: d["Close"] for t, d in prices.items()})
        market = closes.pct_change().mean(axis=1).add(1).cumprod()
    mk = pd.DataFrame(index=market.index)
    for n in (1, 5, 20):
        mk[f"mkt_ret_{n}"] = market.pct_change(n)
    mk["mkt_vol_20"] = market.pct_change().rolling(20).std()
    if vix is not None:
        v = vix.reindex(market.index).ffill()
        mk["vix"] = v
        mk["vix_chg_5"] = v.pct_change(5)
        mk["vix_z"] = (v - v.rolling(252, min_periods=60).mean()) / (
            v.rolling(252, min_periods=60).std() + 1e-9)
    for col in mk.columns:
        panel[col] = mk[col].reindex(dates).values

    shock = (panel["mkt_ret_1"].abs() > 2.5 * panel["mkt_vol_20"]).astype(float)
    shock[panel["mkt_ret_1"].isna() | panel["mkt_vol_20"].isna()] = np.nan
    panel["mkt_shock"] = shock

    if extras.get("global"):
        gf = pd.concat([pd.DataFrame({f"{name}_ret_{n}": s.pct_change(n) for n in (1, 5)})
                        for name, s in extras["global"].items()], axis=1)
        gf = _align(gf, all_dates, 1)
        for col in gf.columns:
            panel[col] = gf[col].reindex(dates).values
    if extras.get("fii_dii") is not None:
        ff = _fii_features(extras["fii_dii"], all_dates)
        for col in ff.columns:
            panel[col] = ff[col].reindex(dates).values

    g = panel.groupby(level="date")
    for c in ("ret_5", "ret_20", "vol_20", "vol_ratio"):
        panel[f"{c}_rank"] = g[c].rank(pct=True)
    panel["ret_5_rel"] = panel["ret_5"] - panel["mkt_ret_5"]
    panel["ret_20_rel"] = panel["ret_20"] - panel["mkt_ret_20"]
    if sector_frames:
        for n in (1, 5, 20):
            panel[f"sector_ret_{n}"] = panel[f"sector_ret_{n}"].fillna(panel[f"mkt_ret_{n}"])
        panel["rel_sector_5"] = panel["ret_5"] - panel["sector_ret_5"]
        panel["rel_sector_20"] = panel["ret_20"] - panel["sector_ret_20"]

    med = panel.groupby(level="date")["fwd_ret"].transform("median")
    if target == "outperform":
        y = (panel["fwd_ret"] > med).astype(float)
    else:
        y = (panel["fwd_ret"] > 0).astype(float)
    y[panel["fwd_ret"].isna()] = np.nan
    panel["y"] = y
    return panel.replace([np.inf, -np.inf], np.nan)


def feature_columns(panel, groups=None):
    """Feature columns, optionally limited to some groups (Technical is always kept)."""
    cols = [c for c in panel.columns
            if c not in ("fwd_ret", "y", "close") and panel[c].notna().any()]
    if groups is None:
        return cols
    keep = set(groups) | {"Technical"}
    return [c for c in cols if group_of(c) in keep]


def required_cols(feats):
    """Only the core technical features must be present; other groups may have gaps."""
    return [c for c in feats if group_of(c) == "Technical"]


# --------------------------------------------------------------------------
# Models and walk-forward validation
# --------------------------------------------------------------------------
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each feature to its [lower, upper] quantiles, learned on training data only."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        with np.errstate(all="ignore"):
            self.lo_ = np.nanquantile(X, self.lower, axis=0)
            self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def make_model(name, winsorize=True):
    steps = [Winsorizer()] if winsorize else []
    if name == "Logistic regression":
        steps += [SimpleImputer(strategy="median", keep_empty_features=True),
                  StandardScaler(), LogisticRegression(C=0.1, max_iter=500)]
    elif name == "Random forest":
        steps += [SimpleImputer(strategy="median", keep_empty_features=True),
                  RandomForestClassifier(n_estimators=150, max_depth=6,
                                         min_samples_leaf=200, n_jobs=-1, random_state=0)]
    else:
        steps += [HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=150,
            min_samples_leaf=300, l2_regularization=1.0, random_state=0)]
    return make_pipeline(*steps)


def walk_forward(panel, horizon, model_name, feats, min_train_days=750,
                 retrain_every=63, winsorize=True, progress=None, keep_blocks=4):
    """
    Expanding-window walk-forward. For each test block starting at index s, the
    model only sees rows whose label was fully known before s (purge gap of
    horizon+1 days), so no future information reaches training.
    Returns (predictions, last_blocks) where last_blocks holds the final few
    (model, test_rows) pairs for feature-group importance.
    """
    req = required_cols(feats)
    pdates = panel.index.get_level_values("date")
    dates = pdates.unique().sort_values()
    labeled = panel.dropna(subset=req + ["y"])
    ldates = labeled.index.get_level_values("date")

    starts = list(range(min_train_days, len(dates), retrain_every))
    if not starts:
        raise ValueError("Not enough history for the chosen training window.")
    out, blocks = [], []
    for i, s in enumerate(starts):
        block = dates[s: s + retrain_every]
        train = labeled[ldates <= dates[s - horizon - 1]]
        test = panel[(pdates >= block[0]) & (pdates <= block[-1])]
        test = test.dropna(subset=req + ["y", "fwd_ret"])
        if len(train) < 2000 or len(test) == 0 or train["y"].nunique() < 2:
            continue
        model = make_model(model_name, winsorize)
        model.fit(train[feats], train["y"])
        p = model.predict_proba(test[feats])[:, 1]
        out.append(pd.DataFrame(
            {"proba": p, "y": test["y"].values, "fwd_ret": test["fwd_ret"].values},
            index=test.index,
        ))
        blocks.append((model, test))
        if progress:
            progress((i + 1) / len(starts))
    if not out:
        raise ValueError("No test blocks could be evaluated.")
    return pd.concat(out).sort_index(), blocks[-keep_blocks:]


def group_importance(blocks, feats, repeats=3, seed=0):
    """
    Drop in out-of-sample AUC when one feature group's values are shuffled.
    ~0 = the model does not use it (or it does not help); larger = more useful.
    """
    rng = np.random.default_rng(seed)
    groups = {}
    for c in feats:
        groups.setdefault(group_of(c), []).append(c)
    drops = {g: [] for g in groups}
    for model, test in blocks:
        X, y = test[feats], test["y"]
        if y.nunique() < 2:
            continue
        base = roc_auc_score(y, model.predict_proba(X)[:, 1])
        for g, cols in groups.items():
            vals = []
            for _ in range(repeats):
                Xp = X.copy()
                Xp[cols] = X[cols].values[rng.permutation(len(X))]
                vals.append(roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
            drops[g].append(base - float(np.mean(vals)))
    return pd.Series({g: float(np.mean(v)) for g, v in drops.items() if v}).sort_values(ascending=False)


# --------------------------------------------------------------------------
# Evaluation and backtest
# --------------------------------------------------------------------------
def evaluate(pred, top_k):
    y, p = pred["y"], pred["proba"]
    acc = float(((p > 0.5).astype(float) == y).mean())
    base = float(max(y.mean(), 1 - y.mean()))
    auc = float(roc_auc_score(y, p))

    g = pred.groupby(level="date")
    pr = g["proba"].rank()
    rr = g["fwd_ret"].rank()
    pr = pr - pr.groupby(level="date").transform("mean")
    rr = rr - rr.groupby(level="date").transform("mean")
    num = (pr * rr).groupby(level="date").sum()
    den = np.sqrt((pr ** 2).groupby(level="date").sum()
                  * (rr ** 2).groupby(level="date").sum())
    ic = (num / den).dropna()
    ic_mean = float(ic.mean())
    ic_t = float(ic.mean() / (ic.std() / np.sqrt(len(ic)) + 1e-12))

    rank = g["proba"].rank(ascending=False, method="first")
    top = pred[rank <= top_k]
    return {
        "accuracy": acc,
        "baseline_accuracy": base,
        "auc": auc,
        "ic_mean": ic_mean,
        "ic_tstat": ic_t,
        "topk_hit_rate": float(top["y"].mean()),
        "topk_avg_ret": float(top["fwd_ret"].mean()),
        "topk_median_ret": float(top["fwd_ret"].median()),
        "all_avg_ret": float(pred["fwd_ret"].mean()),
        "n_predictions": int(len(pred)),
        "n_days": int(pred.index.get_level_values("date").nunique()),
    }


def backtest(pred, horizon, top_k, cost_pct):
    """Hold the top-k picks for `horizon` days, non-overlapping, equal weight."""
    dates = pred.index.get_level_values("date").unique().sort_values()
    rows = []
    for d in dates[::horizon]:
        day = pred.xs(d, level="date")
        picks = day.nlargest(top_k, "proba")
        gross = float(picks["fwd_ret"].mean())
        rows.append((d, gross, gross - cost_pct / 100.0, float(day["fwd_ret"].mean())))
    bt = pd.DataFrame(rows, columns=["date", "gross", "net", "bench"]).set_index("date")
    bt["equity_net"] = (1 + bt["net"]).cumprod()
    bt["equity_gross"] = (1 + bt["gross"]).cumprod()
    bt["equity_bench"] = (1 + bt["bench"]).cumprod()

    ppy = 252 / horizon
    n = len(bt)

    def stats(r, eq):
        cagr = float(eq.iloc[-1] ** (ppy / n) - 1)
        sharpe = float(r.mean() / (r.std() + 1e-12) * np.sqrt(ppy))
        mdd = float((eq / eq.cummax() - 1).min())
        return {"cagr": cagr, "sharpe": sharpe, "max_drawdown": mdd,
                "total_return": float(eq.iloc[-1] - 1)}

    return bt, {
        "strategy_net": stats(bt["net"], bt["equity_net"]),
        "strategy_gross": stats(bt["gross"], bt["equity_gross"]),
        "benchmark": stats(bt["bench"], bt["equity_bench"]),
        "periods": n,
    }


def _alerts(r):
    a = []
    if r.get("big_move", 0) == 1:
        a.append("big move today")
    if r.get("vol_spike", 0) == 1:
        a.append("volume spike")
    if r.get("gap_big", 0) == 1:
        a.append("large gap")
    d = r.get("days_to_earnings", np.nan)
    if pd.notna(d) and d <= 10:
        a.append(f"earnings in {int(d)}d")
    return ", ".join(a)


def latest_picks(panel, horizon, model_name, feats, top_k, winsorize=True):
    """Train on everything with a known outcome; score the most recent day."""
    req = required_cols(feats)
    labeled = panel.dropna(subset=req + ["y"])
    model = make_model(model_name, winsorize)
    model.fit(labeled[feats], labeled["y"])

    dates = panel.index.get_level_values("date")
    last = dates.max()
    cur = panel[dates == last].dropna(subset=req).droplevel("date")
    cur["score"] = model.predict_proba(cur[feats])[:, 1]
    cur = cur.sort_values("score", ascending=False)
    table = pd.DataFrame({
        "score": cur["score"],
        "last_close": cur["close"],
        "ret_1d": cur["ret_1"],
        "ret_5d": cur["ret_5"],
        "ret_20d": cur["ret_20"],
        "rsi_14": cur["rsi_14"],
    })
    table["alerts"] = cur.apply(_alerts, axis=1)
    table["pick"] = np.arange(len(table)) < top_k
    market = {
        "mkt_shock": bool(cur["mkt_shock"].iloc[0] == 1) if "mkt_shock" in cur else False,
        "vix_z": float(cur["vix_z"].iloc[0]) if "vix_z" in cur and cur["vix_z"].notna().any() else None,
    }
    return table, last, market


def outlier_summary(panel):
    """How common outliers/events are in the data (share of stock-days)."""
    s = {}
    for col, label in (("big_move", "Moves larger than 3 sigma"),
                       ("vol_spike", "Volume more than 2.5x normal"),
                       ("gap_big", "Large overnight gaps")):
        if col in panel:
            s[label] = float(panel[col].mean())
    if "mkt_shock" in panel:
        m = panel["mkt_shock"].groupby(level="date").first()
        s["Market-wide shock days"] = float(m.mean())
    return s


def run_pipeline(prices, market=None, vix=None, horizon=5, target="outperform",
                 model_name="Gradient boosting", top_k=5, cost_pct=0.2,
                 min_train_days=750, retrain_every=63, progress=None,
                 extras=None, sector_map=None, groups=None, winsorize=True):
    panel = build_panel(prices, market, vix, horizon, target, extras, sector_map)
    feats = feature_columns(panel, groups)
    pred, blocks = walk_forward(panel, horizon, model_name, feats, min_train_days,
                                retrain_every, winsorize, progress)
    metrics = evaluate(pred, top_k)
    bt, bt_stats = backtest(pred, horizon, top_k, cost_pct)
    picks, as_of, market_state = latest_picks(panel, horizon, model_name, feats, top_k, winsorize)
    return {
        "metrics": metrics, "backtest": bt, "backtest_stats": bt_stats,
        "picks": picks, "as_of": as_of, "pred": pred, "market_state": market_state,
        "importance": group_importance(blocks, feats),
        "feature_groups": sorted({group_of(c) for c in feats}),
        "n_features": len(feats),
        "outliers": outlier_summary(panel),
        "test_start": pred.index.get_level_values("date").min(),
        "test_end": pred.index.get_level_values("date").max(),
    }
