from __future__ import annotations
import logging
logging.getLogger("yfinance").setLevel(logging.ERROR)
import os, time, random
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Optional HTTP cache to reduce repeat hits to Yahoo
try:
    import requests_cache
    _session = requests_cache.CachedSession("yfinance_cache", backend="sqlite", expire_after=300)
except Exception:
    _session = None

DATA_DIR = "data_cache"
os.makedirs(DATA_DIR, exist_ok=True)

@dataclass
class TrainResult:
    pipeline: Pipeline
    metrics: Dict[str, float]
    feature_names: list[str]

def _cache_path(ticker: str, period: str, interval: str) -> str:
    safe = f"{ticker}_{period}_{interval}".replace("/", "-")
    return os.path.join(DATA_DIR, f"{safe}.parquet")

def _limit_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    # Map Streamlit-like periods to days
    days_map = {"1mo": 31, "3mo": 93, "6mo": 186, "1y": 366, "2y": 732, "5y": 1830}
    if period in days_map:
        cutoff = df.index.max() - pd.Timedelta(days=days_map[period])
        df = df[df.index >= cutoff]
    return df

def _fetch_yfinance(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        ticker, period=period, interval=interval,
        auto_adjust=True, progress=False, group_by="column",
        session=_session, timeout=30, threads=False
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def _fetch_stooq(ticker: str, period: str) -> pd.DataFrame:
    # Stooq provides DAILY data only; no key required
    df = pdr.DataReader(ticker, "stooq")  # returns most-recent first
    df = df.sort_index()                  # ascending time
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = _limit_by_period(df, period)
    return df

def fetch_prices(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Try Yahoo → Stooq → last-good parquet. Always return flat columns."""
    path = _cache_path(ticker, period, interval)

    # 1) Try Yahoo (few gentle retries)
    max_tries = 3
    for attempt in range(1, max_tries + 1):
        try:
            df = _fetch_yfinance(ticker, period, interval)
            if df is not None and not df.empty:
                df = df.dropna().copy()
                try: df.to_parquet(path)
                except Exception: pass
                return df
            raise ValueError("Empty dataframe from Yahoo")
        except Exception as e:
            msg = str(e)
            transient = any(s in msg for s in ["Rate limit", "Too Many Requests", "HTTP", "timed out", "Empty"])
            if transient and attempt < max_tries:
                time.sleep((2 ** (attempt - 1)) + random.uniform(0, 0.5))
                continue
            break  # move to stooq

    # 2) Stooq fallback (daily only). If user asked intraday, we still return daily.
    try:
        df = _fetch_stooq(ticker, period)
        if df is not None and not df.empty:
            try: df.to_parquet(path)
            except Exception: pass
            # mimic yfinance auto_adjust behavior already reflected in Stooq close
            return df.dropna().copy()
    except Exception:
        pass

    # 3) Last-good parquet snapshot
    if os.path.exists(path):
        try:
            df = pd.read_parquet(path)
            if not df.empty:
                df.attrs["stale"] = True
                return df
        except Exception:
            pass

    raise ValueError(f"No data returned for {ticker} (period={period}, interval={interval}).")

def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def make_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list[str]]:
    out = df.copy()
    out["ret_1"] = out["Close"].pct_change()
    out["ret_5"] = out["Close"].pct_change(5)
    out["ma_5"] = out["Close"].rolling(5).mean()
    out["ma_10"] = out["Close"].rolling(10).mean()
    out["ma_ratio_5_10"] = out["ma_5"] / (out["ma_10"] + 1e-9)
    out["vol_5"] = out["Volume"].rolling(5).mean()
    out["rsi_14"] = _rsi(out["Close"], 14)
    out["bb_mid"] = out["Close"].rolling(20).mean()
    out["bb_std"] = out["Close"].rolling(20).std()
    out["bb_z"] = (out["Close"] - out["bb_mid"]) / (out["bb_std"] + 1e-9)
    out["future_ret"] = out["Close"].pct_change().shift(-1)
    y = (out["future_ret"] > 0).astype(int)

    features = ["ret_1","ret_5","ma_ratio_5_10","vol_5","rsi_14","bb_z"]
    X = out[features].replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]
    return X, y, features

def train_model(X: pd.DataFrame, y: pd.Series, test_size: float = 0.25) -> TrainResult:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    return TrainResult(pipeline=pipe, metrics=metrics, feature_names=list(X.columns))

def save_model(result: TrainResult, path: str = "model.joblib") -> None:
    joblib.dump({"pipeline": result.pipeline, "features": result.feature_names}, path)

def load_model(path: str = "model.joblib") -> Optional[Pipeline]:
    try:
        saved = joblib.load(path)
        return saved.get("pipeline", None)
    except Exception:
        return None

def latest_prediction(pipe: Pipeline, X: pd.DataFrame) -> Optional[int]:
    if pipe is None or X.empty:
        return None
    return int(pipe.predict(X.tail(1))[0])
