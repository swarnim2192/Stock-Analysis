from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

@dataclass
class TrainResult:
    pipeline: Pipeline
    metrics: Dict[str, float]
    feature_names: list[str]

def fetch_prices(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} (period={period}, interval={interval}).")
    return df.dropna().copy()

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
