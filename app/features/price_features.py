from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def require_ohlcv(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"缺少 OHLCV 字段: {missing}")


def _true_range(frame: pd.DataFrame) -> pd.Series:
    prev_close = frame["close"].shift(1)
    ranges = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def build_price_features(frame: pd.DataFrame) -> pd.DataFrame:
    require_ohlcv(frame)
    df = frame.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])

    for window in (5, 10, 20, 60):
        df[f"ma_{window}"] = df["close"].rolling(window).mean()
        df[f"rolling_high_{window}"] = df["high"].rolling(window).max()
        df[f"rolling_low_{window}"] = df["low"].rolling(window).min()

    df["avg_volume_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["avg_volume_20"].replace(0, np.nan)
    df["body_pct"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
    df["range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["bullish"] = df["close"] > df["open"]
    df["bearish"] = df["close"] < df["open"]
    df["pct_change"] = df["close"].pct_change()
    df["tr"] = _true_range(df)
    df["atr_14"] = df["tr"].rolling(14).mean()
    df["trend_up"] = (df["close"] > df["ma_20"]) & (df["ma_20"] > df["ma_60"])
    df["trend_down"] = (df["close"] < df["ma_20"]) & (df["ma_20"] < df["ma_60"])
    df["breakout_level_20"] = df["rolling_high_20"].shift(1)
    df["support_level_20"] = df["rolling_low_20"].shift(1)
    df["drawdown_from_high_60"] = 1 - df["close"] / df["rolling_high_60"].replace(0, np.nan)
    df["retracement_50_20"] = (df["rolling_high_20"].shift(1) + df["rolling_low_20"].shift(1)) / 2
    df["is_limit_up_like"] = df["pct_change"] >= 0.095
    df["is_limit_down_like"] = df["pct_change"] <= -0.095
    df["swing_low_flag"] = (df["low"] < df["low"].shift(1)) & (df["low"] < df["low"].shift(-1))
    df["swing_high_flag"] = (df["high"] > df["high"].shift(1)) & (df["high"] > df["high"].shift(-1))
    return df


def latest_feature_row(frame: pd.DataFrame) -> pd.Series:
    featured = build_price_features(frame)
    return featured.iloc[-1]
