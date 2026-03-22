from __future__ import annotations

import pandas as pd

from app.us_futu.data import resample_ohlcv
from app.us_futu.indicators import MRMCMacdConfig, build_mrmc_nx_indicators

from app.us_equities.config import USEquitiesSignalConfig


def recent_true(series: pd.Series, lookback: int) -> bool:
    if series.empty:
        return False
    return bool(series.tail(lookback).fillna(False).astype(bool).any())


def crossed_above(series_a: pd.Series, series_b: pd.Series, lookback: int) -> bool:
    if len(series_a) < 2 or len(series_b) < 2:
        return False
    crossed = (series_a.shift(1) <= series_b.shift(1)) & (series_a > series_b)
    return recent_true(crossed, lookback)


def ladder_retest_ok(df: pd.DataFrame, lookback: int, tolerance_pct: float) -> bool:
    recent = df.tail(lookback).copy()
    if recent.empty:
        return False
    support = recent[["blue_upper", "yellow_upper"]].min(axis=1)
    low_touch = recent["low"] <= support * (1 + tolerance_pct)
    close_hold = recent["close"] >= support * (1 - tolerance_pct)
    return bool((low_touch & close_hold).any())


def timeframe_snapshot(df: pd.DataFrame, signal_config: USEquitiesSignalConfig) -> dict[str, object]:
    latest = df.iloc[-1]
    avg_volume_20 = float(df["volume"].tail(20).mean())
    avg_dollar_volume_20 = float((df["close"] * df["volume"]).tail(20).mean())
    sell_recent = recent_true(df["mrmc_sell_signal"], signal_config.sell_lookback_bars)
    return {
        "latest_close": float(latest["close"]),
        "avg_volume_20": avg_volume_20,
        "avg_dollar_volume_20": avg_dollar_volume_20,
        "bottom_recent": recent_true(df["mrmc_bottom_signal"], signal_config.bottom_lookback_bars),
        "sell_recent": sell_recent,
        "blue_above_yellow": bool(latest["blue_above_yellow"]),
        "close_above_blue": bool(latest["close_above_blue"]),
        "close_above_yellow": bool(latest["close_above_yellow"]),
        "breakout_recent": crossed_above(df["close"], df["blue_upper"], signal_config.breakout_lookback_bars),
        "breakout_yellow_recent": crossed_above(
            df["close"], df["yellow_upper"], signal_config.breakout_lookback_bars + 2
        ),
        "blue_cross_yellow_recent": crossed_above(
            df["blue_mid"], df["yellow_mid"], signal_config.breakout_lookback_bars + 3
        ),
        "retest_ok": ladder_retest_ok(df, signal_config.retest_lookback_bars, signal_config.retest_tolerance_pct),
        "trend_ok": bool(latest["blue_above_yellow"] and latest["close_above_blue"] and not sell_recent),
        "bullish_ok": bool(latest["close_above_blue"] and not sell_recent),
    }


def build_symbol_state(
    daily_history: pd.DataFrame,
    signal_config: USEquitiesSignalConfig,
    macd_config: MRMCMacdConfig,
) -> dict[str, object] | None:
    daily = build_mrmc_nx_indicators(daily_history, macd_config)
    weekly_history = resample_ohlcv(daily_history, "W-FRI")
    monthly_history = resample_ohlcv(daily_history, "ME")
    if len(daily) < 120 or len(weekly_history) < 40 or len(monthly_history) < 18:
        return None
    weekly = build_mrmc_nx_indicators(weekly_history, macd_config)
    monthly = build_mrmc_nx_indicators(monthly_history, macd_config)
    return {
        "1d": timeframe_snapshot(daily, signal_config),
        "1w": timeframe_snapshot(weekly, signal_config),
        "1mo": timeframe_snapshot(monthly, signal_config),
    }
