from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from app.us_futu.data import (
    USDataConfig,
    fetch_us_history,
    filter_us_tradeable_universe,
    load_us_universe_snapshot,
    resample_ohlcv,
)
from app.us_futu.indicators import MRMCMacdConfig, build_mrmc_nx_indicators


@dataclass(slots=True)
class USUniverseConfig:
    min_price: float = 5.0
    min_avg_volume_20: float = 2_000_000
    min_avg_dollar_volume_20: float = 100_000_000
    max_symbols: int = 0
    exclude_symbol_patterns: list[str] = field(
        default_factory=lambda: [
            r"\^",
            r"W$",
            r"WS$",
            r"WT$",
            r"U$",
            r"R$",
            r"P$",
        ]
    )


@dataclass(slots=True)
class USMarketConfig:
    index_symbols: list[str] = field(default_factory=lambda: ["^GSPC", "^IXIC", "^DJI"])
    min_positive_count: int = 2


@dataclass(slots=True)
class USSignalConfig:
    bottom_lookback_bars: int = 6
    sell_lookback_bars: int = 3
    breakout_lookback_bars: int = 5
    retest_lookback_bars: int = 10
    retest_tolerance_pct: float = 0.02
    right_side_only: bool = True
    weekly_trend_required: bool = False
    monthly_filter_enabled: bool = True


@dataclass(slots=True)
class USSectorsConfig:
    min_resonance_members: int = 2
    min_breakout_members: int = 1
    baskets: dict[str, list[str]] = field(default_factory=dict)


def load_us_futu_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _recent_true(series: pd.Series, lookback: int) -> bool:
    if series.empty:
        return False
    return bool(series.tail(lookback).fillna(False).astype(bool).any())


def _crossed_above(series_a: pd.Series, series_b: pd.Series, lookback: int) -> bool:
    if len(series_a) < 2 or len(series_b) < 2:
        return False
    crossed = (series_a.shift(1) <= series_b.shift(1)) & (series_a > series_b)
    return _recent_true(crossed, lookback)


def _ladder_retest_ok(df: pd.DataFrame, lookback: int, tolerance_pct: float) -> bool:
    if df.empty:
        return False
    recent = df.tail(lookback).copy()
    support = recent[["blue_upper", "yellow_upper"]].min(axis=1)
    low_touch = recent["low"] <= support * (1 + tolerance_pct)
    close_hold = recent["close"] >= support * (1 - tolerance_pct)
    return bool((low_touch & close_hold).any())


def _timeframe_snapshot(df: pd.DataFrame, signal_config: USSignalConfig) -> dict[str, object]:
    latest = df.iloc[-1]
    avg_volume_20 = float(df["volume"].tail(20).mean())
    avg_dollar_volume_20 = float((df["close"] * df["volume"]).tail(20).mean())
    breakout_recent = _crossed_above(df["close"], df["blue_upper"], signal_config.breakout_lookback_bars)
    breakout_yellow_recent = _crossed_above(df["close"], df["yellow_upper"], signal_config.breakout_lookback_bars + 2)
    blue_cross_yellow_recent = _crossed_above(df["blue_mid"], df["yellow_mid"], signal_config.breakout_lookback_bars + 3)
    return {
        "latest_close": float(latest["close"]),
        "avg_volume_20": avg_volume_20,
        "avg_dollar_volume_20": avg_dollar_volume_20,
        "bottom_recent": _recent_true(df["mrmc_bottom_signal"], signal_config.bottom_lookback_bars),
        "sell_recent": _recent_true(df["mrmc_sell_signal"], signal_config.sell_lookback_bars),
        "blue_above_yellow": bool(latest["blue_above_yellow"]),
        "close_above_blue": bool(latest["close_above_blue"]),
        "close_above_yellow": bool(latest["close_above_yellow"]),
        "breakout_recent": breakout_recent,
        "breakout_yellow_recent": breakout_yellow_recent,
        "blue_cross_yellow_recent": blue_cross_yellow_recent,
        "retest_ok": _ladder_retest_ok(df, signal_config.retest_lookback_bars, signal_config.retest_tolerance_pct),
        "bottom_signal_today": bool(latest["mrmc_bottom_signal"]),
        "sell_signal_today": bool(latest["mrmc_sell_signal"]),
        "trend_ok": bool(latest["blue_above_yellow"] and latest["close_above_blue"] and not _recent_true(df["mrmc_sell_signal"], signal_config.sell_lookback_bars)),
        "bullish_ok": bool(latest["close_above_blue"] and not _recent_true(df["mrmc_sell_signal"], signal_config.sell_lookback_bars)),
    }


def _market_regime(index_snapshots: dict[str, dict[str, object]], market_config: USMarketConfig) -> tuple[str, int]:
    positive_count = 0
    for snapshot in index_snapshots.values():
        if snapshot["blue_above_yellow"] and snapshot["close_above_blue"] and not snapshot["sell_recent"]:
            positive_count += 1
    if positive_count >= market_config.min_positive_count:
        return "risk_on", positive_count
    if positive_count == 0:
        return "risk_off", positive_count
    return "neutral", positive_count


def _basket_membership(symbol: str, sectors_config: USSectorsConfig) -> list[str]:
    baskets: list[str] = []
    for basket_name, members in sectors_config.baskets.items():
        if symbol in members:
            baskets.append(basket_name)
    return baskets


def _compute_sector_summary(
    states: dict[str, dict[str, object]],
    sectors_config: USSectorsConfig,
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for basket_name, members in sectors_config.baskets.items():
        present = [symbol for symbol in members if symbol in states]
        if not present:
            summary[basket_name] = {
                "member_count": 0,
                "trend_count": 0,
                "breakout_count": 0,
                "bottom_count": 0,
                "score": 0.0,
            }
            continue
        trend_count = sum(1 for symbol in present if states[symbol]["1d"]["trend_ok"])
        breakout_count = sum(1 for symbol in present if states[symbol]["1d"]["breakout_recent"])
        bottom_count = sum(1 for symbol in present if states[symbol]["1d"]["bottom_recent"])
        weekly_trend_count = sum(1 for symbol in present if states[symbol]["1w"]["trend_ok"])
        score = trend_count * 2.0 + breakout_count * 3.0 + bottom_count * 1.5 + weekly_trend_count * 1.5
        summary[basket_name] = {
            "member_count": len(present),
            "trend_count": trend_count,
            "breakout_count": breakout_count,
            "bottom_count": bottom_count,
            "weekly_trend_count": weekly_trend_count,
            "score": round(score, 2),
        }
    return summary


def _best_sector_context(
    symbol: str,
    states: dict[str, dict[str, object]],
    sector_summary: dict[str, dict[str, object]],
    sectors_config: USSectorsConfig,
) -> dict[str, object]:
    memberships = _basket_membership(symbol, sectors_config)
    if not memberships:
        return {
            "sector_name": "",
            "sector_member_count": 0,
            "sector_trend_count": 0,
            "sector_breakout_count": 0,
            "sector_bottom_count": 0,
            "sector_weekly_trend_count": 0,
            "sector_score": 0.0,
            "sector_resonance_ok": False,
        }
    ranked = sorted(
        memberships,
        key=lambda basket: sector_summary.get(basket, {}).get("score", 0.0),
        reverse=True,
    )
    best = ranked[0]
    context = sector_summary.get(best, {})
    resonance_ok = bool(
        context.get("trend_count", 0) >= sectors_config.min_resonance_members
        or context.get("breakout_count", 0) >= sectors_config.min_breakout_members
    )
    return {
        "sector_name": best,
        "sector_member_count": int(context.get("member_count", 0) or 0),
        "sector_trend_count": int(context.get("trend_count", 0) or 0),
        "sector_breakout_count": int(context.get("breakout_count", 0) or 0),
        "sector_bottom_count": int(context.get("bottom_count", 0) or 0),
        "sector_weekly_trend_count": int(context.get("weekly_trend_count", 0) or 0),
        "sector_score": float(context.get("score", 0.0) or 0.0),
        "sector_resonance_ok": resonance_ok,
    }


def _build_candidates(
    symbol: str,
    timeframe_snapshots: dict[str, dict[str, object]],
    market_regime: str,
    positive_index_count: int,
    signal_config: USSignalConfig,
    universe_config: USUniverseConfig,
    sector_context: dict[str, object],
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    daily = timeframe_snapshots["1d"]
    weekly = timeframe_snapshots["1w"]
    monthly = timeframe_snapshots["1mo"]

    liquidity_ok = (
        daily["latest_close"] >= universe_config.min_price
        and daily["avg_volume_20"] >= universe_config.min_avg_volume_20
        and daily["avg_dollar_volume_20"] >= universe_config.min_avg_dollar_volume_20
    )
    if not liquidity_ok:
        return candidates

    right_side_ok = daily["close_above_blue"] or weekly["close_above_blue"]
    if signal_config.right_side_only and not right_side_ok:
        return candidates
    if signal_config.weekly_trend_required and not weekly["trend_ok"]:
        return candidates
    if signal_config.monthly_filter_enabled and monthly["sell_recent"]:
        return candidates

    if (
        daily["bottom_recent"]
        and daily["breakout_recent"]
        and daily["close_above_blue"]
        and not daily["sell_recent"]
    ):
        score = 88.0
        if weekly["trend_ok"]:
            score += 4
        if monthly["bullish_ok"]:
            score += 2
        if market_regime == "risk_on":
            score += 4
        score += min(8.0, float(sector_context["sector_score"]) * 0.4)
        candidates.append(
            {
                "symbol": symbol,
                "strategy_type": "daily_bottom_breakout",
                "score": round(score, 2),
                "market_regime": market_regime,
                "market_positive_index_count": positive_index_count,
                "trigger_timeframe": "1d",
                "entry_note": "日线 MRMC 抄底后，价格右侧站上蓝梯",
                "risk_note": "日线蓝梯下边缘失守则离场",
            }
        )

    if (
        daily["blue_above_yellow"]
        and daily["close_above_blue"]
        and not daily["sell_recent"]
    ):
        score = 80.0
        if weekly["blue_above_yellow"]:
            score += 4
        if weekly["close_above_blue"]:
            score += 2
        if monthly["blue_above_yellow"]:
            score += 2
        if market_regime == "risk_on":
            score += 3
        score += min(8.0, float(sector_context["sector_score"]) * 0.35)
        candidates.append(
            {
                "symbol": symbol,
                "strategy_type": "blue_above_yellow_trend_daily",
                "score": round(score, 2),
                "market_regime": market_regime,
                "market_positive_index_count": positive_index_count,
                "trigger_timeframe": "1d",
                "entry_note": "日线蓝梯稳定在黄梯之上，价格维持在蓝梯上方",
                "risk_note": "日线蓝梯下边缘失守则离场",
            }
        )

    if (
        daily["breakout_yellow_recent"]
        and daily["retest_ok"]
        and daily["close_above_blue"]
        and not daily["sell_recent"]
    ):
        score = 86.0
        if daily["blue_cross_yellow_recent"]:
            score += 2
        if weekly["trend_ok"]:
            score += 3
        if market_regime == "risk_on":
            score += 2
        score += min(10.0, float(sector_context["sector_score"]) * 0.5)
        candidates.append(
            {
                "symbol": symbol,
                "strategy_type": "daily_sweet_spot",
                "score": round(score, 2),
                "market_regime": market_regime,
                "market_positive_index_count": positive_index_count,
                "trigger_timeframe": "1d",
                "entry_note": "突破蓝黄梯后回踩支撑，符合日线甜点结构",
                "risk_note": "日线蓝梯下边缘失守则离场",
            }
        )

    if (
        weekly["trend_ok"]
        and weekly["close_above_blue"]
        and not weekly["sell_recent"]
        and daily["bullish_ok"]
    ):
        score = 84.0
        if monthly["blue_above_yellow"]:
            score += 3
        if daily["bottom_recent"]:
            score += 2
        if market_regime == "risk_on":
            score += 2
        score += min(10.0, float(sector_context["sector_score"]) * 0.45)
        candidates.append(
            {
                "symbol": symbol,
                "strategy_type": "weekly_trend_resonance",
                "score": round(score, 2),
                "market_regime": market_regime,
                "market_positive_index_count": positive_index_count,
                "trigger_timeframe": "1w",
                "entry_note": "周线趋势走强，日线同步维持右侧结构",
                "risk_note": "周线蓝梯下边缘失守则离场",
            }
        )

    return candidates


def screen_us_market(
    universe_config: USUniverseConfig,
    market_config: USMarketConfig | None = None,
    signal_config: USSignalConfig | None = None,
    sectors_config: USSectorsConfig | None = None,
    data_config: USDataConfig | None = None,
    macd_config: MRMCMacdConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    market_config = market_config or USMarketConfig()
    signal_config = signal_config or USSignalConfig()
    sectors_config = sectors_config or USSectorsConfig()
    data_config = data_config or USDataConfig()
    macd_config = macd_config or MRMCMacdConfig()

    universe_snapshot = load_us_universe_snapshot(data_config)
    universe = filter_us_tradeable_universe(
        universe_snapshot,
        min_price=universe_config.min_price,
        min_avg_volume_20=universe_config.min_avg_volume_20,
        min_avg_dollar_volume_20=universe_config.min_avg_dollar_volume_20,
        exclude_symbol_patterns=universe_config.exclude_symbol_patterns,
    )
    if universe_config.max_symbols > 0:
        universe = universe.head(universe_config.max_symbols).copy()

    index_snapshots: dict[str, dict[str, object]] = {}
    for index_symbol in market_config.index_symbols:
        daily = fetch_us_history(index_symbol, data_config, index=True)
        if daily.empty:
            continue
        daily_ind = build_mrmc_nx_indicators(daily, macd_config)
        index_snapshots[index_symbol] = _timeframe_snapshot(daily_ind, signal_config)

    market_regime, positive_index_count = _market_regime(index_snapshots, market_config)

    states: dict[str, dict[str, object]] = {}
    for _, row in universe.iterrows():
        symbol = str(row["symbol"])
        daily = fetch_us_history(symbol, data_config, index=False)
        if daily.empty:
            continue
        daily_ind = build_mrmc_nx_indicators(daily, macd_config)
        weekly = resample_ohlcv(daily, "W-FRI")
        monthly = resample_ohlcv(daily, "ME")
        if len(daily_ind) < 120 or len(weekly) < 40 or len(monthly) < 18:
            continue
        weekly_ind = build_mrmc_nx_indicators(weekly, macd_config)
        monthly_ind = build_mrmc_nx_indicators(monthly, macd_config)
        states[symbol] = {
            "snapshot": row.to_dict(),
            "1d": _timeframe_snapshot(daily_ind, signal_config),
            "1w": _timeframe_snapshot(weekly_ind, signal_config),
            "1mo": _timeframe_snapshot(monthly_ind, signal_config),
        }

    sector_summary = _compute_sector_summary(states, sectors_config)
    rows: list[dict[str, object]] = []
    for symbol, state in states.items():
        timeframe_snapshots = {
            "1d": state["1d"],
            "1w": state["1w"],
            "1mo": state["1mo"],
        }
        sector_context = _best_sector_context(symbol, states, sector_summary, sectors_config)
        for candidate in _build_candidates(
            symbol,
            timeframe_snapshots,
            market_regime,
            positive_index_count,
            signal_config,
            universe_config,
            sector_context,
        ):
            candidate["daily_price"] = timeframe_snapshots["1d"]["latest_close"]
            candidate["daily_avg_volume_20"] = timeframe_snapshots["1d"]["avg_volume_20"]
            candidate["daily_avg_dollar_volume_20"] = timeframe_snapshots["1d"]["avg_dollar_volume_20"]
            candidate["daily_blue_above_yellow"] = timeframe_snapshots["1d"]["blue_above_yellow"]
            candidate["daily_close_above_blue"] = timeframe_snapshots["1d"]["close_above_blue"]
            candidate["daily_bottom_recent"] = timeframe_snapshots["1d"]["bottom_recent"]
            candidate["daily_breakout_recent"] = timeframe_snapshots["1d"]["breakout_recent"]
            candidate["weekly_blue_above_yellow"] = timeframe_snapshots["1w"]["blue_above_yellow"]
            candidate["weekly_close_above_blue"] = timeframe_snapshots["1w"]["close_above_blue"]
            candidate["monthly_blue_above_yellow"] = timeframe_snapshots["1mo"]["blue_above_yellow"]
            candidate["sector_name"] = sector_context["sector_name"]
            candidate["sector_member_count"] = sector_context["sector_member_count"]
            candidate["sector_trend_count"] = sector_context["sector_trend_count"]
            candidate["sector_breakout_count"] = sector_context["sector_breakout_count"]
            candidate["sector_bottom_count"] = sector_context["sector_bottom_count"]
            candidate["sector_weekly_trend_count"] = sector_context["sector_weekly_trend_count"]
            candidate["sector_score"] = sector_context["sector_score"]
            candidate["sector_resonance_ok"] = sector_context["sector_resonance_ok"]
            rows.append(candidate)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["sector_resonance_ok", "sector_score", "score", "strategy_type", "symbol"],
            ascending=[False, False, False, True, True],
        ).reset_index(drop=True)
    summary = {
        "market_regime": market_regime,
        "market_positive_index_count": positive_index_count,
        "index_snapshots": index_snapshots,
        "universe_size": int(len(universe)),
        "state_count": int(len(states)),
        "sector_summary": sector_summary,
    }
    return result, summary
