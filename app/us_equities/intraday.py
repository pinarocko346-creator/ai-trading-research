from __future__ import annotations

from app.us_futu.data import USDataConfig, download_us_history, resample_ohlcv
from app.us_futu.indicators import MRMCMacdConfig, build_mrmc_nx_indicators

from app.us_equities.config import USEquitiesIntradayConfig, USEquitiesSignalConfig
from app.us_equities.daily_logic import timeframe_snapshot


def build_intraday_state(
    symbol: str,
    intraday_config: USEquitiesIntradayConfig,
    signal_config: USEquitiesSignalConfig,
    macd_config: MRMCMacdConfig,
) -> dict[str, object] | None:
    if not intraday_config.enabled:
        return None

    data_config = USDataConfig(
        source=intraday_config.source,
        daily_period="2y",
        intraday_30m_period=intraday_config.intraday_30m_period,
        intraday_60m_period=intraday_config.intraday_60m_period,
        refresh_hours=intraday_config.refresh_hours,
    )
    frame_30m = download_us_history(symbol, "30m", data_config)
    frame_60m = download_us_history(symbol, "60m", data_config)
    if frame_30m.empty or frame_60m.empty:
        return None

    ind_30m = build_mrmc_nx_indicators(frame_30m, macd_config)
    ind_1h = build_mrmc_nx_indicators(frame_60m, macd_config)
    ind_2h = build_mrmc_nx_indicators(resample_ohlcv(frame_60m, "2h"), macd_config)
    ind_3h = build_mrmc_nx_indicators(resample_ohlcv(frame_60m, "3h"), macd_config)
    ind_4h = build_mrmc_nx_indicators(resample_ohlcv(frame_60m, "4h"), macd_config)

    enough_history = (
        len(ind_30m) >= intraday_config.min_30m_bars
        and len(ind_1h) >= intraday_config.min_1h_bars
        and len(ind_2h) >= intraday_config.min_2h_bars
        and len(ind_3h) >= intraday_config.min_3h_bars
        and len(ind_4h) >= intraday_config.min_4h_bars
    )
    if not enough_history:
        return None

    return {
        "30m": timeframe_snapshot(ind_30m, signal_config),
        "1h": timeframe_snapshot(ind_1h, signal_config),
        "2h": timeframe_snapshot(ind_2h, signal_config),
        "3h": timeframe_snapshot(ind_3h, signal_config),
        "4h": timeframe_snapshot(ind_4h, signal_config),
    }


def build_4321_candidate(
    symbol: str,
    state: dict[str, object],
    market_regime: str,
    positive_index_count: int,
    sector_context: dict[str, object],
) -> dict[str, object] | None:
    intraday = state.get("intraday")
    if not intraday:
        return None

    tf_30m = intraday["30m"]
    tf_1h = intraday["1h"]
    tf_2h = intraday["2h"]
    tf_3h = intraday["3h"]
    tf_4h = intraday["4h"]
    daily = state["1d"]
    weekly = state["1w"]

    bottom_resonance = (
        tf_1h["bottom_recent"]
        and tf_2h["bottom_recent"]
        and tf_3h["bottom_recent"]
        and tf_4h["bottom_recent"]
    )
    right_side_breakout = tf_30m["breakout_recent"] and tf_1h["close_above_blue"] and not tf_1h["sell_recent"]
    higher_tf_support = daily["bullish_ok"] or weekly["trend_ok"]
    if not (bottom_resonance and right_side_breakout and higher_tf_support):
        return None

    score = 96.0
    if daily["blue_above_yellow"]:
        score += 3
    if weekly["trend_ok"]:
        score += 3
    if market_regime == "risk_on":
        score += 4
    score += min(10.0, float(sector_context["sector_score"]) * 0.5)
    return {
        "symbol": symbol,
        "strategy_type": "4321_intraday_resonance",
        "score": round(score, 2),
        "market_regime": market_regime,
        "market_positive_index_count": positive_index_count,
        "trigger_timeframe": "30m+1h/2h/3h/4h",
        "entry_note": "1/2/3/4小时 MRMC 共振抄底，30分钟右侧突破蓝梯",
        "risk_note": "优先看30分钟或1小时蓝梯下边缘",
    }
