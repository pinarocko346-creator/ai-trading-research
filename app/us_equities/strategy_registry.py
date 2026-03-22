from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.us_equities.config import USEquitiesSignalConfig, USEquitiesStrategyConfig, USEquitiesUniverseConfig


@dataclass(slots=True)
class StrategyContext:
    symbol: str
    state: dict[str, object]
    market_regime: str
    positive_index_count: int
    signal_config: USEquitiesSignalConfig
    universe_config: USEquitiesUniverseConfig
    sector_context: dict[str, object]


@dataclass(slots=True)
class StrategyDefinition:
    code: str
    name: str
    stage: str
    trigger_timeframe: str
    required_timeframes: tuple[str, ...]
    enabled_by_default: bool
    evaluator: Callable[[StrategyContext], dict[str, object] | None]


def _base_candidate(
    context: StrategyContext,
    *,
    strategy_type: str,
    score: float,
    trigger_timeframe: str,
    entry_note: str,
    risk_note: str,
) -> dict[str, object]:
    return {
        "symbol": context.symbol,
        "strategy_type": strategy_type,
        "score": round(score, 2),
        "market_regime": context.market_regime,
        "market_positive_index_count": context.positive_index_count,
        "trigger_timeframe": trigger_timeframe,
        "entry_note": entry_note,
        "risk_note": risk_note,
    }


def _recommended_intraday_plan() -> dict[str, object]:
    return {
        "entry_execution_timeframe": "30m",
        "confirmation_timeframes": "1h/2h/3h/4h",
        "max_resonance_timeframe": "4h",
        "recommended_hold_window": "1-4周",
        "recommended_option_tenor": "月期权优先",
        "recommended_stop_timeframe": "1h",
        "recommended_stop_reference": "1小时蓝梯下边缘",
        "sell_level_aggressive": "30m",
        "sell_level_standard": "1h",
        "sell_level_conservative": "4h",
        "sell_reference_aggressive": "30分钟蓝梯下边缘",
        "sell_reference_standard": "1小时蓝梯下边缘",
        "sell_reference_conservative": "4小时蓝梯下边缘",
        "recommended_sell_level": "standard",
    }


def _evaluate_daily_bottom_breakout(context: StrategyContext) -> dict[str, object] | None:
    daily = context.state["1d"]
    weekly = context.state["1w"]
    monthly = context.state["1mo"]
    if not (
        daily["bottom_recent"]
        and daily["breakout_recent"]
        and daily["close_above_blue"]
        and not daily["sell_recent"]
    ):
        return None
    score = 88.0
    if weekly["trend_ok"]:
        score += 4
    if monthly["bullish_ok"]:
        score += 2
    if context.market_regime == "risk_on":
        score += 4
    score += min(8.0, float(context.sector_context["sector_score"]) * 0.4)
    candidate = _base_candidate(
        context,
        strategy_type="daily_bottom_breakout",
        score=score,
        trigger_timeframe="1d",
        entry_note="日线 MRMC 抄底后，价格右侧站上蓝梯",
        risk_note="日线蓝梯下边缘失守则离场",
    )
    candidate["recommended_stop_timeframe"] = "1d"
    candidate["recommended_stop_reference"] = "日线蓝梯下边缘"
    candidate["recommended_hold_window"] = "数日至数周"
    candidate["sell_level_aggressive"] = "1d"
    candidate["sell_level_standard"] = "1d"
    candidate["sell_level_conservative"] = "1w"
    candidate["sell_reference_aggressive"] = "日线蓝梯下边缘"
    candidate["sell_reference_standard"] = "日线蓝梯下边缘"
    candidate["sell_reference_conservative"] = "周线蓝梯下边缘"
    candidate["recommended_sell_level"] = "standard"
    return candidate


def _evaluate_blue_above_yellow_trend_daily(context: StrategyContext) -> dict[str, object] | None:
    daily = context.state["1d"]
    weekly = context.state["1w"]
    monthly = context.state["1mo"]
    if not (daily["blue_above_yellow"] and daily["close_above_blue"] and not daily["sell_recent"]):
        return None
    score = 80.0
    if weekly["blue_above_yellow"]:
        score += 4
    if weekly["close_above_blue"]:
        score += 2
    if monthly["blue_above_yellow"]:
        score += 2
    if context.market_regime == "risk_on":
        score += 3
    score += min(8.0, float(context.sector_context["sector_score"]) * 0.35)
    candidate = _base_candidate(
        context,
        strategy_type="blue_above_yellow_trend_daily",
        score=score,
        trigger_timeframe="1d",
        entry_note="日线蓝梯稳定在黄梯之上，价格维持在蓝梯上方",
        risk_note="日线蓝梯下边缘失守则离场",
    )
    candidate["recommended_stop_timeframe"] = "1d"
    candidate["recommended_stop_reference"] = "日线蓝梯下边缘"
    candidate["recommended_hold_window"] = "数周至数月"
    candidate["sell_level_aggressive"] = "1d"
    candidate["sell_level_standard"] = "1d"
    candidate["sell_level_conservative"] = "1w"
    candidate["sell_reference_aggressive"] = "日线蓝梯下边缘"
    candidate["sell_reference_standard"] = "日线蓝梯下边缘"
    candidate["sell_reference_conservative"] = "周线蓝梯下边缘"
    candidate["recommended_sell_level"] = "standard"
    return candidate


def _evaluate_daily_sweet_spot(context: StrategyContext) -> dict[str, object] | None:
    daily = context.state["1d"]
    weekly = context.state["1w"]
    if not (
        daily["breakout_yellow_recent"]
        and daily["retest_ok"]
        and daily["close_above_blue"]
        and not daily["sell_recent"]
    ):
        return None
    score = 86.0
    if daily["blue_cross_yellow_recent"]:
        score += 2
    if weekly["trend_ok"]:
        score += 3
    if context.market_regime == "risk_on":
        score += 2
    score += min(10.0, float(context.sector_context["sector_score"]) * 0.5)
    candidate = _base_candidate(
        context,
        strategy_type="daily_sweet_spot",
        score=score,
        trigger_timeframe="1d",
        entry_note="突破蓝黄梯后回踩支撑，符合日线甜点结构",
        risk_note="日线蓝梯下边缘失守则离场",
    )
    candidate["recommended_stop_timeframe"] = "1d"
    candidate["recommended_stop_reference"] = "日线蓝梯或黄梯支撑位"
    candidate["recommended_hold_window"] = "数日至数周"
    candidate["sell_level_aggressive"] = "1d_blue"
    candidate["sell_level_standard"] = "1d_yellow"
    candidate["sell_level_conservative"] = "1w_blue"
    candidate["sell_reference_aggressive"] = "日线蓝梯下边缘"
    candidate["sell_reference_standard"] = "日线黄梯或关键回踩支撑"
    candidate["sell_reference_conservative"] = "周线蓝梯下边缘"
    candidate["recommended_sell_level"] = "standard"
    return candidate


def _evaluate_weekly_trend_resonance(context: StrategyContext) -> dict[str, object] | None:
    daily = context.state["1d"]
    weekly = context.state["1w"]
    monthly = context.state["1mo"]
    if not (weekly["trend_ok"] and weekly["close_above_blue"] and not weekly["sell_recent"] and daily["bullish_ok"]):
        return None
    score = 84.0
    if monthly["blue_above_yellow"]:
        score += 3
    if daily["bottom_recent"]:
        score += 2
    if context.market_regime == "risk_on":
        score += 2
    score += min(10.0, float(context.sector_context["sector_score"]) * 0.45)
    candidate = _base_candidate(
        context,
        strategy_type="weekly_trend_resonance",
        score=score,
        trigger_timeframe="1w",
        entry_note="周线趋势走强，日线同步维持右侧结构",
        risk_note="周线蓝梯下边缘失守则离场",
    )
    candidate["recommended_stop_timeframe"] = "1w"
    candidate["recommended_stop_reference"] = "周线蓝梯下边缘"
    candidate["recommended_hold_window"] = "数周至数月"
    candidate["sell_level_aggressive"] = "1d"
    candidate["sell_level_standard"] = "1w"
    candidate["sell_level_conservative"] = "1w_yellow"
    candidate["sell_reference_aggressive"] = "日线蓝梯下边缘"
    candidate["sell_reference_standard"] = "周线蓝梯下边缘"
    candidate["sell_reference_conservative"] = "周线黄梯下边缘"
    candidate["recommended_sell_level"] = "standard"
    return candidate


def _evaluate_4321_intraday_resonance(context: StrategyContext) -> dict[str, object] | None:
    intraday = context.state.get("intraday")
    if not intraday:
        return None
    tf_30m = intraday["30m"]
    tf_1h = intraday["1h"]
    tf_2h = intraday["2h"]
    tf_3h = intraday["3h"]
    tf_4h = intraday["4h"]
    daily = context.state["1d"]
    weekly = context.state["1w"]

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
    if context.market_regime == "risk_on":
        score += 4
    score += min(10.0, float(context.sector_context["sector_score"]) * 0.5)
    candidate = _base_candidate(
        context,
        strategy_type="4321_intraday_resonance",
        score=score,
        trigger_timeframe="30m+1h/2h/3h/4h",
        entry_note="1/2/3/4小时 MRMC 共振抄底，30分钟右侧突破蓝梯",
        risk_note="优先看30分钟或1小时蓝梯下边缘",
    )
    candidate.update(_recommended_intraday_plan())
    candidate["setup_bottom_resonance_1h"] = bool(tf_1h["bottom_recent"])
    candidate["setup_bottom_resonance_2h"] = bool(tf_2h["bottom_recent"])
    candidate["setup_bottom_resonance_3h"] = bool(tf_3h["bottom_recent"])
    candidate["setup_bottom_resonance_4h"] = bool(tf_4h["bottom_recent"])
    candidate["setup_breakout_30m"] = bool(tf_30m["breakout_recent"])
    candidate["setup_close_above_blue_1h"] = bool(tf_1h["close_above_blue"])
    candidate["setup_sell_recent_1h"] = bool(tf_1h["sell_recent"])
    candidate["higher_tf_support_daily"] = bool(daily["bullish_ok"])
    candidate["higher_tf_support_weekly"] = bool(weekly["trend_ok"])
    candidate["position_style"] = "事件驱动波段"
    return candidate


STRATEGY_REGISTRY: tuple[StrategyDefinition, ...] = (
    StrategyDefinition(
        code="daily_bottom_breakout",
        name="日线抄底右侧突破",
        stage="日线",
        trigger_timeframe="1d",
        required_timeframes=("1d", "1w", "1mo"),
        enabled_by_default=True,
        evaluator=_evaluate_daily_bottom_breakout,
    ),
    StrategyDefinition(
        code="blue_above_yellow_trend_daily",
        name="日线蓝在黄上趋势",
        stage="日线",
        trigger_timeframe="1d",
        required_timeframes=("1d", "1w", "1mo"),
        enabled_by_default=True,
        evaluator=_evaluate_blue_above_yellow_trend_daily,
    ),
    StrategyDefinition(
        code="daily_sweet_spot",
        name="日线甜点结构",
        stage="日线",
        trigger_timeframe="1d",
        required_timeframes=("1d", "1w"),
        enabled_by_default=True,
        evaluator=_evaluate_daily_sweet_spot,
    ),
    StrategyDefinition(
        code="weekly_trend_resonance",
        name="周线趋势共振",
        stage="周线",
        trigger_timeframe="1w",
        required_timeframes=("1d", "1w", "1mo"),
        enabled_by_default=True,
        evaluator=_evaluate_weekly_trend_resonance,
    ),
    StrategyDefinition(
        code="4321_intraday_resonance",
        name="4321 多周期共振",
        stage="多周期",
        trigger_timeframe="30m+1h/2h/3h/4h",
        required_timeframes=("30m", "1h", "2h", "3h", "4h", "1d", "1w"),
        enabled_by_default=False,
        evaluator=_evaluate_4321_intraday_resonance,
    ),
)


def evaluate_registered_strategies(context: StrategyContext) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for strategy in STRATEGY_REGISTRY:
        candidate = strategy.evaluator(context)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def get_enabled_strategies(strategy_config: USEquitiesStrategyConfig | None = None) -> tuple[StrategyDefinition, ...]:
    strategy_config = strategy_config or USEquitiesStrategyConfig()
    default_codes = {strategy.code for strategy in STRATEGY_REGISTRY if strategy.enabled_by_default}
    enabled_codes = default_codes | set(strategy_config.extra_enabled_codes)
    enabled_codes -= set(strategy_config.disabled_codes)
    return tuple(strategy for strategy in STRATEGY_REGISTRY if strategy.code in enabled_codes)


def evaluate_enabled_strategies(
    context: StrategyContext,
    strategy_config: USEquitiesStrategyConfig | None = None,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for strategy in get_enabled_strategies(strategy_config):
        candidate = strategy.evaluator(context)
        if candidate is not None:
            candidates.append(candidate)
    return candidates
