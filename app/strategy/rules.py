from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from app.core.types import ResearchSignal, SignalDefinition
from app.features.price_features import build_price_features


PROGRAMMABLE_SIGNAL_CODES = {
    "selling_climax",
    "2b_structure",
    "false_breakdown",
    "right_shoulder",
    "double_breakout",
    "strength_emergence",
    "jumping_creek",
    "cup_with_handle_watch",
    "cup_with_handle",
    "cup_with_handle_strict",
    "cup_with_handle_leader",
    "pullback_confirmation",
    "n_breakout",
    "support_resistance_flip",
    "spring",
    "pattern_breakout",
    "first_rebound_after_crash",
}

PREPARE_REQUIRED_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ma_10",
    "ma_20",
    "ma_60",
    "rolling_high_20",
    "rolling_low_20",
    "rolling_high_40",
    "rolling_low_40",
    "breakout_level_20",
    "support_level_20",
    "avg_volume_5",
    "avg_volume_20",
    "volume_ratio",
    "volume_dryup_ratio_5_20",
    "body_pct",
    "range_pct",
    "close_in_range",
    "close_to_high_pct",
    "bullish",
    "bearish",
    "pct_change",
    "atr_14",
    "range_atr_ratio",
    "trend_up",
    "trend_down",
    "drawdown_from_high_60",
    "recovery_from_low_40",
    "retracement_50_20",
    "tight_range_pct_5",
    "tight_range_pct_10",
    "tight_range_pct_20",
    "is_limit_up_like",
    "is_limit_down_like",
    "swing_low_flag",
    "swing_high_flag",
]


def build_signal_catalog() -> list[SignalDefinition]:
    return [
        SignalDefinition("selling_climax", "抛售高潮", "底部反转", "极端放量急跌后的止跌试错点。", 3, True),
        SignalDefinition("2b_structure", "2B结构", "底部反转", "假跌破前低后迅速收回关键位。", 1, True),
        SignalDefinition("false_breakdown", "假诱空", "底部反转", "不标准 2B，但属于下破后快速拉回。", 1, True),
        SignalDefinition("right_shoulder", "顺势头肩", "底部反转", "头肩底右肩简化先手点。", 1, True),
        SignalDefinition("double_breakout", "双突破", "反转确认", "趋势线突破叠加第二重确认。", 1, True),
        SignalDefinition("strength_emergence", "强势出现", "反转确认", "形态中轴上放量。", 2, True),
        SignalDefinition("jumping_creek", "跳跃小溪", "趋势启动", "威科夫式放量越过阻力。", 2, True),
        SignalDefinition("cup_with_handle_watch", "疑似杯柄杯", "跟踪预警", "右杯沿接近前高，进入杯柄或强突观察阶段。", 3, False),
        SignalDefinition("cup_with_handle", "杯子与杯柄", "趋势启动", "经典杯柄杯整理后放量突破。", 1, True),
        SignalDefinition("cup_with_handle_strict", "严格杯柄杯", "趋势启动", "按欧奈尔数值规则严格筛选的杯柄杯突破。", 1, True),
        SignalDefinition("cup_with_handle_leader", "领涨大票杯柄杯", "趋势启动", "偏强趋势、大票过滤、慢抬低点的高质量杯柄杯。", 1, True),
        SignalDefinition("pullback_confirmation", "回抽确认", "趋势延续", "突破后回踩不破再确认。", 2, True),
        SignalDefinition("n_breakout", "N字突破", "趋势延续", "突破-回抽-再突破。", 3, True),
        SignalDefinition("support_resistance_flip", "支撑压力互换", "趋势延续", "突破后原压力转支撑。", 3, True),
        SignalDefinition("spring", "箱体弹簧", "特殊结构", "箱体下沿假跌破后拉回。", 3, True),
        SignalDefinition("pattern_breakout", "形态突破", "特殊结构", "箱体、三角形等结构向上突破。", 3, True),
        SignalDefinition("first_rebound_after_crash", "急跌首次反弹", "特殊结构", "趋势急跌后的第一次强反弹。", 3, True),
    ]


@dataclass(slots=True)
class RuleThresholds:
    min_volume_ratio: float = 1.2
    min_body_pct: float = 0.02
    close_reclaim_pct: float = 0.003
    head_gap_pct: float = 0.03
    shoulder_tolerance_pct: float = 0.04
    pullback_tolerance_pct: float = 0.02
    breakout_buffer_pct: float = 0.01
    signal_recency_days: int = 3
    two_b_support_lookback: int = 20
    two_b_break_pct: float = 0.003
    two_b_min_drawdown: float = 0.06
    two_b_min_body_pct: float = 0.01
    two_b_min_volume_ratio: float = 1.0
    two_b_reclaim_pct: float = 0.0
    double_breakout_lookback: int = 20
    double_breakout_high_lookback: int = 10
    double_breakout_close_to_high_pct: float = 0.99
    double_breakout_min_volume_ratio: float = 1.05
    double_breakout_min_body_pct: float = 0.015
    double_breakout_min_close_in_range: float = 0.7
    double_breakout_prior_below_bars: int = 3
    double_breakout_prior_below_tolerance_pct: float = 0.003
    double_breakout_min_breakout_pct: float = 0.01
    double_breakout_prep_tight_range_pct: float = 0.12
    double_breakout_min_ma20_over_ma60_pct: float = 1.0
    double_breakout_max_intraday_pullback_pct: float = 0.003
    double_breakout_prior_volume_dryup_max_ratio: float = 0.95
    climax_volume_ratio: float = 1.8
    climax_drawdown_pct: float = 0.12
    climax_reversal_min_body_pct: float = 0.015
    climax_reversal_min_close_in_range: float = 0.6
    climax_reversal_min_volume_ratio: float = 1.0
    climax_reversal_reclaim_pct: float = 0.01
    climax_max_reversal_bars: int = 3
    swing_lookback: int = 40
    false_break_volume_ratio: float = 0.9
    false_break_reclaim_min_volume_ratio: float = 1.05
    false_break_min_close_in_range: float = 0.6
    false_break_min_reclaim_pct: float = 0.001
    false_break_min_break_pct: float = 0.003
    range_lookback: int = 30
    box_tight_pct: float = 0.18
    strength_midline_buffer_pct: float = 0.01
    strength_upper_zone_frac: float = 0.72
    strength_near_high_pct: float = 0.025
    strength_recent_above_midline_bars: int = 4
    creek_lookback: int = 40
    creek_breakout_pct: float = 0.015
    creek_min_volume_ratio: float = 1.8
    creek_min_body_pct: float = 0.025
    creek_min_range_atr_ratio: float = 0.9
    creek_min_close_in_range: float = 0.75
    creek_prep_tight_range_pct: float = 0.12
    creek_prior_below_bars: int = 4
    creek_prior_below_tolerance_pct: float = 0.005
    creek_min_breakout_pct: float = 0.02
    creek_min_ma20_over_ma60_pct: float = 1.0
    creek_max_intraday_pullback_pct: float = 0.005
    creek_prior_volume_dryup_max_ratio: float = 0.95
    creek_min_breakout_close_to_high_pct: float = 0.99
    cup_base_lookback: int = 50
    cup_base_min_bars: int = 25
    cup_short_handle_min_cup_bars: int = 24
    cup_prior_trend_lookback: int = 20
    cup_prior_uptrend_min_pct: float = 0.12
    cup_min_depth_pct: float = 0.1
    cup_max_depth_pct: float = 0.35
    cup_min_side_bars: int = 8
    cup_min_recovery_pct: float = 0.94
    cup_min_symmetry_ratio: float = 0.45
    cup_round_max_curve_error_pct: float = 0.18
    cup_round_max_low_position_offset: float = 0.16
    cup_handle_short_min_bars: int = 2
    cup_handle_min_bars: int = 3
    cup_handle_max_bars: int = 8
    cup_handle_short_max_depth_pct: float = 0.13
    cup_handle_max_depth_pct: float = 0.12
    cup_handle_short_max_range_pct: float = 0.13
    cup_handle_max_range_pct: float = 0.12
    cup_handle_min_position_pct: float = 0.55
    cup_handle_short_max_volume_dryup_ratio: float = 1.05
    cup_handle_max_volume_dryup_ratio: float = 0.95
    cup_watch_near_resistance_pct: float = 0.01
    cup_watch_min_close_in_range: float = 0.7
    cup_watch_min_volume_ratio: float = 1.1
    cup_short_handle_min_breakout_pct: float = 0.05
    cup_breakout_pct: float = 0.01
    cup_breakout_min_body_pct: float = 0.018
    cup_breakout_min_close_in_range: float = 0.8
    cup_breakout_min_volume_ratio: float = 1.25
    cup_strict_base_lookback: int = 75
    cup_strict_prior_rise_lookback: int = 60
    cup_strict_prior_rise_min_pct: float = 0.3
    cup_strict_min_depth_pct: float = 0.15
    cup_strict_max_depth_pct: float = 0.35
    cup_strict_min_cup_bars: int = 30
    cup_strict_min_recovery_pct: float = 0.9
    cup_strict_handle_min_bars: int = 5
    cup_strict_handle_max_bars: int = 15
    cup_strict_handle_max_depth_vs_cup: float = 0.3333333333
    cup_strict_handle_max_bars_ratio: float = 0.3333333333
    cup_strict_handle_min_position_pct: float = 0.5
    cup_strict_handle_max_volume_dryup_ratio: float = 0.95
    cup_strict_breakout_min_volume_ratio_50: float = 1.4
    cup_leader_base_lookback: int = 95
    cup_leader_prior_rise_lookback: int = 120
    cup_leader_prior_rise_min_pct: float = 0.5
    cup_leader_ma60_rising_lookback: int = 10
    cup_leader_min_market_cap: float = 20_000_000_000.0
    cup_leader_min_down_bars: int = 12
    cup_leader_min_up_bars: int = 12
    cup_leader_min_cup_bars: int = 30
    cup_leader_handle_min_bars: int = 5
    cup_leader_handle_max_bars: int = 15
    cup_leader_handle_min_depth_pct: float = 0.05
    cup_leader_handle_max_depth_pct: float = 0.10
    cup_leader_handle_max_volume_dryup_ratio: float = 0.95
    cup_leader_breakout_min_volume_ratio_50: float = 1.4
    n_breakout_pullback_window: int = 15
    n_breakout_min_volume_ratio: float = 1.05
    n_breakout_min_close_in_range: float = 0.65
    n_breakout_rebreak_buffer_pct: float = 0.003
    n_breakout_min_pullback_bars: int = 2
    n_breakout_max_pullback_bars: int = 10
    support_flip_lookback: int = 30
    support_flip_min_breakout_volume_ratio: float = 1.3
    support_flip_min_breakout_body_pct: float = 0.018
    support_flip_min_breakout_close_in_range: float = 0.8
    support_flip_prior_below_bars: int = 3
    support_flip_prior_below_tolerance_pct: float = 0.003
    support_flip_pullback_min_bars: int = 3
    support_flip_pullback_max_bars: int = 10
    support_flip_pullback_tolerance_pct: float = 0.02
    support_flip_max_close_below_level_pct: float = 0.01
    support_flip_max_pullback_volume_ratio: float = 0.95
    support_flip_reclaim_min_volume_ratio: float = 1.1
    support_flip_reclaim_min_close_in_range: float = 0.75
    support_flip_reclaim_min_body_pct: float = 0.01
    support_flip_reclaim_min_rebound_from_low_pct: float = 0.015
    support_flip_reclaim_above_prev_high_buffer_pct: float = 0.0
    support_flip_reclaim_buffer_pct: float = 0.002
    spring_break_pct: float = 0.01
    pattern_lookback: int = 30
    pattern_min_width_pct: float = 0.08
    pattern_max_width_pct: float = 0.22
    pattern_breakout_pct: float = 0.008
    pattern_min_volume_ratio: float = 1.05
    pattern_close_to_high_pct: float = 0.985
    pattern_prior_below_bars: int = 3
    pattern_prior_below_tolerance_pct: float = 0.003
    first_crash_drop_pct: float = 0.08
    rebound_min_body_pct: float = 0.02
    two_b_reclaim_bars: int = 2
    two_b_prior_low_min_age: int = 5
    two_b_close_above_break_bar: bool = True
    two_b_reclaim_close_in_upper_half: bool = True
    false_break_break_pct: float = 0.001
    false_break_reclaim_bars: int = 3
    false_break_reclaim_tolerance_pct: float = 0.003
    false_break_min_body_pct: float = 0.003
    false_break_prior_low_min_age: int = 3
    false_break_close_in_upper_half: bool = True
    false_break_close_above_break_bar: bool = True
    right_shoulder_similarity_pct: float = 0.08
    right_shoulder_bounce_pct: float = 0.015
    right_shoulder_signal_bars_from_low: int = 5
    right_shoulder_neckline_buffer_pct: float = 0.01


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    featured = build_price_features(frame)
    required = [column for column in PREPARE_REQUIRED_COLUMNS if column in featured.columns]
    return featured.dropna(subset=required).reset_index(drop=True)


def _prior_swing_levels(
    df: pd.DataFrame, recent_date: pd.Timestamp, lookback: int, *, min_age: int = 0
) -> tuple[float | None, float | None]:
    history = df[df["date"] < recent_date].tail(lookback + min_age)
    if min_age > 0 and len(history) > min_age:
        history = history.iloc[:-min_age]
    swing_lows = history[history["swing_low_flag"]]
    swing_highs = history[history["swing_high_flag"]]
    prior_low = float(swing_lows.iloc[-1]["low"]) if not swing_lows.empty else None
    prior_high = float(swing_highs.iloc[-1]["high"]) if not swing_highs.empty else None
    return prior_low, prior_high


def _range_stats(frame: pd.DataFrame) -> tuple[float, float, float]:
    high = float(frame["high"].max())
    low = float(frame["low"].min())
    width_pct = (high - low) / max(low, 1e-6)
    return low, high, width_pct


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / max(denominator, 1e-6)


def _cup_roundness_metrics(
    cup_section: pd.DataFrame, left_peak_idx: int, cup_low_idx: int, thresholds: RuleThresholds
) -> dict[str, float | bool]:
    close_values = cup_section["close"].to_numpy(dtype=float)
    x_axis = np.linspace(-1.0, 1.0, len(close_values))
    curve = np.polyfit(x_axis, close_values, 2)
    fitted = np.polyval(curve, x_axis)
    curve_rmse = float(np.sqrt(np.mean((close_values - fitted) ** 2)))
    cup_depth = max(float(cup_section["high"].max()) - float(cup_section["low"].min()), 1e-6)
    curve_error_pct = curve_rmse / cup_depth
    low_position_ratio = _safe_ratio(cup_low_idx - left_peak_idx, max(len(cup_section) - 1, 1))
    low_position_offset = abs(low_position_ratio - 0.5)
    roundness_ok = bool(
        curve[0] > 0
        and curve_error_pct <= thresholds.cup_round_max_curve_error_pct
        and low_position_offset <= thresholds.cup_round_max_low_position_offset
    )
    return {
        "cup_curve_fit_error_pct": float(curve_error_pct),
        "cup_low_position_ratio": float(low_position_ratio),
        "cup_low_position_offset": float(low_position_offset),
        "cup_roundness_ok": roundness_ok,
    }


def _progressive_segment_lows(section: pd.DataFrame, segments: int = 3) -> tuple[list[float], bool]:
    if len(section) < segments:
        return [], False
    index_groups = [group for group in np.array_split(np.arange(len(section)), segments) if len(group) > 0]
    buckets = [section.iloc[group] for group in index_groups]
    lows = [float(bucket["low"].min()) for bucket in buckets]
    progressive = all(left < right for left, right in zip(lows, lows[1:]))
    return lows, progressive


def _market_cap_info(row: pd.Series, thresholds: RuleThresholds) -> tuple[float | None, bool, bool]:
    for column in ("market_cap", "total_mv", "float_market_cap", "float_mv"):
        value = row.get(column)
        if value is None or pd.isna(value):
            continue
        market_cap = float(value)
        return market_cap, market_cap >= thresholds.cup_leader_min_market_cap, False
    return None, True, True


def _select_recent_candidate(df: pd.DataFrame, predicate: pd.Series, recency_days: int) -> pd.Series | None:
    recent = df.tail(recency_days).copy()
    candidates = recent[predicate.tail(recency_days).values]
    if candidates.empty:
        return None
    return candidates.iloc[-1]


def _latest_breakout_day(df: pd.DataFrame, lookback: int = 15) -> pd.Series | None:
    window = df.iloc[-lookback:].copy()
    breakout_days = window[window["close"] > window["breakout_level_20"]]
    if breakout_days.empty:
        return None
    return breakout_days.iloc[-1]


def _empty_signal(
    signal_type: str,
    symbol: str,
    row: pd.Series,
    *,
    trend_ok: bool,
    location_ok: bool,
    pattern_ok: bool,
    volume_ok: bool,
    factors: dict[str, object],
    invalid_reason: str | None,
    stop_price: float | None = None,
    target_price: float | None = None,
) -> ResearchSignal:
    satisfied = sum([trend_ok, location_ok, pattern_ok, volume_ok])
    confidence = round(min(100.0, satisfied * 20 + (15 if invalid_reason is None else 0)), 2)
    return ResearchSignal(
        signal_type=signal_type,
        symbol=symbol,
        signal_date=row["date"].date(),
        confidence_score=confidence,
        trend_ok=trend_ok,
        location_ok=location_ok,
        pattern_ok=pattern_ok,
        volume_ok=volume_ok,
        entry_price=float(row["close"]),
        stop_price=stop_price,
        target_price=target_price,
        invalid_reason=invalid_reason,
        factors=factors,
        risk_tags=[] if invalid_reason is None else ["needs_review"],
    )


def detect_selling_climax(
    frame: pd.DataFrame, symbol: str, thresholds: RuleThresholds
) -> ResearchSignal | None:
    df = _prepare_frame(frame)
    if len(df) < 30:
        return None
    recent = df.iloc[-1]
    crash_bar = _select_recent_candidate(
        df,
        (df["drawdown_from_high_60"] >= thresholds.climax_drawdown_pct)
        & (df["volume_ratio"] >= thresholds.climax_volume_ratio)
        & (df["is_limit_down_like"] | (df["range_pct"] >= 0.06)),
        thresholds.signal_recency_days,
    )
    if crash_bar is None:
        return None
    recent_slice = df[df["date"] >= crash_bar["date"]].head(thresholds.climax_max_reversal_bars + 1)
    reversal_bar = recent_slice[
        (recent_slice["close"] >= recent_slice["open"])
        & (recent_slice["close"] > recent_slice["low"] * 1.03)
        & (recent_slice["body_pct"] >= thresholds.climax_reversal_min_body_pct)
        & (recent_slice["close_in_range"] >= thresholds.climax_reversal_min_close_in_range)
        & (recent_slice["volume_ratio"] >= thresholds.climax_reversal_min_volume_ratio)
    ]
    signal_bar = reversal_bar.iloc[-1] if not reversal_bar.empty else recent
    trend_ok = bool(crash_bar["drawdown_from_high_60"] >= thresholds.climax_drawdown_pct)
    location_ok = bool(crash_bar["volume_ratio"] >= thresholds.climax_volume_ratio)
    pattern_ok = bool(
        signal_bar["close"] >= crash_bar["low"] * 1.03
        and signal_bar["close"] >= crash_bar["close"] * (1 + thresholds.climax_reversal_reclaim_pct)
        and (signal_bar["date"] - crash_bar["date"]).days <= thresholds.climax_max_reversal_bars + 1
    )
    volume_ok = bool(
        crash_bar["volume_ratio"] >= thresholds.climax_volume_ratio
        and signal_bar["volume_ratio"] >= thresholds.climax_reversal_min_volume_ratio
    )
    invalid_reason = None if all([trend_ok, location_ok, pattern_ok, volume_ok]) else "极端杀跌后尚未看到明确止跌"
    return _empty_signal(
        "selling_climax",
        symbol,
        signal_bar,
        trend_ok=trend_ok,
        location_ok=location_ok,
        pattern_ok=pattern_ok,
        volume_ok=volume_ok,
        stop_price=float(crash_bar["low"]),
        target_price=float(df["ma_20"].iloc[-1]),
        factors={
            "climax_date": crash_bar["date"].date().isoformat(),
            "volume_ratio": float(crash_bar["volume_ratio"]),
            "drawdown_from_high_60": float(crash_bar["drawdown_from_high_60"]),
            "reversal_volume_ratio": float(signal_bar["volume_ratio"]),
            "reversal_close_in_range": float(signal_bar["close_in_range"]),
        },
        invalid_reason=invalid_reason,
    )


def detect_2b_structure(
    frame: pd.DataFrame, symbol: str, thresholds: RuleThresholds
) -> ResearchSignal | None:
    df = _prepare_frame(frame)
    if len(df) < max(30, thresholds.swing_lookback + 5):
        return None
    recent_window = df.tail(thresholds.signal_recency_days).copy()
    prior_swing_low, _ = _prior_swing_levels(
        df,
        recent_window.iloc[0]["date"],
        thresholds.swing_lookback,
        min_age=thresholds.two_b_prior_low_min_age,
    )
    if prior_swing_low is None:
        support_series = df["low"].rolling(thresholds.two_b_support_lookback).min().shift(1)
        recent_window["support_level"] = support_series.tail(thresholds.signal_recency_days).values
    else:
        recent_window["support_level"] = prior_swing_low
    breakdown_candidates = recent_window[recent_window["low"] <= recent_window["support_level"] * (1 - thresholds.two_b_break_pct)]
    if breakdown_candidates.empty:
        return None

    breakdown_bar = breakdown_candidates.iloc[0]
    prior_history = df[df["date"] < breakdown_bar["date"]]
    after_break = recent_window[recent_window["date"] >= breakdown_bar["date"]].head(thresholds.two_b_reclaim_bars + 1)
    close_position = (after_break["close"] - after_break["low"]) / (after_break["high"] - after_break["low"]).clip(lower=1e-6)
    reclaim_candidates = after_break[
        (after_break["close"] >= after_break["support_level"] * (1 + thresholds.two_b_reclaim_pct))
        & (
            after_break["bullish"]
            | (
                thresholds.two_b_reclaim_close_in_upper_half
                & (close_position >= 0.5)
            )
        )
        & (
            (after_break["body_pct"] >= thresholds.two_b_min_body_pct)
            | (after_break["date"] == breakdown_bar["date"])
        )
    ]
    if reclaim_candidates.empty:
        recent = recent_window.iloc[-1]
        return _empty_signal(
            "2b_structure",
            symbol,
            recent,
            trend_ok=bool(prior_history["trend_down"].tail(8).any() or recent["drawdown_from_high_60"] > thresholds.two_b_min_drawdown),
            location_ok=True,
            pattern_ok=False,
            volume_ok=bool(recent["volume_ratio"] >= thresholds.two_b_min_volume_ratio),
            stop_price=float(after_break["low"].min()),
            target_price=float(df["high"].rolling(thresholds.two_b_support_lookback).max().shift(1).iloc[-1]),
            factors={
                "prior_support": float(breakdown_bar["support_level"]),
                "fake_break_low": float(after_break["low"].min()),
                "breakdown_date": breakdown_bar["date"].date().isoformat(),
            },
            invalid_reason="出现了假跌破，但还没有看到足够强的收回动作",
        )

    reclaim_bar = reclaim_candidates.iloc[-1]
    prior_support = float(breakdown_bar["support_level"])
    fake_break_low = float(after_break["low"].min())
    reclaim_in_time = bool((reclaim_bar["date"] - breakdown_bar["date"]).days <= thresholds.two_b_reclaim_bars + 1)
    reclaim_strength_ok = bool(
        reclaim_bar["close"] > breakdown_bar["close"] if thresholds.two_b_close_above_break_bar else True
    )
    trend_ok = bool(
        prior_history["trend_down"].tail(8).any() or reclaim_bar["drawdown_from_high_60"] > thresholds.two_b_min_drawdown
    )
    location_ok = fake_break_low < prior_support * (1 - thresholds.two_b_break_pct)
    pattern_ok = bool(
        reclaim_bar["close"] >= prior_support * (1 + thresholds.two_b_reclaim_pct)
        and reclaim_in_time
        and reclaim_strength_ok
        and (
            reclaim_bar["bullish"]
            or (
                thresholds.two_b_reclaim_close_in_upper_half
                and (reclaim_bar["close"] - reclaim_bar["low"]) / max(reclaim_bar["high"] - reclaim_bar["low"], 1e-6) >= 0.5
            )
        )
    )
    volume_ok = bool(
        max(float(reclaim_bar["volume_ratio"]), float(breakdown_bar["volume_ratio"])) >= thresholds.two_b_min_volume_ratio
    )
    invalid_reason = None
    if not all([trend_ok, location_ok, pattern_ok, volume_ok]):
        invalid_reason = "未同时满足下跌背景、假跌破、收回关键位和放量四个条件"
    return _empty_signal(
        "2b_structure",
        symbol,
        reclaim_bar,
        trend_ok=trend_ok,
        location_ok=location_ok,
        pattern_ok=pattern_ok,
        volume_ok=volume_ok,
        stop_price=float(fake_break_low),
        target_price=float(df["high"].rolling(thresholds.two_b_support_lookback).max().shift(1).iloc[-1]),
        factors={
            "prior_support": prior_support,
            "fake_break_low": fake_break_low,
            "breakdown_date": breakdown_bar["date"].date().isoformat(),
            "support_source": "swing_low" if prior_swing_low is not None else "rolling_low",
            "reclaim_date": reclaim_bar["date"].date().isoformat(),
            "reclaim_in_time": reclaim_in_time,
            "volume_ratio": float(reclaim_bar["volume_ratio"]),
        },
        invalid_reason=invalid_reason,
    )


def detect_false_breakdown(
    frame: pd.DataFrame, symbol: str, thresholds: RuleThresholds
) -> ResearchSignal | None:
    df = _prepare_frame(frame)
    if len(df) < max(25, thresholds.swing_lookback // 2):
        return None
    recent_window = df.tail(thresholds.signal_recency_days).copy()
    recent = recent_window.iloc[-1]
    prior_swing_low, _ = _prior_swing_levels(
        df,
        recent_window.iloc[0]["date"],
        max(20, thresholds.swing_lookback // 2),
        min_age=thresholds.false_break_prior_low_min_age,
    )
    history = df.iloc[:-1]
    if prior_swing_low is None:
        support_series = df["low"].rolling(thresholds.two_b_support_lookback).min().shift(1)
        recent_window["support_level"] = support_series.tail(thresholds.signal_recency_days).values
    else:
        recent_window["support_level"] = prior_swing_low

    breakdown_candidates = recent_window[
        recent_window["low"] <= recent_window["support_level"] * (1 - thresholds.false_break_break_pct)
    ]
    if breakdown_candidates.empty:
        return None

    breakdown_bar = breakdown_candidates.iloc[0]
    after_break = recent_window[recent_window["date"] >= breakdown_bar["date"]].head(thresholds.false_break_reclaim_bars + 1)
    break_pct = (float(breakdown_bar["support_level"]) - float(after_break["low"].min())) / max(
        float(breakdown_bar["support_level"]), 1e-6
    )
    close_position = (after_break["close"] - after_break["low"]) / (after_break["high"] - after_break["low"]).clip(lower=1e-6)
    reclaim_candidates = after_break[
        (after_break["close"] >= after_break["support_level"] * (1 + thresholds.false_break_min_reclaim_pct))
        & (
            after_break["bullish"]
            | (
                (after_break["body_pct"] >= thresholds.false_break_min_body_pct)
                & (after_break["close_in_range"] >= thresholds.false_break_min_close_in_range)
            )
            | (thresholds.false_break_close_in_upper_half & (close_position >= 0.65))
        )
    ]
    if reclaim_candidates.empty:
        return _empty_signal(
            "false_breakdown",
            symbol,
            recent,
            trend_ok=bool(history["drawdown_from_high_60"].iloc[-1] > thresholds.two_b_min_drawdown * 0.8),
            location_ok=True,
            pattern_ok=False,
            volume_ok=bool(recent["volume_ratio"] >= thresholds.false_break_volume_ratio),
            stop_price=float(after_break["low"].min()),
            target_price=float(history["rolling_high_20"].iloc[-1]),
            factors={
                "support": float(breakdown_bar["support_level"]),
                "breakdown_date": breakdown_bar["date"].date().isoformat(),
                "support_source": "swing_low" if prior_swing_low is not None else "rolling_low",
            },
            invalid_reason="出现了假诱空雏形，但回拉确认还不够",
        )

    reclaim_bar = reclaim_candidates.iloc[-1]
    support = float(breakdown_bar["support_level"])
    broke_support = bool((after_break["low"] < support).any())
    reclaim_in_time = bool((reclaim_bar["date"] - breakdown_bar["date"]).days <= thresholds.false_break_reclaim_bars + 1)
    reclaim_strength_ok = bool(
        reclaim_bar["close"] > breakdown_bar["close"] if thresholds.false_break_close_above_break_bar else True
    )

    trend_ok = bool(
        history["drawdown_from_high_60"].iloc[-1] > thresholds.two_b_min_drawdown * 0.8
        and (history["trend_down"].tail(8).any() or history["close"].iloc[-1] < history["ma_20"].iloc[-1])
    )
    location_ok = broke_support and break_pct >= thresholds.false_break_min_break_pct
    pattern_ok = bool(
        reclaim_bar["close"] >= support * (1 + thresholds.false_break_min_reclaim_pct)
        and reclaim_in_time
        and reclaim_strength_ok
        and (
            reclaim_bar["bullish"]
            or (
                reclaim_bar["body_pct"] >= thresholds.false_break_min_body_pct
                and reclaim_bar["close_in_range"] >= thresholds.false_break_min_close_in_range
            )
            or (
                thresholds.false_break_close_in_upper_half
                and (reclaim_bar["close"] - reclaim_bar["low"]) / max(reclaim_bar["high"] - reclaim_bar["low"], 1e-6) >= 0.65
            )
        )
    )
    volume_ok = bool(
        max(float(reclaim_bar["volume_ratio"]), float(breakdown_bar["volume_ratio"]))
        >= thresholds.false_break_reclaim_min_volume_ratio
    )
    invalid_reason = None
    if not all([trend_ok, location_ok, pattern_ok, volume_ok]):
        invalid_reason = "假诱空信号强度不足，可能仍是下跌中继"
    return _empty_signal(
        "false_breakdown",
        symbol,
        reclaim_bar,
        trend_ok=trend_ok,
        location_ok=location_ok,
        pattern_ok=pattern_ok,
        volume_ok=volume_ok,
        stop_price=float(after_break["low"].min()),
        target_price=float(history["rolling_high_20"].iloc[-1]),
        factors={
            "support": support,
            "breakdown_date": breakdown_bar["date"].date().isoformat(),
            "reclaim_date": reclaim_bar["date"].date().isoformat(),
            "reclaim_in_time": reclaim_in_time,
            "support_source": "swing_low" if prior_swing_low is not None else "rolling_low",
            "volume_ratio": float(reclaim_bar["volume_ratio"]),
            "close_in_range": float(reclaim_bar["close_in_range"]),
            "break_pct": float(break_pct),
        },
        invalid_reason=invalid_reason,
    )


def detect_right_shoulder(
    frame: pd.DataFrame, symbol: str, thresholds: RuleThresholds
) -> ResearchSignal | None:
    df = _prepare_frame(frame)
    if len(df) < 30:
        return None
    recent = df.iloc[-1]
    window = df.tail(30).reset_index(drop=True)
    head_idx = int(window["low"].idxmin())
    if head_idx < 5 or head_idx > len(window) - 6:
        return None

    left = window.iloc[:head_idx]
    right = window.iloc[head_idx + 1 :]
    if len(right) < 5:
        return None
    left_shoulder = float(left["low"].min())
    head_low = float(window.loc[head_idx, "low"])
    right_tail = right.tail(8).copy()
    right_low = float(right_tail["low"].min())
    right_low_idx = int(right_tail["low"].idxmin())
    neckline = float(max(left["high"].max(), right["high"].max()))
    retracement_anchor = float(window["retracement_50_20"].iloc[-1])
    right_low_date = window.loc[right_low_idx, "date"]
    signal_bar = recent
    bars_since_right_low = len(window.loc[right_low_idx + 1 :])

    trend_ok = bool(recent["close"] > recent["ma_10"] and recent["drawdown_from_high_60"] > 0.02)
    location_ok = (
        right_low > head_low * (1 + thresholds.head_gap_pct)
        and abs(right_low - left_shoulder) / max(left_shoulder, 1e-6) <= thresholds.right_shoulder_similarity_pct
        and abs(right_low - retracement_anchor) / max(retracement_anchor, 1e-6)
        <= thresholds.shoulder_tolerance_pct
    )
    pattern_ok = bool(
        recent["close"] >= right_low * (1 + thresholds.right_shoulder_bounce_pct)
        and recent["close"] <= neckline * (1 + thresholds.right_shoulder_neckline_buffer_pct)
        and left_shoulder > head_low
        and bars_since_right_low <= thresholds.right_shoulder_signal_bars_from_low
    )
    volume_ok = bool(recent["volume_ratio"] >= 0.85)
    invalid_reason = None
    if not all([trend_ok, location_ok, pattern_ok, volume_ok]):
        invalid_reason = "右肩结构不完整或确认力度不足"
    return _empty_signal(
        "right_shoulder",
        symbol,
        recent,
        trend_ok=bool(trend_ok),
        location_ok=bool(location_ok),
        pattern_ok=bool(pattern_ok),
        volume_ok=bool(volume_ok),
        stop_price=right_low,
        target_price=neckline,
        factors={
            "head_low": head_low,
            "right_low": right_low,
            "left_shoulder": left_shoulder,
            "neckline": neckline,
            "right_low_date": right_low_date.date().isoformat(),
            "bars_since_right_low": bars_since_right_low,
        },
        invalid_reason=invalid_reason,
    )


def detect_double_breakout(
    frame: pd.DataFrame, symbol: str, thresholds: RuleThresholds
) -> ResearchSignal | None:
    df = _prepare_frame(frame)
    if len(df) < max(30, thresholds.double_breakout_lookback + 5):
        return None
    breakout_series = df["high"].rolling(thresholds.double_breakout_lookback).max().shift(1)
    recent_window = df.tail(thresholds.signal_recency_days).copy()
    recent_window["breakout_level"] = breakout_series.tail(thresholds.signal_recency_days).values
    candidates = recent_window[
        recent_window["close"] > recent_window["breakout_level"] * (1 + thresholds.breakout_buffer_pct)
    ]
    if candidates.empty:
        return None

    signal_bar = candidates.iloc[-1]
    prior = df[df["date"] < signal_bar["date"]].iloc[-1]
    prior_window = df[df["date"] < signal_bar["date"]].tail(thresholds.double_breakout_prior_below_bars)
    recent_high = float(df[df["date"] <= signal_bar["date"]]["high"].tail(thresholds.double_breakout_high_lookback).max())
    breakout_level = float(signal_bar["breakout_level"])
    breakout_pct = (float(signal_bar["close"]) - breakout_level) / max(breakout_level, 1e-6)
    prior_below_breakout = bool(
        len(prior_window) == thresholds.double_breakout_prior_below_bars
        and (prior_window["close"] <= breakout_level * (1 + thresholds.double_breakout_prior_below_tolerance_pct)).all()
    )
    prep_tight = bool(
        not prior_window["tight_range_pct_5"].dropna().empty
        and float(prior_window["tight_range_pct_5"].dropna().iloc[-1]) <= thresholds.double_breakout_prep_tight_range_pct
    )
    prior_volume_dryup_ratio = _safe_ratio(
        float(prior_window["volume"].mean()),
        float(df[df["date"] < signal_bar["date"]].tail(20)["volume"].mean()),
    )
    breakout_hold_ok = bool(
        float(signal_bar["low"]) >= breakout_level * (1 - thresholds.double_breakout_max_intraday_pullback_pct)
    )

    trend_ok = bool(
        signal_bar["trend_up"]
        and signal_bar["ma_20"] >= signal_bar["ma_60"] * thresholds.double_breakout_min_ma20_over_ma60_pct
        and signal_bar["ma_20"] >= df.iloc[-5]["ma_20"]
    )
    location_ok = bool(
        signal_bar["close"] > breakout_level * (1 + thresholds.breakout_buffer_pct)
        and breakout_hold_ok
    )
    pattern_ok = bool(
        signal_bar["close"] > prior["high"]
        and signal_bar["close"] >= recent_high * thresholds.double_breakout_close_to_high_pct
        and signal_bar["body_pct"] >= thresholds.double_breakout_min_body_pct
        and signal_bar["close_in_range"] >= thresholds.double_breakout_min_close_in_range
        and prior_below_breakout
        and breakout_pct >= thresholds.double_breakout_min_breakout_pct
        and prep_tight
        and prior_volume_dryup_ratio <= thresholds.double_breakout_prior_volume_dryup_max_ratio
    )
    volume_ok = bool(signal_bar["volume_ratio"] >= thresholds.double_breakout_min_volume_ratio)
    invalid_reason = None
    if not all([trend_ok, location_ok, pattern_ok, volume_ok]):
        invalid_reason = "双突破未形成趋势、位置、结构和量价共振"
    stop_price = float(df[df["date"] <= signal_bar["date"]]["low"].tail(5).min())
    return _empty_signal(
        "double_breakout",
        symbol,
        signal_bar,
        trend_ok=trend_ok,
        location_ok=location_ok,
        pattern_ok=pattern_ok,
        volume_ok=volume_ok,
        stop_price=stop_price,
        target_price=float(signal_bar["close"] + 2 * signal_bar["atr_14"]),
        factors={
            "breakout_level": breakout_level,
            "recent_high_10": recent_high,
            "volume_ratio": float(signal_bar["volume_ratio"]),
            "close_in_range": float(signal_bar["close_in_range"]),
            "prior_below_breakout": prior_below_breakout,
            "breakout_pct": float(breakout_pct),
            "prep_tight": prep_tight,
            "prior_volume_dryup_ratio": float(prior_volume_dryup_ratio),
            "breakout_hold_ok": breakout_hold_ok,
        },
        invalid_reason=invalid_reason,
    )


def detect_strength_emergence(
    frame: pd.DataFrame, symbol: str, thresholds: RuleThresholds
) -> ResearchSignal | None:
    df = _prepare_frame(frame)
    if len(df) < thresholds.range_lookback + 5:
        return None
    recent = df.iloc[-1]
    box = df.tail(thresholds.range_lookback)
    box_low, box_high, width_pct = _range_stats(box)
    midline = (box_high + box_low) / 2
    box_height = max(box_high - box_low, 1e-6)
    upper_zone_level = box_low + box_height * thresholds.strength_upper_zone_frac
    recent_box = box.tail(5)
    close_to_high_pct = (box_high - recent["close"]) / max(box_high, 1e-6)
    closes_above_midline = int((recent_box["close"] > midline).sum())

    trend_ok = bool(
        recent["ma_20"] >= recent["ma_60"] * 0.98
        and recent["ma_20"] >= df.iloc[-5]["ma_20"]
    )
    location_ok = bool(
        width_pct <= thresholds.box_tight_pct
        and recent["close"] > midline * (1 + thresholds.strength_midline_buffer_pct)
        and recent["close"] >= upper_zone_level
    )
    pattern_ok = bool(
        recent["close"] < box_high * (1 + thresholds.breakout_buffer_pct)
        and close_to_high_pct <= thresholds.strength_near_high_pct
        and closes_above_midline >= thresholds.strength_recent_above_midline_bars
        and recent["close"] >= recent["open"]
    )
    volume_ok = bool(recent["volume_ratio"] >= thresholds.min_volume_ratio)
    invalid_reason = None if all([trend_ok, location_ok, pattern_ok, volume_ok]) else "中轴上放量但仍未形成有效强势出现"
    return _empty_signal(
        "strength_emergence",
        symbol,
        recent,
        trend_ok=trend_ok,
        location_ok=location_ok,
        pattern_ok=pattern_ok,
        volume_ok=volume_ok,
        stop_price=box_low,
        target_price=box_high,
        factors={
            "box_low": box_low,
            "box_high": box_high,
            "midline": midline,
            "width_pct": width_pct,
            "upper_zone_level": float(upper_zone_level),
            "close_to_high_pct": float(close_to_high_pct),
            "closes_above_midline": closes_above_midline,
        },
        invalid_reason=invalid_reason,
    )


def detect_jumping_creek(
    frame: pd.DataFrame, symbol: str, thresholds: RuleThresholds
) -> ResearchSignal | None:
    df = _prepare_frame(frame)
    if len(df) < thresholds.creek_lookback + thresholds.creek_prior_below_bars + 2:
        return None
    recent_window = df.tail(thresholds.signal_recency_days).copy()
    resistance_series = df["high"].rolling(thresholds.creek_lookback).max().shift(1)
    recent_window["resistance"] = resistance_series.tail(thresholds.signal_recency_days).values
    candidates = recent_window[
        (recent_window["close"] > recent_window["resistance"] * (1 + thresholds.creek_min_breakout_pct))
        & (recent_window["volume_ratio"] >= thresholds.creek_min_volume_ratio)
    ]
    if candidates.empty:
        return None
    signal_bar = candidates.iloc[-1]
    prior_window = df[df["date"] < signal_bar["date"]].tail(thresholds.creek_prior_below_bars)
    breakout_pct = (float(signal_bar["close"]) - float(signal_bar["resistance"])) / max(float(signal_bar["resistance"]), 1e-6)
    range_atr_ratio = (float(signal_bar["high"]) - float(signal_bar["low"])) / max(float(signal_bar["atr_14"]), 1e-6)
    prior_below_resistance = bool(
        len(prior_window) == thresholds.creek_prior_below_bars
        and (prior_window["close"] <= signal_bar["resistance"] * (1 + thresholds.creek_prior_below_tolerance_pct)).all()
    )
    prep_tight = bool(float(prior_window["tight_range_pct_5"].dropna().iloc[-1]) <= thresholds.creek_prep_tight_range_pct) if not prior_window["tight_range_pct_5"].dropna().empty else False
    prior_volume_dryup_ratio = _safe_ratio(float(prior_window["volume"].mean()), float(df[df["date"] < signal_bar["date"]].tail(20)["volume"].mean()))
    breakout_hold_ok = bool(float(signal_bar["low"]) >= float(signal_bar["resistance"]) * (1 - thresholds.creek_max_intraday_pullback_pct))

    trend_ok = bool(
        signal_bar["trend_up"]
        and signal_bar["ma_20"] >= signal_bar["ma_60"] * thresholds.creek_min_ma20_over_ma60_pct
        and signal_bar["ma_20"] >= df.iloc[-5]["ma_20"]
    )
    location_ok = bool(
        signal_bar["close"] > signal_bar["resistance"] * (1 + thresholds.creek_min_breakout_pct)
        and breakout_hold_ok
    )
    pattern_ok = bool(
        signal_bar["close"] >= signal_bar["high"] * thresholds.creek_min_breakout_close_to_high_pct
        and signal_bar["body_pct"] >= thresholds.creek_min_body_pct
        and range_atr_ratio >= thresholds.creek_min_range_atr_ratio
        and prior_below_resistance
        and breakout_pct >= thresholds.creek_min_breakout_pct
        and signal_bar["close_in_range"] >= thresholds.creek_min_close_in_range
        and prep_tight
        and prior_volume_dryup_ratio <= thresholds.creek_prior_volume_dryup_max_ratio
    )
    volume_ok = bool(signal_bar["volume_ratio"] >= thresholds.creek_min_volume_ratio)
    invalid_reason = None if all([trend_ok, location_ok, pattern_ok, volume_ok]) else "越过阻力但蓄势、站稳或放量质量不足"
    return _empty_signal(
        "jumping_creek",
        symbol,
        signal_bar,
        trend_ok=trend_ok,
        location_ok=location_ok,
        pattern_ok=pattern_ok,
        volume_ok=volume_ok,
        stop_price=float(df[df["date"] <= signal_bar["date"]]["low"].tail(5).min()),
        target_price=float(signal_bar["close"] + 2 * signal_bar["atr_14"]),
        factors={
            "resistance": float(signal_bar["resistance"]),
            "volume_ratio": float(signal_bar["volume_ratio"]),
            "breakout_pct": float(breakout_pct),
            "range_atr_ratio": float(range_atr_ratio),
            "prior_below_resistance": prior_below_resistance,
            "close_in_range": float(signal_bar["close_in_range"]),
            "prep_tight": prep_tight,
            "prior_volume_dryup_ratio": float(prior_volume_dryup_ratio),
            "breakout_hold_ok": breakout_hold_ok,
        },
        invalid_reason=invalid_reason,
    )


def detect_cup_with_handle(
    frame: pd.DataFrame, symbol: str, thresholds: RuleThresholds
) -> ResearchSignal | None:
    df = _prepare_frame(frame)
    minimum_bars = thresholds.cup_base_lookback + thresholds.signal_recency_days
    if len(df) < minimum_bars:
        return None

    recent_window = df.tail(thresholds.signal_recency_days).copy()
    resistance_series = df["high"].rolling(thresholds.cup_base_lookback).max().shift(1)
    recent_window["resistance"] = resistance_series.tail(thresholds.signal_recency_days).values
    candidates = recent_window[
        (recent_window["close"] > recent_window["resistance"] * (1 + thresholds.cup_breakout_pct))
        & (recent_window["volume_ratio"] >= thresholds.cup_breakout_min_volume_ratio)
    ]
    if candidates.empty:
        return None

    signal_bar = candidates.iloc[-1]
    signal_index = int(signal_bar.name)
    pre_signal = df.iloc[:signal_index].copy()
    if len(pre_signal) < thresholds.cup_base_lookback:
        return None

    handle_search = pre_signal.tail(thresholds.cup_handle_max_bars + 4)
    minimum_handle_bars = min(thresholds.cup_handle_short_min_bars, thresholds.cup_handle_min_bars)
    if len(handle_search) <= minimum_handle_bars:
        return None
    right_peak_idx = int(handle_search["high"].idxmax())
    handle_window = pre_signal.iloc[right_peak_idx + 1 :]
    if not (minimum_handle_bars <= len(handle_window) <= thresholds.cup_handle_max_bars):
        return None

    minimum_cup_bars = min(thresholds.cup_base_min_bars, thresholds.cup_short_handle_min_cup_bars)
    cup_window = pre_signal.iloc[max(0, right_peak_idx - thresholds.cup_base_lookback + 1) : right_peak_idx + 1].copy()
    if len(cup_window) < minimum_cup_bars:
        return None
    left_search_end = max(0, len(cup_window) - minimum_handle_bars)
    left_search = cup_window.iloc[:left_search_end]
    if left_search.empty:
        return None
    left_peak_idx = int(left_search["high"].idxmax())
    if left_peak_idx >= right_peak_idx:
        return None

    cup_section = df.iloc[left_peak_idx : right_peak_idx + 1].copy()
    if len(cup_section) < minimum_cup_bars:
        return None
    cup_low_idx = int(cup_section["low"].idxmin())
    if not (left_peak_idx < cup_low_idx < right_peak_idx):
        return None

    left_peak = float(df.loc[left_peak_idx, "high"])
    cup_low = float(df.loc[cup_low_idx, "low"])
    right_peak = float(df.loc[right_peak_idx, "high"])
    handle_low = float(handle_window["low"].min())
    handle_high = float(handle_window["high"].max())
    handle_bars = int(len(handle_window))
    left_bars = int(cup_low_idx - left_peak_idx)
    right_bars = int(right_peak_idx - cup_low_idx)
    cup_depth_pct = _safe_ratio(left_peak - cup_low, left_peak)
    right_peak_recovery_pct = _safe_ratio(right_peak, left_peak)
    handle_depth_pct = _safe_ratio(right_peak - handle_low, right_peak)
    handle_range_pct = _safe_ratio(handle_high - handle_low, right_peak)
    handle_low_position_pct = _safe_ratio(handle_low - cup_low, left_peak - cup_low)
    symmetry_ratio = _safe_ratio(min(left_bars, right_bars), max(left_bars, right_bars))
    handle_volume_dryup_ratio = _safe_ratio(float(handle_window["volume"].mean()), float(cup_section["volume"].mean()))
    resistance = max(float(signal_bar["resistance"]), left_peak, right_peak)
    breakout_pct = _safe_ratio(float(signal_bar["close"]) - resistance, resistance)
    roundness = _cup_roundness_metrics(cup_section, left_peak_idx, cup_low_idx, thresholds)

    prior_trend_start = max(0, left_peak_idx - thresholds.cup_prior_trend_lookback)
    prior_trend_window = df.iloc[prior_trend_start : left_peak_idx + 1]
    if len(prior_trend_window) < max(5, thresholds.cup_prior_trend_lookback // 2):
        return None
    prior_uptrend_pct = _safe_ratio(left_peak - float(prior_trend_window["close"].iloc[0]), float(prior_trend_window["close"].iloc[0]))
    classic_handle = bool(
        handle_bars >= thresholds.cup_handle_min_bars
        and handle_depth_pct <= thresholds.cup_handle_max_depth_pct
        and handle_range_pct <= thresholds.cup_handle_max_range_pct
        and handle_volume_dryup_ratio <= thresholds.cup_handle_max_volume_dryup_ratio
    )
    short_handle = bool(
        thresholds.cup_handle_short_min_bars <= handle_bars < thresholds.cup_handle_min_bars
        and handle_depth_pct <= thresholds.cup_handle_short_max_depth_pct
        and handle_range_pct <= thresholds.cup_handle_short_max_range_pct
        and handle_volume_dryup_ratio <= thresholds.cup_handle_short_max_volume_dryup_ratio
        and breakout_pct >= thresholds.cup_short_handle_min_breakout_pct
    )
    handle_ok = classic_handle or short_handle

    trend_ok = bool(
        signal_bar["trend_up"]
        and signal_bar["ma_20"] >= signal_bar["ma_60"] * 0.99
        and prior_uptrend_pct >= thresholds.cup_prior_uptrend_min_pct
    )
    location_ok = bool(
        thresholds.cup_min_depth_pct <= cup_depth_pct <= thresholds.cup_max_depth_pct
        and right_peak_recovery_pct >= thresholds.cup_min_recovery_pct
        and handle_low_position_pct >= thresholds.cup_handle_min_position_pct
    )
    pattern_ok = bool(
        left_bars >= thresholds.cup_min_side_bars
        and right_bars >= thresholds.cup_min_side_bars
        and symmetry_ratio >= thresholds.cup_min_symmetry_ratio
        and bool(roundness["cup_roundness_ok"])
        and handle_ok
        and breakout_pct >= thresholds.cup_breakout_pct
        and signal_bar["body_pct"] >= thresholds.cup_breakout_min_body_pct
        and signal_bar["close_in_range"] >= thresholds.cup_breakout_min_close_in_range
    )
    volume_ok = bool(
        signal_bar["volume_ratio"] >= thresholds.cup_breakout_min_volume_ratio
        and handle_ok
    )
    invalid_reason = (
        None
        if all([trend_ok, location_ok, pattern_ok, volume_ok])
        else "杯体圆弧度不足，或柄部/突破质量不够理想"
    )
    return _empty_signal(
        "cup_with_handle",
        symbol,
        signal_bar,
        trend_ok=trend_ok,
        location_ok=location_ok,
        pattern_ok=pattern_ok,
        volume_ok=volume_ok,
        stop_price=handle_low,
        target_price=float(signal_bar["close"] + 2 * signal_bar["atr_14"]),
        factors={
            "resistance": resistance,
            "left_peak": left_peak,
            "cup_low": cup_low,
            "right_peak": right_peak,
            "cup_depth_pct": float(cup_depth_pct),
            "right_peak_recovery_pct": float(right_peak_recovery_pct),
            "handle_depth_pct": float(handle_depth_pct),
            "handle_range_pct": float(handle_range_pct),
            "handle_low_position_pct": float(handle_low_position_pct),
            "handle_bars": handle_bars,
            "handle_mode": "classic" if classic_handle else "short" if short_handle else "invalid",
            "handle_volume_dryup_ratio": float(handle_volume_dryup_ratio),
            "symmetry_ratio": float(symmetry_ratio),
            "prior_uptrend_pct": float(prior_uptrend_pct),
            "breakout_pct": float(breakout_pct),
            "close_in_range": float(signal_bar["close_in_range"]),
            "volume_ratio": float(signal_bar["volume_ratio"]),
            **roundness,
        },
        invalid_reason=invalid_reason,
    )


def detect_cup_with_handle_strict(
    frame: pd.DataFrame, symbol: str, thresholds: RuleThresholds
) -> ResearchSignal | None:
    df = _prepare_frame(frame)
    minimum_bars = max(
        thresholds.cup_strict_base_lookback + thresholds.signal_recency_days,
        thresholds.cup_strict_prior_rise_lookback + thresholds.cup_strict_min_cup_bars,
    )
    if len(df) < minimum_bars:
        return None

    recent_window = df.tail(thresholds.signal_recency_days).copy()
    resistance_series = df["high"].rolling(thresholds.cup_strict_base_lookback).max().shift(1)
    recent_window["resistance"] = resistance_series.tail(thresholds.signal_recency_days).values
    breakout_volume_ratio_50 = recent_window["volume"] / recent_window["avg_volume_50"].replace(0, pd.NA)
    candidates = recent_window[
        (recent_window["close"] > recent_window["resistance"])
        & (recent_window["body_pct"] >= thresholds.cup_breakout_min_body_pct)
        & (recent_window["close_in_range"] >= thresholds.cup_breakout_min_close_in_range)
        & (breakout_volume_ratio_50 >= thresholds.cup_strict_breakout_min_volume_ratio_50)
    ]
    if candidates.empty:
        return None

    signal_bar = candidates.iloc[-1]
    signal_index = int(signal_bar.name)
    pre_signal = df.iloc[:signal_index].copy()
    if len(pre_signal) < thresholds.cup_strict_base_lookback:
        return None

    handle_search = pre_signal.tail(thresholds.cup_strict_handle_max_bars + 4)
    if len(handle_search) <= thresholds.cup_strict_handle_min_bars:
        return None
    right_peak_idx = int(handle_search["high"].idxmax())
    handle_window = pre_signal.iloc[right_peak_idx + 1 :]
    if not (thresholds.cup_strict_handle_min_bars <= len(handle_window) <= thresholds.cup_strict_handle_max_bars):
        return None

    cup_window = pre_signal.iloc[max(0, right_peak_idx - thresholds.cup_strict_base_lookback + 1) : right_peak_idx + 1].copy()
    if len(cup_window) < thresholds.cup_strict_min_cup_bars:
        return None
    left_search_end = max(0, len(cup_window) - thresholds.cup_strict_handle_min_bars)
    left_search = cup_window.iloc[:left_search_end]
    if left_search.empty:
        return None
    left_peak_idx = int(left_search["high"].idxmax())
    if left_peak_idx >= right_peak_idx:
        return None

    cup_section = df.iloc[left_peak_idx : right_peak_idx + 1].copy()
    if len(cup_section) < thresholds.cup_strict_min_cup_bars:
        return None
    cup_low_idx = int(cup_section["low"].idxmin())
    if not (left_peak_idx < cup_low_idx < right_peak_idx):
        return None

    left_peak = float(df.loc[left_peak_idx, "high"])
    cup_low = float(df.loc[cup_low_idx, "low"])
    right_peak = float(df.loc[right_peak_idx, "high"])
    handle_low = float(handle_window["low"].min())
    handle_high = float(handle_window["high"].max())
    handle_bars = int(len(handle_window))
    cup_bars = int(len(cup_section))
    left_bars = int(cup_low_idx - left_peak_idx)
    right_bars = int(right_peak_idx - cup_low_idx)
    cup_depth_pct = _safe_ratio(left_peak - cup_low, left_peak)
    right_peak_recovery_pct = _safe_ratio(right_peak, left_peak)
    handle_depth_pct = _safe_ratio(right_peak - handle_low, right_peak)
    handle_depth_vs_cup_pct = _safe_ratio(handle_depth_pct, cup_depth_pct)
    handle_range_pct = _safe_ratio(handle_high - handle_low, right_peak)
    handle_low_position_pct = _safe_ratio(handle_low - cup_low, left_peak - cup_low)
    handle_bars_ratio = _safe_ratio(handle_bars, cup_bars)
    symmetry_ratio = _safe_ratio(min(left_bars, right_bars), max(left_bars, right_bars))
    handle_volume_dryup_ratio = _safe_ratio(float(handle_window["volume"].mean()), float(cup_section["volume"].mean()))
    resistance = max(float(signal_bar["resistance"]), left_peak, right_peak)
    breakout_pct = _safe_ratio(float(signal_bar["close"]) - resistance, resistance)
    breakout_volume_ratio_50_value = _safe_ratio(float(signal_bar["volume"]), float(signal_bar["avg_volume_50"]))
    roundness = _cup_roundness_metrics(cup_section, left_peak_idx, cup_low_idx, thresholds)

    prior_rise_start = max(0, left_peak_idx - thresholds.cup_strict_prior_rise_lookback)
    prior_rise_window = df.iloc[prior_rise_start : left_peak_idx + 1]
    if len(prior_rise_window) < thresholds.cup_strict_prior_rise_lookback:
        return None
    prior_rise_60_pct = _safe_ratio(
        left_peak - float(prior_rise_window["close"].iloc[0]),
        float(prior_rise_window["close"].iloc[0]),
    )

    trend_ok = bool(
        signal_bar["trend_up"]
        and prior_rise_60_pct >= thresholds.cup_strict_prior_rise_min_pct
    )
    location_ok = bool(
        thresholds.cup_strict_min_depth_pct <= cup_depth_pct <= thresholds.cup_strict_max_depth_pct
        and right_peak_recovery_pct >= thresholds.cup_strict_min_recovery_pct
        and handle_low_position_pct >= thresholds.cup_strict_handle_min_position_pct
    )
    pattern_ok = bool(
        cup_bars >= thresholds.cup_strict_min_cup_bars
        and left_bars >= thresholds.cup_min_side_bars
        and right_bars >= thresholds.cup_min_side_bars
        and symmetry_ratio >= thresholds.cup_min_symmetry_ratio
        and bool(roundness["cup_roundness_ok"])
        and handle_bars_ratio <= thresholds.cup_strict_handle_max_bars_ratio
        and handle_depth_vs_cup_pct <= thresholds.cup_strict_handle_max_depth_vs_cup
        and handle_volume_dryup_ratio <= thresholds.cup_strict_handle_max_volume_dryup_ratio
        and breakout_pct > 0
        and signal_bar["close_in_range"] >= thresholds.cup_breakout_min_close_in_range
    )
    volume_ok = bool(
        breakout_volume_ratio_50_value >= thresholds.cup_strict_breakout_min_volume_ratio_50
        and signal_bar["close"] > resistance
    )
    invalid_reason = (
        None
        if all([trend_ok, location_ok, pattern_ok, volume_ok])
        else "未满足严格杯柄杯的趋势、杯深、杯长、柄深或放量突破条件"
    )
    return _empty_signal(
        "cup_with_handle_strict",
        symbol,
        signal_bar,
        trend_ok=trend_ok,
        location_ok=location_ok,
        pattern_ok=pattern_ok,
        volume_ok=volume_ok,
        stop_price=handle_low,
        target_price=float(signal_bar["close"] + 2 * signal_bar["atr_14"]),
        factors={
            "resistance": resistance,
            "left_peak": left_peak,
            "cup_low": cup_low,
            "right_peak": right_peak,
            "cup_depth_pct": float(cup_depth_pct),
            "cup_length_bars": cup_bars,
            "right_peak_recovery_pct": float(right_peak_recovery_pct),
            "handle_depth_pct": float(handle_depth_pct),
            "handle_depth_vs_cup_pct": float(handle_depth_vs_cup_pct),
            "handle_range_pct": float(handle_range_pct),
            "handle_low_position_pct": float(handle_low_position_pct),
            "handle_bars": handle_bars,
            "handle_bars_ratio": float(handle_bars_ratio),
            "handle_volume_dryup_ratio": float(handle_volume_dryup_ratio),
            "symmetry_ratio": float(symmetry_ratio),
            "prior_rise_60_pct": float(prior_rise_60_pct),
            "breakout_pct": float(breakout_pct),
            "close_in_range": float(signal_bar["close_in_range"]),
            "volume_ratio": float(signal_bar["volume_ratio"]),
            "volume_ratio_50": float(breakout_volume_ratio_50_value),
            **roundness,
        },
        invalid_reason=invalid_reason,
    )


def detect_cup_with_handle_leader(
    frame: pd.DataFrame, symbol: str, thresholds: RuleThresholds
) -> ResearchSignal | None:
    df = _prepare_frame(frame)
    minimum_bars = max(
        thresholds.cup_leader_base_lookback + thresholds.signal_recency_days,
        thresholds.cup_leader_prior_rise_lookback + thresholds.cup_leader_min_cup_bars,
    )
    if len(df) < minimum_bars:
        return None

    recent_window = df.tail(thresholds.signal_recency_days).copy()
    resistance_series = df["high"].rolling(thresholds.cup_leader_base_lookback).max().shift(1)
    recent_window["resistance"] = resistance_series.tail(thresholds.signal_recency_days).values
    breakout_volume_ratio_50 = recent_window["volume"] / recent_window["avg_volume_50"].replace(0, pd.NA)
    candidates = recent_window[
        (recent_window["close"] > recent_window["resistance"])
        & (recent_window["body_pct"] >= thresholds.cup_breakout_min_body_pct)
        & (recent_window["close_in_range"] >= thresholds.cup_breakout_min_close_in_range)
        & (breakout_volume_ratio_50 >= thresholds.cup_leader_breakout_min_volume_ratio_50)
    ]
    if candidates.empty:
        return None

    signal_bar = candidates.iloc[-1]
    signal_index = int(signal_bar.name)
    pre_signal = df.iloc[:signal_index].copy()
    if len(pre_signal) < thresholds.cup_leader_base_lookback:
        return None

    handle_search = pre_signal.tail(thresholds.cup_leader_handle_max_bars + 4)
    if len(handle_search) <= thresholds.cup_leader_handle_min_bars:
        return None
    right_peak_idx = int(handle_search["high"].idxmax())
    handle_window = pre_signal.iloc[right_peak_idx + 1 :]
    if not (thresholds.cup_leader_handle_min_bars <= len(handle_window) <= thresholds.cup_leader_handle_max_bars):
        return None

    cup_window = pre_signal.iloc[max(0, right_peak_idx - thresholds.cup_leader_base_lookback + 1) : right_peak_idx + 1].copy()
    if len(cup_window) < thresholds.cup_leader_min_cup_bars:
        return None
    left_search_end = max(0, len(cup_window) - thresholds.cup_leader_handle_min_bars)
    left_search = cup_window.iloc[:left_search_end]
    if left_search.empty:
        return None
    left_peak_idx = int(left_search["high"].idxmax())
    if left_peak_idx >= right_peak_idx:
        return None

    cup_section = df.iloc[left_peak_idx : right_peak_idx + 1].copy()
    if len(cup_section) < thresholds.cup_leader_min_cup_bars:
        return None
    cup_low_idx = int(cup_section["low"].idxmin())
    if not (left_peak_idx < cup_low_idx < right_peak_idx):
        return None

    left_peak = float(df.loc[left_peak_idx, "high"])
    cup_low = float(df.loc[cup_low_idx, "low"])
    right_peak = float(df.loc[right_peak_idx, "high"])
    handle_low = float(handle_window["low"].min())
    handle_high = float(handle_window["high"].max())
    handle_bars = int(len(handle_window))
    cup_bars = int(len(cup_section))
    left_bars = int(cup_low_idx - left_peak_idx)
    right_bars = int(right_peak_idx - cup_low_idx)
    cup_depth_pct = _safe_ratio(left_peak - cup_low, left_peak)
    right_peak_recovery_pct = _safe_ratio(right_peak, left_peak)
    handle_depth_pct = _safe_ratio(right_peak - handle_low, right_peak)
    handle_range_pct = _safe_ratio(handle_high - handle_low, right_peak)
    handle_low_position_pct = _safe_ratio(handle_low - cup_low, left_peak - cup_low)
    symmetry_ratio = _safe_ratio(min(left_bars, right_bars), max(left_bars, right_bars))
    handle_volume_dryup_ratio = _safe_ratio(float(handle_window["volume"].mean()), float(cup_section["volume"].mean()))
    resistance = max(float(signal_bar["resistance"]), left_peak, right_peak)
    breakout_pct = _safe_ratio(float(signal_bar["close"]) - resistance, resistance)
    breakout_volume_ratio_50_value = _safe_ratio(float(signal_bar["volume"]), float(signal_bar["avg_volume_50"]))
    roundness = _cup_roundness_metrics(cup_section, left_peak_idx, cup_low_idx, thresholds)

    six_month_start = signal_index - thresholds.cup_leader_prior_rise_lookback
    if six_month_start < 0:
        return None
    prior_rise_6m_pct = _safe_ratio(
        float(signal_bar["close"]) - float(df.iloc[six_month_start]["close"]),
        float(df.iloc[six_month_start]["close"]),
    )
    ma60_reference_idx = signal_index - thresholds.cup_leader_ma60_rising_lookback
    if ma60_reference_idx < 0:
        return None
    ma60_rising = bool(signal_bar["ma_60"] > df.iloc[ma60_reference_idx]["ma_60"])
    right_section = df.iloc[cup_low_idx : right_peak_idx + 1].copy()
    recovery_lows, progressive_lows_ok = _progressive_segment_lows(right_section, 3)
    market_cap_value, market_cap_ok, market_cap_check_skipped = _market_cap_info(signal_bar, thresholds)

    trend_ok = bool(
        prior_rise_6m_pct >= thresholds.cup_leader_prior_rise_min_pct
        and ma60_rising
        and market_cap_ok
    )
    location_ok = bool(
        left_bars >= thresholds.cup_leader_min_down_bars
        and right_bars >= thresholds.cup_leader_min_up_bars
        and cup_bars >= thresholds.cup_leader_min_cup_bars
        and right_peak_recovery_pct >= thresholds.cup_strict_min_recovery_pct
        and handle_low_position_pct >= thresholds.cup_strict_handle_min_position_pct
    )
    pattern_ok = bool(
        bool(roundness["cup_roundness_ok"])
        and symmetry_ratio >= thresholds.cup_min_symmetry_ratio
        and progressive_lows_ok
        and thresholds.cup_leader_handle_min_depth_pct <= handle_depth_pct <= thresholds.cup_leader_handle_max_depth_pct
        and handle_volume_dryup_ratio <= thresholds.cup_leader_handle_max_volume_dryup_ratio
        and signal_bar["close_in_range"] >= thresholds.cup_breakout_min_close_in_range
        and breakout_pct > 0
    )
    volume_ok = bool(breakout_volume_ratio_50_value >= thresholds.cup_leader_breakout_min_volume_ratio_50)
    invalid_reason = (
        None
        if all([trend_ok, location_ok, pattern_ok, volume_ok])
        else "未满足领涨大票杯柄杯的6个月涨幅、MA60、时长、慢抬低点、柄深或放量条件"
    )
    return _empty_signal(
        "cup_with_handle_leader",
        symbol,
        signal_bar,
        trend_ok=trend_ok,
        location_ok=location_ok,
        pattern_ok=pattern_ok,
        volume_ok=volume_ok,
        stop_price=handle_low,
        target_price=float(signal_bar["close"] + 2 * signal_bar["atr_14"]),
        factors={
            "resistance": resistance,
            "left_peak": left_peak,
            "cup_low": cup_low,
            "right_peak": right_peak,
            "cup_depth_pct": float(cup_depth_pct),
            "cup_length_bars": cup_bars,
            "cup_down_bars": left_bars,
            "cup_up_bars": right_bars,
            "right_peak_recovery_pct": float(right_peak_recovery_pct),
            "handle_depth_pct": float(handle_depth_pct),
            "handle_range_pct": float(handle_range_pct),
            "handle_low_position_pct": float(handle_low_position_pct),
            "handle_bars": handle_bars,
            "handle_volume_dryup_ratio": float(handle_volume_dryup_ratio),
            "symmetry_ratio": float(symmetry_ratio),
            "prior_rise_6m_pct": float(prior_rise_6m_pct),
            "ma60_rising": ma60_rising,
            "breakout_pct": float(breakout_pct),
            "close_in_range": float(signal_bar["close_in_range"]),
            "volume_ratio": float(signal_bar["volume_ratio"]),
            "volume_ratio_50": float(breakout_volume_ratio_50_value),
            "market_cap": market_cap_value,
            "market_cap_ok": market_cap_ok,
            "market_cap_check_skipped": market_cap_check_skipped,
            "right_recovery_segment_lows": [float(value) for value in recovery_lows],
            "progressive_lows_ok": progressive_lows_ok,
            **roundness,
        },
        invalid_reason=invalid_reason,
    )


def detect_cup_with_handle_watch(
    frame: pd.DataFrame, symbol: str, thresholds: RuleThresholds
) -> ResearchSignal | None:
    df = _prepare_frame(frame)
    minimum_bars = thresholds.cup_base_lookback + 1
    if len(df) < minimum_bars:
        return None

    recent_window = df.tail(thresholds.signal_recency_days).copy()
    resistance_series = df["high"].rolling(thresholds.cup_base_lookback).max().shift(1)
    recent_window["resistance"] = resistance_series.tail(thresholds.signal_recency_days).values
    candidates = recent_window[
        (recent_window["high"] >= recent_window["resistance"] * (1 - thresholds.breakout_buffer_pct))
        & (recent_window["close"] >= recent_window["resistance"] * (1 - thresholds.cup_watch_near_resistance_pct))
        & (recent_window["close"] <= recent_window["resistance"] * (1 + thresholds.cup_breakout_pct))
        & (recent_window["volume_ratio"] >= thresholds.cup_watch_min_volume_ratio)
    ]
    if candidates.empty:
        return None

    signal_bar = candidates.iloc[-1]
    signal_index = int(signal_bar.name)
    if signal_index < thresholds.cup_base_min_bars:
        return None
    minimum_cup_bars = min(thresholds.cup_base_min_bars, thresholds.cup_short_handle_min_cup_bars)
    cup_window = df.iloc[max(0, signal_index - thresholds.cup_base_lookback + 1) : signal_index + 1].copy()
    if len(cup_window) < minimum_cup_bars:
        return None

    left_search = cup_window.iloc[: -max(1, thresholds.cup_handle_short_min_bars)]
    if left_search.empty:
        return None
    left_peak_idx = int(left_search["high"].idxmax())
    right_peak_idx = signal_index
    if left_peak_idx >= right_peak_idx:
        return None

    cup_section = df.iloc[left_peak_idx : right_peak_idx + 1].copy()
    if len(cup_section) < minimum_cup_bars:
        return None
    cup_low_idx = int(cup_section["low"].idxmin())
    if not (left_peak_idx < cup_low_idx < right_peak_idx):
        return None

    left_peak = float(df.loc[left_peak_idx, "high"])
    cup_low = float(df.loc[cup_low_idx, "low"])
    right_peak = float(df.loc[right_peak_idx, "high"])
    left_bars = int(cup_low_idx - left_peak_idx)
    right_bars = int(right_peak_idx - cup_low_idx)
    cup_depth_pct = _safe_ratio(left_peak - cup_low, left_peak)
    right_peak_recovery_pct = _safe_ratio(right_peak, left_peak)
    symmetry_ratio = _safe_ratio(min(left_bars, right_bars), max(left_bars, right_bars))
    roundness = _cup_roundness_metrics(cup_section, left_peak_idx, cup_low_idx, thresholds)
    prior_trend_start = max(0, left_peak_idx - thresholds.cup_prior_trend_lookback)
    prior_trend_window = df.iloc[prior_trend_start : left_peak_idx + 1]
    if len(prior_trend_window) < max(5, thresholds.cup_prior_trend_lookback // 2):
        return None
    prior_uptrend_pct = _safe_ratio(left_peak - float(prior_trend_window["close"].iloc[0]), float(prior_trend_window["close"].iloc[0]))
    close_to_resistance_pct = _safe_ratio(float(signal_bar["close"]) - float(signal_bar["resistance"]), float(signal_bar["resistance"]))

    trend_ok = bool(
        signal_bar["trend_up"]
        and signal_bar["ma_20"] >= signal_bar["ma_60"] * 0.99
        and prior_uptrend_pct >= thresholds.cup_prior_uptrend_min_pct
    )
    location_ok = bool(
        thresholds.cup_min_depth_pct <= cup_depth_pct <= thresholds.cup_max_depth_pct
        and right_peak_recovery_pct >= thresholds.cup_min_recovery_pct
    )
    pattern_ok = bool(
        left_bars >= thresholds.cup_min_side_bars
        and right_bars >= thresholds.cup_min_side_bars
        and symmetry_ratio >= thresholds.cup_min_symmetry_ratio
        and bool(roundness["cup_roundness_ok"])
        and signal_bar["close_in_range"] >= thresholds.cup_watch_min_close_in_range
    )
    volume_ok = bool(signal_bar["volume_ratio"] >= thresholds.cup_watch_min_volume_ratio)
    invalid_reason = (
        None
        if all([trend_ok, location_ok, pattern_ok, volume_ok])
        else "杯体圆弧度不足，或仍需等待柄部确认与正式突破"
    )
    return _empty_signal(
        "cup_with_handle_watch",
        symbol,
        signal_bar,
        trend_ok=trend_ok,
        location_ok=location_ok,
        pattern_ok=pattern_ok,
        volume_ok=volume_ok,
        stop_price=cup_low,
        target_price=float(signal_bar["resistance"] * (1 + thresholds.cup_breakout_pct)),
        factors={
            "watch_trigger": "right_peak_touch",
            "watch_reference_resistance": float(signal_bar["resistance"]),
            "watch_close_to_resistance_pct": float(close_to_resistance_pct),
            "left_peak": left_peak,
            "cup_low": cup_low,
            "right_peak": right_peak,
            "cup_depth_pct": float(cup_depth_pct),
            "right_peak_recovery_pct": float(right_peak_recovery_pct),
            "symmetry_ratio": float(symmetry_ratio),
            "prior_uptrend_pct": float(prior_uptrend_pct),
            "close_in_range": float(signal_bar["close_in_range"]),
            "volume_ratio": float(signal_bar["volume_ratio"]),
            **roundness,
        },
        invalid_reason=invalid_reason,
    )


def detect_pullback_confirmation(
    frame: pd.DataFrame, symbol: str, thresholds: RuleThresholds
) -> ResearchSignal | None:
    df = _prepare_frame(frame)
    if len(df) < 30:
        return None
    recent = df.iloc[-1]
    breakout_day = _latest_breakout_day(df.iloc[:-1], 15)
    if breakout_day is None:
        return None
    breakout_level = float(breakout_day["breakout_level_20"])
    recent_pullback_low = float(df["low"].tail(5).min())

    trend_ok = bool(recent["trend_up"])
    location_ok = recent_pullback_low >= breakout_level * (1 - thresholds.pullback_tolerance_pct)
    pattern_ok = bool(recent["bullish"] and recent["close"] > breakout_level)
    volume_ok = bool(recent["volume_ratio"] >= 0.9)
    invalid_reason = None
    if not all([trend_ok, location_ok, pattern_ok, volume_ok]):
        invalid_reason = "突破后回抽未确认，支撑可能失效"
    return _empty_signal(
        "pullback_confirmation",
        symbol,
        recent,
        trend_ok=trend_ok,
        location_ok=location_ok,
        pattern_ok=pattern_ok,
        volume_ok=volume_ok,
        stop_price=recent_pullback_low,
        target_price=float(breakout_day["close"] + 2 * recent["atr_14"]),
        factors={
            "breakout_day": breakout_day["date"].date().isoformat(),
            "breakout_level": breakout_level,
            "pullback_low": recent_pullback_low,
        },
        invalid_reason=invalid_reason,
    )


def detect_n_breakout(
    frame: pd.DataFrame, symbol: str, thresholds: RuleThresholds
) -> ResearchSignal | None:
    df = _prepare_frame(frame)
    if len(df) < thresholds.n_breakout_pullback_window + 20:
        return None
    recent = df.iloc[-1]
    prior_window = df.iloc[-thresholds.n_breakout_pullback_window - 5 : -1]
    first_breakout = prior_window[prior_window["close"] > prior_window["breakout_level_20"]]
    if first_breakout.empty:
        return None
    first_break = first_breakout.iloc[-1]
    post_break_window = df[(df["date"] > first_break["date"]) & (df["date"] < recent["date"])].tail(thresholds.n_breakout_pullback_window)
    if post_break_window.empty:
        return None
    if not (thresholds.n_breakout_min_pullback_bars <= len(post_break_window) <= thresholds.n_breakout_max_pullback_bars):
        return None
    pullback_low = float(post_break_window["low"].min())
    retraced = pullback_low >= float(first_break["breakout_level_20"]) * (1 - thresholds.pullback_tolerance_pct)
    prior_high = float(post_break_window["high"].max())
    trend_ok = bool(recent["trend_up"])
    location_ok = bool(retraced)
    pattern_ok = bool(
        recent["close"] > prior_high * (1 + thresholds.n_breakout_rebreak_buffer_pct)
        and recent["close"] > first_break["close"]
        and recent["close_in_range"] >= thresholds.n_breakout_min_close_in_range
    )
    volume_ok = bool(
        recent["volume_ratio"] >= thresholds.n_breakout_min_volume_ratio
        and first_break["volume_ratio"] >= thresholds.n_breakout_min_volume_ratio
    )
    invalid_reason = None if all([trend_ok, location_ok, pattern_ok, volume_ok]) else "N字推进尚未形成再突破"
    return _empty_signal(
        "n_breakout",
        symbol,
        recent,
        trend_ok=trend_ok,
        location_ok=location_ok,
        pattern_ok=pattern_ok,
        volume_ok=volume_ok,
        stop_price=pullback_low,
        target_price=float(recent["close"] + 2 * recent["atr_14"]),
        factors={
            "first_break_date": first_break["date"].date().isoformat(),
            "pullback_low": pullback_low,
            "prior_high": prior_high,
            "pullback_bars": len(post_break_window),
            "close_in_range": float(recent["close_in_range"]),
            "volume_ratio": float(recent["volume_ratio"]),
        },
        invalid_reason=invalid_reason,
    )


def detect_support_resistance_flip(
    frame: pd.DataFrame, symbol: str, thresholds: RuleThresholds
) -> ResearchSignal | None:
    df = _prepare_frame(frame)
    if len(df) < thresholds.support_flip_lookback + thresholds.support_flip_pullback_max_bars + 2:
        return None
    recent = df.iloc[-1]
    breakout_day = _latest_breakout_day(df.iloc[:-1], thresholds.support_flip_lookback)
    if breakout_day is None:
        return None
    breakout_index = int(breakout_day.name)
    pullback_window = df.iloc[breakout_index + 1 :]
    if not (thresholds.support_flip_pullback_min_bars <= len(pullback_window) <= thresholds.support_flip_pullback_max_bars):
        return None
    level = float(breakout_day["breakout_level_20"])
    pullback_low = float(pullback_window["low"].min())
    breakout_volume_ratio = float(breakout_day["volume_ratio"])
    breakout_body_pct = float(breakout_day["body_pct"])
    breakout_close_in_range = float(breakout_day["close_in_range"])
    breakout_prior_window = df.iloc[max(0, breakout_index - thresholds.support_flip_prior_below_bars) : breakout_index]
    prior_below_breakout = bool(
        len(breakout_prior_window) == thresholds.support_flip_prior_below_bars
        and (breakout_prior_window["close"] <= level * (1 + thresholds.support_flip_prior_below_tolerance_pct)).all()
    )
    breakout_valid = bool(
        breakout_day["close"] > level
        and breakout_volume_ratio >= thresholds.support_flip_min_breakout_volume_ratio
        and breakout_body_pct >= thresholds.support_flip_min_breakout_body_pct
        and breakout_close_in_range >= thresholds.support_flip_min_breakout_close_in_range
        and prior_below_breakout
    )
    pullback_volume_ratio = _safe_ratio(float(pullback_window["volume"].mean()), float(breakout_day["volume"]))
    close_hold_ok = bool(
        (pullback_window["close"] >= level * (1 - thresholds.support_flip_max_close_below_level_pct)).all()
    )
    recent_low = float(pullback_window.tail(5)["low"].min())
    reclaim_body_pct = float(recent["body_pct"])
    reclaim_rebound_from_low_pct = _safe_ratio(float(recent["close"]) - pullback_low, pullback_low)
    reclaim_above_prev_high = bool(
        recent["close"] >= float(df.iloc[-2]["high"]) * (1 + thresholds.support_flip_reclaim_above_prev_high_buffer_pct)
    )
    trend_ok = bool(recent["trend_up"] and recent["ma_20"] >= recent["ma_60"] * 1.0)
    location_ok = bool(
        breakout_valid
        and level * (1 - thresholds.support_flip_pullback_tolerance_pct) <= pullback_low <= level * (1 + thresholds.support_flip_pullback_tolerance_pct)
        and close_hold_ok
    )
    pattern_ok = bool(
        recent["bullish"]
        and recent["close"] >= level * (1 + thresholds.support_flip_reclaim_buffer_pct)
        and recent["close_in_range"] >= thresholds.support_flip_reclaim_min_close_in_range
        and reclaim_body_pct >= thresholds.support_flip_reclaim_min_body_pct
        and reclaim_rebound_from_low_pct >= thresholds.support_flip_reclaim_min_rebound_from_low_pct
        and reclaim_above_prev_high
    )
    volume_ok = bool(
        pullback_volume_ratio <= thresholds.support_flip_max_pullback_volume_ratio
        and recent["volume_ratio"] >= thresholds.support_flip_reclaim_min_volume_ratio
    )
    invalid_reason = None if all([trend_ok, location_ok, pattern_ok, volume_ok]) else "突破后回踩确认不够充分或转强质量不足"
    return _empty_signal(
        "support_resistance_flip",
        symbol,
        recent,
        trend_ok=trend_ok,
        location_ok=location_ok,
        pattern_ok=pattern_ok,
        volume_ok=volume_ok,
        stop_price=recent_low,
        target_price=float(recent["close"] + recent["atr_14"]),
        factors={
            "flip_level": level,
            "recent_low": recent_low,
            "pullback_low": pullback_low,
            "breakout_date": breakout_day["date"].date().isoformat(),
            "pullback_bars": int(len(pullback_window)),
            "breakout_volume_ratio": breakout_volume_ratio,
            "breakout_body_pct": breakout_body_pct,
            "breakout_close_in_range": breakout_close_in_range,
            "prior_below_breakout": prior_below_breakout,
            "pullback_volume_ratio": float(pullback_volume_ratio),
            "close_hold_ok": close_hold_ok,
            "reclaim_body_pct": reclaim_body_pct,
            "reclaim_rebound_from_low_pct": float(reclaim_rebound_from_low_pct),
            "reclaim_above_prev_high": reclaim_above_prev_high,
            "volume_ratio": float(recent["volume_ratio"]),
            "close_in_range": float(recent["close_in_range"]),
        },
        invalid_reason=invalid_reason,
    )


def detect_spring(
    frame: pd.DataFrame, symbol: str, thresholds: RuleThresholds
) -> ResearchSignal | None:
    df = _prepare_frame(frame)
    if len(df) < thresholds.range_lookback + 5:
        return None
    recent_window = df.tail(thresholds.signal_recency_days).copy()
    box_low = float(df.iloc[-thresholds.range_lookback - thresholds.signal_recency_days : -thresholds.signal_recency_days]["low"].min())
    candidates = recent_window[
        (recent_window["low"] < box_low * (1 - thresholds.spring_break_pct))
        & (recent_window["close"] >= box_low)
    ]
    if candidates.empty:
        return None
    signal_bar = candidates.iloc[-1]
    trend_ok = bool(signal_bar["ma_20"] >= signal_bar["ma_60"] * 0.98)
    location_ok = bool(signal_bar["close"] >= box_low)
    pattern_ok = bool(signal_bar["low"] < box_low and signal_bar["bullish"])
    volume_ok = bool(signal_bar["volume_ratio"] >= thresholds.false_break_volume_ratio)
    invalid_reason = None if all([trend_ok, location_ok, pattern_ok, volume_ok]) else "箱体弹簧仍可能是真破位"
    return _empty_signal(
        "spring",
        symbol,
        signal_bar,
        trend_ok=trend_ok,
        location_ok=location_ok,
        pattern_ok=pattern_ok,
        volume_ok=volume_ok,
        stop_price=float(signal_bar["low"]),
        target_price=float(df["high"].tail(thresholds.range_lookback).max()),
        factors={"box_low": box_low, "volume_ratio": float(signal_bar["volume_ratio"])},
        invalid_reason=invalid_reason,
    )


def detect_pattern_breakout(
    frame: pd.DataFrame, symbol: str, thresholds: RuleThresholds
) -> ResearchSignal | None:
    df = _prepare_frame(frame)
    if len(df) < thresholds.pattern_lookback + thresholds.pattern_prior_below_bars + 2:
        return None
    recent_window = df.tail(thresholds.signal_recency_days).copy()
    structure = df.iloc[-thresholds.pattern_lookback - thresholds.signal_recency_days : -thresholds.signal_recency_days]
    if structure.empty:
        return None
    box_low, box_high, width_pct = _range_stats(structure)
    candidates = recent_window[recent_window["close"] > box_high * (1 + thresholds.pattern_breakout_pct)]
    if candidates.empty:
        return None
    signal_bar = candidates.iloc[-1]
    prior_window = df[df["date"] < signal_bar["date"]].tail(thresholds.pattern_prior_below_bars)
    breakout_pct = (float(signal_bar["close"]) - box_high) / max(box_high, 1e-6)
    prior_below_box = bool(
        len(prior_window) == thresholds.pattern_prior_below_bars
        and (prior_window["close"] <= box_high * (1 + thresholds.pattern_prior_below_tolerance_pct)).all()
    )
    trend_ok = bool(signal_bar["ma_20"] >= signal_bar["ma_60"] * 0.98)
    location_ok = bool(thresholds.pattern_min_width_pct <= width_pct <= thresholds.pattern_max_width_pct)
    pattern_ok = bool(
        signal_bar["close"] > box_high * (1 + thresholds.pattern_breakout_pct)
        and signal_bar["close"] >= signal_bar["high"] * thresholds.pattern_close_to_high_pct
        and prior_below_box
    )
    volume_ok = bool(signal_bar["volume_ratio"] >= thresholds.pattern_min_volume_ratio)
    invalid_reason = None if all([trend_ok, location_ok, pattern_ok, volume_ok]) else "形态突破的结构或量价确认不足"
    return _empty_signal(
        "pattern_breakout",
        symbol,
        signal_bar,
        trend_ok=trend_ok,
        location_ok=location_ok,
        pattern_ok=pattern_ok,
        volume_ok=volume_ok,
        stop_price=box_high,
        target_price=float(signal_bar["close"] + 2 * signal_bar["atr_14"]),
        factors={
            "box_low": box_low,
            "box_high": box_high,
            "width_pct": width_pct,
            "breakout_pct": float(breakout_pct),
            "prior_below_box": prior_below_box,
        },
        invalid_reason=invalid_reason,
    )


def detect_first_rebound_after_crash(
    frame: pd.DataFrame, symbol: str, thresholds: RuleThresholds
) -> ResearchSignal | None:
    df = _prepare_frame(frame)
    if len(df) < 20:
        return None
    recent_window = df.tail(thresholds.signal_recency_days + 2).copy()
    recent = recent_window.iloc[-1]
    recent_window["drop_from_prev"] = (recent_window["close"] - recent_window["close"].shift(1)) / recent_window["close"].shift(1)
    crash_candidates = recent_window.iloc[:-1]
    crash_candidates = crash_candidates[crash_candidates["drop_from_prev"] <= -thresholds.first_crash_drop_pct]
    if crash_candidates.empty:
        return None
    crash_bar = crash_candidates.iloc[-1]
    after_crash = recent_window[recent_window["date"] > crash_bar["date"]]
    rebound_candidates = after_crash[
        (after_crash["bullish"])
        & (after_crash["body_pct"] >= thresholds.rebound_min_body_pct)
        & (after_crash["close"] > after_crash["open"])
    ]
    if rebound_candidates.empty:
        return None
    signal_bar = rebound_candidates.iloc[0]
    trend_ok = bool(df.iloc[-10]["ma_20"] < df.iloc[-10]["close"] or df.iloc[-10]["trend_up"])
    location_ok = bool(crash_bar["drop_from_prev"] <= -thresholds.first_crash_drop_pct)
    pattern_ok = bool(signal_bar["date"] == rebound_candidates.iloc[0]["date"])
    volume_ok = bool(signal_bar["volume_ratio"] >= thresholds.false_break_volume_ratio)
    invalid_reason = None if all([trend_ok, location_ok, pattern_ok, volume_ok]) else "急跌后的第一次反弹力度不足"
    return _empty_signal(
        "first_rebound_after_crash",
        symbol,
        signal_bar,
        trend_ok=trend_ok,
        location_ok=location_ok,
        pattern_ok=pattern_ok,
        volume_ok=volume_ok,
        stop_price=float(crash_bar["low"]),
        target_price=float(signal_bar["close"] + signal_bar["atr_14"]),
        factors={"crash_date": crash_bar["date"].date().isoformat(), "crash_drop_pct": float(crash_bar["drop_from_prev"])},
        invalid_reason=invalid_reason,
    )


DETECTORS: dict[str, Callable[[pd.DataFrame, str, RuleThresholds], ResearchSignal | None]] = {
    "selling_climax": detect_selling_climax,
    "2b_structure": detect_2b_structure,
    "false_breakdown": detect_false_breakdown,
    "right_shoulder": detect_right_shoulder,
    "double_breakout": detect_double_breakout,
    "strength_emergence": detect_strength_emergence,
    "jumping_creek": detect_jumping_creek,
    "cup_with_handle_watch": detect_cup_with_handle_watch,
    "cup_with_handle": detect_cup_with_handle,
    "cup_with_handle_strict": detect_cup_with_handle_strict,
    "cup_with_handle_leader": detect_cup_with_handle_leader,
    "pullback_confirmation": detect_pullback_confirmation,
    "n_breakout": detect_n_breakout,
    "support_resistance_flip": detect_support_resistance_flip,
    "spring": detect_spring,
    "pattern_breakout": detect_pattern_breakout,
    "first_rebound_after_crash": detect_first_rebound_after_crash,
}


def scan_signals(
    frame: pd.DataFrame,
    symbol: str,
    enabled_signals: Iterable[str] | None = None,
    thresholds: RuleThresholds | None = None,
    include_invalid: bool = False,
) -> list[ResearchSignal]:
    thresholds = thresholds or RuleThresholds()
    enabled = list(enabled_signals or PROGRAMMABLE_SIGNAL_CODES)
    results: list[ResearchSignal] = []
    for signal_code in enabled:
        detector = DETECTORS.get(signal_code)
        if detector is None:
            continue
        signal = detector(frame, symbol, thresholds)
        if signal is None:
            continue
        if include_invalid or signal.is_valid:
            results.append(signal)
    return results


def scan_signal_history(
    frame: pd.DataFrame,
    symbol: str,
    enabled_signals: Iterable[str] | None = None,
    thresholds: RuleThresholds | None = None,
    include_invalid: bool = False,
    min_history_bars: int = 90,
    step: int = 1,
) -> list[ResearchSignal]:
    thresholds = thresholds or RuleThresholds()
    if len(frame) < min_history_bars:
        return []

    history_signals: list[ResearchSignal] = []
    seen: set[tuple[str, str]] = set()
    for end_idx in range(min_history_bars, len(frame) + 1, step):
        window = frame.iloc[:end_idx]
        current_signals = scan_signals(
            window,
            symbol=symbol,
            enabled_signals=enabled_signals,
            thresholds=thresholds,
            include_invalid=include_invalid,
        )
        for signal in current_signals:
            key = (signal.signal_type, signal.signal_date.isoformat())
            if key in seen:
                continue
            seen.add(key)
            history_signals.append(signal)
    return history_signals
