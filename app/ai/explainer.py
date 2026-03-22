from __future__ import annotations
from textwrap import dedent

import pandas as pd

from app.core.types import ResearchSignal


SIGNAL_LABELS = {
    "selling_climax": "抛售高潮",
    "2b_structure": "2B结构",
    "false_breakdown": "假诱空",
    "right_shoulder": "顺势头肩右肩",
    "double_breakout": "双突破",
    "strength_emergence": "强势出现",
    "jumping_creek": "跳跃小溪",
    "cup_with_handle_watch": "疑似杯柄杯",
    "cup_with_handle": "杯子与杯柄",
    "cup_with_handle_strict": "严格杯柄杯",
    "cup_with_handle_leader": "领涨大票杯柄杯",
    "pullback_confirmation": "回抽确认",
    "n_breakout": "N字突破",
    "support_resistance_flip": "支撑压力互换",
    "spring": "箱体弹簧",
    "pattern_breakout": "形态突破",
    "first_rebound_after_crash": "趋势急跌后的第一次反弹",
}


SECONDARY_SIGNAL_FACTOR_KEYS = {
    "secondary_signal_names",
    "secondary_signal_types",
    "secondary_signal_count",
}

FACTOR_LABELS = {
    "prior_support": "前支撑位",
    "fake_break_low": "假破低点",
    "breakdown_date": "跌破日期",
    "support_source": "支撑来源",
    "reclaim_date": "收回日期",
    "reclaim_in_time": "是否及时收回",
    "volume_ratio": "量比",
    "support": "支撑位",
    "head_low": "头部低点",
    "right_low": "右肩低点",
    "left_shoulder": "左肩低点",
    "neckline": "颈线位",
    "right_low_date": "右肩形成日",
    "bars_since_right_low": "右肩后经过K线数",
    "breakout_level": "突破位",
    "recent_high_10": "近10日高点",
    "box_low": "箱体下沿",
    "box_high": "箱体上沿",
    "midline": "箱体中轴",
    "width_pct": "箱体宽度占比",
    "upper_zone_level": "上沿强势区",
    "close_to_high_pct": "距高点比例",
    "closes_above_midline": "近阶段中轴上方收盘数",
    "resistance": "阻力位",
    "watch_trigger": "预警触发方式",
    "watch_reference_resistance": "预警参考阻力",
    "watch_close_to_resistance_pct": "收盘距阻力比例",
    "left_peak": "左杯沿",
    "cup_low": "杯底",
    "right_peak": "右杯沿",
    "breakout_pct": "突破幅度",
    "range_atr_ratio": "波动/ATR",
    "prior_below_resistance": "突破前压制成立",
    "prior_volume_dryup_ratio": "突破前缩量比",
    "breakout_hold_ok": "突破日是否站稳阻力",
    "cup_depth_pct": "杯体深度",
    "cup_length_bars": "杯体K线数",
    "cup_down_bars": "杯子下跌K线数",
    "cup_up_bars": "杯子回升K线数",
    "right_peak_recovery_pct": "右侧回升接近前高比例",
    "handle_depth_pct": "杯柄深度",
    "handle_depth_vs_cup_pct": "杯柄回撤/杯深比例",
    "handle_range_pct": "杯柄波动范围",
    "handle_low_position_pct": "杯柄位置占比",
    "handle_bars": "杯柄K线数",
    "handle_bars_ratio": "杯柄长度/杯体长度",
    "handle_volume_dryup_ratio": "杯柄缩量比",
    "symmetry_ratio": "杯体对称度",
    "cup_curve_fit_error_pct": "杯体圆弧拟合误差",
    "cup_low_position_ratio": "杯底位置占比",
    "cup_low_position_offset": "杯底偏离中轴程度",
    "cup_roundness_ok": "圆弧度过滤结果",
    "prior_uptrend_pct": "前置上涨幅度",
    "prior_rise_60_pct": "前60日上涨幅度",
    "prior_rise_6m_pct": "前6个月上涨幅度",
    "ma60_rising": "MA60是否上升",
    "pullback_low": "回踩低点",
    "first_break_date": "首次突破日",
    "prior_high": "前高",
    "flip_level": "互换价位",
    "recent_low": "近端低点",
    "pullback_low": "回踩低点",
    "breakout_date": "突破日期",
    "pullback_bars": "回踩K线数",
    "breakout_volume_ratio": "突破日量比",
    "breakout_body_pct": "突破日实体涨幅",
    "breakout_close_in_range": "突破日收盘位置",
    "reclaim_body_pct": "确认日实体涨幅",
    "reclaim_rebound_from_low_pct": "确认日距回踩低点反弹幅度",
    "reclaim_above_prev_high": "确认日是否站上前一日高点",
    "pullback_volume_ratio": "回踩均量/突破量",
    "close_hold_ok": "回踩期间是否守住突破位",
    "prior_below_box": "突破前箱顶压制成立",
    "crash_date": "急跌日",
    "crash_drop_pct": "急跌幅度",
    "climax_date": "高潮日",
    "volume_ratio_50": "量比(对50日均量)",
    "market_cap": "总市值",
    "market_cap_ok": "市值过滤结果",
    "market_cap_check_skipped": "市值条件是否跳过",
    "right_recovery_segment_lows": "右侧分段低点",
    "progressive_lows_ok": "右侧低点逐步抬高",
    "drawdown_from_high_60": "距60日高点回撤",
    "market_ok": "市场层通过",
    "market_score": "市场得分",
    "market_regime": "市场环境",
    "market_up_ratio": "上涨家数占比",
    "market_limit_up_count": "涨停家数",
    "market_limit_down_count": "跌停家数",
    "sector_ok": "板块层通过",
    "sector_score": "板块综合分",
    "sector_band": "板块强度分层",
    "industry_name": "所属行业",
    "industry_score": "行业强度分",
    "concept_names": "相关概念",
    "concept_scores": "概念强度分",
    "filter_ok": "三层滤网通过",
    "quality_score": "漂亮度分",
    "quality_bucket": "漂亮度分层",
    "pretty_ok": "漂亮度通过",
}


def _format_factor_value(value: object) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        if abs(value) < 10 and any(token in f"{value}" for token in (".", "e")):
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if isinstance(value, list):
        return "、".join(str(item) for item in value)
    return str(value)


def _format_signal_factors(factors: dict[str, object]) -> list[str]:
    lines: list[str] = []
    for key, value in factors.items():
        label = FACTOR_LABELS.get(key, key)
        if key == "market_regime":
            value = {
                "risk_on": "偏多",
                "neutral": "中性",
                "risk_off": "偏弱",
            }.get(str(value), value)
        if key == "sector_band":
            value = {
                "crowded": "过热拥挤",
                "strong": "强势主线",
                "edge_high": "边缘活跃-高位",
                "edge_low": "边缘活跃-低位",
                "weak": "弱关联",
                "none": "缺少映射",
            }.get(str(value), value)
        lines.append(f"- {label}：{_format_factor_value(value)}")
    return lines


def _format_reference_pairs(factors: dict[str, object], keys: tuple[str, ...]) -> str:
    items: list[str] = []
    for key in keys:
        if key not in factors:
            continue
        label = FACTOR_LABELS.get(key, key)
        items.append(f"{label}={_format_factor_value(factors[key])}")
    return "，".join(items)


def explain_signal(signal: ResearchSignal) -> str:
    label = SIGNAL_LABELS.get(signal.signal_type, signal.signal_type)
    factors = signal.factors
    display_factors = {
        key: value for key, value in factors.items() if key not in SECONDARY_SIGNAL_FACTOR_KEYS
    }
    summary = [
        f"信号类型：{label}",
        f"标的：{signal.symbol}",
        f"日期：{signal.signal_date.isoformat()}",
        f"评分：{signal.confidence_score}",
        f"趋势/位置/结构/量价：{signal.trend_ok}/{signal.location_ok}/{signal.pattern_ok}/{signal.volume_ok}",
    ]
    secondary_signal_names = factors.get("secondary_signal_names", [])
    if secondary_signal_names:
        summary.append("次级标签：" + "、".join(str(name) for name in secondary_signal_names))
    if signal.invalid_reason:
        summary.append(f"失效原因：{signal.invalid_reason}")
    if display_factors:
        summary.append("关键因子：")
        summary.extend(_format_signal_factors(display_factors))
    return "\n".join(summary)


def explain_failure(signal: ResearchSignal) -> str:
    unmet = []
    if not signal.trend_ok:
        unmet.append("大周期趋势不匹配")
    if not signal.location_ok:
        unmet.append("关键位置不成立")
    if not signal.pattern_ok:
        unmet.append("结构不完整")
    if not signal.volume_ok:
        unmet.append("量价确认不足")
    base = "、".join(unmet) if unmet else "规则未能通过最终确认"
    return f"{SIGNAL_LABELS.get(signal.signal_type, signal.signal_type)}失败归因：{base}。"


def generate_ai_review(signal: ResearchSignal) -> str:
    label = SIGNAL_LABELS.get(signal.signal_type, signal.signal_type)
    positives = []
    cautions = []

    if signal.trend_ok:
        positives.append("大周期方向与该买点定义基本一致")
    else:
        cautions.append("大周期方向还不够顺")
    if signal.location_ok:
        positives.append("关键支撑或阻力位置已经进入有效区域")
    else:
        cautions.append("位置优势不明显")
    if signal.pattern_ok:
        positives.append("结构动作已经走出来，不是纯猜测")
    else:
        cautions.append("结构还不够完整")
    if signal.volume_ok:
        positives.append("量价有一定确认")
    else:
        cautions.append("量能确认偏弱")

    factors = signal.factors
    market_regime = str(factors.get("market_regime", ""))
    market_score = float(factors.get("market_score", 0.0) or 0.0)
    sector_score = float(factors.get("sector_score", 0.0) or 0.0)
    sector_band = str(factors.get("sector_band", ""))
    if market_regime == "risk_on" or bool(factors.get("market_ok", False)):
        positives.append("市场层环境允许当前策略继续筛票")
    elif market_regime == "neutral":
        positives.append("市场层处于中性区间，需要更精选地做主线个股")
    elif "market_ok" in factors:
        cautions.append("市场层环境偏弱，个股信号需要进一步降级看待")
    if sector_band == "crowded":
        cautions.append("所属板块热度过高且拥挤，容易进入一致性兑现阶段")
    elif bool(factors.get("sector_ok", False)):
        positives.append("所属行业或概念具备一定热度支持")
    elif sector_band == "edge_high":
        positives.append("所属板块处于边缘活跃高位区，可以保留继续跟踪")
    elif sector_band == "edge_low":
        cautions.append("所属板块仅处于边缘活跃低位区，暂放观察优先")
    elif sector_score >= 43:
        cautions.append("板块热度处于边缘区，最好再等个股确认")
    elif "sector_ok" in factors:
        cautions.append("板块/概念支持度一般")
    quality_score = float(factors.get("quality_score", 0.0) or 0.0)
    if "pretty_ok" in factors:
        if bool(factors.get("pretty_ok", False)):
            positives.append(f"形态漂亮度通过统一过滤，当前漂亮度分约为 {quality_score:.1f}")
        else:
            cautions.append(f"形态漂亮度未通过统一过滤，当前漂亮度分约为 {quality_score:.1f}")
    extra = _format_reference_pairs(
        factors,
        ("prior_support", "breakout_level", "flip_level", "neckline", "resistance", "box_high", "box_low", "left_peak", "cup_low", "right_peak"),
    )

    lines = [
        f"AI解读：{label} 当前更像是一个{'有效' if signal.is_valid else '待确认'}候选。",
        "支持点：" + ("；".join(positives) if positives else "暂无明显支持点。"),
        "风险点：" + ("；".join(cautions) if cautions else "暂未发现明显结构性硬伤，但仍需结合市场环境复核。"),
    ]
    if extra:
        lines.append("结构参考：" + extra)
    if signal.invalid_reason:
        lines.append("规则提示：" + signal.invalid_reason)
    secondary_signal_names = factors.get("secondary_signal_names", [])
    if secondary_signal_names:
        lines.append("标签补充：该候选同时具备" + "、".join(str(name) for name in secondary_signal_names) + "特征。")
    return "\n".join(lines)


def build_llm_prompt(signal: ResearchSignal, window_frame: pd.DataFrame) -> str:
    preview = window_frame.tail(15)[["date", "open", "high", "low", "close", "volume"]].copy()
    preview["date"] = preview["date"].astype(str)
    return dedent(
        f"""
        你是一名 A 股技术形态研究助手，请审阅下面的 {SIGNAL_LABELS.get(signal.signal_type, signal.signal_type)} 候选信号。

        任务：
        1. 判断这个信号是否符合“先看趋势、再找位置、再等结构、最后看量价”的定义。
        2. 给出支持点和反对点。
        3. 若信号质量一般，指出应该收紧哪些阈值。

        信号摘要：
        {explain_signal(signal)}

        最近 15 根 K 线：
        {preview.to_json(orient="records", force_ascii=False)}
        """
    ).strip()
