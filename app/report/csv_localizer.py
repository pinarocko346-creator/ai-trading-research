from __future__ import annotations

import ast
import json
from typing import Any

import pandas as pd

from app.strategy.rules import build_signal_catalog


COLUMN_LABELS = {
    "signal_type": "买点代码",
    "signal_name": "买点名称",
    "symbol": "股票代码",
    "name": "股票名称",
    "signal_date": "信号日期",
    "confidence_score": "信心分",
    "trend_ok": "趋势通过",
    "location_ok": "位置通过",
    "pattern_ok": "结构通过",
    "volume_ok": "量价通过",
    "entry_price": "入场价",
    "stop_price": "止损价",
    "target_price": "目标价",
    "invalid_reason": "失效原因",
    "risk_tags": "风险标签",
    "factors": "关键因子",
    "base_score": "基础分",
    "score": "总分",
    "market_ok": "市场层通过",
    "market_score": "市场得分",
    "market_regime": "市场环境",
    "market_positive_index_count": "趋势通过指数数",
    "market_up_ratio": "上涨家数占比",
    "market_limit_up_count": "涨停家数",
    "market_limit_down_count": "跌停家数",
    "market_limit_up_down_ratio": "涨跌停比",
    "market_index_details": "指数明细",
    "sector_ok": "板块层通过",
    "sector_score": "板块综合分",
    "industry_name": "所属行业",
    "industry_score": "行业强度分",
    "concept_names": "相关概念",
    "concept_scores": "概念强度分",
    "sector_band": "板块强度分层",
    "filter_ok": "三层滤网通过",
    "quality_score": "漂亮度分",
    "quality_bucket": "漂亮度分层",
    "pretty_ok": "漂亮度通过",
    "secondary_signal_types": "次级买点代码",
    "secondary_signal_names": "次级买点名称",
    "secondary_signal_count": "次级买点数量",
}

VALUE_LABELS = {
    "risk_on": "偏多",
    "neutral": "中性",
    "risk_off": "偏弱",
    "strong": "强势主线",
    "edge_high": "边缘活跃高位区",
    "edge_low": "边缘活跃低位区",
    "weak": "弱势区",
    "none": "无板块加分",
    "crowded": "过热拥挤",
    "needs_review": "需复核",
    "high": "高质量",
    "medium": "中等质量",
    "low": "低质量",
}

NESTED_KEY_LABELS = {
    "name": "名称",
    "close": "收盘价",
    "ma20": "20日均线",
    "ma60": "60日均线",
    "trend_ok": "趋势通过",
    "resistance": "阻力位",
    "left_peak": "左杯沿",
    "cup_low": "杯底",
    "right_peak": "右杯沿",
    "support": "支撑位",
    "support_level": "支撑位",
    "breakout_level": "突破位",
    "recent_high_10": "近10日高点",
    "volume_ratio": "量比",
    "breakout_pct": "突破幅度",
    "range_atr_ratio": "波动ATR比",
    "cup_depth_pct": "杯体深度",
    "right_peak_recovery_pct": "右侧回升接近前高比例",
    "handle_depth_pct": "杯柄深度",
    "handle_range_pct": "杯柄波动范围",
    "handle_low_position_pct": "杯柄位置占比",
    "handle_bars": "杯柄K线数",
    "handle_volume_dryup_ratio": "杯柄缩量比",
    "symmetry_ratio": "杯体对称度",
    "prior_uptrend_pct": "前置上涨幅度",
    "prior_below_resistance": "突破前压制成立",
    "first_break_date": "首次突破日期",
    "pullback_low": "回抽低点",
    "prior_high": "前高",
    "market_ok": "市场层通过",
    "market_score": "市场得分",
    "market_regime": "市场环境",
    "market_up_ratio": "上涨家数占比",
    "market_limit_up_count": "涨停家数",
    "market_limit_down_count": "跌停家数",
    "sector_ok": "板块层通过",
    "sector_score": "板块综合分",
    "sector_band": "板块强度分层",
    "quality_score": "漂亮度分",
    "quality_bucket": "漂亮度分层",
    "pretty_ok": "漂亮度通过",
    "industry_name": "所属行业",
    "industry_score": "行业强度分",
    "concept_names": "相关概念",
    "concept_scores": "概念强度分",
    "secondary_signal_names": "次级标签",
    "secondary_signal_types": "次级买点代码",
    "secondary_signal_count": "次级买点数量",
}

DROP_COLUMNS = {"secondary_signal_types"}


def _signal_label_map() -> dict[str, str]:
    return {item.code: item.name for item in build_signal_catalog()}


def _maybe_parse(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text in {"nan", "None", "null"}:
        return value
    if text[0] not in "[{(":
        return value
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return value


def _translate_value(value: Any, *, signal_labels: dict[str, str]) -> Any:
    value = _maybe_parse(value)
    if isinstance(value, bool):
        return "是" if value else "否"
    if value is None:
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value in signal_labels:
            return signal_labels[value]
        return VALUE_LABELS.get(value, value)
    if isinstance(value, list):
        translated = [_translate_value(item, signal_labels=signal_labels) for item in value]
        if not translated:
            return ""
        if all(not isinstance(item, (dict, list, tuple)) for item in translated):
            return "、".join(str(item) for item in translated)
        return "；".join(str(item) for item in translated)
    if isinstance(value, tuple):
        return "、".join(str(_translate_value(item, signal_labels=signal_labels)) for item in value)
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            label = NESTED_KEY_LABELS.get(str(key), COLUMN_LABELS.get(str(key), str(key)))
            translated = _translate_value(item, signal_labels=signal_labels)
            parts.append(f"{label}={translated}")
        return "；".join(parts)
    return str(value)


def localize_csv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.rename(columns=COLUMN_LABELS)

    localized = frame.copy()
    signal_labels = _signal_label_map()

    if "signal_name" not in localized.columns and "signal_type" in localized.columns:
        localized["signal_name"] = localized["signal_type"].map(signal_labels).fillna(localized["signal_type"])
    if "secondary_signal_names" in localized.columns:
        localized["secondary_signal_names"] = localized["secondary_signal_names"].apply(
            lambda value: _translate_value(value, signal_labels=signal_labels)
        )
    if "secondary_signal_types" in localized.columns:
        localized["secondary_signal_types"] = localized["secondary_signal_types"].apply(
            lambda value: _translate_value(value, signal_labels=signal_labels)
        )

    for column in list(localized.columns):
        if column in DROP_COLUMNS:
            localized = localized.drop(columns=column)
            continue
        if column == "signal_type":
            localized[column] = localized[column].apply(lambda value: _translate_value(value, signal_labels=signal_labels))
            continue
        localized[column] = localized[column].apply(lambda value: _translate_value(value, signal_labels=signal_labels))

    preferred_order = [
        "symbol",
        "name",
        "signal_name",
        "signal_type",
        "signal_date",
        "score",
        "base_score",
        "confidence_score",
        "filter_ok",
        "pretty_ok",
        "quality_score",
        "quality_bucket",
        "market_regime",
        "sector_band",
    ]
    ordered = [column for column in preferred_order if column in localized.columns]
    remaining = [column for column in localized.columns if column not in ordered]
    localized = localized[ordered + remaining]
    return localized.rename(columns=COLUMN_LABELS)


def write_localized_csv(frame: pd.DataFrame, output_path: str) -> None:
    localized = localize_csv_frame(frame)
    localized.to_csv(output_path, index=False, encoding="utf-8-sig")


def localized_csv_preview(frame: pd.DataFrame) -> str:
    localized = localize_csv_frame(frame)
    return json.dumps(localized.head(5).to_dict(orient="records"), ensure_ascii=False, indent=2)
