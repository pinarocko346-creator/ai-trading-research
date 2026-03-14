from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.ai.explainer import explain_failure, explain_signal, generate_ai_review
from app.backtest.metrics import summarize_trades, summarize_by_signal_type
from app.core.types import BacktestTrade, ResearchSignal


def _market_regime_label(value: object) -> str:
    mapping = {
        "risk_on": "偏多",
        "neutral": "中性",
        "risk_off": "偏弱",
    }
    return mapping.get(str(value), str(value))


def build_daily_report(
    signals: list[ResearchSignal],
    trades: list[BacktestTrade],
    *,
    chart_map: dict[str, str] | None = None,
    report_context: dict[str, object] | None = None,
) -> str:
    lines = ["# 13买点研究报告", ""]
    chart_map = chart_map or {}
    report_context = report_context or {}
    metrics = summarize_trades(trades)
    lines.extend(
        [
            "## 回测摘要",
            f"- 交易次数：{metrics['trade_count']}",
            f"- 胜率：{metrics['win_rate']:.2%}",
            f"- 平均收益：{metrics['avg_return_pct']:.2%}",
            f"- 最大回撤：{metrics['max_drawdown']:.2%}",
            "",
        ]
    )
    if report_context:
        lines.extend(
            [
                "## 三层滤网摘要",
                f"- 市场环境：{_market_regime_label(report_context.get('market_regime', 'unknown'))}",
                f"- 市场得分：{report_context.get('market_score', 0)}",
                f"- 趋势通过指数数：{report_context.get('market_positive_index_count', 0)}",
                f"- 上涨家数占比：{report_context.get('market_up_ratio', 0)}",
                f"- 涨停/跌停：{report_context.get('market_limit_up_count', 0)}/{report_context.get('market_limit_down_count', 0)}",
                "",
            ]
        )
    lines.append("## 候选信号")
    if not signals:
        lines.append("- 今日无有效信号。")
    for signal in signals:
        lines.append(f"### {signal.symbol} / {signal.signal_type}")
        lines.append("```text")
        lines.append(explain_signal(signal) if signal.is_valid else explain_failure(signal))
        lines.append("```")
        lines.append("```text")
        lines.append(generate_ai_review(signal))
        lines.append("```")
        chart_key = f"{signal.symbol}:{signal.signal_type}:{signal.signal_date.isoformat()}"
        if chart_key in chart_map:
            lines.append(f"- 图路径：`{chart_map[chart_key]}`")
    by_type = summarize_by_signal_type(trades)
    if not by_type.empty:
        lines.extend(["", "## 分买点表现", "```json", by_type.to_json(orient='records', force_ascii=False), "```"])
    return "\n".join(lines)


def save_report(report_text: str, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_text, encoding="utf-8")
    return path


def dump_signals_json(signals: list[ResearchSignal], output_path: str | Path) -> Path:
    payload = [signal.to_dict() for signal in signals]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
