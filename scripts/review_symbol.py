from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from app.data.ingest import DataIngestConfig, fetch_a_share_history
from app.report.charting import plot_signal_context
from app.strategy.rules import RuleThresholds, scan_signal_history, scan_signals
from app.strategy.scoring import rank_signals


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_thresholds() -> RuleThresholds:
    config = yaml.safe_load((PROJECT_ROOT / "config" / "strategy_13_points.yaml").read_text(encoding="utf-8"))
    return RuleThresholds(**config["thresholds"])


def main() -> None:
    parser = argparse.ArgumentParser(description="单票信号复核与可视化")
    parser.add_argument("symbol", help="A股代码，例如 600036")
    parser.add_argument("--history", action="store_true", help="扫描历史信号，而不是只看最近信号")
    parser.add_argument("--signal", dest="signals", action="append", help="只查看指定信号，可重复传入")
    parser.add_argument("--limit", type=int, default=5, help="最多导出多少张图")
    args = parser.parse_args()

    thresholds = _load_thresholds()
    history = fetch_a_share_history(args.symbol, DataIngestConfig(cache_dir=PROJECT_ROOT / "data" / "cache"))
    if args.history:
        signals = scan_signal_history(
            history,
            symbol=args.symbol,
            enabled_signals=args.signals,
            thresholds=thresholds,
            include_invalid=False,
        )
    else:
        signals = scan_signals(
            history,
            symbol=args.symbol,
            enabled_signals=args.signals,
            thresholds=thresholds,
            include_invalid=True,
        )
    ranked = rank_signals(signals)[: args.limit]
    output_dir = PROJECT_ROOT / "reports" / "charts" / args.symbol
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ranked:
        print("没有找到匹配信号。")
        return

    for signal in ranked:
        file_name = f"{signal.signal_date.isoformat()}_{signal.signal_type}.png"
        chart_path = plot_signal_context(history, signal, output_dir / file_name)
        print(signal.to_dict())
        print(f"chart={chart_path}")


if __name__ == "__main__":
    main()
