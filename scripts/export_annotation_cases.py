from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from app.data.ingest import DataIngestConfig, fetch_a_share_history
from app.report.charting import plot_signal_context
from app.strategy.rules import RuleThresholds, scan_signal_history


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="导出历史案例图和标注清单")
    parser.add_argument("symbols", nargs="+", help="股票代码列表")
    parser.add_argument("--signal", dest="signals", action="append", help="只导出指定信号，可重复传入")
    parser.add_argument("--limit-per-symbol", type=int, default=10, help="每只股票最多导出多少案例")
    args = parser.parse_args()

    config = yaml.safe_load((PROJECT_ROOT / "config" / "strategy_13_points.yaml").read_text(encoding="utf-8"))
    thresholds = RuleThresholds(**config["thresholds"])
    ingest_config = DataIngestConfig(
        cache_dir=PROJECT_ROOT / "data" / "cache",
        **config.get("ingest", {}),
    )

    rows: list[dict[str, object]] = []
    output_dir = PROJECT_ROOT / "reports" / "annotation_cases"
    output_dir.mkdir(parents=True, exist_ok=True)

    for symbol in args.symbols:
        history = fetch_a_share_history(symbol, ingest_config)
        signals = scan_signal_history(
            history,
            symbol=symbol,
            enabled_signals=args.signals or config["signals"]["enabled"],
            thresholds=thresholds,
            include_invalid=False,
        )[: args.limit_per_symbol]
        symbol_dir = output_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        for signal in signals:
            chart_name = f"{signal.signal_date.isoformat()}_{signal.signal_type}.png"
            chart_path = plot_signal_context(history, signal, symbol_dir / chart_name)
            rows.append(
                {
                    "symbol": symbol,
                    "signal_type": signal.signal_type,
                    "signal_date": signal.signal_date.isoformat(),
                    "confidence_score": signal.confidence_score,
                    "chart_path": str(chart_path),
                    "manual_label": "",
                    "notes": "",
                }
            )

    review_frame = pd.DataFrame(rows)
    csv_path = output_dir / "annotation_index.csv"
    review_frame.to_csv(csv_path, index=False)
    print(f"cases={len(review_frame)}")
    print(f"output={csv_path}")


if __name__ == "__main__":
    main()
