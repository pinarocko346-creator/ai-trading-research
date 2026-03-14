from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from app.data.ingest import DataIngestConfig, fetch_a_share_history
from app.data.universe import UniverseConfig
from app.report.charting import plot_signal_context
from app.strategy.rules import RuleThresholds, scan_signal_history
from app.strategy.scanner import load_default_universe


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="批量验证某个买点的历史样本")
    parser.add_argument("--signal", required=True, help="信号代码，例如 2b_structure")
    parser.add_argument("--max-symbols", type=int, default=10, help="最多扫描多少只股票")
    parser.add_argument("--step", type=int, default=10, help="历史扫描步长")
    parser.add_argument("--limit-charts", type=int, default=8, help="最多导出多少张图")
    args = parser.parse_args()

    config = yaml.safe_load((PROJECT_ROOT / "config" / "strategy_13_points.yaml").read_text(encoding="utf-8"))
    thresholds = RuleThresholds(**config["thresholds"])
    universe_config = UniverseConfig(**config["universe"])
    ingest_config = DataIngestConfig(
        cache_dir=PROJECT_ROOT / "data" / "cache",
        **config.get("ingest", {}),
    )
    spot = load_default_universe(
        universe_config,
        max_symbols=args.max_symbols,
        ingest_config=ingest_config,
    )

    all_signals = []
    charts_dir = PROJECT_ROOT / "reports" / "validation" / args.signal
    charts_dir.mkdir(parents=True, exist_ok=True)

    for _, row in spot.iterrows():
        symbol = str(row["symbol"])
        history = fetch_a_share_history(symbol, ingest_config)
        signals = scan_signal_history(
            history,
            symbol=symbol,
            enabled_signals=[args.signal],
            thresholds=thresholds,
            include_invalid=False,
            step=args.step,
        )
        for signal in signals:
            payload = signal.to_dict()
            payload["name"] = row.get("name", "")
            all_signals.append(payload)
            if len(all_signals) <= args.limit_charts:
                plot_signal_context(history, signal, charts_dir / f"{symbol}_{signal.signal_date.isoformat()}_{signal.signal_type}.png")

    output_path = charts_dir / "summary.csv"
    pd.DataFrame(all_signals).to_csv(output_path, index=False)
    print(f"signal={args.signal}")
    print(f"count={len(all_signals)}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
