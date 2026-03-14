from __future__ import annotations

from pathlib import Path

import yaml

from app.backtest.engine import BacktestConfig, run_signal_backtest
from app.data.ingest import DataIngestConfig, fetch_a_share_history
from app.features.price_features import build_price_features
from app.report.report_builder import build_daily_report, save_report
from app.strategy.rules import RuleThresholds, scan_signals
from app.strategy.scoring import top_valid_signals


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_config() -> dict:
    config_path = PROJECT_ROOT / "config" / "strategy_13_points.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def main() -> None:
    config = _load_config()
    symbols = ["000001", "600036", "300750"]
    signal_config = config["signals"]
    thresholds = RuleThresholds(**config["thresholds"])
    backtest_config = BacktestConfig(**config["backtest"])
    ingest_config = DataIngestConfig(
        cache_dir=PROJECT_ROOT / "data" / "cache",
        **config.get("ingest", {}),
    )

    all_signals = []
    price_map = {}
    for symbol in symbols:
        try:
            history = fetch_a_share_history(symbol, ingest_config)
        except Exception as exc:  # pragma: no cover - 依赖外部数据源
            print(f"[warn] 跳过 {symbol}: {exc}")
            continue
        featured = build_price_features(history)
        price_map[symbol] = featured
        all_signals.extend(
            scan_signals(
                featured,
                symbol=symbol,
                enabled_signals=signal_config["enabled"],
                thresholds=thresholds,
                include_invalid=False,
            )
        )

    ranked = top_valid_signals(all_signals)
    trades = run_signal_backtest(price_map, ranked, backtest_config)
    report = build_daily_report(ranked, trades)
    output_path = PROJECT_ROOT / "reports" / "latest_report.md"
    save_report(report, output_path)
    print(f"已生成研究报告: {output_path}")
    print(f"有效信号数量: {len(ranked)}")
    print(f"回测交易数量: {len(trades)}")


if __name__ == "__main__":
    main()
