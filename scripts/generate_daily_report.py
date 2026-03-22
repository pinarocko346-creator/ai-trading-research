from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import yaml

from app.data.market_context import MarketFilterConfig
from app.data.sector_context import SectorFilterConfig
from app.backtest.engine import BacktestConfig, run_signal_backtest
from app.data.ingest import DataIngestConfig, fetch_a_share_history
from app.report.csv_localizer import write_localized_csv
from app.data.universe import UniverseConfig
from app.report.charting import plot_signal_context
from app.report.report_builder import build_daily_report, dump_signals_json, save_report
from app.strategy.rules import RuleThresholds, scan_signals
from app.strategy.scanner import ScanConfig, load_default_universe, scan_market, select_diverse_candidates
from app.strategy.scoring import rank_signals


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_config() -> dict:
    return yaml.safe_load((PROJECT_ROOT / "config" / "strategy_13_points.yaml").read_text(encoding="utf-8"))


def _match_signal(history, symbol: str, signal_type: str, signal_date: str, thresholds: RuleThresholds, enabled: list[str]):
    signals = scan_signals(
        history,
        symbol=symbol,
        enabled_signals=enabled,
        thresholds=thresholds,
        include_invalid=False,
    )
    for signal in rank_signals(signals):
        if signal.signal_type == signal_type and signal.signal_date.isoformat() == signal_date:
            return signal
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="生成A股14买点日报")
    parser.add_argument("--max-symbols", type=int, default=100, help="扫描股票池数量，传 0 表示全量")
    parser.add_argument("--top", type=int, default=20, help="日报保留候选数量")
    args = parser.parse_args()

    config = _load_config()
    thresholds = RuleThresholds(**config["thresholds"])
    universe_config = UniverseConfig(**config["universe"])
    backtest_config = BacktestConfig(**config["backtest"])
    market_filter = MarketFilterConfig(
        cache_dir=PROJECT_ROOT / "data" / "cache" / "market",
        **config.get("market_filter", {}),
    )
    sector_filter = SectorFilterConfig(
        cache_dir=PROJECT_ROOT / "data" / "cache" / "sector",
        **config.get("sector_filter", {}),
    )
    scan_config = ScanConfig(
        **config.get("scan", {}),
        max_symbols=args.max_symbols,
        cache_dir=PROJECT_ROOT / "data" / "cache",
        ingest_config=DataIngestConfig(
            cache_dir=PROJECT_ROOT / "data" / "cache",
            **config.get("ingest", {}),
        ),
        market_filter=market_filter,
        sector_filter=sector_filter,
    )
    ingest_config = scan_config.ingest_config

    universe = load_default_universe(
        universe_config,
        max_symbols=args.max_symbols,
        ingest_config=ingest_config,
    )
    scan_results = scan_market(
        universe,
        thresholds=thresholds,
        signal_types=config["signals"]["enabled"],
        scan_config=scan_config,
    )
    top_rows = select_diverse_candidates(
        scan_results,
        top_n=args.top,
        per_signal_limit=scan_config.per_signal_limit,
    )

    price_map = {}
    signals = []
    chart_map: dict[str, str] = {}
    chart_dir = PROJECT_ROOT / "reports" / "daily_charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    for _, row in top_rows.iterrows():
        symbol = str(row["symbol"])
        history = price_map.get(symbol)
        if history is None:
            history = fetch_a_share_history(symbol, ingest_config)
            price_map[symbol] = history
        signal = _match_signal(
            history,
            symbol=symbol,
            signal_type=str(row["signal_type"]),
            signal_date=str(row["signal_date"]),
            thresholds=thresholds,
            enabled=config["signals"]["enabled"],
        )
        if signal is None:
            continue
        secondary_signal_names = row.get("secondary_signal_names", [])
        secondary_signal_types = row.get("secondary_signal_types", [])
        secondary_signal_count = int(row.get("secondary_signal_count", 0) or 0)
        if secondary_signal_count > 0:
            signal.factors["secondary_signal_names"] = list(secondary_signal_names)
            signal.factors["secondary_signal_types"] = list(secondary_signal_types)
            signal.factors["secondary_signal_count"] = secondary_signal_count
        signal.factors["market_ok"] = bool(row.get("market_ok", False))
        signal.factors["market_score"] = float(row.get("market_score", 0.0) or 0.0)
        signal.factors["market_regime"] = str(row.get("market_regime", "unknown"))
        signal.factors["market_up_ratio"] = float(row.get("market_up_ratio", 0.0) or 0.0)
        signal.factors["market_limit_up_count"] = int(row.get("market_limit_up_count", 0) or 0)
        signal.factors["market_limit_down_count"] = int(row.get("market_limit_down_count", 0) or 0)
        signal.factors["filter_ok"] = bool(row.get("filter_ok", False))
        signal.factors["sector_ok"] = bool(row.get("sector_ok", False))
        signal.factors["sector_score"] = float(row.get("sector_score", 0.0) or 0.0)
        signal.factors["sector_band"] = str(row.get("sector_band", "") or "")
        signal.factors["industry_name"] = str(row.get("industry_name", "") or "")
        signal.factors["industry_score"] = float(row.get("industry_score", 0.0) or 0.0)
        signal.factors["concept_names"] = list(row.get("concept_names", []) or [])
        signal.factors["concept_scores"] = list(row.get("concept_scores", []) or [])
        signal.factors["quality_score"] = float(row.get("quality_score", 0.0) or 0.0)
        signal.factors["quality_bucket"] = str(row.get("quality_bucket", "") or "")
        signal.factors["pretty_ok"] = bool(row.get("pretty_ok", False))
        signals.append(signal)
        chart_path = chart_dir / f"{symbol}_{signal.signal_date.isoformat()}_{signal.signal_type}.png"
        plot_signal_context(history, signal, chart_path)
        chart_map[f"{symbol}:{signal.signal_type}:{signal.signal_date.isoformat()}"] = str(chart_path)

    trades = run_signal_backtest(price_map, signals, backtest_config)
    report_context = None
    if not top_rows.empty:
        first_row = top_rows.iloc[0]
        report_context = {
            "market_regime": str(first_row.get("market_regime", "unknown")),
            "market_score": float(first_row.get("market_score", 0.0) or 0.0),
            "market_positive_index_count": int(first_row.get("market_positive_index_count", 0) or 0),
            "market_up_ratio": float(first_row.get("market_up_ratio", 0.0) or 0.0),
            "market_limit_up_count": int(first_row.get("market_limit_up_count", 0) or 0),
            "market_limit_down_count": int(first_row.get("market_limit_down_count", 0) or 0),
        }
    report_text = build_daily_report(signals, trades, chart_map=chart_map, report_context=report_context)

    output_dir = PROJECT_ROOT / "reports" / "daily"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"daily_report_{stamp}.md"
    scan_path = output_dir / f"daily_candidates_{stamp}.csv"
    json_path = output_dir / f"daily_signals_{stamp}.json"

    save_report(report_text, report_path)
    write_localized_csv(top_rows, str(scan_path))
    dump_signals_json(signals, json_path)

    print(f"universe={len(universe)}")
    print(f"scan_results={len(scan_results)}")
    print(f"report_signals={len(signals)}")
    print(f"report={report_path}")
    print(f"candidates={scan_path}")
    print(f"signals_json={json_path}")


if __name__ == "__main__":
    main()
