from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from app.backtest.engine import BacktestConfig, run_signal_backtest
from app.data.ingest import DataIngestConfig, fetch_a_share_history, resolve_sqlite_db_path
from app.data.market_context import MarketFilterConfig
from app.data.sector_context import SectorFilterConfig
from app.data.universe import UniverseConfig
from app.report.charting import plot_signal_context
from app.report.csv_localizer import write_localized_csv
from app.report.report_builder import build_daily_report, dump_signals_json, save_report
from app.strategy.rules import RuleThresholds, scan_signals
from app.strategy.scanner import ScanConfig, load_default_universe, scan_market, select_diverse_candidates
from app.strategy.scoring import rank_signals


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _safe_tag(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "-", value).strip("-")


def _resolve_universe_config(config: dict, scope: str) -> UniverseConfig:
    if scope == "research" and config.get("research_universe"):
        return UniverseConfig(**config["research_universe"])
    return UniverseConfig(**config["universe"])


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


def _build_sqlite_status(ingest_config: DataIngestConfig, *, max_staleness_days: int) -> dict[str, object]:
    status: dict[str, object] = {
        "source": ingest_config.source,
        "sqlite_db_path": ingest_config.sqlite_db_path,
        "latest_trade_date": None,
        "symbol_count": None,
        "staleness_days": None,
        "is_stale": False,
    }
    if ingest_config.source != "sqlite":
        return status

    db_path = resolve_sqlite_db_path(ingest_config)
    query = """
        SELECT
            MAX(date) AS latest_trade_date,
            COUNT(DISTINCT code) AS symbol_count
        FROM kline_data
    """
    with closing(sqlite3.connect(db_path)) as conn:
        row = pd.read_sql_query(query, conn).iloc[0]

    latest_trade_date = row.get("latest_trade_date")
    if pd.notna(latest_trade_date):
        latest_ts = pd.Timestamp(str(latest_trade_date)).normalize()
        today = pd.Timestamp.today().normalize()
        staleness_days = int((today - latest_ts).days)
        status["latest_trade_date"] = latest_ts.date().isoformat()
        status["staleness_days"] = staleness_days
        status["is_stale"] = staleness_days > max_staleness_days
    symbol_count = row.get("symbol_count")
    if pd.notna(symbol_count):
        status["symbol_count"] = int(symbol_count)
    return status


def _build_summary(top_rows: pd.DataFrame, manifest: dict[str, object]) -> str:
    lines = [
        "A股13买点 OpenClaw 日任务摘要",
        f"run_id: {manifest['run_id']}",
        f"生成时间: {manifest['generated_at']}",
        f"股票池范围: {manifest['universe_scope']}",
        f"股票池数量: {manifest['universe_size']}",
        f"扫描信号数: {manifest['scan_result_count']}",
        f"日报候选数: {manifest['report_signal_count']}",
    ]
    latest_trade_date = manifest.get("sqlite_latest_trade_date")
    if latest_trade_date:
        lines.append(f"SQLite 最新交易日: {latest_trade_date}")
    if manifest.get("sqlite_is_stale"):
        lines.append(f"数据状态: 警告，距今 {manifest['sqlite_staleness_days']} 天")
    else:
        lines.append("数据状态: 正常")
    if top_rows.empty:
        lines.append("今日候选: 无")
        return "\n".join(lines) + "\n"

    lines.append("今日候选:")
    for _, row in top_rows.head(10).iterrows():
        lines.append(
            f"- {row['symbol']} {row.get('signal_name') or row['signal_type']} "
            f"score={row['score']} regime={row.get('market_regime', '')} "
            f"sector={row.get('sector_band', '')}"
        )
    return "\n".join(lines) + "\n"


def _refresh_latest(run_dir: Path, latest_dir: Path) -> None:
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenClaw 每日运行入口")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "strategy_13_points.yaml"),
        help="策略配置文件路径",
    )
    parser.add_argument("--max-symbols", type=int, default=0, help="扫描股票数量上限，传 0 表示全量")
    parser.add_argument("--top", type=int, default=20, help="日报保留候选数量")
    parser.add_argument(
        "--universe-scope",
        choices=["tradeable", "research"],
        default="tradeable",
        help="每日任务使用的股票池范围",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "reports" / "openclaw_daily"),
        help="OpenClaw 日任务输出根目录",
    )
    parser.add_argument("--latest-dir", default="", help="最新结果目录，默认 output-root/latest")
    parser.add_argument("--tag", default="", help="附加在 run_id 后面的标签")
    parser.add_argument("--sqlite-db-path", default="", help="覆盖配置中的 sqlite_db_path")
    parser.add_argument("--max-staleness-days", type=int, default=3, help="SQLite 数据允许滞后天数")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    output_root = Path(args.output_root).expanduser()
    latest_dir = Path(args.latest_dir).expanduser() if args.latest_dir else output_root / "latest"

    config = _load_config(config_path)
    if args.sqlite_db_path:
        config.setdefault("ingest", {})
        config["ingest"]["sqlite_db_path"] = args.sqlite_db_path

    thresholds = RuleThresholds(**config["thresholds"])
    universe_config = _resolve_universe_config(config, args.universe_scope)
    backtest_config = BacktestConfig(**config["backtest"])
    ingest_config = DataIngestConfig(
        cache_dir=PROJECT_ROOT / "data" / "cache",
        **config.get("ingest", {}),
    )
    market_filter = MarketFilterConfig(
        cache_dir=PROJECT_ROOT / "data" / "cache" / "market",
        **config.get("market_filter", {}),
    )
    sector_filter = SectorFilterConfig(
        cache_dir=PROJECT_ROOT / "data" / "cache" / "sector",
        **config.get("sector_filter", {}),
    )

    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = _safe_tag(args.tag)
    run_id = f"{stamp}_{suffix}" if suffix else stamp
    run_dir = output_root / run_id
    chart_dir = run_dir / "charts"
    run_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    universe = load_default_universe(
        universe_config,
        max_symbols=args.max_symbols,
        ingest_config=ingest_config,
    )
    scan_limit = len(universe) if args.max_symbols <= 0 else min(args.max_symbols, len(universe))
    scan_config = ScanConfig(
        **config.get("scan", {}),
        max_symbols=scan_limit,
        cache_dir=PROJECT_ROOT / "data" / "cache",
        ingest_config=ingest_config,
        market_filter=market_filter,
        sector_filter=sector_filter,
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

    price_map: dict[str, pd.DataFrame] = {}
    signals = []
    chart_map: dict[str, str] = {}
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

    full_scan_path = run_dir / "full_scan.csv"
    candidates_path = run_dir / "daily_candidates.csv"
    signals_path = run_dir / "daily_signals.json"
    report_path = run_dir / "daily_report.md"
    summary_path = run_dir / "summary.txt"
    manifest_path = run_dir / "manifest.json"

    write_localized_csv(scan_results, str(full_scan_path))
    write_localized_csv(top_rows, str(candidates_path))
    dump_signals_json(signals, signals_path)
    save_report(report_text, report_path)

    sqlite_status = _build_sqlite_status(ingest_config, max_staleness_days=args.max_staleness_days)
    top_candidates = []
    for _, row in top_rows.head(10).iterrows():
        top_candidates.append(
            {
                "symbol": str(row["symbol"]),
                "signal_type": str(row["signal_type"]),
                "signal_name": str(row.get("signal_name") or row["signal_type"]),
                "score": float(row.get("score", 0.0) or 0.0),
                "market_regime": str(row.get("market_regime", "")),
                "sector_band": str(row.get("sector_band", "")),
                "filter_ok": bool(row.get("filter_ok", False)),
            }
        )
    manifest = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "project_root": str(PROJECT_ROOT),
        "universe_scope": args.universe_scope,
        "max_symbols": args.max_symbols,
        "top_n": args.top,
        "universe_size": int(len(universe)),
        "scan_result_count": int(len(scan_results)),
        "report_signal_count": int(len(signals)),
        "filter_ok_count": int(scan_results["filter_ok"].fillna(False).sum()) if "filter_ok" in scan_results else 0,
        "market_regime": str(top_rows.iloc[0]["market_regime"]) if not top_rows.empty else "",
        "sqlite_source": sqlite_status["source"],
        "sqlite_db_path": sqlite_status["sqlite_db_path"],
        "sqlite_latest_trade_date": sqlite_status["latest_trade_date"],
        "sqlite_symbol_count": sqlite_status["symbol_count"],
        "sqlite_staleness_days": sqlite_status["staleness_days"],
        "sqlite_is_stale": sqlite_status["is_stale"],
        "status": "warning" if sqlite_status["is_stale"] else "ok",
        "top_candidates": top_candidates,
        "outputs": {
            "run_dir": str(run_dir),
            "full_scan_csv": str(full_scan_path),
            "daily_candidates_csv": str(candidates_path),
            "daily_signals_json": str(signals_path),
            "daily_report_md": str(report_path),
            "summary_txt": str(summary_path),
            "charts_dir": str(chart_dir),
        },
    }
    summary_text = _build_summary(top_rows, manifest)
    summary_path.write_text(summary_text, encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    _refresh_latest(run_dir, latest_dir)

    print(f"run_id={run_id}")
    print(f"universe_size={len(universe)}")
    print(f"scan_result_count={len(scan_results)}")
    print(f"report_signal_count={len(signals)}")
    print(f"status={manifest['status']}")
    print(f"run_dir={run_dir}")
    print(f"manifest={manifest_path}")
    print(f"latest_dir={latest_dir}")


if __name__ == "__main__":
    main()
