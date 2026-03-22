from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from app.backtest.engine import BacktestConfig, run_signal_backtest
from app.backtest.metrics import summarize_trade_frame
from app.core.types import ResearchSignal
from app.data.ingest import DataIngestConfig, fetch_a_share_history, load_sqlite_breadth_history
from app.data.market_context import MarketFilterConfig, fetch_index_history, score_market_snapshot
from app.data.sector_context import SectorFilterConfig, load_sector_snapshot
from app.report.charting import plot_signal_context
from app.strategy.rules import RuleThresholds, scan_signal_history
from app.strategy.scanner import (
    _filter_ok,
    _sector_band,
    load_default_universe,
    pretty_signal_ok,
    quality_bucket_label,
    score_signal_quality,
)
from scripts.validate_three_layer_filter_history import (
    _build_daily_breadth,
    _build_historical_theme_payload,
    _fetch_board_history,
    _resolve_research_universe_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_config() -> dict:
    return yaml.safe_load((PROJECT_ROOT / "config" / "strategy_13_points.yaml").read_text(encoding="utf-8"))


def _factor_bucket_label(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty or valid.nunique() < 3:
        return pd.Series(["all"] * len(series), index=series.index)
    try:
        buckets = pd.qcut(valid, q=min(3, valid.nunique()), labels=["低", "中", "高"][: min(3, valid.nunique())], duplicates="drop")
    except ValueError:
        return pd.Series(["all"] * len(series), index=series.index)
    labeled = pd.Series(["all"] * len(series), index=series.index, dtype="object")
    labeled.loc[valid.index] = buckets.astype(str)
    return labeled


def _summarize_groups(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if frame.empty:
        return pd.DataFrame(rows)
    for keys, group in frame.groupby(group_columns, dropna=False):
        summary = summarize_trade_frame(group)
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: value for column, value in zip(group_columns, keys)}
        row.update(summary)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="批量验证某个买点的历史样本")
    parser.add_argument("--signal", dest="signals", action="append", help="信号代码，例如 2b_structure，可重复传入")
    parser.add_argument("--symbol", dest="symbols", action="append", help="股票代码，例如 600036，可重复传入")
    parser.add_argument("--max-symbols", type=int, default=10, help="最多扫描多少只股票，传 0 表示全量")
    parser.add_argument("--step", type=int, default=10, help="历史扫描步长")
    parser.add_argument("--limit-charts", type=int, default=8, help="最多导出多少张图")
    parser.add_argument("--start-date", default="2024-01-01", help="验证起始日期")
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y-%m-%d"), help="验证结束日期")
    args = parser.parse_args()

    config = _load_config()
    thresholds = RuleThresholds(**config["thresholds"])
    universe_config = _resolve_research_universe_config(config)
    ingest_config = DataIngestConfig(
        cache_dir=PROJECT_ROOT / "data" / "cache",
        start_date=args.start_date,
        end_date=args.end_date,
        **config.get("ingest", {}),
    )
    backtest_config = BacktestConfig(**config["backtest"])
    market_filter = MarketFilterConfig(
        cache_dir=PROJECT_ROOT / "data" / "cache" / "market",
        **config.get("market_filter", {}),
    )
    sector_filter = SectorFilterConfig(
        cache_dir=PROJECT_ROOT / "data" / "cache" / "sector",
        **config.get("sector_filter", {}),
    )
    pretty_min_quality_score = float(config.get("scan", {}).get("pretty_min_quality_score", 60.0))
    pretty_hard_filter_score = float(config.get("scan", {}).get("pretty_hard_filter_score", 45.0))
    enabled_signals = list(args.signals or config["signals"]["enabled"])
    requested_symbols = [str(symbol).zfill(6) for symbol in (args.symbols or [])]
    if requested_symbols:
        spot = pd.DataFrame({"symbol": requested_symbols, "name": [""] * len(requested_symbols)})
    else:
        spot = load_default_universe(
            universe_config,
            max_symbols=args.max_symbols,
            ingest_config=ingest_config,
        )

    signal_key = "_".join(enabled_signals) if len(enabled_signals) <= 3 else "multi_signal"
    charts_dir = PROJECT_ROOT / "reports" / "validation" / signal_key
    charts_dir.mkdir(parents=True, exist_ok=True)
    start_ts = pd.Timestamp(args.start_date)

    price_map: dict[str, pd.DataFrame] = {}
    for _, row in spot.iterrows():
        symbol = str(row["symbol"])
        price_map[symbol] = fetch_a_share_history(symbol, ingest_config)

    derived_breadth_by_date = _build_daily_breadth(price_map)
    if ingest_config.source == "sqlite":
        breadth_by_date = derived_breadth_by_date
        breadth_by_date.update(load_sqlite_breadth_history(ingest_config))
    else:
        breadth_by_date = derived_breadth_by_date
    index_histories = {
        name: fetch_index_history(code, market_filter)
        for name, code in market_filter.index_symbols.items()
    }
    sector_snapshot = load_sector_snapshot(sector_filter)
    theme_map = sector_snapshot.get("symbol_theme_map", {})
    selected_symbols = {str(symbol) for symbol in spot["symbol"].astype(str)}
    unique_industries = {
        str(payload.get("industry_name", "") or "")
        for symbol, payload in theme_map.items()
        if str(symbol) in selected_symbols and isinstance(payload, dict) and payload.get("industry_name")
    }
    unique_concepts = {
        str(concept)
        for symbol, payload in theme_map.items()
        if str(symbol) in selected_symbols and isinstance(payload, dict)
        for concept in payload.get("concept_names", []) or []
    }
    board_histories: dict[tuple[str, str], pd.DataFrame] = {}
    board_cache_dir = PROJECT_ROOT / "data" / "cache" / "sector_history"
    start_date = args.start_date.replace("-", "")
    end_date = args.end_date.replace("-", "")
    for board_name in sorted(unique_industries):
        board_histories[("industry", board_name)] = _fetch_board_history(
            kind="industry",
            board_name=board_name,
            start_date=start_date,
            end_date=end_date,
            cache_dir=board_cache_dir,
        )
    for board_name in sorted(unique_concepts):
        board_histories[("concept", board_name)] = _fetch_board_history(
            kind="concept",
            board_name=board_name,
            start_date=start_date,
            end_date=end_date,
            cache_dir=board_cache_dir,
        )

    all_signals = []
    detail_rows: list[dict[str, object]] = []
    for _, row in spot.iterrows():
        symbol = str(row["symbol"])
        history = price_map[symbol]
        signals = scan_signal_history(
            history,
            symbol=symbol,
            enabled_signals=enabled_signals,
            thresholds=thresholds,
            include_invalid=False,
            step=args.step,
        )
        for signal in signals:
            signal_date = pd.Timestamp(signal.signal_date)
            if signal_date < start_ts:
                continue
            breadth_frame = breadth_by_date.get(signal_date)
            if breadth_frame is None or breadth_frame.empty:
                continue
            market_snapshot = score_market_snapshot(
                {
                    name: history_frame[history_frame["date"] <= signal_date]
                    for name, history_frame in index_histories.items()
                },
                breadth_frame,
                market_filter,
            )
            theme_payload = _build_historical_theme_payload(
                symbol,
                signal_date,
                theme_map,
                board_histories,
                sector_filter,
            )
            sector_band = _sector_band(theme_payload, sector_filter)
            filter_ok = _filter_ok(market_snapshot, theme_payload, sector_filter, signal.signal_type)
            quality_score = score_signal_quality(signal)
            quality_bucket = quality_bucket_label(
                quality_score,
                pretty_min_quality_score=pretty_min_quality_score,
                pretty_hard_filter_score=pretty_hard_filter_score,
            )
            pretty_ok = pretty_signal_ok(
                quality_score,
                pretty_hard_filter_score=pretty_hard_filter_score,
            )
            payload = signal.to_dict()
            payload["name"] = row.get("name", "")
            payload["market_score"] = market_snapshot["market_score"]
            payload["market_regime"] = market_snapshot["market_regime"]
            payload["market_ok"] = market_snapshot["market_ok"]
            payload["sector_score"] = theme_payload["sector_score"]
            payload["sector_ok"] = theme_payload["sector_ok"]
            payload["sector_band"] = sector_band
            payload["filter_ok"] = filter_ok
            payload["quality_score"] = quality_score
            payload["quality_bucket"] = quality_bucket
            payload["pretty_ok"] = pretty_ok
            payload["industry_name"] = theme_payload["industry_name"]
            payload["concept_names"] = theme_payload["concept_names"]
            payload["volume_ratio_factor"] = signal.factors.get("volume_ratio")
            payload["breakout_pct_factor"] = signal.factors.get("breakout_pct")
            payload["range_atr_ratio_factor"] = signal.factors.get("range_atr_ratio")
            payload["drawdown_factor"] = signal.factors.get("drawdown_from_high_60")
            all_signals.append(payload)
            detail_rows.append(payload)
            if len(all_signals) <= args.limit_charts:
                plot_signal_context(history, signal, charts_dir / f"{symbol}_{signal.signal_date.isoformat()}_{signal.signal_type}.png")

    signal_objects = []
    for payload in detail_rows:
        signal_objects.append(
            ResearchSignal(
                signal_type=str(payload["signal_type"]),
                symbol=str(payload["symbol"]),
                signal_date=pd.Timestamp(payload["signal_date"]).date(),
                confidence_score=float(payload["confidence_score"]),
                trend_ok=bool(payload["trend_ok"]),
                location_ok=bool(payload["location_ok"]),
                pattern_ok=bool(payload["pattern_ok"]),
                volume_ok=bool(payload["volume_ok"]),
                entry_price=float(payload["entry_price"]) if payload.get("entry_price") is not None and not pd.isna(payload["entry_price"]) else None,
                stop_price=float(payload["stop_price"]) if payload.get("stop_price") is not None and not pd.isna(payload["stop_price"]) else None,
                target_price=float(payload["target_price"]) if payload.get("target_price") is not None and not pd.isna(payload["target_price"]) else None,
                invalid_reason=payload.get("invalid_reason"),
                risk_tags=list(payload.get("risk_tags", []) or []),
                factors=dict(payload.get("factors", {}) or {}),
            )
        )

    detail_frame = pd.DataFrame(detail_rows)
    trades = run_signal_backtest(price_map, signal_objects, backtest_config)
    trade_frame = pd.DataFrame([trade.to_dict() for trade in trades])
    if not trade_frame.empty:
        trade_frame["signal_date"] = trade_frame["signal_date"].astype(str)
        merged = detail_frame.merge(
            trade_frame,
            on=["symbol", "signal_type", "signal_date"],
            how="left",
            suffixes=("", "_trade"),
        )
    else:
        merged = detail_frame.copy()

    if not merged.empty:
        merged["factor_confidence_bucket"] = _factor_bucket_label(merged["confidence_score"])
        merged["factor_volume_ratio_bucket"] = _factor_bucket_label(merged["volume_ratio_factor"])
        merged["factor_breakout_pct_bucket"] = _factor_bucket_label(merged["breakout_pct_factor"])
        merged["factor_sector_score_bucket"] = _factor_bucket_label(merged["sector_score"])
        merged["factor_market_score_bucket"] = _factor_bucket_label(merged["market_score"])
        merged["factor_quality_score_bucket"] = _factor_bucket_label(merged["quality_score"])

    detail_path = charts_dir / "validation_details.csv"
    trade_path = charts_dir / "trades.csv"
    signal_summary_path = charts_dir / "signal_type_summary.csv"
    context_summary_path = charts_dir / "context_summary.csv"
    factor_summary_path = charts_dir / "factor_bucket_summary.csv"

    merged.to_csv(detail_path, index=False)
    trade_frame.to_csv(trade_path, index=False)

    executed_frame = merged.dropna(subset=["return_pct"]) if "return_pct" in merged.columns else pd.DataFrame()
    signal_summary = _summarize_groups(executed_frame, ["signal_type"]) if not executed_frame.empty else pd.DataFrame()
    if not signal_summary.empty and not detail_frame.empty:
        signal_counts = detail_frame.groupby("signal_type", dropna=False).size().reset_index(name="signal_count")
        signal_summary = signal_counts.merge(signal_summary, on="signal_type", how="left")
    signal_summary.to_csv(signal_summary_path, index=False)

    context_summary = (
        _summarize_groups(executed_frame, ["signal_type", "market_regime", "sector_band"])
        if not executed_frame.empty
        else pd.DataFrame()
    )
    context_summary.to_csv(context_summary_path, index=False)

    factor_frames = []
    if not executed_frame.empty:
        for factor_column in [
            "factor_confidence_bucket",
            "factor_volume_ratio_bucket",
            "factor_breakout_pct_bucket",
            "factor_sector_score_bucket",
            "factor_market_score_bucket",
            "factor_quality_score_bucket",
        ]:
            if factor_column not in executed_frame.columns:
                continue
            factor_frame = _summarize_groups(executed_frame, [factor_column])
            if factor_frame.empty:
                continue
            factor_frame.insert(0, "factor", factor_column.replace("factor_", "").replace("_bucket", ""))
            factor_frames.append(factor_frame.rename(columns={factor_column: "bucket"}))
    factor_summary = pd.concat(factor_frames, ignore_index=True) if factor_frames else pd.DataFrame()
    factor_summary.to_csv(factor_summary_path, index=False)

    print(f"signals={','.join(enabled_signals)}")
    print(f"signal_count={len(detail_rows)}")
    print(f"trade_count={len(trades)}")
    print(f"detail={detail_path}")
    print(f"trades={trade_path}")
    print(f"signal_summary={signal_summary_path}")
    print(f"context_summary={context_summary_path}")
    print(f"factor_summary={factor_summary_path}")


if __name__ == "__main__":
    main()
