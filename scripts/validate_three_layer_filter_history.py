from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import yaml

from app.backtest.engine import BacktestConfig, run_signal_backtest
from app.backtest.metrics import summarize_trades
from app.data.ingest import DataIngestConfig, fetch_a_share_history, load_sqlite_breadth_history
from app.data.market_context import MarketFilterConfig, fetch_index_history, score_market_snapshot
from app.data.sector_context import SectorFilterConfig, load_sector_snapshot
from app.data.universe import UniverseConfig
from app.strategy.rules import RuleThresholds, scan_signal_history
from app.strategy.scanner import _filter_ok, _sector_band, load_default_universe


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOARD_HISTORY_FETCH_DISABLED = False


def _import_akshare():
    try:
        import akshare as ak  # type: ignore
    except ImportError as exc:
        raise RuntimeError("缺少 akshare，无法加载历史板块数据。") from exc
    return ak


def _safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", value)


def _load_config() -> dict:
    return yaml.safe_load((PROJECT_ROOT / "config" / "strategy_13_points.yaml").read_text(encoding="utf-8"))


def _resolve_research_universe_config(config: dict) -> UniverseConfig:
    universe_section = config.get("research_universe") or config["universe"]
    return UniverseConfig(**universe_section)


def _normalize_board_history(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "涨跌幅": "pct_chg",
            "换手率": "turnover_rate",
        }
    ).copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    for column in ("open", "close", "high", "low", "volume", "amount", "pct_chg", "turnover_rate"):
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.sort_values("date").reset_index(drop=True)
    normalized["ret_5"] = normalized["close"].pct_change(5)
    normalized["ret_20"] = normalized["close"].pct_change(20)
    normalized["amount_ratio_20"] = normalized["amount"] / normalized["amount"].rolling(20).mean()
    normalized["hist_score"] = (
        normalized["ret_5"].clip(lower=0, upper=0.12).fillna(0) / 0.12 * 35
        + normalized["ret_20"].clip(lower=0, upper=0.2).fillna(0) / 0.2 * 30
        + normalized["amount_ratio_20"].clip(lower=0, upper=2).fillna(0) / 2 * 20
        + normalized["turnover_rate"].clip(lower=0, upper=8).fillna(0) / 8 * 15
    ).round(2)
    return normalized


def _board_cache_path(kind: str, board_name: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{kind}_{_safe_name(board_name)}.parquet"


def _fetch_board_history(
    *,
    kind: str,
    board_name: str,
    start_date: str,
    end_date: str,
    cache_dir: Path,
) -> pd.DataFrame:
    global BOARD_HISTORY_FETCH_DISABLED
    cache_file = _board_cache_path(kind, board_name, cache_dir)
    if cache_file.exists():
        return pd.read_parquet(cache_file)
    if BOARD_HISTORY_FETCH_DISABLED:
        return pd.DataFrame(columns=["date", "hist_score"])

    ak = _import_akshare()
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            if kind == "industry":
                raw = ak.stock_board_industry_hist_em(symbol=board_name, start_date=start_date, end_date=end_date)
            else:
                raw = ak.stock_board_concept_hist_em(symbol=board_name, start_date=start_date, end_date=end_date)
            normalized = _normalize_board_history(raw)
            normalized.to_parquet(cache_file, index=False)
            return normalized
        except Exception as exc:  # pragma: no cover - depends on flaky network
            last_error = exc
            time.sleep(1 + attempt)
    BOARD_HISTORY_FETCH_DISABLED = True
    print(f"warning: 无法获取 {kind} 板块历史 {board_name}: {last_error}")
    return pd.DataFrame(columns=["date", "hist_score"])


def _latest_board_score(history: pd.DataFrame | None, signal_date: pd.Timestamp) -> float:
    if history is None or history.empty:
        return 0.0
    eligible = history[history["date"] <= signal_date]
    if eligible.empty:
        return 0.0
    latest = eligible.iloc[-1]
    return float(latest.get("hist_score", 0.0) or 0.0)


def _build_historical_theme_payload(
    symbol: str,
    signal_date: pd.Timestamp,
    theme_map: dict[str, dict[str, object]],
    board_histories: dict[tuple[str, str], pd.DataFrame],
    config: SectorFilterConfig,
) -> dict[str, object]:
    raw = dict(theme_map.get(symbol, {}))
    industry_name = str(raw.get("industry_name", "") or "")
    concept_names = list(raw.get("concept_names", []) or [])
    industry_score = _latest_board_score(board_histories.get(("industry", industry_name)), signal_date) if industry_name else 0.0
    concept_scores = [
        _latest_board_score(board_histories.get(("concept", str(name))), signal_date)
        for name in concept_names
    ]
    top_concept_score = max(concept_scores) if concept_scores else 0.0
    avg_concept_score = sum(concept_scores) / len(concept_scores) if concept_scores else 0.0
    if industry_score > 0 and top_concept_score > 0:
        blended_score = industry_score * 0.65 + top_concept_score * 0.35
    else:
        blended_score = max(industry_score, top_concept_score, avg_concept_score)
    sector_score = round(max(industry_score, top_concept_score, avg_concept_score, blended_score), 2)
    return {
        "industry_name": industry_name,
        "industry_score": round(industry_score, 2),
        "concept_names": concept_names,
        "concept_scores": [round(item, 2) for item in concept_scores],
        "sector_score": sector_score,
        "sector_ok": bool(sector_score >= config.min_sector_score),
    }


def _build_daily_breadth(price_map: dict[str, pd.DataFrame]) -> dict[pd.Timestamp, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    for symbol, history in price_map.items():
        frame = history[["date", "close"]].copy().sort_values("date")
        frame["pct_chg"] = frame["close"].pct_change() * 100
        frame["symbol"] = symbol
        rows.append(frame[["date", "pct_chg", "symbol"]])
    combined = pd.concat(rows, ignore_index=True)
    combined = combined.dropna(subset=["pct_chg"])
    return {date: group[["pct_chg"]].reset_index(drop=True) for date, group in combined.groupby("date")}


def _summarize_group(name: str, trades) -> dict[str, float | str]:
    summary = summarize_trades(trades)
    return {
        "group": name,
        "trade_count": summary["trade_count"],
        "win_rate": summary["win_rate"],
        "avg_return_pct": summary["avg_return_pct"],
        "profit_factor": summary["profit_factor"],
        "max_drawdown": summary["max_drawdown"],
        "avg_hold_days": summary["avg_hold_days"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="三层滤网历史滚动验证")
    parser.add_argument("--max-symbols", type=int, default=20, help="验证股票数量")
    parser.add_argument("--step", type=int, default=10, help="历史扫描步长")
    parser.add_argument("--start-date", default="2024-01-01", help="验证起始日期")
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y-%m-%d"), help="验证结束日期")
    args = parser.parse_args()

    config = _load_config()
    thresholds = RuleThresholds(**config["thresholds"])
    universe_config = _resolve_research_universe_config(config)
    backtest_config = BacktestConfig(**config["backtest"])
    market_filter = MarketFilterConfig(
        cache_dir=PROJECT_ROOT / "data" / "cache" / "market",
        **config.get("market_filter", {}),
    )
    sector_filter = SectorFilterConfig(
        cache_dir=PROJECT_ROOT / "data" / "cache" / "sector",
        **config.get("sector_filter", {}),
    )
    ingest_config = DataIngestConfig(
        cache_dir=PROJECT_ROOT / "data" / "cache",
        start_date=args.start_date,
        end_date=args.end_date,
        **config.get("ingest", {}),
    )

    universe = load_default_universe(
        universe_config,
        max_symbols=args.max_symbols,
        ingest_config=ingest_config,
    )
    price_map: dict[str, pd.DataFrame] = {}
    for symbol in universe["symbol"].astype(str):
        price_map[symbol] = fetch_a_share_history(symbol, ingest_config)

    if ingest_config.source == "sqlite":
        breadth_by_date = load_sqlite_breadth_history(ingest_config)
    else:
        breadth_by_date = _build_daily_breadth(price_map)
    index_histories = {
        name: fetch_index_history(code, market_filter)
        for name, code in market_filter.index_symbols.items()
    }

    sector_snapshot = load_sector_snapshot(sector_filter)
    theme_map = sector_snapshot.get("symbol_theme_map", {})
    selected_symbols = {str(symbol) for symbol in universe["symbol"].astype(str)}
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

    detailed_rows: list[dict[str, object]] = []
    all_signals = []
    grouped_signals: dict[str, list] = {
        "filter_ok_true": [],
        "filter_ok_false": [],
        "crowded": [],
        "strong": [],
        "edge_high": [],
        "edge_low": [],
        "weak": [],
        "none": [],
    }

    start_ts = pd.Timestamp(args.start_date)
    for symbol, history in price_map.items():
        history_signals = scan_signal_history(
            history,
            symbol=symbol,
            enabled_signals=config["signals"]["enabled"],
            thresholds=thresholds,
            include_invalid=False,
            step=args.step,
        )
        for signal in history_signals:
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
            theme_payload = _build_historical_theme_payload(symbol, signal_date, theme_map, board_histories, sector_filter)
            sector_band = _sector_band(theme_payload, sector_filter)
            filter_ok = _filter_ok(
                market_snapshot,
                theme_payload,
                sector_filter,
                signal.signal_type,
            )

            signal.factors.update(
                {
                    "market_score": market_snapshot["market_score"],
                    "market_regime": market_snapshot["market_regime"],
                    "market_ok": market_snapshot["market_ok"],
                    "sector_score": theme_payload["sector_score"],
                    "sector_ok": theme_payload["sector_ok"],
                    "sector_band": sector_band,
                    "filter_ok": filter_ok,
                }
            )
            detailed_rows.append(
                {
                    "symbol": symbol,
                    "signal_type": signal.signal_type,
                    "signal_date": signal.signal_date.isoformat(),
                    "market_score": market_snapshot["market_score"],
                    "market_regime": market_snapshot["market_regime"],
                    "sector_score": theme_payload["sector_score"],
                    "sector_ok": theme_payload["sector_ok"],
                    "sector_band": sector_band,
                    "filter_ok": filter_ok,
                    "industry_name": theme_payload["industry_name"],
                    "concept_names": theme_payload["concept_names"],
                }
            )
            all_signals.append(signal)
            grouped_signals["filter_ok_true" if filter_ok else "filter_ok_false"].append(signal)
            grouped_signals[sector_band].append(signal)

    summary_rows = [_summarize_group("all", run_signal_backtest(price_map, all_signals, backtest_config))]
    for group_name, signals in grouped_signals.items():
        summary_rows.append(_summarize_group(group_name, run_signal_backtest(price_map, signals, backtest_config)))

    output_dir = PROJECT_ROOT / "reports" / "validation" / "three_layer_filter"
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "rolling_validation_details.csv"
    summary_path = output_dir / "rolling_validation_summary.csv"
    pd.DataFrame(detailed_rows).to_csv(detail_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"symbols={len(price_map)}")
    print(f"signals={len(all_signals)}")
    print(f"detail={detail_path}")
    print(f"summary={summary_path}")
    if summary_rows:
        print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
