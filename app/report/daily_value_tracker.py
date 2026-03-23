from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from contextlib import closing

import pandas as pd

from app.data.ingest import (
    DataIngestConfig,
    normalize_ohlcv,
    resolve_sqlite_db_path,
    sqlite_price_table,
    sqlite_table_columns,
)


SNAPSHOT_KEY_COLUMNS = ["symbol", "signal_type", "signal_date"]
FORWARD_HORIZONS = (1, 3, 5, 10)
SCOREBOARD_WINDOWS = (20, 60, 120)


@dataclass(slots=True)
class ValueTrackerArtifacts:
    snapshot_today: pd.DataFrame
    snapshot_history: pd.DataFrame
    forward_frame: pd.DataFrame
    scoreboard: pd.DataFrame
    today_expectancy: pd.DataFrame


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _snapshot_layer(row: pd.Series) -> str:
    pretty_ok = bool(row.get("pretty_ok", False))
    filter_ok = bool(row.get("filter_ok", False))
    if pretty_ok and filter_ok:
        return "executable"
    if pretty_ok or filter_ok:
        return "candidate"
    return "watch"


def build_daily_signal_snapshot(
    scan_results: pd.DataFrame,
    top_rows: pd.DataFrame,
    *,
    run_id: str,
    generated_at: str,
    universe_scope: str,
    latest_trade_date: str,
) -> pd.DataFrame:
    if scan_results.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "generated_at",
                "trade_date",
                "universe_scope",
                "symbol",
                "name",
                "signal_date",
                "signal_type",
                "signal_name",
                "layer",
                "is_top_candidate",
                "candidate_rank",
                "score",
                "base_score",
                "quality_score",
                "quality_bucket",
                "confidence_score",
                "pretty_ok",
                "filter_ok",
                "market_ok",
                "sector_ok",
                "market_regime",
                "sector_band",
            ]
        )

    snapshot = scan_results.copy()
    snapshot["run_id"] = run_id
    snapshot["generated_at"] = generated_at
    snapshot["trade_date"] = latest_trade_date
    snapshot["universe_scope"] = universe_scope
    snapshot["layer"] = snapshot.apply(_snapshot_layer, axis=1)
    snapshot["is_top_candidate"] = False
    snapshot["candidate_rank"] = pd.NA

    if not top_rows.empty:
        top_lookup: dict[tuple[str, str, str], tuple[bool, int]] = {}
        for idx, (_, row) in enumerate(top_rows.iterrows(), start=1):
            top_lookup[(str(row["symbol"]), str(row["signal_type"]), str(row["signal_date"]))] = (True, idx)

        def _mark_top(row: pd.Series) -> tuple[bool, object]:
            return top_lookup.get((str(row["symbol"]), str(row["signal_type"]), str(row["signal_date"])), (False, pd.NA))

        top_payload = snapshot.apply(_mark_top, axis=1)
        snapshot["is_top_candidate"] = top_payload.map(lambda item: item[0])
        snapshot["candidate_rank"] = top_payload.map(lambda item: item[1])

    columns = [
        "run_id",
        "generated_at",
        "trade_date",
        "universe_scope",
        "symbol",
        "name",
        "signal_date",
        "signal_type",
        "signal_name",
        "layer",
        "is_top_candidate",
        "candidate_rank",
        "score",
        "base_score",
        "quality_score",
        "quality_bucket",
        "confidence_score",
        "pretty_ok",
        "filter_ok",
        "market_ok",
        "sector_ok",
        "market_regime",
        "sector_band",
    ]
    for column in columns:
        if column not in snapshot.columns:
            snapshot[column] = pd.NA
    snapshot = snapshot[columns].copy()
    snapshot["trade_date"] = snapshot["trade_date"].astype(str)
    snapshot["signal_date"] = snapshot["signal_date"].astype(str)
    return snapshot.sort_values(["layer", "score", "quality_score"], ascending=[True, False, False]).reset_index(drop=True)


def upsert_snapshot_history(snapshot_today: pd.DataFrame, history_path: Path) -> pd.DataFrame:
    existing = _read_csv(history_path)
    merged = pd.concat([existing, snapshot_today], ignore_index=True)
    if merged.empty:
        return merged
    for column in ("trade_date", "signal_date", "generated_at"):
        if column in merged.columns:
            merged[column] = merged[column].astype(str)
    merged = merged.drop_duplicates(subset=SNAPSHOT_KEY_COLUMNS, keep="last")
    merged = merged.sort_values(["signal_date", "symbol", "signal_type"]).reset_index(drop=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(history_path, index=False)
    return merged


def _load_price_history_for_symbols(
    symbols: list[str],
    ingest_config: DataIngestConfig,
    *,
    start_date: str,
) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])

    db_path = resolve_sqlite_db_path(ingest_config)
    normalized_symbols = sorted({str(symbol).zfill(6) for symbol in symbols})
    frames: list[pd.DataFrame] = []
    chunk_size = 500

    with closing(sqlite3.connect(db_path)) as conn:
        table_name = sqlite_price_table(conn)
        if table_name == "kline_data":
            for offset in range(0, len(normalized_symbols), chunk_size):
                chunk = normalized_symbols[offset : offset + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                query = f"""
                    SELECT code, date, open, high, low, close, volume, amount, pct_chg, turnover
                    FROM kline_data
                    WHERE code IN ({placeholders})
                      AND date >= ?
                    ORDER BY code, date
                """
                frames.append(pd.read_sql_query(query, conn, params=[*chunk, start_date]))
        else:
            columns = sqlite_table_columns(conn, table_name)
            use_adjusted = ingest_config.adjust == "hfq" and {
                "open_adj",
                "high_adj",
                "low_adj",
                "close_adj",
            }.issubset(columns)
            open_column = "COALESCE(open_adj, open)" if use_adjusted else "open"
            high_column = "COALESCE(high_adj, high)" if use_adjusted else "high"
            low_column = "COALESCE(low_adj, low)" if use_adjusted else "low"
            close_column = "COALESCE(close_adj, close)" if use_adjusted else "close"
            turnover_column = "turn" if "turn" in columns else "turnover"
            for offset in range(0, len(normalized_symbols), chunk_size):
                chunk = normalized_symbols[offset : offset + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                query = f"""
                    SELECT
                        code,
                        date,
                        {open_column} AS open,
                        {high_column} AS high,
                        {low_column} AS low,
                        {close_column} AS close,
                        volume,
                        amount,
                        pct_chg,
                        {turnover_column} AS turnover
                    FROM {table_name}
                    WHERE code IN ({placeholders})
                      AND date >= ?
                    ORDER BY code, date
                """
                frames.append(pd.read_sql_query(query, conn, params=[*chunk, start_date]))

    if not frames:
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    return normalize_ohlcv(pd.concat(frames, ignore_index=True))


def compute_forward_returns(
    snapshot_history: pd.DataFrame,
    ingest_config: DataIngestConfig,
    *,
    horizons: tuple[int, ...] = FORWARD_HORIZONS,
) -> pd.DataFrame:
    if snapshot_history.empty:
        return snapshot_history.copy()

    result = snapshot_history.copy()
    result["signal_date"] = pd.to_datetime(result["signal_date"], errors="coerce")
    min_signal_date = result["signal_date"].min()
    if pd.isna(min_signal_date):
        return result

    price_frame = _load_price_history_for_symbols(
        result["symbol"].astype(str).tolist(),
        ingest_config,
        start_date=(min_signal_date - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
    )
    if price_frame.empty:
        return result

    for horizon in horizons:
        result[f"return_{horizon}d"] = pd.NA
        result[f"mfe_{horizon}d"] = pd.NA
        result[f"mae_{horizon}d"] = pd.NA
    result["entry_date"] = pd.NA
    result["entry_open"] = pd.NA

    price_map = {symbol: frame.reset_index(drop=True) for symbol, frame in price_frame.groupby("symbol", sort=False)}
    row_indexes: list[int] = []
    forward_rows: list[dict[str, object]] = []

    for idx, row in result.iterrows():
        symbol = str(row["symbol"])
        signal_date = pd.Timestamp(row["signal_date"])
        history = price_map.get(symbol)
        if history is None or history.empty:
            continue
        matched = history.index[history["date"] == signal_date].tolist()
        if not matched:
            continue
        signal_index = matched[0]
        entry_index = signal_index + 1
        if entry_index >= len(history):
            continue
        entry_bar = history.iloc[entry_index]
        entry_open = float(entry_bar["open"])
        payload: dict[str, object] = {
            "entry_date": entry_bar["date"].date().isoformat(),
            "entry_open": round(entry_open, 4),
        }
        for horizon in horizons:
            end_index = entry_index + horizon - 1
            if end_index >= len(history):
                payload[f"return_{horizon}d"] = pd.NA
                payload[f"mfe_{horizon}d"] = pd.NA
                payload[f"mae_{horizon}d"] = pd.NA
                continue
            lookahead = history.iloc[entry_index : end_index + 1]
            end_bar = lookahead.iloc[-1]
            payload[f"return_{horizon}d"] = round((float(end_bar["close"]) - entry_open) / entry_open, 4)
            payload[f"mfe_{horizon}d"] = round((float(lookahead["high"].max()) - entry_open) / entry_open, 4)
            payload[f"mae_{horizon}d"] = round((float(lookahead["low"].min()) - entry_open) / entry_open, 4)
        row_indexes.append(idx)
        forward_rows.append(payload)

    if not row_indexes:
        return result

    forward_frame = pd.DataFrame(forward_rows, index=row_indexes)
    for column in forward_frame.columns:
        result.loc[forward_frame.index, column] = forward_frame[column]
    result["signal_date"] = result["signal_date"].dt.date.astype(str)
    return result


def build_strategy_scoreboard(
    forward_frame: pd.DataFrame,
    *,
    windows: tuple[int, ...] = SCOREBOARD_WINDOWS,
    horizons: tuple[int, ...] = FORWARD_HORIZONS,
) -> pd.DataFrame:
    if forward_frame.empty:
        return pd.DataFrame()

    frame = forward_frame.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce")
    unique_dates = sorted(frame["signal_date"].dropna().dt.date.unique())
    if not unique_dates:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for window in windows:
        selected_dates = set(unique_dates[-window:])
        scoped = frame[frame["signal_date"].dt.date.isin(selected_dates)].copy()
        if scoped.empty:
            continue
        for keys, group in scoped.groupby(["layer", "signal_type"], dropna=False):
            layer, signal_type = keys
            row: dict[str, object] = {
                "window_days": int(window),
                "layer": str(layer),
                "signal_type": str(signal_type),
                "signal_count": int(len(group)),
            }
            quality_score = pd.to_numeric(group.get("quality_score"), errors="coerce")
            if not quality_score.dropna().empty:
                row["avg_quality_score"] = round(float(quality_score.mean()), 2)
            for horizon in horizons:
                series = pd.to_numeric(group.get(f"return_{horizon}d"), errors="coerce").dropna()
                if series.empty:
                    row[f"trade_count_{horizon}d"] = 0
                    row[f"win_rate_{horizon}d"] = 0.0
                    row[f"avg_return_{horizon}d"] = 0.0
                    row[f"median_return_{horizon}d"] = 0.0
                else:
                    row[f"trade_count_{horizon}d"] = int(len(series))
                    row[f"win_rate_{horizon}d"] = round(float((series > 0).mean()), 4)
                    row[f"avg_return_{horizon}d"] = round(float(series.mean()), 4)
                    row[f"median_return_{horizon}d"] = round(float(series.median()), 4)
                mfe_series = pd.to_numeric(group.get(f"mfe_{horizon}d"), errors="coerce").dropna()
                mae_series = pd.to_numeric(group.get(f"mae_{horizon}d"), errors="coerce").dropna()
                row[f"avg_mfe_{horizon}d"] = round(float(mfe_series.mean()), 4) if not mfe_series.empty else 0.0
                row[f"avg_mae_{horizon}d"] = round(float(mae_series.mean()), 4) if not mae_series.empty else 0.0
            rows.append(row)

    scoreboard = pd.DataFrame(rows)
    if scoreboard.empty:
        return scoreboard
    return scoreboard.sort_values(
        ["window_days", "layer", "avg_return_5d", "win_rate_5d", "signal_count"],
        ascending=[True, True, False, False, False],
    ).reset_index(drop=True)


def build_today_expectancy(
    snapshot_today: pd.DataFrame,
    scoreboard: pd.DataFrame,
    *,
    reference_window: int = 60,
) -> pd.DataFrame:
    if snapshot_today.empty:
        return snapshot_today.copy()

    today = snapshot_today.copy()
    today["candidate_rank"] = pd.to_numeric(today["candidate_rank"], errors="coerce")
    scoped = scoreboard[scoreboard["window_days"] == reference_window].copy()
    if scoped.empty:
        today["expected_win_rate_5d"] = pd.NA
        today["expected_avg_return_5d"] = pd.NA
        today["expected_signal_count_5d"] = pd.NA
        return today

    scoped = scoped.rename(
        columns={
            "win_rate_5d": "expected_win_rate_5d",
            "avg_return_5d": "expected_avg_return_5d",
            "trade_count_5d": "expected_signal_count_5d",
        }
    )
    keep_columns = [
        "layer",
        "signal_type",
        "expected_win_rate_5d",
        "expected_avg_return_5d",
        "expected_signal_count_5d",
    ]
    merged = today.merge(scoped[keep_columns], on=["layer", "signal_type"], how="left")
    return merged.sort_values(
        ["is_top_candidate", "expected_avg_return_5d", "quality_score", "score"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def build_value_report(
    today_expectancy: pd.DataFrame,
    scoreboard: pd.DataFrame,
    *,
    latest_trade_date: str,
) -> str:
    lines = [
        "# 每日价值验证看板",
        "",
        f"- 交易日：{latest_trade_date}",
        f"- 今日信号数：{len(today_expectancy)}",
        "",
    ]

    if today_expectancy.empty:
        lines.extend(["## 今日结论", "- 今日无原始有效信号。", ""])
    else:
        lines.extend(["## 今日最值得看的信号", ""])
        interesting = today_expectancy.copy()
        interesting = interesting[
            [
                "symbol",
                "signal_type",
                "layer",
                "score",
                "quality_score",
                "expected_avg_return_5d",
                "expected_win_rate_5d",
                "expected_signal_count_5d",
            ]
        ].head(12)
        for _, row in interesting.iterrows():
            expected_return = row.get("expected_avg_return_5d")
            expected_win_rate = row.get("expected_win_rate_5d")
            sample_count = row.get("expected_signal_count_5d")
            lines.append(
                f"- {row['symbol']} {row['signal_type']} layer={row['layer']} "
                f"score={row['score']} quality={row['quality_score']} "
                f"exp5d={expected_return if pd.notna(expected_return) else 'NA'} "
                f"win5d={expected_win_rate if pd.notna(expected_win_rate) else 'NA'} "
                f"n={sample_count if pd.notna(sample_count) else 'NA'}"
            )
        lines.append("")

    if scoreboard.empty:
        lines.extend(["## 历史统计", "- 暂无可用的历史前瞻收益样本。", ""])
        return "\n".join(lines)

    lines.append("## 策略统计")
    for window in SCOREBOARD_WINDOWS:
        scoped = scoreboard[scoreboard["window_days"] == window].copy()
        if scoped.empty:
            continue
        lines.append("")
        lines.append(f"### 最近 {window} 个交易日")
        for layer in ("executable", "candidate", "watch"):
            layer_frame = scoped[scoped["layer"] == layer].head(8)
            if layer_frame.empty:
                continue
            lines.append(f"- {layer}")
            for _, row in layer_frame.iterrows():
                lines.append(
                    f"  - {row['signal_type']}: n={row['signal_count']} "
                    f"win5d={row.get('win_rate_5d', 0.0)} "
                    f"avg5d={row.get('avg_return_5d', 0.0)} "
                    f"mfe5d={row.get('avg_mfe_5d', 0.0)} "
                    f"mae5d={row.get('avg_mae_5d', 0.0)}"
                )
    return "\n".join(lines)


def build_value_tracker_artifacts(
    scan_results: pd.DataFrame,
    top_rows: pd.DataFrame,
    *,
    run_id: str,
    generated_at: str,
    universe_scope: str,
    latest_trade_date: str,
    ingest_config: DataIngestConfig,
    history_dir: Path,
) -> ValueTrackerArtifacts:
    history_dir.mkdir(parents=True, exist_ok=True)
    snapshot_today = build_daily_signal_snapshot(
        scan_results,
        top_rows,
        run_id=run_id,
        generated_at=generated_at,
        universe_scope=universe_scope,
        latest_trade_date=latest_trade_date,
    )
    snapshot_history_path = history_dir / "signal_snapshot_history.csv"
    snapshot_history = upsert_snapshot_history(snapshot_today, snapshot_history_path)
    forward_frame = compute_forward_returns(snapshot_history, ingest_config)
    forward_path = history_dir / "signal_forward_returns.csv"
    forward_frame.to_csv(forward_path, index=False)
    scoreboard = build_strategy_scoreboard(forward_frame)
    scoreboard_path = history_dir / "strategy_value_scoreboard.csv"
    scoreboard.to_csv(scoreboard_path, index=False)
    today_expectancy = build_today_expectancy(snapshot_today, scoreboard)
    return ValueTrackerArtifacts(
        snapshot_today=snapshot_today,
        snapshot_history=snapshot_history,
        forward_frame=forward_frame,
        scoreboard=scoreboard,
        today_expectancy=today_expectancy,
    )
