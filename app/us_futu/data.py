from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
import re

import pandas as pd
import yfinance as yf


@dataclass(slots=True)
class USDataConfig:
    source: str = "sqlite"
    sqlite_db_path: str = "~/us_stock_daily_data/us_stock_daily.db"
    price_table: str = "daily"
    index_table: str = "index_daily"
    cache_dir: Path = field(default_factory=lambda: Path("data/cache/us_futu"))
    daily_period: str = "2y"
    intraday_30m_period: str = "60d"
    intraday_60m_period: str = "720d"
    refresh_hours: int = 8
    adjust_price: bool = True


def resolve_us_sqlite_db_path(config: USDataConfig) -> Path:
    return Path(config.sqlite_db_path).expanduser().resolve()


def _normalize_local_history(frame: pd.DataFrame, *, adjust_price: bool) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "amount", "turn"])
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"])
    for column in ("open", "high", "low", "close", "adj_close", "volume", "amount", "turn"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if adjust_price and "adj_close" in df.columns:
        factor = df["adj_close"] / df["close"].replace(0, pd.NA)
        valid_factor = factor.notna() & (factor > 0)
        for column in ("open", "high", "low", "close"):
            if column in df.columns:
                df.loc[valid_factor, column] = df.loc[valid_factor, column] * factor.loc[valid_factor]
    keep_columns = [column for column in ["date", "open", "high", "low", "close", "volume", "amount", "turn"] if column in df.columns]
    return df[keep_columns].dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)


def fetch_us_history(symbol: str, config: USDataConfig | None = None, *, index: bool = False) -> pd.DataFrame:
    config = config or USDataConfig()
    if config.source != "sqlite":
        return download_us_history(symbol, "1d", config)
    db_path = resolve_us_sqlite_db_path(config)
    table_name = config.index_table if index else config.price_table
    query = f"""
        SELECT symbol, date, open, high, low, close, volume, adj_close, amount, turn
        FROM {table_name}
        WHERE symbol = ?
        ORDER BY date
    """
    with closing(sqlite3.connect(db_path)) as conn:
        frame = pd.read_sql_query(query, conn, params=[symbol])
    return _normalize_local_history(frame, adjust_price=config.adjust_price)


def load_us_universe_snapshot(config: USDataConfig | None = None) -> pd.DataFrame:
    config = config or USDataConfig()
    if config.source != "sqlite":
        raise RuntimeError("全市场快照当前只支持 sqlite 数据源。")
    db_path = resolve_us_sqlite_db_path(config)
    query = f"""
        WITH recent_dates AS (
            SELECT date
            FROM (
                SELECT DISTINCT date
                FROM {config.price_table}
                ORDER BY date DESC
                LIMIT 20
            )
        ),
        recent_rows AS (
            SELECT symbol, date, close, volume, amount
            FROM {config.price_table}
            WHERE date IN (SELECT date FROM recent_dates)
        ),
        latest_date AS (
            SELECT MAX(date) AS max_date FROM {config.price_table}
        )
        SELECT
            r.symbol,
            MAX(CASE WHEN r.date = (SELECT max_date FROM latest_date) THEN r.close END) AS close,
            MAX(CASE WHEN r.date = (SELECT max_date FROM latest_date) THEN r.volume END) AS volume,
            MAX(CASE WHEN r.date = (SELECT max_date FROM latest_date) THEN r.amount END) AS amount,
            AVG(r.volume) AS avg_volume_20,
            AVG(r.amount) AS avg_amount_20
        FROM recent_rows r
        GROUP BY r.symbol
    """
    with closing(sqlite3.connect(db_path)) as conn:
        frame = pd.read_sql_query(query, conn)
    for column in ("close", "volume", "amount", "avg_volume_20", "avg_amount_20"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str)
    return frame


def load_index_snapshot(config: USDataConfig | None = None, index_symbols: list[str] | None = None) -> pd.DataFrame:
    config = config or USDataConfig()
    if config.source != "sqlite":
        raise RuntimeError("指数快照当前只支持 sqlite 数据源。")
    index_symbols = index_symbols or []
    db_path = resolve_us_sqlite_db_path(config)
    params: list[str] = []
    where_sql = ""
    if index_symbols:
        placeholders = ",".join("?" for _ in index_symbols)
        where_sql = f"WHERE symbol IN ({placeholders})"
        params = list(index_symbols)
    query = f"""
        SELECT symbol, date, open, high, low, close, volume, adj_close, amount, turn
        FROM {config.index_table}
        {where_sql}
        ORDER BY symbol, date
    """
    with closing(sqlite3.connect(db_path)) as conn:
        frame = pd.read_sql_query(query, conn, params=params)
    return frame


def filter_us_tradeable_universe(
    frame: pd.DataFrame,
    *,
    min_price: float,
    min_avg_volume_20: float,
    min_avg_dollar_volume_20: float,
    exclude_symbol_patterns: list[str] | None = None,
) -> pd.DataFrame:
    filtered = frame.copy()
    filtered = filtered[
        (filtered["close"] >= min_price)
        & (filtered["avg_volume_20"] >= min_avg_volume_20)
        & (filtered["avg_amount_20"] >= min_avg_dollar_volume_20)
    ].copy()
    patterns = exclude_symbol_patterns or []
    if patterns:
        combined = "|".join(f"(?:{pattern})" for pattern in patterns)
        filtered = filtered[~filtered["symbol"].astype(str).str.contains(combined, regex=True, na=False)].copy()
    filtered = filtered.sort_values(["avg_amount_20", "avg_volume_20"], ascending=False).reset_index(drop=True)
    return filtered


def _cache_path(config: USDataConfig, symbol: str, interval: str) -> Path:
    safe_symbol = symbol.replace("/", "_")
    return config.cache_dir / interval / f"{safe_symbol}.csv"


def _is_cache_fresh(path: Path, refresh_hours: int) -> bool:
    if not path.exists():
        return False
    age = pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")
    return age <= pd.Timedelta(hours=refresh_hours)


def _normalize_download(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    normalized = frame.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = normalized.columns.get_level_values(0)
    normalized = normalized.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Datetime": "date",
            "Date": "date",
        }
    )
    normalized = normalized.reset_index()
    if "date" not in normalized.columns:
        first_column = normalized.columns[0]
        normalized = normalized.rename(columns={first_column: "date"})
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.tz_localize(None)
    for column in ("open", "high", "low", "close", "volume"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=["date", "open", "high", "low", "close"])
    return normalized[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)


def download_us_history(symbol: str, interval: str, config: USDataConfig | None = None) -> pd.DataFrame:
    config = config or USDataConfig()
    cache_file = _cache_path(config, symbol, interval)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if _is_cache_fresh(cache_file, config.refresh_hours):
        cached = pd.read_csv(cache_file)
        cached["date"] = pd.to_datetime(cached["date"])
        return cached

    period = config.daily_period
    if interval == "30m":
        period = config.intraday_30m_period
    elif interval == "60m":
        period = config.intraday_60m_period

    try:
        downloaded = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
            prepost=False,
        )
    except Exception:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    normalized = _normalize_download(downloaded)
    normalized.to_csv(cache_file, index=False)
    return normalized


def resample_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    indexed = frame.copy().set_index("date")
    resampled = (
        indexed.resample(rule, label="right", closed="right")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    return resampled
