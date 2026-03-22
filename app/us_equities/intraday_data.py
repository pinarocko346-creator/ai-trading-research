from __future__ import annotations

from contextlib import closing
from pathlib import Path
import sqlite3

import pandas as pd

from app.us_futu.data import USDataConfig, download_us_history

from app.us_equities.config import USEquitiesIntradayConfig


def _normalize_intraday_frame(
    frame: pd.DataFrame,
    *,
    datetime_column: str,
    open_column: str,
    high_column: str,
    low_column: str,
    close_column: str,
    volume_column: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    normalized = frame.rename(
        columns={
            datetime_column: "date",
            open_column: "open",
            high_column: "high",
            low_column: "low",
            close_column: "close",
            volume_column: "volume",
        }
    ).copy()
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.tz_localize(None)
    for column in ("open", "high", "low", "close", "volume"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=["date", "open", "high", "low", "close"])
    normalized = normalized.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return normalized[["date", "open", "high", "low", "close", "volume"]]


def _load_intraday_from_yfinance(symbol: str, timeframe: str, config: USEquitiesIntradayConfig) -> pd.DataFrame:
    data_config = USDataConfig(
        source="yfinance",
        daily_period="2y",
        intraday_30m_period=config.intraday_30m_period,
        intraday_60m_period=config.intraday_60m_period,
        refresh_hours=config.refresh_hours,
    )
    return download_us_history(symbol, timeframe, data_config)


def _load_intraday_from_sqlite(symbol: str, timeframe: str, config: USEquitiesIntradayConfig) -> pd.DataFrame:
    db_path = Path(config.sqlite_db_path).expanduser().resolve()
    table_name = config.sqlite_table_by_timeframe.get(timeframe)
    if not table_name:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    query = f"""
        SELECT
            {config.sqlite_symbol_column} AS symbol,
            {config.sqlite_datetime_column} AS bar_time,
            {config.sqlite_open_column} AS open,
            {config.sqlite_high_column} AS high,
            {config.sqlite_low_column} AS low,
            {config.sqlite_close_column} AS close,
            {config.sqlite_volume_column} AS volume
        FROM {table_name}
        WHERE {config.sqlite_symbol_column} = ?
        ORDER BY {config.sqlite_datetime_column}
    """
    with closing(sqlite3.connect(db_path)) as conn:
        frame = pd.read_sql_query(query, conn, params=[symbol])
    return _normalize_intraday_frame(
        frame,
        datetime_column="bar_time",
        open_column="open",
        high_column="high",
        low_column="low",
        close_column="close",
        volume_column="volume",
    )


def load_intraday_history(symbol: str, timeframe: str, config: USEquitiesIntradayConfig) -> pd.DataFrame:
    if timeframe not in {"30m", "60m"}:
        raise ValueError(f"不支持的时间周期: {timeframe}")
    if config.source == "sqlite":
        return _load_intraday_from_sqlite(symbol, timeframe, config)
    return _load_intraday_from_yfinance(symbol, timeframe, config)
