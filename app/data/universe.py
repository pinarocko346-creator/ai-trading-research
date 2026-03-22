from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import sqlite3

import pandas as pd

from app.data.ingest import (
    DataIngestConfig,
    resolve_sqlite_db_path,
    sqlite_price_table,
    sqlite_table_columns,
    sqlite_uses_legacy_schema,
    strip_exchange_prefix,
)


@dataclass(slots=True)
class UniverseConfig:
    min_close: float = 3.0
    min_avg_volume: float = 3_000_000
    min_turnover_rate: float = 0.5
    exclude_st: bool = True
    exclude_beijing: bool = False


def _import_akshare():
    try:
        import akshare as ak  # type: ignore
    except ImportError as exc:
        raise RuntimeError("缺少 akshare，无法加载 A 股股票池。") from exc
    return ak


def _load_sqlite_spot(config: DataIngestConfig) -> pd.DataFrame:
    db_path = resolve_sqlite_db_path(config)
    with closing(sqlite3.connect(db_path)) as conn:
        table_name = sqlite_price_table(conn)
        trade_dates_query = f"""
            SELECT date
            FROM (
                SELECT DISTINCT date
                FROM {table_name}
                ORDER BY date DESC
                LIMIT 20
            )
            ORDER BY date
        """
        trade_dates = pd.read_sql_query(trade_dates_query, conn)["date"].astype(str).tolist()
        if not trade_dates:
            return pd.DataFrame(columns=["symbol", "name", "close", "volume", "avg_volume_20", "turnover_rate", "pct_chg"])
        placeholders = ",".join("?" for _ in trade_dates)
        if sqlite_uses_legacy_schema(conn):
            spot_query = f"""
                SELECT
                    k.code AS symbol,
                    COALESCE(s.name, k.code) AS name,
                    k.date,
                    k.close,
                    k.volume,
                    k.turnover AS turnover_rate,
                    k.pct_chg
                FROM kline_data k
                LEFT JOIN stock_list s
                  ON s.code = k.code
                WHERE k.date IN ({placeholders})
            """
        else:
            columns = sqlite_table_columns(conn, table_name)
            close_column = "COALESCE(k.close_adj, k.close)" if config.adjust == "hfq" and "close_adj" in columns else "k.close"
            turnover_column = "turn" if "turn" in columns else "turnover"
            spot_query = f"""
                SELECT
                    k.code AS symbol,
                    k.code AS name,
                    k.date,
                    {close_column} AS close,
                    k.volume,
                    k.{turnover_column} AS turnover_rate,
                    k.pct_chg
                FROM {table_name} k
                WHERE k.date IN ({placeholders})
            """
        frame = pd.read_sql_query(spot_query, conn, params=trade_dates)

    frame["date"] = pd.to_datetime(frame["date"])
    frame["symbol"] = frame["symbol"].astype(str).map(strip_exchange_prefix)
    for column in ("close", "volume", "turnover_rate", "pct_chg"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)

    latest = frame.groupby("symbol", as_index=False).tail(1).copy()
    avg_volume = (
        frame.groupby("symbol", as_index=False)["volume"]
        .mean()
        .rename(columns={"volume": "avg_volume_20"})
    )
    recent_turnover = (
        frame.assign(turnover_recent=frame["turnover_rate"].where(frame["turnover_rate"] > 0))
        .groupby("symbol", as_index=False)["turnover_recent"]
        .agg(lambda series: series.dropna().iloc[-1] if not series.dropna().empty else 0.0)
        .rename(columns={"turnover_recent": "recent_turnover_rate"})
    )
    latest = latest.merge(avg_volume, on="symbol", how="left")
    latest = latest.merge(recent_turnover, on="symbol", how="left")
    latest["turnover_rate"] = latest["turnover_rate"].where(
        latest["turnover_rate"] > 0,
        latest["recent_turnover_rate"],
    )
    latest["turnover_rate"] = latest["turnover_rate"].fillna(0.0)
    return latest[["symbol", "name", "close", "volume", "avg_volume_20", "turnover_rate", "pct_chg"]].reset_index(drop=True)


def filter_tradeable_universe(frame: pd.DataFrame, config: UniverseConfig | None = None) -> pd.DataFrame:
    config = config or UniverseConfig()
    filtered = frame.copy()

    if config.exclude_st and "name" in filtered.columns:
        filtered = filtered[~filtered["name"].astype(str).str.upper().str.contains("ST")]
    if config.exclude_beijing and "symbol" in filtered.columns:
        filtered = filtered[~filtered["symbol"].astype(str).str.startswith("8")]
    if "close" in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered["close"], errors="coerce") >= config.min_close]
    if "avg_volume_20" in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered["avg_volume_20"], errors="coerce") >= config.min_avg_volume]
    if "turnover_rate" in filtered.columns:
        filtered = filtered[
            pd.to_numeric(filtered["turnover_rate"], errors="coerce") >= config.min_turnover_rate
        ]
    if "is_halted" in filtered.columns:
        filtered = filtered[~filtered["is_halted"].fillna(False)]
    return filtered.reset_index(drop=True)


def load_a_share_spot(ingest_config: DataIngestConfig | None = None) -> pd.DataFrame:
    if ingest_config and ingest_config.source == "sqlite":
        return _load_sqlite_spot(ingest_config)

    ak = _import_akshare()
    spot = ak.stock_zh_a_spot_em()
    column_map = {
        "代码": "symbol",
        "名称": "name",
        "最新价": "close",
        "成交量": "volume",
        "换手率": "turnover_rate",
        "涨跌幅": "pct_chg",
        "总市值": "market_cap",
        "流通市值": "float_market_cap",
    }
    normalized = spot.rename(columns=column_map).copy()
    return normalized


def build_universe_snapshot(
    latest_feature_frame: pd.DataFrame, config: UniverseConfig | None = None
) -> pd.DataFrame:
    config = config or UniverseConfig()
    columns = [
        column
        for column in ["symbol", "name", "close", "avg_volume_20", "turnover_rate", "is_halted", "trend_up"]
        if column in latest_feature_frame.columns
    ]
    snapshot = latest_feature_frame[columns].copy()
    return filter_tradeable_universe(snapshot, config)
