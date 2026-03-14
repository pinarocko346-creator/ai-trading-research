from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
import sqlite3

import pandas as pd


@dataclass(slots=True)
class DataIngestConfig:
    cache_dir: Path = Path("data/cache")
    adjust: str = "qfq"
    start_date: str = "2018-01-01"
    end_date: str | None = None
    refresh: bool = False
    source: str = "akshare"
    sqlite_db_path: str | None = None
    warmup_days: int = 120


REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def _import_akshare():
    try:
        import akshare as ak  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "缺少 akshare，请先执行 `python3 -m pip install -e .` 或单独安装 akshare。"
        ) from exc
    return ak


def resolve_sqlite_db_path(config: DataIngestConfig) -> Path:
    if not config.sqlite_db_path:
        raise RuntimeError("SQLite 数据源已启用，但未配置 sqlite_db_path。")
    db_path = Path(config.sqlite_db_path).expanduser()
    if not db_path.exists():
        raise RuntimeError(f"SQLite 数据库不存在: {db_path}")
    return db_path


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    column_map = {
        "code": "symbol",
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "涨跌幅": "pct_chg",
        "turnover": "turnover_rate",
        "换手率": "turnover_rate",
    }
    normalized = frame.rename(columns=column_map).copy()
    for column in REQUIRED_COLUMNS:
        if column not in normalized.columns:
            raise ValueError(f"缺少必要列: {column}")
    normalized["date"] = pd.to_datetime(normalized["date"])
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "pct_chg",
        "turnover_rate",
    ]
    for column in numeric_columns:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized.sort_values("date").reset_index(drop=True)


def cache_path(symbol: str, config: DataIngestConfig) -> Path:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    return config.cache_dir / f"{symbol}_{config.adjust}.parquet"


def _fetch_akshare_history(symbol: str, config: DataIngestConfig) -> pd.DataFrame:
    cache_file = cache_path(symbol, config)
    if cache_file.exists() and not config.refresh:
        return pd.read_parquet(cache_file)

    ak = _import_akshare()
    raw = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=config.start_date.replace("-", ""),
        end_date=(config.end_date or pd.Timestamp.today().strftime("%Y%m%d")).replace("-", ""),
        adjust=config.adjust,
    )
    normalized = normalize_ohlcv(raw)
    normalized.to_parquet(cache_file, index=False)
    return normalized


def _fetch_sqlite_history(symbol: str, config: DataIngestConfig) -> pd.DataFrame:
    end_date = config.end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    query = """
        SELECT
            code,
            date,
            open,
            high,
            low,
            close,
            volume,
            amount,
            pct_chg,
            turnover
        FROM kline_data
        WHERE code = ?
          AND date >= ?
          AND date <= ?
        ORDER BY date
    """
    with closing(sqlite3.connect(resolve_sqlite_db_path(config))) as conn:
        frame = pd.read_sql_query(query, conn, params=[symbol, config.start_date, end_date])
    return normalize_ohlcv(frame)


def _fetch_history_with_sqlite_warmup(symbol: str, config: DataIngestConfig) -> pd.DataFrame:
    sqlite_history = _fetch_sqlite_history(symbol, config)
    if sqlite_history.empty or config.warmup_days <= 0:
        return sqlite_history

    sqlite_start = sqlite_history["date"].min()
    requested_start = pd.Timestamp(config.start_date)
    warmup_start = (requested_start - pd.Timedelta(days=config.warmup_days)).strftime("%Y-%m-%d")
    if sqlite_start <= pd.Timestamp(warmup_start):
        return sqlite_history

    warmup_config = DataIngestConfig(
        cache_dir=config.cache_dir,
        adjust=config.adjust,
        start_date=warmup_start,
        end_date=(sqlite_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        refresh=config.refresh,
        source="akshare",
        warmup_days=0,
    )
    try:
        warmup_history = _fetch_akshare_history(symbol, warmup_config)
    except Exception:
        return sqlite_history

    combined = pd.concat(
        [warmup_history[warmup_history["date"] < sqlite_start], sqlite_history],
        ignore_index=True,
    )
    combined = combined.drop_duplicates(subset=["date"], keep="last")
    return combined.sort_values("date").reset_index(drop=True)


def load_sqlite_breadth_history(config: DataIngestConfig) -> dict[pd.Timestamp, pd.DataFrame]:
    end_date = config.end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    query = """
        SELECT date, pct_chg
        FROM kline_data
        WHERE date >= ?
          AND date <= ?
    """
    with closing(sqlite3.connect(resolve_sqlite_db_path(config))) as conn:
        frame = pd.read_sql_query(query, conn, params=[config.start_date, end_date])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["pct_chg"] = pd.to_numeric(frame["pct_chg"], errors="coerce")
    frame = frame.dropna(subset=["date", "pct_chg"])
    return {
        date: group[["pct_chg"]].reset_index(drop=True)
        for date, group in frame.groupby("date")
    }


def fetch_a_share_history(symbol: str, config: DataIngestConfig | None = None) -> pd.DataFrame:
    config = config or DataIngestConfig()
    if config.source == "sqlite":
        return _fetch_history_with_sqlite_warmup(symbol, config)
    return _fetch_akshare_history(symbol, config)


def load_csv_history(csv_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    return normalize_ohlcv(frame)


def update_history_cache(symbols: list[str], config: DataIngestConfig | None = None) -> dict[str, Path]:
    config = config or DataIngestConfig()
    written: dict[str, Path] = {}
    for symbol in symbols:
        fetch_a_share_history(symbol, config)
        written[symbol] = cache_path(symbol, config)
    return written
