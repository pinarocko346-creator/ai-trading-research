from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class DataIngestConfig:
    cache_dir: Path = Path("data/cache")
    adjust: str = "qfq"
    start_date: str = "2018-01-01"
    end_date: str | None = None
    refresh: bool = False


REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def _import_akshare():
    try:
        import akshare as ak  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "缺少 akshare，请先执行 `python3 -m pip install -e .` 或单独安装 akshare。"
        ) from exc
    return ak


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    column_map = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "涨跌幅": "pct_chg",
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


def fetch_a_share_history(symbol: str, config: DataIngestConfig | None = None) -> pd.DataFrame:
    config = config or DataIngestConfig()
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
