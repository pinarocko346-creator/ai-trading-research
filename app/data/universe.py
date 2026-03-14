from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


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


def load_a_share_spot() -> pd.DataFrame:
    ak = _import_akshare()
    spot = ak.stock_zh_a_spot_em()
    column_map = {
        "代码": "symbol",
        "名称": "name",
        "最新价": "close",
        "成交量": "volume",
        "换手率": "turnover_rate",
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
