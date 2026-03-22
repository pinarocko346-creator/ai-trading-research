from __future__ import annotations

import pandas as pd

from app.us_futu.data import (
    USDataConfig,
    fetch_us_history,
    filter_us_tradeable_universe,
    load_us_universe_snapshot,
)

from app.us_equities.config import USEquitiesDatabaseConfig, USEquitiesUniverseConfig


def build_data_config(config: USEquitiesDatabaseConfig) -> USDataConfig:
    return USDataConfig(
        source="sqlite",
        sqlite_db_path=config.sqlite_db_path,
        price_table=config.price_table,
        index_table=config.index_table,
        adjust_price=config.adjust_price,
    )


def load_tradeable_universe(
    database_config: USEquitiesDatabaseConfig,
    universe_config: USEquitiesUniverseConfig,
) -> pd.DataFrame:
    data_config = build_data_config(database_config)
    snapshot = load_us_universe_snapshot(data_config)
    filtered = filter_us_tradeable_universe(
        snapshot,
        min_price=universe_config.min_price,
        min_avg_volume_20=universe_config.min_avg_volume_20,
        min_avg_dollar_volume_20=universe_config.min_avg_dollar_volume_20,
        exclude_symbol_patterns=universe_config.exclude_symbol_patterns,
    )
    if universe_config.max_symbols > 0:
        filtered = filtered.head(universe_config.max_symbols).copy()
    return filtered.reset_index(drop=True)


def load_symbol_history(
    symbol: str,
    database_config: USEquitiesDatabaseConfig,
    *,
    index: bool = False,
) -> pd.DataFrame:
    return fetch_us_history(symbol, build_data_config(database_config), index=index)
