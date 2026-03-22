from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class USEquitiesDatabaseConfig:
    sqlite_db_path: str = "~/us_stock_daily_data/us_stock_daily.db"
    price_table: str = "daily"
    index_table: str = "index_daily"
    adjust_price: bool = True


@dataclass(slots=True)
class USEquitiesUniverseConfig:
    min_price: float = 5.0
    min_avg_volume_20: float = 2_000_000
    min_avg_dollar_volume_20: float = 100_000_000
    max_symbols: int = 0
    exclude_symbol_patterns: list[str] = field(
        default_factory=lambda: [
            r"W$",
            r"WS$",
            r"WT$",
            r"U$",
            r"R$",
            r"P$",
        ]
    )


@dataclass(slots=True)
class USEquitiesMarketConfig:
    index_symbols: list[str] = field(default_factory=lambda: ["^GSPC", "^IXIC", "^DJI"])
    min_positive_count: int = 2


@dataclass(slots=True)
class USEquitiesSignalConfig:
    bottom_lookback_bars: int = 6
    sell_lookback_bars: int = 3
    breakout_lookback_bars: int = 5
    retest_lookback_bars: int = 10
    retest_tolerance_pct: float = 0.02
    right_side_only: bool = True
    weekly_trend_required: bool = False
    monthly_filter_enabled: bool = True
    require_sector_resonance: bool = False


@dataclass(slots=True)
class USEquitiesStrategyConfig:
    extra_enabled_codes: list[str] = field(default_factory=list)
    disabled_codes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class USEquitiesIntradayConfig:
    enabled: bool = False
    source: str = "yfinance"
    intraday_30m_period: str = "60d"
    intraday_60m_period: str = "720d"
    refresh_hours: int = 8
    max_symbols: int = 150
    min_30m_bars: int = 40
    min_1h_bars: int = 80
    min_2h_bars: int = 40
    min_3h_bars: int = 30
    min_4h_bars: int = 25
    sqlite_db_path: str = "~/us_stock_intraday_data/us_stock_intraday.db"
    sqlite_table_by_timeframe: dict[str, str] = field(
        default_factory=lambda: {
            "30m": "bars_30m",
            "60m": "bars_60m",
        }
    )
    sqlite_symbol_column: str = "symbol"
    sqlite_datetime_column: str = "datetime"
    sqlite_open_column: str = "open"
    sqlite_high_column: str = "high"
    sqlite_low_column: str = "low"
    sqlite_close_column: str = "close"
    sqlite_volume_column: str = "volume"


@dataclass(slots=True)
class USEquitiesSectorConfig:
    min_resonance_members: int = 2
    min_breakout_members: int = 1
    baskets: dict[str, list[str]] = field(default_factory=dict)
