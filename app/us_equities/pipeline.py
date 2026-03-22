from __future__ import annotations

import pandas as pd

from app.us_futu.indicators import MRMCMacdConfig

from app.us_equities.config import (
    USEquitiesDatabaseConfig,
    USEquitiesIntradayConfig,
    USEquitiesMarketConfig,
    USEquitiesSectorConfig,
    USEquitiesSignalConfig,
    USEquitiesStrategyConfig,
    USEquitiesUniverseConfig,
)
from app.us_equities.daily_logic import build_symbol_state
from app.us_equities.database import load_symbol_history, load_tradeable_universe
from app.us_equities.intraday import build_intraday_state
from app.us_equities.sectors import build_sector_context, compute_sector_summary
from app.us_equities.strategy_registry import StrategyContext, evaluate_enabled_strategies, get_enabled_strategies


PREFERRED_RESULT_COLUMNS = [
    "symbol",
    "strategy_type",
    "score",
    "market_regime",
    "market_positive_index_count",
    "trigger_timeframe",
    "entry_note",
    "risk_note",
    "entry_execution_timeframe",
    "confirmation_timeframes",
    "max_resonance_timeframe",
    "position_style",
    "recommended_hold_window",
    "recommended_option_tenor",
    "recommended_sell_level",
    "recommended_stop_timeframe",
    "recommended_stop_reference",
    "sell_level_aggressive",
    "sell_reference_aggressive",
    "sell_level_standard",
    "sell_reference_standard",
    "sell_level_conservative",
    "sell_reference_conservative",
    "daily_price",
    "daily_avg_volume_20",
    "daily_avg_dollar_volume_20",
    "daily_blue_above_yellow",
    "daily_close_above_blue",
    "daily_bottom_recent",
    "daily_breakout_recent",
    "weekly_blue_above_yellow",
    "weekly_close_above_blue",
    "monthly_blue_above_yellow",
    "has_intraday_state",
    "intraday_30m_breakout_recent",
    "intraday_1h_bottom_recent",
    "intraday_2h_bottom_recent",
    "intraday_3h_bottom_recent",
    "intraday_4h_bottom_recent",
    "setup_bottom_resonance_1h",
    "setup_bottom_resonance_2h",
    "setup_bottom_resonance_3h",
    "setup_bottom_resonance_4h",
    "setup_breakout_30m",
    "setup_close_above_blue_1h",
    "setup_sell_recent_1h",
    "higher_tf_support_daily",
    "higher_tf_support_weekly",
    "sector_name",
    "sector_member_count",
    "sector_trend_count",
    "sector_breakout_count",
    "sector_bottom_count",
    "sector_weekly_trend_count",
    "sector_score",
    "sector_resonance_ok",
]


def _reorder_result_columns(results: pd.DataFrame) -> pd.DataFrame:
    for column in PREFERRED_RESULT_COLUMNS:
        if column not in results.columns:
            results[column] = None
    remaining_columns = [column for column in results.columns if column not in PREFERRED_RESULT_COLUMNS]
    return results[PREFERRED_RESULT_COLUMNS + remaining_columns]


def _market_regime(index_states: dict[str, dict[str, object]], market_config: USEquitiesMarketConfig) -> tuple[str, int]:
    positive_count = 0
    for state in index_states.values():
        daily = state["1d"]
        if daily["blue_above_yellow"] and daily["close_above_blue"] and not daily["sell_recent"]:
            positive_count += 1
    if positive_count >= market_config.min_positive_count:
        return "risk_on", positive_count
    if positive_count == 0:
        return "risk_off", positive_count
    return "neutral", positive_count


def run_daily_pipeline(
    database_config: USEquitiesDatabaseConfig,
    universe_config: USEquitiesUniverseConfig,
    market_config: USEquitiesMarketConfig,
    signal_config: USEquitiesSignalConfig,
    sector_config: USEquitiesSectorConfig,
    strategy_config: USEquitiesStrategyConfig | None = None,
    intraday_config: USEquitiesIntradayConfig | None = None,
    macd_config: MRMCMacdConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    macd_config = macd_config or MRMCMacdConfig()
    strategy_config = strategy_config or USEquitiesStrategyConfig()
    intraday_config = intraday_config or USEquitiesIntradayConfig()
    enabled_strategies = get_enabled_strategies(strategy_config)
    universe = load_tradeable_universe(database_config, universe_config)

    index_states: dict[str, dict[str, object]] = {}
    for symbol in market_config.index_symbols:
        history = load_symbol_history(symbol, database_config, index=True)
        if history.empty:
            continue
        state = build_symbol_state(history, signal_config, macd_config)
        if state is not None:
            index_states[symbol] = state
    market_regime, positive_index_count = _market_regime(index_states, market_config)

    symbol_states: dict[str, dict[str, object]] = {}
    intraday_symbols_processed = 0
    for _, row in universe.iterrows():
        symbol = str(row["symbol"])
        history = load_symbol_history(symbol, database_config, index=False)
        if history.empty:
            continue
        state = build_symbol_state(history, signal_config, macd_config)
        if state is None:
            continue
        state["snapshot"] = row.to_dict()
        if intraday_config.enabled and intraday_symbols_processed < intraday_config.max_symbols:
            intraday_state = build_intraday_state(symbol, intraday_config, signal_config, macd_config)
            if intraday_state is not None:
                state["intraday"] = intraday_state
            intraday_symbols_processed += 1
        symbol_states[symbol] = state

    sector_summary = compute_sector_summary(symbol_states, sector_config)
    rows: list[dict[str, object]] = []
    intraday_candidate_count = 0
    for symbol, state in symbol_states.items():
        sector_context = build_sector_context(symbol, sector_summary, sector_config)
        context = StrategyContext(
            symbol=symbol,
            state=state,
            market_regime=market_regime,
            positive_index_count=positive_index_count,
            signal_config=signal_config,
            universe_config=universe_config,
            sector_context=sector_context,
        )
        candidates = evaluate_enabled_strategies(context, strategy_config)
        for candidate in candidates:
            candidate["daily_price"] = state["1d"]["latest_close"]
            candidate["daily_avg_volume_20"] = state["1d"]["avg_volume_20"]
            candidate["daily_avg_dollar_volume_20"] = state["1d"]["avg_dollar_volume_20"]
            candidate["daily_blue_above_yellow"] = state["1d"]["blue_above_yellow"]
            candidate["daily_close_above_blue"] = state["1d"]["close_above_blue"]
            candidate["daily_bottom_recent"] = state["1d"]["bottom_recent"]
            candidate["daily_breakout_recent"] = state["1d"]["breakout_recent"]
            candidate["weekly_blue_above_yellow"] = state["1w"]["blue_above_yellow"]
            candidate["weekly_close_above_blue"] = state["1w"]["close_above_blue"]
            candidate["monthly_blue_above_yellow"] = state["1mo"]["blue_above_yellow"]
            candidate["has_intraday_state"] = "intraday" in state
            if candidate["strategy_type"] == "4321_intraday_resonance" and "intraday" in state:
                candidate["intraday_30m_breakout_recent"] = state["intraday"]["30m"]["breakout_recent"]
                candidate["intraday_1h_bottom_recent"] = state["intraday"]["1h"]["bottom_recent"]
                candidate["intraday_2h_bottom_recent"] = state["intraday"]["2h"]["bottom_recent"]
                candidate["intraday_3h_bottom_recent"] = state["intraday"]["3h"]["bottom_recent"]
                candidate["intraday_4h_bottom_recent"] = state["intraday"]["4h"]["bottom_recent"]
                intraday_candidate_count += 1
            candidate.update(sector_context)
            rows.append(candidate)

    results = pd.DataFrame(rows)
    results = _reorder_result_columns(results)
    if not results.empty:
        results = results.sort_values(
            ["sector_resonance_ok", "sector_score", "score", "strategy_type", "symbol"],
            ascending=[False, False, False, True, True],
        ).reset_index(drop=True)
    summary = {
        "market_regime": market_regime,
        "market_positive_index_count": positive_index_count,
        "index_state_count": int(len(index_states)),
        "universe_size": int(len(universe)),
        "state_count": int(len(symbol_states)),
        "enabled_strategy_codes": [strategy.code for strategy in enabled_strategies],
        "intraday_symbols_processed": int(intraday_symbols_processed),
        "intraday_candidate_count": int(intraday_candidate_count),
        "sector_summary": sector_summary,
    }
    return results, summary
