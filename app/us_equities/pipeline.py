from __future__ import annotations

import pandas as pd

from app.us_futu.indicators import MRMCMacdConfig

from app.us_equities.config import (
    USEquitiesDatabaseConfig,
    USEquitiesIntradayConfig,
    USEquitiesMarketConfig,
    USEquitiesSectorConfig,
    USEquitiesSignalConfig,
    USEquitiesUniverseConfig,
)
from app.us_equities.daily_logic import build_symbol_state
from app.us_equities.database import load_symbol_history, load_tradeable_universe
from app.us_equities.intraday import build_intraday_state
from app.us_equities.sectors import build_sector_context, compute_sector_summary
from app.us_equities.strategy_registry import StrategyContext, evaluate_registered_strategies


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
    intraday_config: USEquitiesIntradayConfig | None = None,
    macd_config: MRMCMacdConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    macd_config = macd_config or MRMCMacdConfig()
    intraday_config = intraday_config or USEquitiesIntradayConfig()
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
        candidates = evaluate_registered_strategies(context)
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
        "intraday_symbols_processed": int(intraday_symbols_processed),
        "intraday_candidate_count": int(intraday_candidate_count),
        "sector_summary": sector_summary,
    }
    return results, summary
