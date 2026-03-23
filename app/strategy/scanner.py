from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app.data.market_context import MarketFilterConfig, load_market_snapshot
from app.data.sector_context import SectorFilterConfig, build_symbol_theme_payload, load_sector_snapshot
from app.data.ingest import DataIngestConfig, fetch_a_share_history
from app.data.universe import UniverseConfig, filter_tradeable_universe, load_a_share_spot
from app.strategy.rules import RuleThresholds, build_signal_catalog, scan_signals
from app.strategy.scoring import rank_signals, score_signal

BREAKOUT_NORMALIZATION_PRIORITY = {
    "cup_with_handle_leader": 6,
    "cup_with_handle_strict": 5,
    "cup_with_handle": 4,
    "jumping_creek": 3,
    "n_breakout": 2,
    "double_breakout": 1,
}

TREND_SIGNAL_TYPES = {
    "double_breakout",
    "jumping_creek",
    "cup_with_handle_watch",
    "cup_with_handle",
    "cup_with_handle_strict",
    "cup_with_handle_leader",
    "pullback_confirmation",
    "n_breakout",
    "support_resistance_flip",
    "pattern_breakout",
    "strength_emergence",
}

REVERSAL_SIGNAL_TYPES = {
    "selling_climax",
    "2b_structure",
    "false_breakdown",
    "right_shoulder",
    "spring",
    "first_rebound_after_crash",
}

CANDIDATE_DISABLED_SIGNAL_TYPES = {
    "selling_climax",
    "cup_with_handle_watch",
    "pullback_confirmation",
}

CANDIDATE_FILTER_REQUIRED_SIGNAL_TYPES = {
    "double_breakout",
    "jumping_creek",
    "n_breakout",
}

CANDIDATE_SIGNAL_LIMITS = {
    "jumping_creek": 1,
    "n_breakout": 1,
}


@dataclass(slots=True)
class ScanConfig:
    max_symbols: int = 100
    cache_dir: Path = Path("data/cache")
    per_signal_limit: int = 3
    pretty_min_quality_score: float = 60.0
    pretty_hard_filter_score: float = 55.0
    ingest_config: DataIngestConfig = field(default_factory=DataIngestConfig)
    market_filter: MarketFilterConfig = field(default_factory=MarketFilterConfig)
    sector_filter: SectorFilterConfig = field(default_factory=SectorFilterConfig)
    apply_market_filter: bool = True
    apply_sector_filter: bool = True


def _market_score_adjustment(market_snapshot: dict[str, object]) -> float:
    regime = str(market_snapshot.get("market_regime", "neutral"))
    market_score = float(market_snapshot.get("market_score", 0.0) or 0.0)
    if regime == "risk_on":
        return 8.0
    if regime == "risk_off":
        return -10.0
    return 2.0 if market_score >= 55 else -2.0


def _sector_score_adjustment(theme_payload: dict[str, object], config: SectorFilterConfig) -> float:
    sector_score = float(theme_payload.get("sector_score", 0.0) or 0.0)
    if sector_score >= config.crowded_min_score:
        return -3.5
    if sector_score >= config.min_sector_score:
        return 1.5
    if sector_score >= config.edge_high_min_score:
        return 2.5
    if sector_score >= config.edge_low_min_score:
        return 1.0
    if sector_score > 0:
        return -1.5
    return -4.0


def _sector_band(theme_payload: dict[str, object], config: SectorFilterConfig) -> str:
    sector_score = float(theme_payload.get("sector_score", 0.0) or 0.0)
    if sector_score >= config.crowded_min_score:
        return "crowded"
    if sector_score >= config.min_sector_score:
        return "strong"
    if sector_score >= config.edge_high_min_score:
        return "edge_high"
    if sector_score >= config.edge_low_min_score:
        return "edge_low"
    if sector_score > 0:
        return "weak"
    return "none"


def _filter_ok(
    market_snapshot: dict[str, object],
    theme_payload: dict[str, object],
    config: SectorFilterConfig,
    signal_type: str | None = None,
) -> bool:
    regime = str(market_snapshot.get("market_regime", "neutral"))
    sector_band = _sector_band(theme_payload, config)
    if sector_band == "crowded":
        return False
    if signal_type in REVERSAL_SIGNAL_TYPES:
        if regime == "risk_off":
            return sector_band in {"edge_low", "weak"}
        return sector_band in {"edge_high", "edge_low", "weak"}
    if signal_type in TREND_SIGNAL_TYPES:
        if regime == "risk_off":
            return sector_band == "edge_high"
        return sector_band in {"strong", "edge_high"}
    if regime == "risk_off":
        return sector_band in {"edge_high", "edge_low"}
    if regime == "risk_on":
        return sector_band in {"strong", "edge_high", "edge_low", "weak"}
    return sector_band in {"strong", "edge_high", "edge_low"}


def _numeric_factor(signal, key: str) -> float | None:
    value = signal.factors.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def score_signal_quality(signal) -> float:
    score = 12.0
    score += {
        "false_breakdown": 4.0,
        "2b_structure": 3.0,
        "right_shoulder": 4.0,
        "support_resistance_flip": 4.0,
        "strength_emergence": 4.0,
        "pattern_breakout": 2.0,
        "first_rebound_after_crash": 2.0,
        "spring": 2.0,
        "cup_with_handle_watch": -10.0,
        "cup_with_handle": 6.0,
        "cup_with_handle_strict": 8.0,
        "cup_with_handle_leader": 10.0,
        "pullback_confirmation": -6.0,
        "n_breakout": -12.0,
        "double_breakout": -18.0,
        "jumping_creek": -20.0,
        "selling_climax": -18.0,
    }.get(signal.signal_type, 0.0)
    score += 10.0 if signal.trend_ok else -10.0
    score += 10.0 if signal.location_ok else -10.0
    score += 12.0 if signal.pattern_ok else -12.0
    score += 8.0 if signal.volume_ok else -10.0

    volume_ratio = _numeric_factor(signal, "volume_ratio")
    if volume_ratio is not None:
        if volume_ratio >= 2.0:
            score += 6.0
        elif volume_ratio >= 1.3:
            score += 4.0
        elif volume_ratio >= 1.0:
            score += 2.0
        else:
            score -= 4.0

    close_in_range = _numeric_factor(signal, "close_in_range")
    if close_in_range is not None:
        if close_in_range >= 0.85:
            score += 5.0
        elif close_in_range >= 0.7:
            score += 2.5
        elif close_in_range < 0.55:
            score -= 5.0

    breakout_pct = _numeric_factor(signal, "breakout_pct")
    if breakout_pct is not None:
        if breakout_pct >= 0.04:
            score += 5.0
        elif breakout_pct >= 0.015:
            score += 3.0
        elif breakout_pct < 0.008:
            score -= 4.0

    if signal.factors.get("prep_tight") is True:
        score += 3.0
    if signal.factors.get("prior_below_resistance") is True or signal.factors.get("prior_below_breakout") is True:
        score += 3.0
    if signal.factors.get("prior_below_box") is True:
        score += 2.0
    if signal.factors.get("reclaim_in_time") is True:
        score += 2.0

    if signal.signal_type in {"cup_with_handle", "cup_with_handle_strict", "cup_with_handle_leader"}:
        handle_depth_pct = _numeric_factor(signal, "handle_depth_pct")
        if handle_depth_pct is not None:
            if handle_depth_pct <= 0.06:
                score += 6.0
            elif handle_depth_pct <= 0.1:
                score += 3.0
            else:
                score -= 8.0
        handle_volume_dryup_ratio = _numeric_factor(signal, "handle_volume_dryup_ratio")
        if handle_volume_dryup_ratio is not None:
            if handle_volume_dryup_ratio <= 0.85:
                score += 5.0
            elif handle_volume_dryup_ratio <= 1.0:
                score += 2.0
            else:
                score -= 5.0
        right_peak_recovery_pct = _numeric_factor(signal, "right_peak_recovery_pct")
        if right_peak_recovery_pct is not None:
            if right_peak_recovery_pct >= 0.97:
                score += 4.0
            elif right_peak_recovery_pct < 0.93:
                score -= 5.0
        handle_low_position_pct = _numeric_factor(signal, "handle_low_position_pct")
        if handle_low_position_pct is not None:
            if handle_low_position_pct >= 0.65:
                score += 4.0
            elif handle_low_position_pct < 0.55:
                score -= 6.0
        if signal.signal_type in {"cup_with_handle_strict", "cup_with_handle_leader"}:
            prior_rise_60_pct = _numeric_factor(signal, "prior_rise_60_pct")
            if prior_rise_60_pct is not None:
                if prior_rise_60_pct >= 0.45:
                    score += 4.0
                elif prior_rise_60_pct < 0.3:
                    score -= 8.0
            handle_depth_vs_cup_pct = _numeric_factor(signal, "handle_depth_vs_cup_pct")
            if handle_depth_vs_cup_pct is not None:
                if handle_depth_vs_cup_pct <= 0.25:
                    score += 4.0
                elif handle_depth_vs_cup_pct > 0.3334:
                    score -= 8.0
            volume_ratio_50 = _numeric_factor(signal, "volume_ratio_50")
            if volume_ratio_50 is not None:
                if volume_ratio_50 >= 1.8:
                    score += 5.0
                elif volume_ratio_50 >= 1.4:
                    score += 3.0
                else:
                    score -= 8.0
        if signal.signal_type == "cup_with_handle_leader":
            prior_rise_6m_pct = _numeric_factor(signal, "prior_rise_6m_pct")
            if prior_rise_6m_pct is not None:
                if prior_rise_6m_pct >= 0.8:
                    score += 5.0
                elif prior_rise_6m_pct >= 0.5:
                    score += 3.0
                else:
                    score -= 8.0
            if signal.factors.get("ma60_rising") is True:
                score += 3.0
            if signal.factors.get("progressive_lows_ok") is True:
                score += 5.0
            if signal.factors.get("market_cap_ok") is True and signal.factors.get("market_cap_check_skipped") is False:
                score += 2.0
    elif signal.signal_type == "support_resistance_flip":
        pullback_volume_ratio = _numeric_factor(signal, "pullback_volume_ratio")
        if pullback_volume_ratio is not None:
            if pullback_volume_ratio <= 0.75:
                score += 5.0
            elif pullback_volume_ratio <= 0.95:
                score += 2.0
            else:
                score -= 6.0
        pullback_bars = _numeric_factor(signal, "pullback_bars")
        if pullback_bars is not None:
            if 3 <= pullback_bars <= 7:
                score += 4.0
            elif pullback_bars > 10:
                score -= 6.0
        if signal.factors.get("close_hold_ok") is True:
            score += 4.0
        breakout_volume_ratio = _numeric_factor(signal, "breakout_volume_ratio")
        if breakout_volume_ratio is not None:
            if breakout_volume_ratio >= 1.8:
                score += 3.0
            elif breakout_volume_ratio < 1.2:
                score -= 4.0
    elif signal.signal_type in {"double_breakout", "jumping_creek", "n_breakout", "pattern_breakout"}:
        if signal.factors.get("prep_tight") is False:
            score -= 4.0
    elif signal.signal_type in {"false_breakdown", "2b_structure", "spring"}:
        break_pct = _numeric_factor(signal, "break_pct")
        if break_pct is not None:
            if break_pct >= 0.015:
                score += 4.0
            elif break_pct < 0.005:
                score -= 4.0
        if signal.signal_type == "false_breakdown":
            breakdown_close_in_range = _numeric_factor(signal, "breakdown_close_in_range")
            if breakdown_close_in_range is not None:
                if breakdown_close_in_range <= 0.2:
                    score += 2.0
                elif breakdown_close_in_range > 0.5:
                    score -= 3.0
            rebound_from_low_pct = _numeric_factor(signal, "rebound_from_low_pct")
            if rebound_from_low_pct is not None:
                if rebound_from_low_pct >= 0.03:
                    score += 2.0
                elif rebound_from_low_pct < 0.012:
                    score -= 2.0
            close_in_range = _numeric_factor(signal, "close_in_range")
            if close_in_range is not None:
                if close_in_range >= 0.75:
                    score += 2.0
                elif close_in_range < 0.6:
                    score -= 2.0
            confirm_close_vs_ma10 = _numeric_factor(signal, "confirm_close_vs_ma10")
            if confirm_close_vs_ma10 is not None:
                if confirm_close_vs_ma10 >= 1.01:
                    score += 2.0
                elif confirm_close_vs_ma10 < 0.995:
                    score -= 3.0
            confirm_close_vs_ma20 = _numeric_factor(signal, "confirm_close_vs_ma20")
            if confirm_close_vs_ma20 is not None:
                if confirm_close_vs_ma20 >= 1.0:
                    score += 2.0
                elif confirm_close_vs_ma20 < 0.995:
                    score -= 2.0
            if signal.factors.get("reclaim_above_break_open") is True:
                score += 2.0
            elif signal.factors.get("reclaim_above_break_open") is False:
                score -= 3.0
            if signal.factors.get("no_new_low_after_reclaim") is True:
                score += 2.0
            elif signal.factors.get("no_new_low_after_reclaim") is False:
                score -= 4.0
            if signal.factors.get("ma20_flat_enough") is True:
                score += 1.0
            elif signal.factors.get("ma20_flat_enough") is False:
                score -= 3.0
    elif signal.signal_type == "first_rebound_after_crash":
        crash_drop_pct = _numeric_factor(signal, "crash_drop_pct")
        if crash_drop_pct is not None:
            if crash_drop_pct <= -0.1:
                score += 4.0
            elif crash_drop_pct > -0.08:
                score -= 4.0

    return round(max(0.0, min(100.0, score)), 2)


def quality_bucket_label(
    quality_score: float,
    *,
    pretty_min_quality_score: float = 60.0,
    pretty_hard_filter_score: float = 55.0,
) -> str:
    if quality_score >= pretty_min_quality_score:
        return "high"
    if quality_score >= pretty_hard_filter_score:
        return "medium"
    return "low"


def pretty_signal_ok(quality_score: float, *, pretty_hard_filter_score: float = 55.0) -> bool:
    return quality_score >= pretty_hard_filter_score


def _candidate_signal_allowed(row: pd.Series) -> bool:
    signal_type = str(row.get("signal_type", "") or "")
    if signal_type in CANDIDATE_DISABLED_SIGNAL_TYPES:
        return False
    if signal_type in CANDIDATE_FILTER_REQUIRED_SIGNAL_TYPES:
        if "filter_ok" not in row.index or pd.isna(row.get("filter_ok")):
            return True
        return bool(row.get("filter_ok", False))
    return True


def load_default_universe(
    universe_config: UniverseConfig | None = None,
    *,
    max_symbols: int | None = 100,
    ingest_config: DataIngestConfig | None = None,
) -> pd.DataFrame:
    universe_config = universe_config or UniverseConfig()
    spot = load_a_share_spot(ingest_config)
    if "volume" in spot.columns and "avg_volume_20" not in spot.columns:
        spot["avg_volume_20"] = pd.to_numeric(spot["volume"], errors="coerce")
    filtered = filter_tradeable_universe(spot, universe_config)
    if "volume" in filtered.columns:
        filtered["volume"] = pd.to_numeric(filtered["volume"], errors="coerce")
        filtered = filtered.sort_values("volume", ascending=False)
    if max_symbols is None or max_symbols <= 0:
        return filtered.reset_index(drop=True)
    return filtered.head(max_symbols).reset_index(drop=True)


def scan_market(
    universe: pd.DataFrame,
    *,
    thresholds: RuleThresholds | None = None,
    signal_types: list[str] | None = None,
    scan_config: ScanConfig | None = None,
) -> pd.DataFrame:
    thresholds = thresholds or RuleThresholds()
    scan_config = scan_config or ScanConfig()
    ingest_config = scan_config.ingest_config
    signal_names = {item.code: item.name for item in build_signal_catalog()}
    results: list[dict[str, object]] = []
    full_spot = load_a_share_spot(ingest_config)
    market_snapshot = load_market_snapshot(scan_config.market_filter, spot_frame=full_spot)
    sector_snapshot = load_sector_snapshot(scan_config.sector_filter)

    for _, row in universe.head(scan_config.max_symbols).iterrows():
        symbol = str(row["symbol"])
        try:
            history = fetch_a_share_history(symbol, ingest_config)
        except Exception:
            continue
        for extra_column in ("market_cap", "float_market_cap", "total_mv", "float_mv"):
            if extra_column in row.index and pd.notna(row[extra_column]):
                history[extra_column] = row[extra_column]
        signals = scan_signals(
            history,
            symbol=symbol,
            enabled_signals=signal_types,
            thresholds=thresholds,
            include_invalid=False,
        )
        for signal in rank_signals(signals):
            theme_payload = build_symbol_theme_payload(symbol, sector_snapshot, scan_config.sector_filter)
            payload = signal.to_dict()
            base_score = score_signal(signal)
            quality_score = score_signal_quality(signal)
            quality_bucket = quality_bucket_label(
                quality_score,
                pretty_min_quality_score=scan_config.pretty_min_quality_score,
                pretty_hard_filter_score=scan_config.pretty_hard_filter_score,
            )
            pretty_ok = pretty_signal_ok(
                quality_score,
                pretty_hard_filter_score=scan_config.pretty_hard_filter_score,
            )
            market_ok = bool(market_snapshot["market_ok"])
            sector_ok = bool(theme_payload["sector_ok"])
            sector_band = _sector_band(theme_payload, scan_config.sector_filter)
            score = float(base_score)
            if scan_config.apply_market_filter:
                score += _market_score_adjustment(market_snapshot)
            if scan_config.apply_sector_filter:
                score += _sector_score_adjustment(theme_payload, scan_config.sector_filter)
            payload["base_score"] = base_score
            payload["score"] = round(max(0.0, score), 2)
            payload["name"] = row.get("name", "")
            payload["signal_name"] = signal_names.get(signal.signal_type, signal.signal_type)
            payload["quality_score"] = quality_score
            payload["quality_bucket"] = quality_bucket
            payload["pretty_ok"] = pretty_ok
            payload["market_ok"] = market_ok
            payload["market_score"] = market_snapshot["market_score"]
            payload["market_regime"] = market_snapshot["market_regime"]
            payload["market_positive_index_count"] = market_snapshot["market_positive_index_count"]
            payload["market_up_ratio"] = market_snapshot["market_up_ratio"]
            payload["market_limit_up_count"] = market_snapshot["market_limit_up_count"]
            payload["market_limit_down_count"] = market_snapshot["market_limit_down_count"]
            payload["market_limit_up_down_ratio"] = market_snapshot["market_limit_up_down_ratio"]
            payload["market_index_details"] = market_snapshot["market_index_details"]
            payload["sector_ok"] = sector_ok
            payload["sector_score"] = theme_payload["sector_score"]
            payload["industry_name"] = theme_payload["industry_name"]
            payload["industry_score"] = theme_payload["industry_score"]
            payload["concept_names"] = theme_payload["concept_names"]
            payload["concept_scores"] = theme_payload["concept_scores"]
            payload["sector_band"] = sector_band
            payload["filter_ok"] = _filter_ok(
                market_snapshot,
                theme_payload,
                scan_config.sector_filter,
                signal.signal_type,
            )
            results.append(payload)

    frame = pd.DataFrame(results)
    if frame.empty:
        return frame
    sort_columns = ["score", "confidence_score", "signal_date"]
    if "quality_score" in frame.columns:
        sort_columns = ["pretty_ok", "quality_score"] + sort_columns
    return frame.sort_values(sort_columns, ascending=False).reset_index(drop=True)


def normalize_signal_candidates(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results

    signal_names = {item.code: item.name for item in build_signal_catalog()}
    normalized_rows: list[pd.Series] = []

    for _, group in results.groupby("symbol", sort=False):
        breakout_group = group[group["signal_type"].isin(BREAKOUT_NORMALIZATION_PRIORITY)]
        other_group = group[~group["signal_type"].isin(BREAKOUT_NORMALIZATION_PRIORITY)]

        if not breakout_group.empty:
            ranked_breakouts = breakout_group.sort_values(
                ["signal_date", "score", "confidence_score"],
                ascending=False,
            ).copy()
            ranked_breakouts["_normalization_priority"] = ranked_breakouts["signal_type"].map(
                BREAKOUT_NORMALIZATION_PRIORITY
            )
            ranked_breakouts = ranked_breakouts.sort_values(
                ["_normalization_priority", "score", "confidence_score", "signal_date"],
                ascending=False,
            )
            primary = ranked_breakouts.iloc[0].copy()
            secondary = ranked_breakouts.iloc[1:]
            primary["secondary_signal_types"] = list(secondary["signal_type"])
            primary["secondary_signal_names"] = [
                signal_names.get(str(signal_type), str(signal_type)) for signal_type in secondary["signal_type"]
            ]
            primary["secondary_signal_count"] = int(len(secondary))
            normalized_rows.append(primary.drop(labels="_normalization_priority", errors="ignore"))

        for _, row in other_group.iterrows():
            enriched = row.copy()
            enriched["secondary_signal_types"] = []
            enriched["secondary_signal_names"] = []
            enriched["secondary_signal_count"] = 0
            normalized_rows.append(enriched)

    normalized = pd.DataFrame(normalized_rows)
    if normalized.empty:
        return normalized
    return normalized.sort_values(["score", "confidence_score", "signal_date"], ascending=False).reset_index(drop=True)


def select_diverse_candidates(results: pd.DataFrame, *, top_n: int, per_signal_limit: int = 3) -> pd.DataFrame:
    if results.empty:
        return results
    normalized_results = normalize_signal_candidates(results)
    ranked_results = normalized_results.copy()
    ranked_results = ranked_results[ranked_results.apply(_candidate_signal_allowed, axis=1)].copy()
    if ranked_results.empty:
        return ranked_results
    if "pretty_ok" in ranked_results.columns:
        pretty_candidates = ranked_results[ranked_results["pretty_ok"].fillna(False)]
        if not pretty_candidates.empty:
            ranked_results = pretty_candidates.copy()
    if "sector_band" in ranked_results.columns:
        sector_priority = {
            "edge_high": 5,
            "strong": 4,
            "edge_low": 3,
            "weak": 2,
            "none": 1,
            "crowded": 0,
        }
        ranked_results["_sector_priority"] = ranked_results["sector_band"].map(sector_priority).fillna(0)
    else:
        ranked_results["_sector_priority"] = 0
    if "filter_ok" in ranked_results.columns:
        ranked_results["_filter_priority"] = ranked_results["filter_ok"].fillna(False).astype(int)
    else:
        ranked_results["_filter_priority"] = 0
    if "pretty_ok" in ranked_results.columns:
        ranked_results["_pretty_priority"] = ranked_results["pretty_ok"].fillna(False).astype(int)
    else:
        ranked_results["_pretty_priority"] = 0
    if "quality_score" in ranked_results.columns:
        ranked_results["_quality_priority"] = pd.to_numeric(ranked_results["quality_score"], errors="coerce").fillna(0.0)
    else:
        ranked_results["_quality_priority"] = 0.0
    ranked_results = ranked_results.sort_values(
        ["_filter_priority", "_pretty_priority", "_quality_priority", "_sector_priority", "score", "confidence_score", "signal_date"],
        ascending=False,
    ).reset_index(drop=True)
    picked_rows = []
    counts: dict[str, int] = {}
    for _, row in ranked_results.iterrows():
        signal_type = str(row["signal_type"])
        signal_limit = CANDIDATE_SIGNAL_LIMITS.get(signal_type, per_signal_limit)
        if counts.get(signal_type, 0) >= signal_limit:
            continue
        counts[signal_type] = counts.get(signal_type, 0) + 1
        picked_rows.append(row)
        if len(picked_rows) >= top_n:
            break
    if not picked_rows:
        return ranked_results.drop(
            columns=["_sector_priority", "_filter_priority", "_pretty_priority", "_quality_priority"],
            errors="ignore",
        ).head(top_n)
    return pd.DataFrame(picked_rows).drop(
        columns=["_sector_priority", "_filter_priority", "_pretty_priority", "_quality_priority"],
        errors="ignore",
    ).reset_index(drop=True)
