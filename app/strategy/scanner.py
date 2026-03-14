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
    "jumping_creek": 3,
    "n_breakout": 2,
    "double_breakout": 1,
}


@dataclass(slots=True)
class ScanConfig:
    max_symbols: int = 100
    cache_dir: Path = Path("data/cache")
    per_signal_limit: int = 3
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
    if sector_score >= config.min_sector_score + 10:
        return 6.0
    if sector_score >= config.min_sector_score:
        return 3.0
    if sector_score >= config.edge_high_min_score:
        return 0.5
    if sector_score >= config.edge_low_min_score:
        return -2.5
    if sector_score > 0:
        return -4.0
    return -6.0


def _sector_band(theme_payload: dict[str, object], config: SectorFilterConfig) -> str:
    sector_score = float(theme_payload.get("sector_score", 0.0) or 0.0)
    if sector_score >= config.min_sector_score:
        return "strong"
    if sector_score >= config.edge_high_min_score:
        return "edge_high"
    if sector_score >= config.edge_low_min_score:
        return "edge_low"
    if sector_score > 0:
        return "weak"
    return "none"


def _filter_ok(market_snapshot: dict[str, object], theme_payload: dict[str, object], config: SectorFilterConfig) -> bool:
    regime = str(market_snapshot.get("market_regime", "neutral"))
    sector_band = _sector_band(theme_payload, config)
    if regime == "risk_off":
        return sector_band == "strong"
    if regime == "risk_on":
        return sector_band in {"strong", "edge_high"}
    return sector_band in {"strong", "edge_high"}


def load_default_universe(
    universe_config: UniverseConfig | None = None,
    *,
    max_symbols: int = 100,
) -> pd.DataFrame:
    universe_config = universe_config or UniverseConfig()
    spot = load_a_share_spot()
    if "volume" in spot.columns:
        spot["avg_volume_20"] = pd.to_numeric(spot["volume"], errors="coerce")
    filtered = filter_tradeable_universe(spot, universe_config)
    if "volume" in filtered.columns:
        filtered["volume"] = pd.to_numeric(filtered["volume"], errors="coerce")
        filtered = filtered.sort_values("volume", ascending=False)
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
    ingest_config = DataIngestConfig(cache_dir=scan_config.cache_dir)
    signal_names = {item.code: item.name for item in build_signal_catalog()}
    results: list[dict[str, object]] = []
    full_spot = load_a_share_spot()
    market_snapshot = load_market_snapshot(scan_config.market_filter, spot_frame=full_spot)
    sector_snapshot = load_sector_snapshot(scan_config.sector_filter)

    for _, row in universe.head(scan_config.max_symbols).iterrows():
        symbol = str(row["symbol"])
        try:
            history = fetch_a_share_history(symbol, ingest_config)
        except Exception:
            continue
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
            payload["filter_ok"] = _filter_ok(market_snapshot, theme_payload, scan_config.sector_filter)
            results.append(payload)

    frame = pd.DataFrame(results)
    if frame.empty:
        return frame
    return frame.sort_values(["score", "confidence_score", "signal_date"], ascending=False).reset_index(drop=True)


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
    picked_rows = []
    counts: dict[str, int] = {}
    for _, row in normalized_results.iterrows():
        signal_type = str(row["signal_type"])
        if counts.get(signal_type, 0) >= per_signal_limit:
            continue
        counts[signal_type] = counts.get(signal_type, 0) + 1
        picked_rows.append(row)
        if len(picked_rows) >= top_n:
            break
    if not picked_rows:
        return normalized_results.head(top_n)
    return pd.DataFrame(picked_rows).reset_index(drop=True)
