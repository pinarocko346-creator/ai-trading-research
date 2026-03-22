from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


def _import_akshare():
    try:
        import akshare as ak  # type: ignore
    except ImportError as exc:
        raise RuntimeError("缺少 akshare，无法加载行业和概念数据。") from exc
    return ak


@dataclass(slots=True)
class SectorFilterConfig:
    cache_dir: Path = Path("data/cache/sector")
    top_industries: int = 35
    top_concepts: int = 50
    max_concepts_per_stock: int = 3
    crowded_min_score: float = 65.0
    min_sector_score: float = 50.0
    edge_high_min_score: float = 40.0
    edge_low_min_score: float = 30.0
    exclude_concept_keywords: list[str] = field(
        default_factory=lambda: ["昨日", "昨日连板", "昨日打二板", "昨日首板"]
    )


def _date_stamp() -> str:
    return pd.Timestamp.today().strftime("%Y%m%d")


def _rankings_cache_path(name: str, config: SectorFilterConfig) -> Path:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    return config.cache_dir / f"{name}_{_date_stamp()}.parquet"


def _mapping_cache_path(config: SectorFilterConfig) -> Path:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    return config.cache_dir / f"symbol_themes_v2_{_date_stamp()}.json"


def _latest_cache_path(pattern: str, config: SectorFilterConfig) -> Path | None:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    matches = sorted(config.cache_dir.glob(pattern), reverse=True)
    return matches[0] if matches else None


def _score_board_row(row: pd.Series) -> float:
    pct_chg = float(pd.to_numeric(row.get("涨跌幅"), errors="coerce") or 0.0)
    turnover = float(pd.to_numeric(row.get("换手率"), errors="coerce") or 0.0)
    up_count = float(pd.to_numeric(row.get("上涨家数"), errors="coerce") or 0.0)
    down_count = float(pd.to_numeric(row.get("下跌家数"), errors="coerce") or 0.0)
    leader_pct = float(pd.to_numeric(row.get("领涨股票-涨跌幅"), errors="coerce") or 0.0)

    breadth_ratio = up_count / max(up_count + down_count, 1.0)
    pct_score = min(max(pct_chg, 0.0), 8.0) / 8.0 * 35
    breadth_score = breadth_ratio * 30
    leader_score = min(max(leader_pct, 0.0), 10.0) / 10.0 * 20
    turnover_score = min(max(turnover, 0.0), 12.0) / 12.0 * 15
    return round(pct_score + breadth_score + leader_score + turnover_score, 2)


def _load_or_fetch_rankings(
    cache_name: str,
    fetcher,
    config: SectorFilterConfig,
) -> pd.DataFrame:
    cache_file = _rankings_cache_path(cache_name, config)
    fallback_cache = _latest_cache_path(f"{cache_name}_*.parquet", config)
    if cache_file.exists():
        return pd.read_parquet(cache_file)
    if fallback_cache is not None:
        return pd.read_parquet(fallback_cache)
    frame = fetcher().copy()
    frame["score"] = frame.apply(_score_board_row, axis=1)
    frame.to_parquet(cache_file, index=False)
    return frame


def fetch_industry_rankings(config: SectorFilterConfig | None = None) -> pd.DataFrame:
    config = config or SectorFilterConfig()
    ak = _import_akshare()
    return _load_or_fetch_rankings("industry_rankings", ak.stock_board_industry_name_em, config)


def fetch_concept_rankings(config: SectorFilterConfig | None = None) -> pd.DataFrame:
    config = config or SectorFilterConfig()
    ak = _import_akshare()

    def _fetch() -> pd.DataFrame:
        frame = ak.stock_board_concept_name_em()
        if config.exclude_concept_keywords:
            pattern = "|".join(config.exclude_concept_keywords)
            frame = frame[~frame["板块名称"].astype(str).str.contains(pattern, na=False)]
        return frame.reset_index(drop=True)

    return _load_or_fetch_rankings("concept_rankings", _fetch, config)


def load_sector_snapshot(config: SectorFilterConfig | None = None) -> dict[str, object]:
    config = config or SectorFilterConfig()
    mapping_cache = _mapping_cache_path(config)
    fallback_mapping_cache = _latest_cache_path("symbol_themes_v2_*.json", config)
    industry_rankings = fetch_industry_rankings(config)
    concept_rankings = fetch_concept_rankings(config)
    industry_score_map = {
        str(row["板块名称"]): float(row["score"])
        for _, row in industry_rankings.iterrows()
    }
    concept_score_map = {
        str(row["板块名称"]): float(row["score"])
        for _, row in concept_rankings.iterrows()
    }

    if mapping_cache.exists() or fallback_mapping_cache is not None:
        cache_to_use = mapping_cache if mapping_cache.exists() else fallback_mapping_cache
        symbol_theme_map = json.loads(cache_to_use.read_text(encoding="utf-8"))
        return {
            "industry_rankings": industry_rankings,
            "concept_rankings": concept_rankings,
            "industry_score_map": industry_score_map,
            "concept_score_map": concept_score_map,
            "symbol_theme_map": symbol_theme_map,
        }

    ak = _import_akshare()
    symbol_theme_map: dict[str, dict[str, object]] = {}

    for _, board in industry_rankings.iterrows():
        board_name = str(board["板块名称"])
        try:
            constituents = ak.stock_board_industry_cons_em(symbol=board_name)
        except Exception:
            continue
        for _, row in constituents.iterrows():
            symbol = str(row["代码"])
            symbol_theme_map.setdefault(symbol, {})
            current = symbol_theme_map[symbol]
            if not current.get("industry_name"):
                current["industry_name"] = board_name

    for _, board in concept_rankings.iterrows():
        board_name = str(board["板块名称"])
        try:
            constituents = ak.stock_board_concept_cons_em(symbol=board_name)
        except Exception:
            continue
        for _, row in constituents.iterrows():
            symbol = str(row["代码"])
            symbol_theme_map.setdefault(symbol, {})
            current = symbol_theme_map[symbol]
            concepts = list(current.get("concept_names", []))
            if board_name not in concepts:
                concepts.append(board_name)
            current["concept_names"] = concepts

    mapping_cache.write_text(json.dumps(symbol_theme_map, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "industry_rankings": industry_rankings,
        "concept_rankings": concept_rankings,
        "industry_score_map": industry_score_map,
        "concept_score_map": concept_score_map,
        "symbol_theme_map": symbol_theme_map,
    }


def build_symbol_theme_payload(symbol: str, snapshot: dict[str, object], config: SectorFilterConfig | None = None) -> dict[str, object]:
    config = config or SectorFilterConfig()
    theme_map = snapshot.get("symbol_theme_map", {})
    payload = dict(theme_map.get(symbol, {})) if isinstance(theme_map, dict) else {}
    industry_name = str(payload.get("industry_name", "") or "")
    industry_score_map = snapshot.get("industry_score_map", {})
    concept_score_map = snapshot.get("concept_score_map", {})
    industry_score = (
        float(industry_score_map.get(industry_name, 0.0))
        if isinstance(industry_score_map, dict)
        else 0.0
    )
    concept_names = list(payload.get("concept_names", []) or [])
    concept_scores = [
        float(concept_score_map.get(str(name), 0.0))
        for name in concept_names
    ] if isinstance(concept_score_map, dict) else []
    top_concept_score = max(concept_scores) if concept_scores else 0.0
    avg_concept_score = sum(concept_scores) / len(concept_scores) if concept_scores else 0.0
    if industry_score > 0 and top_concept_score > 0:
        blended_score = industry_score * 0.65 + top_concept_score * 0.35
    else:
        blended_score = max(industry_score, top_concept_score, avg_concept_score)
    sector_score = round(max(industry_score, top_concept_score, avg_concept_score, blended_score), 2)
    ranked_concepts = sorted(
        zip(concept_names, concept_scores),
        key=lambda item: item[1],
        reverse=True,
    )[: config.max_concepts_per_stock]
    payload["sector_score"] = sector_score
    payload["sector_ok"] = bool(sector_score >= config.min_sector_score)
    payload["industry_name"] = industry_name
    payload["industry_score"] = round(industry_score, 2)
    payload["concept_names"] = [name for name, _ in ranked_concepts]
    payload["concept_scores"] = [round(score, 2) for _, score in ranked_concepts]
    return payload
