from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app.data.universe import load_a_share_spot


def _import_akshare():
    try:
        import akshare as ak  # type: ignore
    except ImportError as exc:
        raise RuntimeError("缺少 akshare，无法加载市场层数据。") from exc
    return ak


@dataclass(slots=True)
class MarketFilterConfig:
    cache_dir: Path = Path("data/cache/market")
    index_symbols: dict[str, str] = field(
        default_factory=lambda: {
            "上证指数": "000001",
            "沪深300": "000300",
            "创业板指": "399006",
        }
    )
    index_start_date: str = "2023-01-01"
    min_positive_index_count: int = 1
    min_up_ratio: float = 0.45
    min_limit_up_down_ratio: float = 1.0
    limit_up_pct: float = 9.5
    limit_down_pct: float = -9.5
    min_market_score: float = 50.0
    risk_on_score: float = 70.0
    risk_off_score: float = 45.0


def _normalize_index_history(frame: pd.DataFrame) -> pd.DataFrame:
    column_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "涨跌幅": "pct_chg",
    }
    normalized = frame.rename(columns=column_map).copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    for column in ("open", "close", "high", "low", "volume", "amount", "pct_chg"):
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.sort_values("date").reset_index(drop=True)
    normalized["ma_20"] = normalized["close"].rolling(20).mean()
    normalized["ma_60"] = normalized["close"].rolling(60).mean()
    return normalized


def _index_cache_path(index_code: str, config: MarketFilterConfig) -> Path:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    return config.cache_dir / f"index_{index_code}.parquet"


def fetch_index_history(index_code: str, config: MarketFilterConfig) -> pd.DataFrame:
    cache_file = _index_cache_path(index_code, config)
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    ak = _import_akshare()
    raw = ak.index_zh_a_hist(
        symbol=index_code,
        period="daily",
        start_date=config.index_start_date.replace("-", ""),
        end_date=pd.Timestamp.today().strftime("%Y%m%d"),
    )
    normalized = _normalize_index_history(raw)
    normalized.to_parquet(cache_file, index=False)
    return normalized


def score_market_snapshot(
    index_histories: dict[str, pd.DataFrame],
    breadth_frame: pd.DataFrame,
    config: MarketFilterConfig,
) -> dict[str, object]:
    index_details: list[dict[str, object]] = []
    positive_index_count = 0
    for name, history in index_histories.items():
        latest = history.dropna().iloc[-1]
        trend_ok = bool(latest["close"] > latest["ma_20"] and latest["ma_20"] > latest["ma_60"])
        positive_index_count += int(trend_ok)
        index_details.append(
            {
                "name": name,
                "close": float(latest["close"]),
                "ma20": float(latest["ma_20"]),
                "ma60": float(latest["ma_60"]),
                "trend_ok": trend_ok,
            }
        )

    breadth = breadth_frame.copy()
    breadth["pct_chg"] = pd.to_numeric(breadth["pct_chg"], errors="coerce")
    breadth = breadth.dropna(subset=["pct_chg"])
    total_count = max(len(breadth), 1)
    up_count = int((breadth["pct_chg"] > 0).sum())
    limit_up_count = int((breadth["pct_chg"] >= config.limit_up_pct).sum())
    limit_down_count = int((breadth["pct_chg"] <= config.limit_down_pct).sum())
    up_ratio = up_count / total_count
    limit_up_down_ratio = limit_up_count / max(limit_down_count, 1)

    index_score = positive_index_count / max(len(index_histories), 1) * 50
    breadth_score = min(up_ratio / max(config.min_up_ratio, 1e-6), 1.0) * 25
    limit_score = min(limit_up_down_ratio / max(config.min_limit_up_down_ratio, 1e-6), 1.0) * 25
    market_score = round(index_score + breadth_score + limit_score, 2)

    market_ok = bool(
        positive_index_count >= config.min_positive_index_count
        and up_ratio >= config.min_up_ratio
        and limit_up_down_ratio >= config.min_limit_up_down_ratio
        and market_score >= config.min_market_score
    )
    if market_score >= config.risk_on_score:
        market_regime = "risk_on"
    elif market_score >= config.risk_off_score:
        market_regime = "neutral"
    else:
        market_regime = "risk_off"

    return {
        "market_ok": market_ok,
        "market_score": market_score,
        "market_regime": market_regime,
        "market_positive_index_count": positive_index_count,
        "market_up_ratio": round(up_ratio, 4),
        "market_limit_up_count": limit_up_count,
        "market_limit_down_count": limit_down_count,
        "market_limit_up_down_ratio": round(limit_up_down_ratio, 4),
        "market_index_details": index_details,
    }


def load_market_snapshot(
    config: MarketFilterConfig | None = None,
    *,
    spot_frame: pd.DataFrame | None = None,
) -> dict[str, object]:
    config = config or MarketFilterConfig()
    breadth_frame = spot_frame if spot_frame is not None else load_a_share_spot()
    breadth_frame = breadth_frame.rename(columns={"涨跌幅": "pct_chg"}).copy()
    index_histories = {
        name: fetch_index_history(index_code, config)
        for name, index_code in config.index_symbols.items()
    }
    return score_market_snapshot(index_histories, breadth_frame, config)
