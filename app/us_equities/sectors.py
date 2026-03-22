from __future__ import annotations

from app.us_equities.config import USEquitiesSectorConfig


def sector_memberships(symbol: str, config: USEquitiesSectorConfig) -> list[str]:
    memberships: list[str] = []
    for basket_name, members in config.baskets.items():
        if symbol in members:
            memberships.append(basket_name)
    return memberships


def compute_sector_summary(
    symbol_states: dict[str, dict[str, object]],
    config: USEquitiesSectorConfig,
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for basket_name, members in config.baskets.items():
        present = [symbol for symbol in members if symbol in symbol_states]
        if not present:
            summary[basket_name] = {
                "member_count": 0,
                "trend_count": 0,
                "breakout_count": 0,
                "bottom_count": 0,
                "weekly_trend_count": 0,
                "score": 0.0,
            }
            continue
        trend_count = sum(1 for symbol in present if symbol_states[symbol]["1d"]["trend_ok"])
        breakout_count = sum(1 for symbol in present if symbol_states[symbol]["1d"]["breakout_recent"])
        bottom_count = sum(1 for symbol in present if symbol_states[symbol]["1d"]["bottom_recent"])
        weekly_trend_count = sum(1 for symbol in present if symbol_states[symbol]["1w"]["trend_ok"])
        score = trend_count * 2.0 + breakout_count * 3.0 + bottom_count * 1.5 + weekly_trend_count * 1.5
        summary[basket_name] = {
            "member_count": len(present),
            "trend_count": trend_count,
            "breakout_count": breakout_count,
            "bottom_count": bottom_count,
            "weekly_trend_count": weekly_trend_count,
            "score": round(score, 2),
        }
    return summary


def build_sector_context(
    symbol: str,
    sector_summary: dict[str, dict[str, object]],
    config: USEquitiesSectorConfig,
) -> dict[str, object]:
    memberships = sector_memberships(symbol, config)
    if not memberships:
        return {
            "sector_name": "",
            "sector_member_count": 0,
            "sector_trend_count": 0,
            "sector_breakout_count": 0,
            "sector_bottom_count": 0,
            "sector_weekly_trend_count": 0,
            "sector_score": 0.0,
            "sector_resonance_ok": False,
        }
    best = max(memberships, key=lambda basket: sector_summary.get(basket, {}).get("score", 0.0))
    context = sector_summary.get(best, {})
    resonance_ok = bool(
        context.get("trend_count", 0) >= config.min_resonance_members
        or context.get("breakout_count", 0) >= config.min_breakout_members
    )
    return {
        "sector_name": best,
        "sector_member_count": int(context.get("member_count", 0) or 0),
        "sector_trend_count": int(context.get("trend_count", 0) or 0),
        "sector_breakout_count": int(context.get("breakout_count", 0) or 0),
        "sector_bottom_count": int(context.get("bottom_count", 0) or 0),
        "sector_weekly_trend_count": int(context.get("weekly_trend_count", 0) or 0),
        "sector_score": float(context.get("score", 0.0) or 0.0),
        "sector_resonance_ok": resonance_ok,
    }
