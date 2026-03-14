from __future__ import annotations

from typing import Iterable

from app.core.types import ResearchSignal


SIGNAL_PRIORITY_WEIGHT = {
    "selling_climax": 0.55,
    "2b_structure": 1.05,
    "false_breakdown": 1.15,
    "right_shoulder": 1.2,
    "double_breakout": 1.1,
    "strength_emergence": 1.15,
    "jumping_creek": 1.05,
    "pullback_confirmation": 1.15,
    "n_breakout": 0.95,
    "support_resistance_flip": 1.2,
    "spring": 0.95,
    "pattern_breakout": 1.0,
    "first_rebound_after_crash": 0.85,
}


def score_signal(signal: ResearchSignal) -> float:
    base = signal.confidence_score
    weight = SIGNAL_PRIORITY_WEIGHT.get(signal.signal_type, 1.0)
    penalty = 0 if signal.is_valid else 15
    return round(max(0.0, base * weight - penalty), 2)


def rank_signals(signals: Iterable[ResearchSignal]) -> list[ResearchSignal]:
    return sorted(
        signals,
        key=lambda signal: (
            score_signal(signal),
            signal.confidence_score,
            signal.signal_date,
        ),
        reverse=True,
    )


def top_valid_signals(signals: Iterable[ResearchSignal], limit: int = 20) -> list[ResearchSignal]:
    ranked = [signal for signal in rank_signals(signals) if signal.is_valid]
    return ranked[:limit]
