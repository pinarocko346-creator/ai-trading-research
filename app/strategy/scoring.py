from __future__ import annotations

from typing import Iterable

from app.core.types import ResearchSignal


SIGNAL_PRIORITY_WEIGHT = {
    "selling_climax": 0.2,
    "2b_structure": 1.05,
    "false_breakdown": 0.9,
    "right_shoulder": 1.25,
    "double_breakout": 0.8,
    "strength_emergence": 1.2,
    "jumping_creek": 0.65,
    "cup_with_handle_watch": 0.45,
    "cup_with_handle": 1.3,
    "cup_with_handle_strict": 1.45,
    "cup_with_handle_leader": 1.6,
    "pullback_confirmation": 0.9,
    "n_breakout": 0.6,
    "support_resistance_flip": 1.25,
    "spring": 0.95,
    "pattern_breakout": 1.0,
    "first_rebound_after_crash": 0.9,
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
