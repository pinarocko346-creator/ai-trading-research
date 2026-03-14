from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any


@dataclass(slots=True)
class SignalDefinition:
    code: str
    name: str
    stage: str
    description: str
    priority: int
    programmable: bool
    notes: str = ""


@dataclass(slots=True)
class ResearchSignal:
    signal_type: str
    symbol: str
    signal_date: date
    confidence_score: float
    trend_ok: bool
    location_ok: bool
    pattern_ok: bool
    volume_ok: bool
    entry_price: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    invalid_reason: str | None = None
    risk_tags: list[str] = field(default_factory=list)
    factors: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.invalid_reason is None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["signal_date"] = self.signal_date.isoformat()
        return payload


@dataclass(slots=True)
class BacktestTrade:
    symbol: str
    signal_type: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    return_pct: float
    hold_days: int
    exit_reason: str
    confidence_score: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["entry_date"] = self.entry_date.isoformat()
        payload["exit_date"] = self.exit_date.isoformat()
        return payload
