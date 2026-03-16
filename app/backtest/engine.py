from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from app.core.types import BacktestTrade, ResearchSignal


@dataclass(slots=True)
class BacktestConfig:
    max_hold_days: int = 10
    stop_buffer_pct: float = 0.002
    target_multiple: float = 2.0
    skip_limit_up_open: bool = True
    skip_limit_down_open: bool = True


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["date"] = pd.to_datetime(prepared["date"])
    prepared = prepared.sort_values("date").reset_index(drop=True)
    return prepared


def _locate_signal_index(frame: pd.DataFrame, signal: ResearchSignal) -> int | None:
    matched = frame.index[frame["date"].dt.date == signal.signal_date].tolist()
    return matched[0] if matched else None


def _trade_is_blocked(bar: pd.Series, config: BacktestConfig) -> bool:
    pct_gap = (bar["open"] - bar["close_prev"]) / max(bar["close_prev"], 1e-6)
    if config.skip_limit_up_open and pct_gap >= 0.095:
        return True
    if config.skip_limit_down_open and pct_gap <= -0.095:
        return True
    return False


def run_signal_backtest(
    price_map: dict[str, pd.DataFrame],
    signals: Iterable[ResearchSignal],
    config: BacktestConfig | None = None,
) -> list[BacktestTrade]:
    config = config or BacktestConfig()
    trades: list[BacktestTrade] = []
    for signal in signals:
        if not signal.is_valid:
            continue
        frame = price_map.get(signal.symbol)
        if frame is None:
            continue
        prepared = _prepare_frame(frame)
        prepared["close_prev"] = prepared["close"].shift(1)
        signal_index = _locate_signal_index(prepared, signal)
        if signal_index is None or signal_index + 1 >= len(prepared):
            continue

        entry_bar = prepared.iloc[signal_index + 1]
        if _trade_is_blocked(entry_bar, config):
            continue
        entry_price = float(entry_bar["open"])
        stop_price = float(signal.stop_price or entry_price * 0.95) * (1 - config.stop_buffer_pct)
        risk_per_share = max(entry_price - stop_price, entry_price * 0.01)
        target_price = float(signal.target_price or (entry_price + risk_per_share * config.target_multiple))

        exit_price = float(entry_bar["close"])
        exit_date = entry_bar["date"].date()
        exit_reason = "time_exit"
        max_high = float(entry_bar["high"])
        min_low = float(entry_bar["low"])

        lookahead = prepared.iloc[signal_index + 1 : signal_index + 1 + config.max_hold_days]
        for _, bar in lookahead.iterrows():
            exit_date = bar["date"].date()
            max_high = max(max_high, float(bar["high"]))
            min_low = min(min_low, float(bar["low"]))
            if bar["low"] <= stop_price:
                exit_price = stop_price
                exit_reason = "stop_loss"
                break
            if bar["high"] >= target_price:
                exit_price = target_price
                exit_reason = "target_hit"
                break
            exit_price = float(bar["close"])
            if bar["close"] < bar.get("ma_10", bar["close"]):
                exit_reason = "trend_break"
                break

        trades.append(
            BacktestTrade(
                symbol=signal.symbol,
                signal_type=signal.signal_type,
                signal_date=signal.signal_date,
                entry_date=entry_bar["date"].date(),
                exit_date=exit_date,
                entry_price=entry_price,
                exit_price=exit_price,
                return_pct=round((exit_price - entry_price) / entry_price, 4),
                hold_days=max((exit_date - entry_bar["date"].date()).days, 1),
                exit_reason=exit_reason,
                confidence_score=signal.confidence_score,
                mfe_pct=round((max_high - entry_price) / entry_price, 4),
                mae_pct=round((min_low - entry_price) / entry_price, 4),
            )
        )
    return trades
