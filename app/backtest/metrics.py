from __future__ import annotations

import pandas as pd

from app.core.types import BacktestTrade


def _sorted_trade_frame(frame: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [column for column in ("signal_date", "entry_date", "exit_date", "symbol") if column in frame.columns]
    if not sort_columns:
        return frame.reset_index(drop=True)
    sorted_frame = frame.copy()
    for column in ("signal_date", "entry_date", "exit_date"):
        if column in sorted_frame.columns:
            sorted_frame[column] = pd.to_datetime(sorted_frame[column], errors="coerce")
    return sorted_frame.sort_values(sort_columns).reset_index(drop=True)


def summarize_trade_frame(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_return_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "avg_hold_days": 0.0,
            "avg_mfe_pct": 0.0,
            "avg_mae_pct": 0.0,
        }

    frame = _sorted_trade_frame(frame)
    gross_profit = frame.loc[frame["return_pct"] > 0, "return_pct"].sum()
    gross_loss = -frame.loc[frame["return_pct"] < 0, "return_pct"].sum()
    equity = (1 + frame["return_pct"]).cumprod()
    drawdown = equity / equity.cummax() - 1
    return {
        "trade_count": int(len(frame)),
        "win_rate": round(float((frame["return_pct"] > 0).mean()), 4),
        "avg_return_pct": round(float(frame["return_pct"].mean()), 4),
        "profit_factor": round(float(gross_profit / gross_loss), 4) if gross_loss else float("inf"),
        "max_drawdown": round(float(drawdown.min()), 4),
        "avg_hold_days": round(float(frame["hold_days"].mean()), 2),
        "avg_mfe_pct": round(float(frame["mfe_pct"].mean()), 4) if "mfe_pct" in frame.columns else 0.0,
        "avg_mae_pct": round(float(frame["mae_pct"].mean()), 4) if "mae_pct" in frame.columns else 0.0,
    }


def trades_to_frame(trades: list[BacktestTrade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "symbol",
                "signal_type",
                "signal_date",
                "entry_date",
                "exit_date",
                "entry_price",
                "exit_price",
                "return_pct",
                "hold_days",
                "exit_reason",
                "confidence_score",
                "mfe_pct",
                "mae_pct",
            ]
        )
    return pd.DataFrame([trade.to_dict() for trade in trades])


def summarize_trades(trades: list[BacktestTrade]) -> dict[str, float]:
    frame = trades_to_frame(trades)
    return summarize_trade_frame(frame)


def summarize_by_signal_type(trades: list[BacktestTrade]) -> pd.DataFrame:
    frame = trades_to_frame(trades)
    if frame.empty:
        return frame
    rows = []
    for signal_type, group in frame.groupby("signal_type", dropna=False):
        summary = summarize_trade_frame(group)
        summary["signal_type"] = signal_type
        rows.append(summary)
    return pd.DataFrame(rows).sort_values("avg_return_pct", ascending=False).reset_index(drop=True)
