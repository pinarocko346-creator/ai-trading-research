from __future__ import annotations

import pandas as pd

from app.core.types import BacktestTrade


def trades_to_frame(trades: list[BacktestTrade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "symbol",
                "signal_type",
                "entry_date",
                "exit_date",
                "entry_price",
                "exit_price",
                "return_pct",
                "hold_days",
                "exit_reason",
                "confidence_score",
            ]
        )
    return pd.DataFrame([trade.to_dict() for trade in trades])


def summarize_trades(trades: list[BacktestTrade]) -> dict[str, float]:
    frame = trades_to_frame(trades)
    if frame.empty:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_return_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "avg_hold_days": 0.0,
        }

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
    }


def summarize_by_signal_type(trades: list[BacktestTrade]) -> pd.DataFrame:
    frame = trades_to_frame(trades)
    if frame.empty:
        return frame
    summary = (
        frame.groupby("signal_type")
        .agg(
            trade_count=("signal_type", "count"),
            win_rate=("return_pct", lambda series: float((series > 0).mean())),
            avg_return_pct=("return_pct", "mean"),
            avg_hold_days=("hold_days", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values("avg_return_pct", ascending=False).reset_index(drop=True)
