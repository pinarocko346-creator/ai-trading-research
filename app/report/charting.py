from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from app.core.types import ResearchSignal
from app.features.price_features import build_price_features


def plot_signal_context(
    frame: pd.DataFrame,
    signal: ResearchSignal,
    output_path: str | Path,
    *,
    lookback_bars: int = 80,
) -> Path:
    featured = build_price_features(frame).copy()
    featured["date"] = pd.to_datetime(featured["date"])
    signal_ts = pd.Timestamp(signal.signal_date)
    context = featured[featured["date"] <= signal_ts].tail(lookback_bars).copy()
    if context.empty:
        raise ValueError("没有可用于绘图的上下文数据")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(context["date"], context["close"], label="close", color="#1f77b4", linewidth=1.8)
    if "ma_20" in context.columns:
        ax.plot(context["date"], context["ma_20"], label="ma20", color="#ff7f0e", linewidth=1.2)
    if "ma_60" in context.columns:
        ax.plot(context["date"], context["ma_60"], label="ma60", color="#2ca02c", linewidth=1.2)

    signal_row = context[context["date"] == signal_ts]
    if not signal_row.empty:
        ax.scatter(
            signal_row["date"],
            signal_row["close"],
            color="#d62728",
            s=60,
            zorder=5,
            label=f"signal:{signal.signal_type}",
        )
    if signal.stop_price is not None:
        ax.axhline(signal.stop_price, color="#9467bd", linestyle="--", linewidth=1, label="stop")
    if signal.target_price is not None:
        ax.axhline(signal.target_price, color="#8c564b", linestyle="--", linewidth=1, label="target")
    if "prior_support" in signal.factors:
        ax.axhline(float(signal.factors["prior_support"]), color="#17becf", linestyle=":", linewidth=1, label="support")
    if "breakout_level" in signal.factors:
        ax.axhline(float(signal.factors["breakout_level"]), color="#17becf", linestyle=":", linewidth=1, label="breakout")
    if "flip_level" in signal.factors:
        ax.axhline(float(signal.factors["flip_level"]), color="#17becf", linestyle=":", linewidth=1, label="flip")
    if "resistance" in signal.factors:
        ax.axhline(float(signal.factors["resistance"]), color="#17becf", linestyle=":", linewidth=1, label="resistance")
    if "neckline" in signal.factors:
        ax.axhline(float(signal.factors["neckline"]), color="#17becf", linestyle=":", linewidth=1, label="neckline")
    if "box_low" in signal.factors:
        ax.axhline(float(signal.factors["box_low"]), color="#7f7f7f", linestyle=":", linewidth=1, label="box_low")
    if "box_high" in signal.factors:
        ax.axhline(float(signal.factors["box_high"]), color="#7f7f7f", linestyle=":", linewidth=1, label="box_high")

    title = f"{signal.symbol} {signal.signal_type} {signal.signal_date.isoformat()} score={signal.confidence_score}"
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("price")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=144)
    plt.close(fig)
    return path
