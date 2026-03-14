from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from app.backtest.engine import BacktestConfig, run_signal_backtest
from app.backtest.metrics import summarize_trades
from app.core.types import ResearchSignal


class BacktestTests(unittest.TestCase):
    def test_backtest_generates_trade_summary(self) -> None:
        frame = pd.DataFrame(
            [
                {"date": "2024-01-01", "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1, "volume": 1},
                {"date": "2024-01-02", "open": 10.1, "high": 10.4, "low": 10.0, "close": 10.3, "volume": 1},
                {"date": "2024-01-03", "open": 10.3, "high": 10.8, "low": 10.2, "close": 10.7, "volume": 1},
                {"date": "2024-01-04", "open": 10.6, "high": 10.9, "low": 10.5, "close": 10.8, "volume": 1},
            ]
        )
        signal = ResearchSignal(
            signal_type="double_breakout",
            symbol="000001",
            signal_date=date(2024, 1, 1),
            confidence_score=90,
            trend_ok=True,
            location_ok=True,
            pattern_ok=True,
            volume_ok=True,
            entry_price=10.1,
            stop_price=9.8,
            target_price=10.7,
        )
        trades = run_signal_backtest({"000001": frame}, [signal], BacktestConfig(max_hold_days=3))
        summary = summarize_trades(trades)
        self.assertEqual(summary["trade_count"], 1)
        self.assertGreater(summary["avg_return_pct"], 0)


if __name__ == "__main__":
    unittest.main()
