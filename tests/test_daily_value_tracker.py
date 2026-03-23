from __future__ import annotations

import unittest

import pandas as pd

from app.report.daily_value_tracker import (
    build_daily_signal_snapshot,
    build_strategy_scoreboard,
    build_today_expectancy,
)


class DailyValueTrackerTests(unittest.TestCase):
    def test_build_daily_signal_snapshot_assigns_layers(self) -> None:
        scan_results = pd.DataFrame(
            [
                {
                    "symbol": "000001",
                    "name": "A",
                    "signal_date": "2026-03-19",
                    "signal_type": "pattern_breakout",
                    "signal_name": "形态突破",
                    "score": 80.0,
                    "base_score": 60.0,
                    "quality_score": 78.0,
                    "quality_bucket": "A",
                    "confidence_score": 0.9,
                    "pretty_ok": True,
                    "filter_ok": True,
                    "market_ok": True,
                    "sector_ok": True,
                    "market_regime": "risk_on",
                    "sector_band": "strong",
                },
                {
                    "symbol": "000002",
                    "name": "B",
                    "signal_date": "2026-03-19",
                    "signal_type": "false_breakdown",
                    "signal_name": "假诱空",
                    "score": 55.0,
                    "base_score": 40.0,
                    "quality_score": 61.0,
                    "quality_bucket": "B",
                    "confidence_score": 0.7,
                    "pretty_ok": True,
                    "filter_ok": False,
                    "market_ok": True,
                    "sector_ok": False,
                    "market_regime": "neutral",
                    "sector_band": "weak",
                },
                {
                    "symbol": "000003",
                    "name": "C",
                    "signal_date": "2026-03-19",
                    "signal_type": "2b_structure",
                    "signal_name": "2B结构",
                    "score": 32.0,
                    "base_score": 25.0,
                    "quality_score": 45.0,
                    "quality_bucket": "C",
                    "confidence_score": 0.6,
                    "pretty_ok": False,
                    "filter_ok": False,
                    "market_ok": False,
                    "sector_ok": False,
                    "market_regime": "risk_off",
                    "sector_band": "none",
                },
            ]
        )
        top_rows = scan_results.iloc[[0]].copy()

        snapshot = build_daily_signal_snapshot(
            scan_results,
            top_rows,
            run_id="20260319",
            generated_at="2026-03-19T16:40:00",
            universe_scope="tradeable",
            latest_trade_date="2026-03-19",
        )

        layer_map = dict(zip(snapshot["symbol"], snapshot["layer"]))
        self.assertEqual(layer_map["000001"], "executable")
        self.assertEqual(layer_map["000002"], "candidate")
        self.assertEqual(layer_map["000003"], "watch")
        self.assertTrue(bool(snapshot.loc[snapshot["symbol"] == "000001", "is_top_candidate"].iloc[0]))

    def test_scoreboard_and_expectancy_merge(self) -> None:
        forward_frame = pd.DataFrame(
            [
                {
                    "symbol": "000001",
                    "signal_type": "pattern_breakout",
                    "signal_date": "2026-03-17",
                    "layer": "executable",
                    "quality_score": 80.0,
                    "return_1d": 0.01,
                    "return_3d": 0.03,
                    "return_5d": 0.05,
                    "return_10d": 0.08,
                    "mfe_1d": 0.02,
                    "mfe_3d": 0.04,
                    "mfe_5d": 0.06,
                    "mfe_10d": 0.09,
                    "mae_1d": -0.01,
                    "mae_3d": -0.02,
                    "mae_5d": -0.03,
                    "mae_10d": -0.04,
                },
                {
                    "symbol": "000002",
                    "signal_type": "pattern_breakout",
                    "signal_date": "2026-03-18",
                    "layer": "executable",
                    "quality_score": 76.0,
                    "return_1d": -0.01,
                    "return_3d": 0.02,
                    "return_5d": 0.04,
                    "return_10d": 0.06,
                    "mfe_1d": 0.01,
                    "mfe_3d": 0.03,
                    "mfe_5d": 0.05,
                    "mfe_10d": 0.07,
                    "mae_1d": -0.02,
                    "mae_3d": -0.03,
                    "mae_5d": -0.04,
                    "mae_10d": -0.05,
                },
                {
                    "symbol": "000003",
                    "signal_type": "false_breakdown",
                    "signal_date": "2026-03-18",
                    "layer": "candidate",
                    "quality_score": 60.0,
                    "return_1d": 0.0,
                    "return_3d": 0.01,
                    "return_5d": 0.02,
                    "return_10d": 0.01,
                    "mfe_1d": 0.02,
                    "mfe_3d": 0.03,
                    "mfe_5d": 0.04,
                    "mfe_10d": 0.05,
                    "mae_1d": -0.01,
                    "mae_3d": -0.01,
                    "mae_5d": -0.02,
                    "mae_10d": -0.03,
                },
            ]
        )

        scoreboard = build_strategy_scoreboard(forward_frame, windows=(20,))
        self.assertFalse(scoreboard.empty)
        executable_row = scoreboard[
            (scoreboard["layer"] == "executable") & (scoreboard["signal_type"] == "pattern_breakout")
        ].iloc[0]
        self.assertEqual(int(executable_row["signal_count"]), 2)
        self.assertAlmostEqual(float(executable_row["avg_return_5d"]), 0.045, places=4)
        self.assertAlmostEqual(float(executable_row["win_rate_5d"]), 1.0, places=4)

        today = pd.DataFrame(
            [
                {
                    "symbol": "000010",
                    "signal_type": "pattern_breakout",
                    "layer": "executable",
                    "quality_score": 79.0,
                    "score": 88.0,
                    "is_top_candidate": True,
                    "candidate_rank": 1,
                }
            ]
        )
        expectancy = build_today_expectancy(today, scoreboard, reference_window=20)
        self.assertAlmostEqual(float(expectancy.iloc[0]["expected_avg_return_5d"]), 0.045, places=4)
        self.assertAlmostEqual(float(expectancy.iloc[0]["expected_win_rate_5d"]), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
