from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from app.data.universe import UniverseConfig
from app.strategy.scanner import load_default_universe, normalize_signal_candidates, select_diverse_candidates


class ScannerTests(unittest.TestCase):
    def test_load_default_universe_allows_full_universe_with_zero_limit(self) -> None:
        spot = pd.DataFrame(
            [
                {"symbol": "000001", "name": "平安银行", "close": 12.0, "volume": 1000, "turnover_rate": 1.2},
                {"symbol": "000002", "name": "万科A", "close": 11.0, "volume": 2000, "turnover_rate": 1.1},
                {"symbol": "000003", "name": "测试股", "close": 10.0, "volume": 1500, "turnover_rate": 1.3},
            ]
        )
        universe_config = UniverseConfig(min_close=1.0, min_avg_volume=0.0, min_turnover_rate=0.0)

        with patch("app.strategy.scanner.load_a_share_spot", return_value=spot):
            universe = load_default_universe(universe_config, max_symbols=0)

        self.assertEqual(len(universe), 3)
        self.assertListEqual(universe["symbol"].tolist(), ["000002", "000003", "000001"])

    def test_normalize_signal_candidates_collapses_breakout_cluster(self) -> None:
        results = pd.DataFrame(
            [
                {
                    "signal_type": "double_breakout",
                    "symbol": "000001",
                    "signal_date": "2026-03-13",
                    "score": 118.75,
                    "confidence_score": 95,
                },
                {
                    "signal_type": "jumping_creek",
                    "symbol": "000001",
                    "signal_date": "2026-03-13",
                    "score": 104.5,
                    "confidence_score": 95,
                },
                {
                    "signal_type": "n_breakout",
                    "symbol": "000001",
                    "signal_date": "2026-03-13",
                    "score": 90.25,
                    "confidence_score": 95,
                },
                {
                    "signal_type": "false_breakdown",
                    "symbol": "000002",
                    "signal_date": "2026-03-13",
                    "score": 104.5,
                    "confidence_score": 95,
                },
            ]
        )

        normalized = normalize_signal_candidates(results)

        self.assertEqual(len(normalized), 2)
        primary = normalized[normalized["symbol"] == "000001"].iloc[0]
        self.assertEqual(primary["signal_type"], "jumping_creek")
        self.assertEqual(primary["secondary_signal_count"], 2)
        self.assertCountEqual(primary["secondary_signal_types"], ["double_breakout", "n_breakout"])

    def test_select_diverse_candidates_uses_normalized_rows(self) -> None:
        results = pd.DataFrame(
            [
                {
                    "signal_type": "double_breakout",
                    "symbol": "000001",
                    "signal_date": "2026-03-13",
                    "score": 118.75,
                    "confidence_score": 95,
                },
                {
                    "signal_type": "jumping_creek",
                    "symbol": "000001",
                    "signal_date": "2026-03-13",
                    "score": 104.5,
                    "confidence_score": 95,
                },
                {
                    "signal_type": "double_breakout",
                    "symbol": "000002",
                    "signal_date": "2026-03-13",
                    "score": 117.0,
                    "confidence_score": 94,
                },
            ]
        )

        selected = select_diverse_candidates(results, top_n=3, per_signal_limit=3)

        self.assertEqual(len(selected), 2)
        self.assertEqual(selected.iloc[0]["signal_type"], "double_breakout")
        self.assertEqual(selected.iloc[1]["signal_type"], "jumping_creek")

    def test_select_diverse_candidates_prioritizes_filter_ok_rows(self) -> None:
        results = pd.DataFrame(
            [
                {
                    "signal_type": "double_breakout",
                    "symbol": "000001",
                    "signal_date": "2026-03-13",
                    "score": 110.0,
                    "confidence_score": 95,
                    "filter_ok": False,
                    "sector_band": "crowded",
                },
                {
                    "signal_type": "jumping_creek",
                    "symbol": "000002",
                    "signal_date": "2026-03-13",
                    "score": 105.0,
                    "confidence_score": 94,
                    "filter_ok": True,
                    "sector_band": "edge_high",
                },
            ]
        )

        selected = select_diverse_candidates(results, top_n=1, per_signal_limit=3)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["symbol"], "000002")

    def test_select_diverse_candidates_filters_out_ugly_shapes(self) -> None:
        results = pd.DataFrame(
            [
                {
                    "signal_type": "double_breakout",
                    "symbol": "000001",
                    "signal_date": "2026-03-13",
                    "score": 120.0,
                    "confidence_score": 95,
                    "filter_ok": True,
                    "pretty_ok": False,
                    "quality_score": 40.0,
                    "sector_band": "edge_high",
                },
                {
                    "signal_type": "cup_with_handle",
                    "symbol": "000002",
                    "signal_date": "2026-03-13",
                    "score": 112.0,
                    "confidence_score": 95,
                    "filter_ok": True,
                    "pretty_ok": True,
                    "quality_score": 78.0,
                    "sector_band": "edge_high",
                },
            ]
        )

        selected = select_diverse_candidates(results, top_n=2, per_signal_limit=3)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["symbol"], "000002")

    def test_select_diverse_candidates_blocks_disabled_weak_signals(self) -> None:
        results = pd.DataFrame(
            [
                {
                    "signal_type": "selling_climax",
                    "symbol": "000001",
                    "signal_date": "2026-03-13",
                    "score": 120.0,
                    "confidence_score": 95,
                    "filter_ok": True,
                    "pretty_ok": True,
                    "quality_score": 80.0,
                    "sector_band": "edge_low",
                },
                {
                    "signal_type": "pullback_confirmation",
                    "symbol": "000002",
                    "signal_date": "2026-03-13",
                    "score": 110.0,
                    "confidence_score": 94,
                    "filter_ok": True,
                    "pretty_ok": True,
                    "quality_score": 78.0,
                    "sector_band": "edge_high",
                },
                {
                    "signal_type": "false_breakdown",
                    "symbol": "000003",
                    "signal_date": "2026-03-13",
                    "score": 108.0,
                    "confidence_score": 95,
                    "filter_ok": True,
                    "pretty_ok": True,
                    "quality_score": 79.0,
                    "sector_band": "edge_low",
                },
            ]
        )

        selected = select_diverse_candidates(results, top_n=3, per_signal_limit=3)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["signal_type"], "false_breakdown")

    def test_select_diverse_candidates_requires_filter_for_certain_weak_signals(self) -> None:
        results = pd.DataFrame(
            [
                {
                    "signal_type": "jumping_creek",
                    "symbol": "000001",
                    "signal_date": "2026-03-13",
                    "score": 112.0,
                    "confidence_score": 95,
                    "filter_ok": False,
                    "pretty_ok": True,
                    "quality_score": 82.0,
                    "sector_band": "weak",
                },
                {
                    "signal_type": "n_breakout",
                    "symbol": "000002",
                    "signal_date": "2026-03-13",
                    "score": 111.0,
                    "confidence_score": 94,
                    "filter_ok": True,
                    "pretty_ok": True,
                    "quality_score": 76.0,
                    "sector_band": "edge_high",
                },
            ]
        )

        selected = select_diverse_candidates(results, top_n=3, per_signal_limit=3)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["signal_type"], "n_breakout")

    def test_select_diverse_candidates_requires_filter_for_double_breakout(self) -> None:
        results = pd.DataFrame(
            [
                {
                    "signal_type": "double_breakout",
                    "symbol": "000001",
                    "signal_date": "2026-03-13",
                    "score": 118.0,
                    "confidence_score": 95,
                    "filter_ok": False,
                    "pretty_ok": True,
                    "quality_score": 78.0,
                    "sector_band": "edge_low",
                },
                {
                    "signal_type": "double_breakout",
                    "symbol": "000002",
                    "signal_date": "2026-03-13",
                    "score": 112.0,
                    "confidence_score": 94,
                    "filter_ok": True,
                    "pretty_ok": True,
                    "quality_score": 76.0,
                    "sector_band": "edge_high",
                },
            ]
        )

        selected = select_diverse_candidates(results, top_n=3, per_signal_limit=3)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["symbol"], "000002")


if __name__ == "__main__":
    unittest.main()
