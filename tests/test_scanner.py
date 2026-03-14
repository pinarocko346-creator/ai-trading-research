from __future__ import annotations

import unittest

import pandas as pd

from app.strategy.scanner import normalize_signal_candidates, select_diverse_candidates


class ScannerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
