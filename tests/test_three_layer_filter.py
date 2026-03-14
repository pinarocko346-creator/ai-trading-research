from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from app.core.types import ResearchSignal
from app.data.market_context import MarketFilterConfig, score_market_snapshot
from app.data.sector_context import SectorFilterConfig, build_symbol_theme_payload
from app.report.report_builder import build_daily_report
from app.strategy.scanner import _filter_ok, _market_score_adjustment, _sector_band, _sector_score_adjustment


class ThreeLayerFilterTests(unittest.TestCase):
    def test_score_market_snapshot_marks_risk_on(self) -> None:
        history = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3, freq="D"),
                "close": [10.0, 11.0, 12.0],
                "ma_20": [9.0, 10.0, 11.0],
                "ma_60": [8.0, 9.0, 10.0],
            }
        )
        breadth = pd.DataFrame({"pct_chg": [1.2, 0.8, 0.5, 10.0, -1.0]})
        result = score_market_snapshot(
            {"上证指数": history, "沪深300": history, "创业板指": history},
            breadth,
            MarketFilterConfig(min_positive_index_count=2, min_up_ratio=0.5, min_limit_up_down_ratio=1.0),
        )

        self.assertTrue(result["market_ok"])
        self.assertEqual(result["market_regime"], "risk_on")

    def test_build_symbol_theme_payload_scores_sector(self) -> None:
        snapshot = {
            "industry_score_map": {"电力": 72.0},
            "concept_score_map": {"中特估": 68.0, "国企改革": 62.0},
            "symbol_theme_map": {
                "000001": {
                    "industry_name": "电力",
                    "concept_names": ["中特估", "国企改革"],
                }
            }
        }

        payload = build_symbol_theme_payload("000001", snapshot, SectorFilterConfig(min_sector_score=60.0))

        self.assertTrue(payload["sector_ok"])
        self.assertEqual(payload["industry_name"], "电力")
        self.assertEqual(payload["concept_names"], ["中特估", "国企改革"])

    def test_market_adjustment_is_soft_in_neutral_regime(self) -> None:
        adjustment = _market_score_adjustment({"market_regime": "neutral", "market_score": 56.0})
        self.assertEqual(adjustment, 2.0)

    def test_sector_adjustment_uses_calibrated_bands(self) -> None:
        config = SectorFilterConfig(min_sector_score=55.0)
        strong = _sector_score_adjustment({"sector_score": 68.0}, config)
        medium = _sector_score_adjustment({"sector_score": 55.0}, config)
        edge_high = _sector_score_adjustment({"sector_score": 49.0}, config)
        edge_low = _sector_score_adjustment({"sector_score": 44.0}, config)

        self.assertEqual(strong, 6.0)
        self.assertEqual(medium, 3.0)
        self.assertEqual(edge_high, 0.5)
        self.assertEqual(edge_low, -2.5)

    def test_filter_ok_only_allows_edge_high_in_neutral_regime(self) -> None:
        config = SectorFilterConfig(min_sector_score=55.0)
        allowed = _filter_ok(
            {"market_regime": "neutral", "market_score": 56.89},
            {"sector_score": 48.5},
            config,
        )
        blocked = _filter_ok(
            {"market_regime": "neutral", "market_score": 56.89},
            {"sector_score": 43.96},
            config,
        )
        self.assertTrue(allowed)
        self.assertFalse(blocked)

    def test_sector_band_marks_edge_zones(self) -> None:
        config = SectorFilterConfig(min_sector_score=55.0)
        high_band = _sector_band({"sector_score": 48.5}, config)
        low_band = _sector_band({"sector_score": 43.96}, config)
        self.assertEqual(high_band, "edge_high")
        self.assertEqual(low_band, "edge_low")

    def test_daily_report_renders_market_summary(self) -> None:
        signal = ResearchSignal(
            signal_type="double_breakout",
            symbol="000001",
            signal_date=date(2026, 3, 14),
            confidence_score=95,
            trend_ok=True,
            location_ok=True,
            pattern_ok=True,
            volume_ok=True,
        )
        report = build_daily_report(
            [signal],
            [],
            report_context={
                "market_regime": "risk_on",
                "market_score": 82.5,
                "market_positive_index_count": 3,
                "market_up_ratio": 0.63,
                "market_limit_up_count": 78,
                "market_limit_down_count": 5,
            },
        )

        self.assertIn("## 三层滤网摘要", report)
        self.assertIn("- 市场环境：偏多", report)
        self.assertIn("- 市场得分：82.5", report)


if __name__ == "__main__":
    unittest.main()
