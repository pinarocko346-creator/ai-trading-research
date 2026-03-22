from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from app.core.types import ResearchSignal
from app.data.market_context import MarketFilterConfig, score_market_snapshot
from app.data.sector_context import (
    SectorFilterConfig,
    build_symbol_theme_payload,
    fetch_concept_rankings,
    fetch_industry_rankings,
    load_sector_snapshot,
)
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
        config = SectorFilterConfig(crowded_min_score=65.0, min_sector_score=50.0, edge_high_min_score=40.0, edge_low_min_score=30.0)
        crowded = _sector_score_adjustment({"sector_score": 68.0}, config)
        strong = _sector_score_adjustment({"sector_score": 55.0}, config)
        edge_high = _sector_score_adjustment({"sector_score": 49.0}, config)
        edge_low = _sector_score_adjustment({"sector_score": 35.0}, config)
        weak = _sector_score_adjustment({"sector_score": 20.0}, config)

        self.assertEqual(crowded, -3.5)
        self.assertEqual(strong, 1.5)
        self.assertEqual(edge_high, 2.5)
        self.assertEqual(edge_low, 1.0)
        self.assertEqual(weak, -1.5)

    def test_filter_ok_avoids_crowded_and_allows_edge_low_for_reversal(self) -> None:
        config = SectorFilterConfig(crowded_min_score=65.0, min_sector_score=50.0, edge_high_min_score=40.0, edge_low_min_score=30.0)
        allowed = _filter_ok(
            {"market_regime": "neutral", "market_score": 56.89},
            {"sector_score": 35.0},
            config,
            "false_breakdown",
        )
        blocked = _filter_ok(
            {"market_regime": "neutral", "market_score": 56.89},
            {"sector_score": 68.0},
            config,
            "false_breakdown",
        )
        self.assertTrue(allowed)
        self.assertFalse(blocked)

    def test_filter_ok_keeps_trend_signal_on_strong_or_edge_high(self) -> None:
        config = SectorFilterConfig(crowded_min_score=65.0, min_sector_score=50.0, edge_high_min_score=40.0, edge_low_min_score=30.0)
        strong_allowed = _filter_ok(
            {"market_regime": "neutral", "market_score": 56.89},
            {"sector_score": 55.0},
            config,
            "double_breakout",
        )
        edge_high_allowed = _filter_ok(
            {"market_regime": "neutral", "market_score": 56.89},
            {"sector_score": 45.0},
            config,
            "jumping_creek",
        )
        edge_low_blocked = _filter_ok(
            {"market_regime": "neutral", "market_score": 56.89},
            {"sector_score": 35.0},
            config,
            "double_breakout",
        )
        self.assertTrue(strong_allowed)
        self.assertTrue(edge_high_allowed)
        self.assertFalse(edge_low_blocked)

    def test_filter_ok_treats_cup_with_handle_as_trend_signal(self) -> None:
        config = SectorFilterConfig(crowded_min_score=65.0, min_sector_score=50.0, edge_high_min_score=40.0, edge_low_min_score=30.0)
        strong_allowed = _filter_ok(
            {"market_regime": "neutral", "market_score": 60.0},
            {"sector_score": 55.0},
            config,
            "cup_with_handle",
        )
        weak_blocked = _filter_ok(
            {"market_regime": "neutral", "market_score": 60.0},
            {"sector_score": 20.0},
            config,
            "cup_with_handle",
        )
        self.assertTrue(strong_allowed)
        self.assertFalse(weak_blocked)

    def test_sector_snapshot_falls_back_to_latest_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            industry_rankings = pd.DataFrame(
                [
                    {"板块名称": "电力", "涨跌幅": 2.0, "换手率": 3.0, "上涨家数": 20, "下跌家数": 5, "领涨股票-涨跌幅": 6.0, "score": 72.0},
                ]
            )
            concept_rankings = pd.DataFrame(
                [
                    {"板块名称": "中特估", "涨跌幅": 1.5, "换手率": 2.5, "上涨家数": 18, "下跌家数": 6, "领涨股票-涨跌幅": 5.0, "score": 68.0},
                ]
            )
            industry_rankings.to_parquet(cache_dir / "industry_rankings_20260318.parquet", index=False)
            concept_rankings.to_parquet(cache_dir / "concept_rankings_20260318.parquet", index=False)
            (cache_dir / "symbol_themes_v2_20260318.json").write_text(
                json.dumps(
                    {
                        "000001": {
                            "industry_name": "电力",
                            "concept_names": ["中特估"],
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            config = SectorFilterConfig(cache_dir=cache_dir)
            snapshot = load_sector_snapshot(config)

            self.assertEqual(snapshot["symbol_theme_map"]["000001"]["industry_name"], "电力")
            self.assertEqual(fetch_industry_rankings(config).iloc[0]["板块名称"], "电力")
            self.assertEqual(fetch_concept_rankings(config).iloc[0]["板块名称"], "中特估")

    def test_sector_band_marks_crowded_and_edge_zones(self) -> None:
        config = SectorFilterConfig(crowded_min_score=65.0, min_sector_score=50.0, edge_high_min_score=40.0, edge_low_min_score=30.0)
        crowded_band = _sector_band({"sector_score": 68.0}, config)
        high_band = _sector_band({"sector_score": 49.0}, config)
        low_band = _sector_band({"sector_score": 35.0}, config)
        self.assertEqual(crowded_band, "crowded")
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
