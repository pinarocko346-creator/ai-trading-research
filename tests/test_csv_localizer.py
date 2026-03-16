from __future__ import annotations

import unittest

import pandas as pd

from app.report.csv_localizer import localize_csv_frame


class CsvLocalizerTests(unittest.TestCase):
    def test_localize_csv_frame_translates_columns_and_values(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "symbol": "601669",
                    "name": "中国电建",
                    "signal_type": "jumping_creek",
                    "signal_name": "跳跃小溪",
                    "signal_date": "2026-03-13",
                    "score": 103.25,
                    "base_score": 99.75,
                    "confidence_score": 95,
                    "filter_ok": True,
                    "pretty_ok": True,
                    "quality_score": 78.0,
                    "quality_bucket": "high",
                    "market_regime": "neutral",
                    "sector_band": "strong",
                    "concept_names": ["工程建设", "长期破净"],
                    "risk_tags": ["needs_review"],
                    "factors": {
                        "resistance": 6.65,
                        "volume_ratio": 3.32,
                        "prior_below_resistance": True,
                    },
                    "market_index_details": [
                        {"name": "上证指数", "close": 4095.45, "ma20": 4121.87, "trend_ok": False},
                    ],
                    "secondary_signal_types": ["double_breakout", "n_breakout"],
                    "secondary_signal_names": ["双突破", "N字突破"],
                }
            ]
        )

        localized = localize_csv_frame(frame)

        self.assertIn("股票代码", localized.columns)
        self.assertIn("买点名称", localized.columns)
        self.assertIn("买点代码", localized.columns)
        self.assertIn("三层滤网通过", localized.columns)
        self.assertIn("漂亮度通过", localized.columns)
        self.assertNotIn("secondary_signal_types", localized.columns)
        self.assertEqual(localized.iloc[0]["买点代码"], "跳跃小溪")
        self.assertEqual(localized.iloc[0]["市场环境"], "中性")
        self.assertEqual(localized.iloc[0]["板块强度分层"], "强势主线")
        self.assertEqual(localized.iloc[0]["三层滤网通过"], "是")
        self.assertEqual(localized.iloc[0]["漂亮度通过"], "是")
        self.assertEqual(localized.iloc[0]["漂亮度分层"], "高质量")
        self.assertEqual(localized.iloc[0]["风险标签"], "需复核")
        self.assertEqual(localized.iloc[0]["相关概念"], "工程建设、长期破净")
        self.assertIn("阻力位=6.65", localized.iloc[0]["关键因子"])
        self.assertIn("量比=3.32", localized.iloc[0]["关键因子"])
        self.assertIn("趋势通过=否", localized.iloc[0]["指数明细"])


if __name__ == "__main__":
    unittest.main()
