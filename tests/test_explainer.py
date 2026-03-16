from __future__ import annotations

import unittest
from datetime import date

from app.ai.explainer import explain_signal, generate_ai_review
from app.core.types import ResearchSignal


class ExplainerTests(unittest.TestCase):
    def test_secondary_signals_render_in_text(self) -> None:
        signal = ResearchSignal(
            signal_type="jumping_creek",
            symbol="000001",
            signal_date=date(2026, 3, 14),
            confidence_score=95,
            trend_ok=True,
            location_ok=True,
            pattern_ok=True,
            volume_ok=True,
            factors={
                "resistance": 10.5,
                "secondary_signal_names": ["双突破", "N字突破"],
                "secondary_signal_count": 2,
            },
        )

        summary = explain_signal(signal)
        review = generate_ai_review(signal)

        self.assertIn("次级标签：双突破、N字突破", summary)
        self.assertIn("标签补充：该候选同时具备双突破、N字突破特征。", review)
        self.assertIn("结构参考：阻力位=10.5", review)
        self.assertIn("关键因子：", summary)
        self.assertIn("- 阻力位：10.5", summary)
        self.assertNotIn("secondary_signal_names", summary)
        self.assertNotIn("secondary_signal_types", summary)
        self.assertNotIn("secondary_signal_count", summary)
        self.assertNotIn('{"resistance"', summary)

    def test_neutral_market_review_is_not_described_as_weak(self) -> None:
        signal = ResearchSignal(
            signal_type="double_breakout",
            symbol="000002",
            signal_date=date(2026, 3, 14),
            confidence_score=90,
            trend_ok=True,
            location_ok=True,
            pattern_ok=True,
            volume_ok=True,
            factors={
                "market_ok": False,
                "market_regime": "neutral",
                "market_score": 56.89,
                "sector_ok": False,
                "sector_score": 48.5,
                "sector_band": "edge_high",
            },
        )

        review = generate_ai_review(signal)

        self.assertIn("市场层处于中性区间", review)
        self.assertIn("所属板块处于边缘活跃高位区", review)
        self.assertNotIn("市场层环境偏弱", review)

    def test_crowded_sector_review_adds_overheat_caution(self) -> None:
        signal = ResearchSignal(
            signal_type="double_breakout",
            symbol="000003",
            signal_date=date(2026, 3, 14),
            confidence_score=90,
            trend_ok=True,
            location_ok=True,
            pattern_ok=True,
            volume_ok=True,
            factors={
                "market_ok": True,
                "market_regime": "risk_on",
                "sector_ok": False,
                "sector_score": 82.0,
                "sector_band": "crowded",
            },
        )

        summary = explain_signal(signal)
        review = generate_ai_review(signal)

        self.assertIn("过热拥挤", summary)
        self.assertIn("所属板块热度过高且拥挤", review)

    def test_quality_fields_render_in_signal_and_review(self) -> None:
        signal = ResearchSignal(
            signal_type="cup_with_handle",
            symbol="000004",
            signal_date=date(2026, 3, 14),
            confidence_score=96,
            trend_ok=True,
            location_ok=True,
            pattern_ok=True,
            volume_ok=True,
            factors={
                "left_peak": 12.2,
                "cup_low": 10.8,
                "right_peak": 12.05,
                "quality_score": 78.0,
                "quality_bucket": "high",
                "pretty_ok": True,
            },
        )

        summary = explain_signal(signal)
        review = generate_ai_review(signal)

        self.assertIn("杯子与杯柄", summary)
        self.assertIn("- 漂亮度分：78", summary)
        self.assertIn("- 漂亮度通过：是", summary)
        self.assertIn("形态漂亮度通过统一过滤", review)


if __name__ == "__main__":
    unittest.main()
