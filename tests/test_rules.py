from __future__ import annotations

import unittest

import pandas as pd

from app.features.price_features import build_price_features
from app.strategy.rules import RuleThresholds, scan_signals


def _synthetic_breakout_frame() -> pd.DataFrame:
    rows = []
    close = 10.0
    for index in range(120):
        if index < 80:
            close += 0.05
        elif index < 100:
            close += 0.01 if index % 2 == 0 else -0.01
        else:
            close += 0.18
        open_price = close * 0.99
        high = close * 1.01
        low = open_price * 0.99
        volume = 2_000_000 if index < 70 else 5_000_000
        rows.append(
            {
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": volume,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_2b_frame() -> pd.DataFrame:
    rows = []
    close = 12.0
    for index in range(90):
        if index < 50:
            close -= 0.08
        elif index < 75:
            close += 0.03 if index % 2 == 0 else -0.02
        elif index == 87:
            close = 8.20
        elif index == 88:
            close = 8.92
        else:
            close += 0.04
        if index == 87:
            open_price = 8.55
        elif index == 88:
            open_price = 8.72
        else:
            open_price = close * 0.995
        high = max(close, open_price) * 1.015
        low = min(close, open_price) * (0.96 if index == 87 else 0.99)
        volume = 2_000_000 if index < 87 else 4_500_000
        rows.append(
            {
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": volume,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_false_break_frame() -> pd.DataFrame:
    rows = []
    close = 11.0
    for index in range(90):
        if index < 45:
            close -= 0.06
        elif index < 82:
            close += 0.015 if index % 2 == 0 else -0.01
        elif index == 87:
            close = 8.18
        elif index == 88:
            close = 8.34
        else:
            close += 0.03
        if index == 87:
            open_price = 8.30
        elif index == 88:
            open_price = 8.28
        else:
            open_price = close * 0.997
        high = max(close, open_price) * 1.012
        low = min(close, open_price) * (0.985 if index != 87 else 0.99)
        volume = 1_900_000 if index < 87 else 3_400_000
        rows.append(
            {
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": volume,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_right_shoulder_frame() -> pd.DataFrame:
    closes = [10.5 + 0.02 * index for index in range(60)] + [
        10.2, 10.0, 9.8, 9.6, 9.3, 9.0, 9.4, 9.9, 10.3, 10.0,
        9.5, 9.0, 8.4, 8.8, 9.3, 9.8, 10.1, 9.9, 9.6, 9.4,
        9.2, 9.12, 9.18, 9.1, 9.05, 9.12, 9.2, 9.32, 9.42, 9.5,
    ]
    rows = []
    for index, close in enumerate(closes):
        if index < 42:
            open_price = close * 0.995
        else:
            open_price = close * 0.998
        high = max(close, open_price) * 1.012
        low = min(close, open_price) * 0.988
        volume = 2_000_000 if index < 84 else 2_600_000
        rows.append(
            {
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": volume,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_strength_emergence_frame() -> pd.DataFrame:
    closes = [7.5 + 0.06 * index for index in range(70)] + [
        10.9, 11.05, 11.2, 11.4, 11.6, 11.75, 11.55, 11.7, 11.82, 11.68,
        11.76, 11.88, 11.72, 11.8, 11.9, 11.78, 11.84, 11.92, 11.86, 11.9,
        11.82, 11.88, 11.9, 11.84, 11.88, 11.92, 11.86, 11.9, 11.95, 11.99,
    ]
    rows = []
    for index, close in enumerate(closes):
        open_price = close * (0.995 if index < len(closes) - 1 else 0.99)
        high = min(max(close, open_price) * 1.01, 12.02)
        low = max(min(close, open_price) * 0.99, 10.85)
        volume = 2_000_000 if index < len(closes) - 1 else 3_200_000
        rows.append(
            {
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": volume,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_jumping_creek_frame() -> pd.DataFrame:
    closes = [7.2 + 0.05 * index for index in range(95)] + [
        13.0, 13.2, 13.4, 13.55, 13.7, 13.8, 13.65, 13.72, 13.78, 13.74,
        13.7, 13.76, 13.72, 13.79, 13.76, 13.74, 13.78, 13.75, 13.77, 13.74,
        13.76, 13.79, 13.77, 13.78, 13.76, 13.8, 13.78, 13.79, 13.77, 13.76,
        13.82, 13.84, 13.85, 14.25,
    ]
    rows = []
    last_index = len(closes) - 1
    for index, close in enumerate(closes):
        if index == last_index:
            open_price = 13.82
            high = 14.32
            low = 13.78
            volume = 4_800_000
        else:
            open_price = close * 0.996
            high = min(max(close, open_price) * 1.008, 13.98)
            low = max(min(close, open_price) * 0.992, 12.95)
            volume = 2_100_000
        rows.append(
            {
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": volume,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_pattern_breakout_frame() -> pd.DataFrame:
    closes = [8.5 + 0.05 * index for index in range(90)] + [
        12.1, 12.35, 12.6, 12.3, 12.05, 11.8, 11.55, 11.75, 11.95, 12.1,
        12.28, 12.12, 11.98, 11.84, 11.92, 12.04, 12.18, 12.12, 11.96, 11.88,
        11.94, 12.02, 12.08, 12.12, 12.16, 12.22, 12.26, 12.29, 12.31, 12.52,
    ]
    rows = []
    last_index = len(closes) - 1
    for index, close in enumerate(closes):
        if index == last_index:
            open_price = 12.28
            high = 12.58
            low = 12.22
            volume = 3_600_000
        else:
            open_price = close * 0.997
            high = min(max(close, open_price) * 1.008, 12.42)
            low = max(min(close, open_price) * 0.992, 11.45)
            volume = 2_200_000
        rows.append(
            {
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": volume,
            }
        )
    return pd.DataFrame(rows)


class RuleTests(unittest.TestCase):
    def test_scan_signals_returns_known_signal_types(self) -> None:
        frame = build_price_features(_synthetic_breakout_frame())
        signals = scan_signals(
            frame,
            symbol="000001",
            enabled_signals=["double_breakout", "pullback_confirmation"],
            thresholds=RuleThresholds(),
            include_invalid=True,
        )
        self.assertTrue(signals)
        self.assertIn(signals[0].signal_type, {"double_breakout", "pullback_confirmation"})

    def test_detects_2b_structure_on_fake_break_reclaim(self) -> None:
        frame = build_price_features(_synthetic_2b_frame())
        signals = scan_signals(
            frame,
            symbol="000001",
            enabled_signals=["2b_structure"],
            thresholds=RuleThresholds(swing_lookback=20),
            include_invalid=False,
        )
        self.assertTrue(signals)
        self.assertEqual(signals[0].signal_type, "2b_structure")

    def test_detects_false_breakdown_on_shallow_reclaim(self) -> None:
        frame = build_price_features(_synthetic_false_break_frame())
        signals = scan_signals(
            frame,
            symbol="000001",
            enabled_signals=["false_breakdown"],
            thresholds=RuleThresholds(swing_lookback=20),
            include_invalid=False,
        )
        self.assertTrue(signals)
        self.assertEqual(signals[0].signal_type, "false_breakdown")

    def test_detects_right_shoulder_on_recent_bounce(self) -> None:
        frame = build_price_features(_synthetic_right_shoulder_frame())
        signals = scan_signals(
            frame,
            symbol="000001",
            enabled_signals=["right_shoulder"],
            thresholds=RuleThresholds(),
            include_invalid=False,
        )
        self.assertTrue(signals)
        self.assertEqual(signals[0].signal_type, "right_shoulder")

    def test_detects_strength_emergence_near_box_high(self) -> None:
        frame = _synthetic_strength_emergence_frame()
        signals = scan_signals(
            frame,
            symbol="000001",
            enabled_signals=["strength_emergence"],
            thresholds=RuleThresholds(),
            include_invalid=False,
        )
        self.assertTrue(signals)
        self.assertEqual(signals[0].signal_type, "strength_emergence")

    def test_detects_jumping_creek_on_expansion_breakout(self) -> None:
        frame = _synthetic_jumping_creek_frame()
        signals = scan_signals(
            frame,
            symbol="000001",
            enabled_signals=["jumping_creek"],
            thresholds=RuleThresholds(),
            include_invalid=False,
        )
        self.assertTrue(signals)
        self.assertEqual(signals[0].signal_type, "jumping_creek")

    def test_detects_pattern_breakout_on_box_escape(self) -> None:
        frame = _synthetic_pattern_breakout_frame()
        signals = scan_signals(
            frame,
            symbol="000001",
            enabled_signals=["pattern_breakout"],
            thresholds=RuleThresholds(),
            include_invalid=False,
        )
        self.assertTrue(signals)
        self.assertEqual(signals[0].signal_type, "pattern_breakout")


if __name__ == "__main__":
    unittest.main()
