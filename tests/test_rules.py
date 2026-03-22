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
            low = 13.90
            volume = 4_800_000
        elif index >= last_index - 4:
            open_price = close * 0.996
            high = min(max(close, open_price) * 1.008, 13.98)
            low = max(min(close, open_price) * 0.992, 12.95)
            volume = 1_650_000
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


def _synthetic_support_flip_frame() -> pd.DataFrame:
    closes = [7.9 + 0.022 * index for index in range(105)] + [
        10.92, 10.98, 11.02, 11.0, 10.96, 11.01, 11.6, 11.42, 11.28, 11.22, 11.3, 11.74,
    ]
    rows = []
    last_index = len(closes) - 1
    breakout_index = len(closes) - 6
    for index, close in enumerate(closes):
        if index == breakout_index:
            open_price = 11.08
            high = 11.72
            low = 11.02
            volume = 4_600_000
        elif index == last_index:
            open_price = 11.34
            high = 11.82
            low = 11.30
            volume = 3_000_000
        elif breakout_index < index < last_index:
            open_price = close * 0.998
            high = close * 1.006
            low = close * 0.992
            volume = 1_850_000
        else:
            open_price = close * 0.996
            high = close * 1.01
            low = close * 0.99
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


def _synthetic_support_flip_frame_failed_pullback() -> pd.DataFrame:
    frame = _synthetic_support_flip_frame().copy()
    pullback_indices = list(range(len(frame) - 5, len(frame) - 1))
    frame.loc[pullback_indices, "close"] = [11.18, 11.02, 10.94, 11.08]
    frame.loc[pullback_indices, "open"] = [11.26, 11.12, 11.02, 11.0]
    frame.loc[pullback_indices, "high"] = [11.30, 11.18, 11.08, 11.12]
    frame.loc[pullback_indices, "low"] = [11.08, 10.88, 10.82, 10.98]
    frame.loc[pullback_indices, "volume"] = [3_000_000, 3_200_000, 3_250_000, 3_100_000]
    return frame


def _synthetic_false_break_frame_shallow_break() -> pd.DataFrame:
    frame = _synthetic_false_break_frame().copy()
    frame.loc[len(frame) - 3, "low"] = 8.21
    frame.loc[len(frame) - 2, "low"] = 8.21
    return frame


def _synthetic_breakout_frame_loose_setup() -> pd.DataFrame:
    frame = _synthetic_breakout_frame().copy()
    frame.loc[len(frame) - 2, "high"] = round(frame.loc[len(frame) - 2, "high"] * 1.08, 2)
    frame.loc[len(frame) - 2, "low"] = round(frame.loc[len(frame) - 2, "low"] * 0.92, 2)
    return frame


def _synthetic_breakout_frame_failed_hold() -> pd.DataFrame:
    frame = _synthetic_breakout_frame().copy()
    last_index = len(frame) - 1
    frame.loc[last_index, "low"] = round(frame.loc[last_index, "breakout_level_20"] * 0.995, 2) if "breakout_level_20" in frame.columns else 13.2
    frame.loc[last_index, "close"] = 14.28
    frame.loc[last_index, "high"] = 14.52
    frame.loc[last_index, "open"] = 13.86
    return frame


def _synthetic_jumping_creek_frame_weak_break() -> pd.DataFrame:
    frame = _synthetic_jumping_creek_frame().copy()
    last_index = len(frame) - 1
    frame.loc[last_index, "close"] = 14.0
    frame.loc[last_index, "high"] = 14.6
    frame.loc[last_index, "low"] = 13.72
    frame.loc[last_index, "open"] = 13.88
    return frame


def _synthetic_jumping_creek_frame_failed_hold() -> pd.DataFrame:
    frame = _synthetic_jumping_creek_frame().copy()
    last_index = len(frame) - 1
    frame.loc[last_index, "open"] = 13.84
    frame.loc[last_index, "high"] = 14.38
    frame.loc[last_index, "low"] = 13.45
    frame.loc[last_index, "close"] = 14.26
    frame.loc[last_index, "volume"] = 5_100_000
    return frame


def _synthetic_cup_with_handle_frame() -> pd.DataFrame:
    closes = [6.0 + 0.04 * index for index in range(45)] + [8.0 + 0.06 * index for index in range(35)] + [
        10.6, 10.45, 10.3, 10.1, 9.9, 9.7, 9.55, 9.4, 9.3, 9.22, 9.18, 9.15,
        9.18, 9.24, 9.32, 9.45, 9.6, 9.78, 9.95, 10.12, 10.25, 10.35, 10.42, 10.5,
        10.55, 10.58, 10.48, 10.42, 10.36, 10.4, 10.45, 10.95,
    ]
    rows = []
    last_index = len(closes) - 1
    for index, close in enumerate(closes):
        if index == last_index:
            open_price = 10.55
            high = 11.02
            low = 10.52
            volume = 5_200_000
        elif index >= last_index - 5:
            open_price = close * 0.998
            high = close * 1.006
            low = close * 0.992
            volume = 1_600_000
        else:
            open_price = close * 0.996
            high = close * 1.01
            low = close * 0.99
            volume = 2_400_000
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


def _synthetic_cup_with_handle_frame_deep_handle() -> pd.DataFrame:
    frame = _synthetic_cup_with_handle_frame().copy()
    handle_indices = list(range(len(frame) - 6, len(frame) - 1))
    frame.loc[handle_indices, "close"] = [10.2, 9.95, 9.82, 9.9, 9.96]
    frame.loc[handle_indices, "open"] = [10.28, 10.05, 9.9, 9.92, 9.98]
    frame.loc[handle_indices, "high"] = [10.3, 10.08, 9.95, 10.0, 10.02]
    frame.loc[handle_indices, "low"] = [10.05, 9.82, 9.68, 9.8, 9.9]
    frame.loc[handle_indices, "volume"] = [2_800_000, 2_900_000, 3_000_000, 2_850_000, 2_900_000]
    return frame


def _synthetic_cup_with_short_handle_frame() -> pd.DataFrame:
    frame = _synthetic_cup_with_handle_frame().copy()
    short_handle_indices = list(range(len(frame) - 5, len(frame)))
    frame.loc[short_handle_indices[:-1], "close"] = [10.48, 10.52, 10.58, 10.46]
    frame.loc[short_handle_indices[:-1], "open"] = [10.44, 10.48, 10.54, 10.50]
    frame.loc[short_handle_indices[:-1], "high"] = [10.54, 10.58, 10.64, 10.52]
    frame.loc[short_handle_indices[:-1], "low"] = [10.40, 10.46, 10.50, 10.38]
    frame.loc[short_handle_indices[:-1], "volume"] = [2_100_000, 2_050_000, 2_000_000, 1_550_000]
    frame.loc[short_handle_indices[-1], "open"] = 10.55
    frame.loc[short_handle_indices[-1], "high"] = 11.08
    frame.loc[short_handle_indices[-1], "low"] = 10.52
    frame.loc[short_handle_indices[-1], "close"] = 11.0
    frame.loc[short_handle_indices[-1], "volume"] = 5_400_000
    return frame


def _synthetic_cup_with_watch_frame() -> pd.DataFrame:
    frame = _synthetic_cup_with_short_handle_frame().copy().iloc[:-1].reset_index(drop=True)
    last_index = len(frame) - 1
    frame.loc[last_index, "open"] = 10.56
    frame.loc[last_index, "high"] = 10.74
    frame.loc[last_index, "low"] = 10.50
    frame.loc[last_index, "close"] = 10.70
    frame.loc[last_index, "volume"] = 3_200_000
    return frame


def _synthetic_cup_with_watch_frame_unround() -> pd.DataFrame:
    frame = _synthetic_cup_with_watch_frame().copy()
    pattern_indices = list(range(len(frame) - 32, len(frame)))
    closes = [
        10.55, 10.42, 10.28, 10.15, 10.02, 9.9, 9.82, 9.76,
        9.7, 9.66, 9.62, 9.58, 9.54, 9.56, 9.6, 9.64,
        9.68, 9.6, 9.52, 9.34, 9.18, 9.05, 9.12, 9.2,
        9.35, 9.52, 9.7, 9.88, 10.05, 10.22, 10.42, 10.7,
    ]
    for offset, index in enumerate(pattern_indices):
        close = closes[offset]
        if offset == len(pattern_indices) - 1:
            frame.loc[index, "open"] = 10.56
            frame.loc[index, "high"] = 10.74
            frame.loc[index, "low"] = 10.50
            frame.loc[index, "close"] = close
            frame.loc[index, "volume"] = 3_200_000
            continue
        frame.loc[index, "open"] = round(close * 0.998, 2)
        frame.loc[index, "high"] = round(close * 1.008, 2)
        frame.loc[index, "low"] = round(close * 0.992, 2)
        frame.loc[index, "close"] = close
        frame.loc[index, "volume"] = 1_900_000
    return frame


def _synthetic_cup_with_handle_strict_frame() -> pd.DataFrame:
    pretrend = (
        [4.2 + 0.015 * index for index in range(60)]
        + [5.1 + 0.026 * index for index in range(70)]
        + [6.92 + 0.035 * index for index in range(60)]
    )
    closes = pretrend + [
        8.95, 8.9, 8.8, 8.7, 8.55, 8.4, 8.25, 8.12,
        8.0, 7.9, 7.82, 7.76, 7.72, 7.68, 7.66, 7.65,
        7.66, 7.69, 7.73, 7.79, 7.87, 7.96, 8.08, 8.2,
        8.32, 8.44, 8.56, 8.66, 8.74, 8.82, 8.88, 8.92,
        8.82, 8.74, 8.78, 8.8, 8.83, 9.25,
    ]
    rows = []
    last_index = len(closes) - 1
    handle_start = last_index - 5
    for index, close in enumerate(closes):
        if index == last_index:
            open_price = 8.9
            high = 9.32
            low = 8.88
            volume = 5_500_000
        elif index >= handle_start:
            open_price = close * 0.998
            high = close * 1.006
            low = close * 0.992
            volume = 1_550_000
        else:
            open_price = close * 0.996
            high = close * 1.01
            low = close * 0.99
            volume = 2_450_000
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


def _synthetic_cup_with_handle_leader_frame() -> pd.DataFrame:
    frame = _synthetic_cup_with_handle_strict_frame().copy()
    handle_indices = list(range(len(frame) - 6, len(frame) - 1))
    frame.loc[handle_indices, "close"] = [8.72, 8.60, 8.52, 8.58, 8.66]
    frame.loc[handle_indices, "open"] = [8.78, 8.66, 8.58, 8.54, 8.62]
    frame.loc[handle_indices, "high"] = [8.80, 8.68, 8.60, 8.62, 8.72]
    frame.loc[handle_indices, "low"] = [8.66, 8.54, 8.46, 8.52, 8.60]
    frame.loc[handle_indices, "volume"] = [1_520_000, 1_480_000, 1_420_000, 1_430_000, 1_470_000]
    frame.loc[:, "market_cap"] = 35_000_000_000.0
    frame.loc[len(frame) - 1, "open"] = 8.82
    frame.loc[len(frame) - 1, "high"] = 9.56
    frame.loc[len(frame) - 1, "low"] = 8.76
    frame.loc[len(frame) - 1, "close"] = 9.50
    frame.loc[len(frame) - 1, "volume"] = 5_800_000
    return frame


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

    def test_false_breakdown_requires_meaningful_break_depth(self) -> None:
        frame = build_price_features(_synthetic_false_break_frame_shallow_break())
        signals = scan_signals(
            frame,
            symbol="000001",
            enabled_signals=["false_breakdown"],
            thresholds=RuleThresholds(swing_lookback=20),
            include_invalid=False,
        )
        self.assertFalse(signals)

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

    def test_jumping_creek_rejects_weak_breakout_close(self) -> None:
        frame = _synthetic_jumping_creek_frame_weak_break()
        signals = scan_signals(
            frame,
            symbol="000001",
            enabled_signals=["jumping_creek"],
            thresholds=RuleThresholds(),
            include_invalid=False,
        )
        self.assertFalse(signals)

    def test_jumping_creek_rejects_breakout_that_fails_to_hold_resistance(self) -> None:
        frame = _synthetic_jumping_creek_frame_failed_hold()
        signals = scan_signals(
            frame,
            symbol="000001",
            enabled_signals=["jumping_creek"],
            thresholds=RuleThresholds(),
            include_invalid=False,
        )
        self.assertFalse(signals)

    def test_detects_cup_with_handle_on_classic_breakout(self) -> None:
        frame = _synthetic_cup_with_handle_frame()
        signals = scan_signals(
            frame,
            symbol="000001",
            enabled_signals=["cup_with_handle"],
            thresholds=RuleThresholds(),
            include_invalid=False,
        )
        self.assertTrue(signals)
        self.assertEqual(signals[0].signal_type, "cup_with_handle")

    def test_cup_with_handle_rejects_deep_handle(self) -> None:
        frame = _synthetic_cup_with_handle_frame_deep_handle()
        signals = scan_signals(
            frame,
            symbol="000001",
            enabled_signals=["cup_with_handle"],
            thresholds=RuleThresholds(),
            include_invalid=False,
        )
        self.assertFalse(signals)

    def test_detects_cup_with_short_handle_on_strong_breakout(self) -> None:
        frame = _synthetic_cup_with_short_handle_frame()
        signals = scan_signals(
            frame,
            symbol="000001",
            enabled_signals=["cup_with_handle"],
            thresholds=RuleThresholds(),
            include_invalid=False,
        )
        self.assertTrue(signals)
        self.assertEqual(signals[0].signal_type, "cup_with_handle")

    def test_detects_strict_cup_with_handle_on_valid_breakout(self) -> None:
        frame = _synthetic_cup_with_handle_strict_frame()
        signals = scan_signals(
            frame,
            symbol="000001",
            enabled_signals=["cup_with_handle_strict"],
            thresholds=RuleThresholds(),
            include_invalid=False,
        )
        self.assertTrue(signals)
        self.assertEqual(signals[0].signal_type, "cup_with_handle_strict")

    def test_strict_cup_with_handle_rejects_short_cup(self) -> None:
        frame = _synthetic_cup_with_handle_frame()
        signals = scan_signals(
            frame,
            symbol="000001",
            enabled_signals=["cup_with_handle_strict"],
            thresholds=RuleThresholds(),
            include_invalid=False,
        )
        self.assertFalse(signals)

    def test_detects_leader_cup_with_handle_on_valid_breakout(self) -> None:
        frame = _synthetic_cup_with_handle_leader_frame()
        signals = scan_signals(
            frame,
            symbol="000001",
            enabled_signals=["cup_with_handle_leader"],
            thresholds=RuleThresholds(),
            include_invalid=False,
        )
        self.assertTrue(signals)
        self.assertEqual(signals[0].signal_type, "cup_with_handle_leader")

    def test_leader_cup_with_handle_requires_5_to_10_pct_handle(self) -> None:
        frame = _synthetic_cup_with_handle_strict_frame().copy()
        frame.loc[:, "market_cap"] = 35_000_000_000.0
        signals = scan_signals(
            frame,
            symbol="000001",
            enabled_signals=["cup_with_handle_leader"],
            thresholds=RuleThresholds(),
            include_invalid=False,
        )
        self.assertFalse(signals)

    def test_detects_cup_with_handle_watch_before_breakout(self) -> None:
        frame = _synthetic_cup_with_watch_frame()
        signals = scan_signals(
            frame,
            symbol="000001",
            enabled_signals=["cup_with_handle_watch"],
            thresholds=RuleThresholds(),
            include_invalid=False,
        )
        self.assertTrue(signals)
        self.assertEqual(signals[0].signal_type, "cup_with_handle_watch")

    def test_cup_with_handle_watch_rejects_unround_bottom(self) -> None:
        frame = _synthetic_cup_with_watch_frame_unround()
        signals = scan_signals(
            frame,
            symbol="000001",
            enabled_signals=["cup_with_handle_watch"],
            thresholds=RuleThresholds(),
            include_invalid=False,
        )
        self.assertFalse(signals)

    def test_double_breakout_requires_tight_pre_breakout_setup(self) -> None:
        frame = build_price_features(_synthetic_breakout_frame_loose_setup())
        signals = scan_signals(
            frame,
            symbol="000001",
            enabled_signals=["double_breakout"],
            thresholds=RuleThresholds(),
            include_invalid=False,
        )
        self.assertFalse(signals)

    def test_double_breakout_rejects_breakout_that_fails_to_hold_level(self) -> None:
        frame = _synthetic_breakout_frame_failed_hold()
        signals = scan_signals(
            frame,
            symbol="000001",
            enabled_signals=["double_breakout"],
            thresholds=RuleThresholds(),
            include_invalid=False,
        )
        self.assertFalse(signals)

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

    def test_detects_support_resistance_flip_after_shallow_pullback(self) -> None:
        frame = _synthetic_support_flip_frame()
        signals = scan_signals(
            frame,
            symbol="000001",
            enabled_signals=["support_resistance_flip"],
            thresholds=RuleThresholds(),
            include_invalid=False,
        )
        self.assertTrue(signals)
        self.assertEqual(signals[0].signal_type, "support_resistance_flip")

    def test_support_resistance_flip_rejects_heavy_pullback(self) -> None:
        frame = _synthetic_support_flip_frame_failed_pullback()
        signals = scan_signals(
            frame,
            symbol="000001",
            enabled_signals=["support_resistance_flip"],
            thresholds=RuleThresholds(),
            include_invalid=False,
        )
        self.assertFalse(signals)


if __name__ == "__main__":
    unittest.main()
