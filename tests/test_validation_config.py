from __future__ import annotations

import unittest

from scripts.validate_three_layer_filter_history import _resolve_research_universe_config


class ValidationConfigTests(unittest.TestCase):
    def test_research_universe_takes_precedence(self) -> None:
        config = {
            "universe": {
                "min_close": 3.0,
                "min_avg_volume": 3_000_000,
                "min_turnover_rate": 0.5,
                "exclude_st": True,
                "exclude_beijing": False,
            },
            "research_universe": {
                "min_close": 2.0,
                "min_avg_volume": 500_000,
                "min_turnover_rate": 0.1,
                "exclude_st": True,
                "exclude_beijing": False,
            },
        }

        universe_config = _resolve_research_universe_config(config)

        self.assertEqual(universe_config.min_close, 2.0)
        self.assertEqual(universe_config.min_avg_volume, 500_000)
        self.assertEqual(universe_config.min_turnover_rate, 0.1)

    def test_default_universe_is_used_as_fallback(self) -> None:
        config = {
            "universe": {
                "min_close": 3.0,
                "min_avg_volume": 3_000_000,
                "min_turnover_rate": 0.5,
                "exclude_st": True,
                "exclude_beijing": False,
            }
        }

        universe_config = _resolve_research_universe_config(config)

        self.assertEqual(universe_config.min_close, 3.0)
        self.assertEqual(universe_config.min_avg_volume, 3_000_000)
        self.assertEqual(universe_config.min_turnover_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
