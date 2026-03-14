from __future__ import annotations

from contextlib import closing
import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.data.ingest import DataIngestConfig, fetch_a_share_history
from app.data.universe import UniverseConfig, load_a_share_spot
from app.strategy.scanner import load_default_universe


class SQLiteDataSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "a_share_historical.db"
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE kline_data (
                    code TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    amount REAL,
                    amplitude REAL,
                    pct_chg REAL,
                    change REAL,
                    turnover REAL,
                    created_at TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE stock_list (
                    code TEXT,
                    name TEXT,
                    industry TEXT,
                    list_date TEXT
                )
                """
            )
            conn.executemany(
                "INSERT INTO stock_list (code, name, industry, list_date) VALUES (?, ?, ?, ?)",
                [
                    ("000001", "平安银行", "", ""),
                    ("000002", "万科A", "", ""),
                    ("000003", "ST测试", "", ""),
                ],
            )
            conn.executemany(
                """
                INSERT INTO kline_data (
                    code, date, open, high, low, close, volume, amount,
                    amplitude, pct_chg, change, turnover, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                [
                    ("000001", "2026-03-11", 10.0, 10.3, 9.9, 10.1, 1000, 10000, 1.0, 1.0, 0.1, 1.2),
                    ("000001", "2026-03-12", 10.1, 10.5, 10.0, 10.4, 2000, 20800, 2.0, 2.97, 0.3, 1.5),
                    ("000001", "2026-03-13", 10.4, 10.8, 10.3, 10.7, 3000, 32100, 2.5, 2.88, 0.3, 1.8),
                    ("000002", "2026-03-12", 8.0, 8.2, 7.9, 8.1, 500, 4050, 1.5, 1.25, 0.1, 0.6),
                    ("000002", "2026-03-13", 8.1, 8.4, 8.0, 8.3, 700, 5810, 2.0, 2.47, 0.2, 0.7),
                    ("000003", "2026-03-11", 4.0, 4.1, 3.9, 4.0, 5000, 20000, 1.0, 0.0, 0.0, 2.0),
                    ("000003", "2026-03-12", 4.0, 4.2, 3.95, 4.1, 6000, 24600, 2.0, 2.5, 0.1, 2.1),
                    ("000003", "2026-03-13", 4.1, 4.3, 4.0, 4.2, 7000, 29400, 2.5, 2.44, 0.1, 2.2),
                ],
            )
            conn.commit()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_fetch_a_share_history_reads_from_sqlite(self) -> None:
        config = DataIngestConfig(
            source="sqlite",
            sqlite_db_path=str(self.db_path),
            start_date="2026-03-11",
            end_date="2026-03-13",
            warmup_days=0,
        )

        history = fetch_a_share_history("000001", config)

        self.assertEqual(list(history["close"]), [10.1, 10.4, 10.7])
        self.assertEqual(list(history["turnover_rate"]), [1.2, 1.5, 1.8])
        self.assertEqual(history["date"].dt.strftime("%Y-%m-%d").tolist(), ["2026-03-11", "2026-03-12", "2026-03-13"])

    def test_load_a_share_spot_and_universe_use_sqlite_snapshot(self) -> None:
        ingest_config = DataIngestConfig(
            source="sqlite",
            sqlite_db_path=str(self.db_path),
        )

        spot = load_a_share_spot(ingest_config)
        row = spot.set_index("symbol").loc["000001"]
        self.assertEqual(row["name"], "平安银行")
        self.assertAlmostEqual(row["close"], 10.7)
        self.assertAlmostEqual(row["avg_volume_20"], 2000.0)
        self.assertAlmostEqual(row["pct_chg"], 2.88)

        universe = load_default_universe(
            UniverseConfig(min_close=3.0, min_avg_volume=600.0, min_turnover_rate=0.5, exclude_st=True),
            max_symbols=10,
            ingest_config=ingest_config,
        )
        self.assertCountEqual(universe["symbol"].tolist(), ["000001", "000002"])


if __name__ == "__main__":
    unittest.main()
