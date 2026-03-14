from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import yaml

from app.data.market_context import MarketFilterConfig
from app.data.sector_context import SectorFilterConfig
from app.data.ingest import DataIngestConfig
from app.data.universe import UniverseConfig
from app.strategy.rules import RuleThresholds
from app.strategy.scanner import ScanConfig, load_default_universe, scan_market


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="A股批量扫描")
    parser.add_argument("--max-symbols", type=int, default=100, help="扫描股票数量上限")
    args = parser.parse_args()

    config = yaml.safe_load((PROJECT_ROOT / "config" / "strategy_13_points.yaml").read_text(encoding="utf-8"))
    thresholds = RuleThresholds(**config["thresholds"])
    universe_config = UniverseConfig(**config["universe"])
    ingest_config = DataIngestConfig(
        cache_dir=PROJECT_ROOT / "data" / "cache",
        **config.get("ingest", {}),
    )
    market_filter = MarketFilterConfig(
        cache_dir=PROJECT_ROOT / "data" / "cache" / "market",
        **config.get("market_filter", {}),
    )
    sector_filter = SectorFilterConfig(
        cache_dir=PROJECT_ROOT / "data" / "cache" / "sector",
        **config.get("sector_filter", {}),
    )
    universe = load_default_universe(
        universe_config,
        max_symbols=args.max_symbols,
        ingest_config=ingest_config,
    )
    results = scan_market(
        universe,
        thresholds=thresholds,
        signal_types=config["signals"]["enabled"],
        scan_config=ScanConfig(
            max_symbols=args.max_symbols,
            cache_dir=PROJECT_ROOT / "data" / "cache",
            ingest_config=ingest_config,
            market_filter=market_filter,
            sector_filter=sector_filter,
        ),
    )

    output_dir = PROJECT_ROOT / "reports" / "scans"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"scan_{stamp}.csv"
    results.to_csv(output_path, index=False)
    print(f"universe={len(universe)}")
    print(f"signals={len(results)}")
    print(f"output={output_path}")
    if not results.empty:
        print(results.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
