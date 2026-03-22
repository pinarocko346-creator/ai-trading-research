from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.us_futu.indicators import MRMCMacdConfig
from app.us_equities.config import (
    USEquitiesDatabaseConfig,
    USEquitiesIntradayConfig,
    USEquitiesMarketConfig,
    USEquitiesSectorConfig,
    USEquitiesSignalConfig,
    USEquitiesUniverseConfig,
)
from app.us_equities.pipeline import run_daily_pipeline


def _build_summary(results, summary: dict[str, object], top_n: int) -> str:
    lines = [
        "美股日线量化筛选摘要",
        f"生成时间: {datetime.now().isoformat(timespec='seconds')}",
        f"市场环境: {summary['market_regime']}",
        f"正向指数数量: {summary['market_positive_index_count']}",
        f"股票池数量: {summary.get('universe_size', 0)}",
        f"有效扫描数量: {summary.get('state_count', 0)}",
        f"多周期扫描数量: {summary.get('intraday_symbols_processed', 0)}",
        f"4321 候选数量: {summary.get('intraday_candidate_count', 0)}",
    ]
    if results.empty:
        lines.append("候选清单: 无")
        return "\n".join(lines) + "\n"
    lines.append("候选清单:")
    for _, row in results.head(top_n).iterrows():
        lines.append(
            f"- {row['symbol']} | {row['strategy_type']} | score={row['score']} | "
            f"tf={row['trigger_timeframe']} | sector={row.get('sector_name', '')}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="美股日线量化筛选")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "us_equities_daily.yaml"),
        help="配置文件路径",
    )
    parser.add_argument("--top", type=int, default=30, help="输出前 N 条候选")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    database_config = USEquitiesDatabaseConfig(**config["database"])
    universe_config = USEquitiesUniverseConfig(**config["universe"])
    market_config = USEquitiesMarketConfig(**config["market"])
    signal_config = USEquitiesSignalConfig(**config["signal"])
    sector_config = USEquitiesSectorConfig(**config["sectors"])
    intraday_config = USEquitiesIntradayConfig(**config.get("intraday", {}))
    macd_config = MRMCMacdConfig(**config.get("macd", {}))

    results, summary = run_daily_pipeline(
        database_config=database_config,
        universe_config=universe_config,
        market_config=market_config,
        signal_config=signal_config,
        sector_config=sector_config,
        intraday_config=intraday_config,
        macd_config=macd_config,
    )

    output_dir = PROJECT_ROOT / "reports" / "us_equities_daily"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"candidates_{stamp}.csv"
    json_path = output_dir / f"summary_{stamp}.json"
    txt_path = output_dir / f"summary_{stamp}.txt"

    results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "summary": summary,
        "candidate_count": int(len(results)),
        "top_candidates": results.head(args.top).to_dict(orient="records"),
        "outputs": {
            "csv": str(csv_path),
            "summary_txt": str(txt_path),
        },
    }
    txt_path.write_text(_build_summary(results, summary, args.top), encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"market_regime={summary['market_regime']}")
    print(f"positive_indexes={summary['market_positive_index_count']}")
    print(f"universe={summary.get('universe_size', 0)}")
    print(f"scanned={summary.get('state_count', 0)}")
    print(f"candidates={len(results)}")
    print(f"csv={csv_path}")
    print(f"summary={txt_path}")
    if not results.empty:
        print(results.head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
