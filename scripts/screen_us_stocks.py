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

from app.us_futu.data import USDataConfig
from app.us_futu.indicators import MRMCMacdConfig
from app.us_futu.screener import (
    USMarketConfig,
    USSignalConfig,
    USSectorsConfig,
    USUniverseConfig,
    screen_us_market,
)


def _build_summary(results, summary: dict[str, object], top_n: int) -> str:
    lines = [
        "美股 MRMC + NX 自动筛选摘要",
        f"生成时间: {datetime.now().isoformat(timespec='seconds')}",
        f"市场环境: {summary['market_regime']}",
        f"正向指数数量: {summary['market_positive_index_count']}",
        f"股票池数量: {summary.get('universe_size', 0)}",
        f"有效扫描数量: {summary.get('state_count', 0)}",
    ]
    if results.empty:
        lines.append("候选清单: 无")
        return "\n".join(lines) + "\n"

    lines.append("候选清单:")
    for _, row in results.head(top_n).iterrows():
        lines.append(
            f"- {row['symbol']} | {row['strategy_type']} | score={row['score']} | "
            f"tf={row['trigger_timeframe']} | sector={row.get('sector_name', '')} | note={row['entry_note']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="美股 MRMC + NX 自动筛选")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "us_futu_screener.yaml"),
        help="配置文件路径",
    )
    parser.add_argument("--top", type=int, default=20, help="输出前 N 条候选")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    data_config = USDataConfig(
        cache_dir=PROJECT_ROOT / "data" / "cache" / "us_futu",
        **config.get("data", {}),
    )
    macd_config = MRMCMacdConfig(**config.get("macd", {}))
    market_config = USMarketConfig(**config.get("market", {}))
    signal_config = USSignalConfig(**config.get("signal", {}))
    sectors_config = USSectorsConfig(**config.get("sectors", {}))
    universe_config = USUniverseConfig(**config["universe"])

    results, summary = screen_us_market(
        universe_config=universe_config,
        market_config=market_config,
        signal_config=signal_config,
        sectors_config=sectors_config,
        data_config=data_config,
        macd_config=macd_config,
    )

    output_dir = PROJECT_ROOT / "reports" / "us_futu"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"us_candidates_{stamp}.csv"
    json_path = output_dir / f"us_summary_{stamp}.json"
    txt_path = output_dir / f"us_summary_{stamp}.txt"

    results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "market_regime": summary["market_regime"],
        "market_positive_index_count": summary["market_positive_index_count"],
        "index_snapshots": summary["index_snapshots"],
        "universe_size": summary.get("universe_size", 0),
        "state_count": summary.get("state_count", 0),
        "sector_summary": summary.get("sector_summary", {}),
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
    print(f"candidates={len(results)}")
    print(f"csv={csv_path}")
    print(f"summary={txt_path}")
    if not results.empty:
        print(results.head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
