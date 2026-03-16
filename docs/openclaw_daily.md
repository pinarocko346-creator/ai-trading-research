# OpenClaw 每日任务接入说明

## 目标

把当前项目整理成一条适合 `OpenClaw` 定时执行的日任务流水线，固定输出目录，便于后续自动读取、复盘和继续迭代。

## 推荐执行顺序

1. 先完成 A 股 SQLite 日线库更新。
2. 再执行本项目的每日扫描任务。
3. 读取 `latest/manifest.json`、`latest/daily_candidates.csv` 和 `latest/daily_report.md`。

如果 SQLite 数据还没更新，脚本仍会执行，但会在 `manifest.json` 里标记 `sqlite_is_stale=true`。

## 统一入口

```bash
cd /Users/apple/ai-trading-research
bash scripts/run_openclaw_daily.sh
```

底层实际调用：

```bash
.venv/bin/python scripts/run_openclaw_daily.py --max-symbols 0 --top 20 --universe-scope tradeable
```

说明：

- `--max-symbols 0`：表示全量股票池扫描
- `--universe-scope tradeable`：使用日常可操作股票池
- `--universe-scope research`：使用更宽的研究股票池

## 环境变量

`OpenClaw` 里建议通过环境变量覆写运行参数：

```bash
export OPENCLAW_OUTPUT_ROOT="$HOME/.openclaw/workspace/ai-trading-research/results/13points"
export OPENCLAW_UNIVERSE_SCOPE="tradeable"
export OPENCLAW_MAX_SYMBOLS="0"
export OPENCLAW_TOP_N="20"
export OPENCLAW_SQLITE_DB_PATH="$HOME/.openclaw/workspace/a-stock-strategy/a_share_historical.db"
bash /Users/apple/ai-trading-research/scripts/run_openclaw_daily.sh
```

可用变量：

- `OPENCLAW_OUTPUT_ROOT`
- `OPENCLAW_LATEST_DIR`
- `OPENCLAW_UNIVERSE_SCOPE`
- `OPENCLAW_MAX_SYMBOLS`
- `OPENCLAW_TOP_N`
- `OPENCLAW_SQLITE_DB_PATH`
- `OPENCLAW_RUN_TAG`
- `OPENCLAW_MAX_STALENESS_DAYS`
- `OPENCLAW_LOG_FILE`
- `PYTHON_BIN`

## 输出结构

每次运行会生成一个独立目录，例如：

```text
results/13points/
  20260314_210000/
    charts/
    daily_candidates.csv
    daily_report.md
    daily_signals.json
    full_scan.csv
    manifest.json
    summary.txt
  latest/
    charts/
    daily_candidates.csv
    daily_report.md
    daily_signals.json
    full_scan.csv
    manifest.json
    summary.txt
  cron.log
```

关键文件：

- `full_scan.csv`：全量扫描结果，适合二次筛选
- `daily_candidates.csv`：最终候选清单
- `daily_report.md`：人读日报
- `daily_signals.json`：机器可读信号详情
- `manifest.json`：本次运行的摘要、状态和输出路径
- `summary.txt`：短摘要，适合消息通知或日志查看

## manifest 字段

`manifest.json` 会包含这些核心字段：

- `run_id`
- `generated_at`
- `universe_scope`
- `universe_size`
- `scan_result_count`
- `report_signal_count`
- `filter_ok_count`
- `market_regime`
- `sqlite_latest_trade_date`
- `sqlite_staleness_days`
- `sqlite_is_stale`
- `status`
- `top_candidates`
- `outputs`

## OpenClaw 定时任务示例

建议在收盘后执行，例如工作日 `16:40`：

```bash
cd /Users/apple/ai-trading-research && bash scripts/run_openclaw_daily.sh
```

推荐流程：

1. `16:10-16:30` 更新 SQLite 数据。
2. `16:40` 执行本项目日任务。
3. 后续 agent 或外部流程读取 `latest/manifest.json` 和 `latest/daily_report.md` 做复盘或通知。
