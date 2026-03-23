# OpenClaw 每日任务接入说明

## 目标

把当前项目整理成一条适合 `OpenClaw` 定时执行的日任务流水线，固定输出目录，便于后续自动读取、复盘和继续迭代。

## 推荐执行顺序

1. 先完成 A 股 SQLite 日线库更新。
2. 再执行本项目的每日扫描任务。
3. 读取 `latest/manifest.json`、`latest/daily_candidates.csv`、`latest/daily_report.md` 和 `latest/daily_value_report.md`。

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
export OPENCLAW_SQLITE_DB_PATH="$HOME/a_share_daily_data/a_share_daily.db"
bash /Users/apple/ai-trading-research/scripts/run_openclaw_daily.sh
```

### Telegram（勿写进 cron 明文）

`TELEGRAM_BOT_TOKEN`、`TELEGRAM_CHAT_ID` 应放在本机 `~/.openclaw/secrets.env`（与 Gateway 一致），**不要**写进 `~/.openclaw/cron/jobs.json` 的 `payload.message`，以免备份或日志泄露。

定时任务里可在执行脚本前加载密钥文件（仅引用路径，不含 token）：

```bash
set -a && [ -f "$HOME/.openclaw/secrets.env" ] && . "$HOME/.openclaw/secrets.env" && set +a
```

然后在同一 shell 中 `export OPENCLAW_SEND_TELEGRAM=1` 并调用 `scripts/run_openclaw_daily.sh`。`openclaw.json` 的 `env` 与 `secrets.providers.default.allowlist` 中可加入上述变量名，便于与 OpenClaw 密钥体系对齐。

**若 token 曾出现在 cron 或会话日志中，建议在 BotFather 轮换 bot token。**

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
    daily_signal_snapshot.csv
    today_expectancy.csv
    strategy_value_scoreboard.csv
    daily_value_report.md
    full_scan.csv
    manifest.json
    summary.txt
  latest/
    charts/
    daily_candidates.csv
    daily_report.md
    daily_signals.json
    daily_signal_snapshot.csv
    today_expectancy.csv
    strategy_value_scoreboard.csv
    daily_value_report.md
    full_scan.csv
    manifest.json
    summary.txt
  history/
    signal_snapshot_history.csv
    signal_forward_returns.csv
    strategy_value_scoreboard.csv
  cron.log
```

关键文件：

- `full_scan.csv`：全量扫描结果，适合二次筛选
- `daily_candidates.csv`：最终候选清单
- `daily_report.md`：人读日报
- `daily_signal_snapshot.csv`：当天全量有效信号快照，按 `watch/candidate/executable` 分层
- `today_expectancy.csv`：当天信号对应的历史期望收益参考
- `strategy_value_scoreboard.csv`：最近 `20/60/120` 个交易日分策略统计
- `daily_value_report.md`：更适合给 `OpenClaw/Kimi` 读取的价值验证摘要
- `daily_signals.json`：机器可读信号详情
- `manifest.json`：本次运行的摘要、状态和输出路径
- `summary.txt`：短摘要，适合消息通知或日志查看
- `history/signal_snapshot_history.csv`：累计信号事件历史
- `history/signal_forward_returns.csv`：累计信号事件的前瞻收益回填

## manifest 字段

`manifest.json` 会包含这些核心字段：

- `run_id`
- `generated_at`
- `universe_scope`
- `universe_size`
- `scan_result_count`
- `report_signal_count`
- `filter_ok_count`
- `value_signal_count`
- `value_history_count`
- `value_executable_signal_count`
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

## A 股 / 美股日线下载：自动重试与失败告警

本机 OpenClaw 里 **A 股 SQLite 每日更新**、**美股 quick 更新**、**美股全量 evening 下载** 三条 cron 已改为调用带 **指数退避重试** 的壳脚本（无需每天盯进度；偶发网络/接口抖动会自动多跑几轮）。

| 脚本 | 作用 |
|------|------|
| `~/.openclaw/workspace/shell-lib/run_with_retry.sh` | 通用重试（日志在 `~/.openclaw/workspace/logs/data_retry/`） |
| `~/.openclaw/workspace/a-stock-strategy/run_quick_update_resilient.sh` | 包装 `quick_update.py` |
| `~/.openclaw/workspace/uscd/run_quick_update_us_resilient.sh` | 包装 `quick_update_us.py` |
| `~/.openclaw/workspace/market_data_cache/run_daily_full_resilient.sh` | 包装 `daily_full_download.sh` |

可调环境变量（在 LaunchAgent / shell 中设置均可）：

- A 股：`A_SHARE_QUICK_RETRIES`（默认 5）、`A_SHARE_QUICK_FIRST_WAIT_SEC`（默认 120）
- 美股 quick：`US_QUICK_RETRIES`、`US_QUICK_FIRST_WAIT_SEC`
- 美股全量 evening：`US_FULL_RETRIES`、`US_FULL_FIRST_WAIT_SEC`
- 全局：`RETRY_LOG_ROOT`、`RETRY_BACKOFF_CAP`（单次等待上限秒，默认 900）

上述三条 cron 已开启 OpenClaw **`failureAlert`**：**连续失败 2 次**且距上次告警超过 **6 小时** 再提醒（具体投递渠道需在 OpenClaw 里配置好 `failure-alert-to` 等，否则仅网关内记录）。

说明：**重试不能替代「数据源长期不可用」**；若多次仍失败，需看 `logs/data_retry/` 与原有 `*.log` 排查根因。
