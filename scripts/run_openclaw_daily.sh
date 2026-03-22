#!/bin/bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
CONFIG_PATH="${OPENCLAW_CONFIG_PATH:-$PROJECT_DIR/config/strategy_13_points.yaml}"
OUTPUT_ROOT="${OPENCLAW_OUTPUT_ROOT:-$HOME/.openclaw/workspace/ai-trading-research/results/13points}"
LATEST_DIR="${OPENCLAW_LATEST_DIR:-$OUTPUT_ROOT/latest}"
UNIVERSE_SCOPE="${OPENCLAW_UNIVERSE_SCOPE:-tradeable}"
MAX_SYMBOLS="${OPENCLAW_MAX_SYMBOLS:-0}"
TOP_N="${OPENCLAW_TOP_N:-20}"
RUN_TAG="${OPENCLAW_RUN_TAG:-}"
SQLITE_DB_PATH="${OPENCLAW_SQLITE_DB_PATH:-}"
MAX_STALENESS_DAYS="${OPENCLAW_MAX_STALENESS_DAYS:-3}"
LOG_FILE="${OPENCLAW_LOG_FILE:-$OUTPUT_ROOT/cron.log}"
SEND_TELEGRAM="${OPENCLAW_SEND_TELEGRAM:-1}"

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$(dirname "$LOG_FILE")"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "未找到可执行 Python: $PYTHON_BIN" >&2
  exit 1
fi

echo "==========================================" | tee -a "$LOG_FILE"
echo "A股14买点 OpenClaw 日任务 - $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "PROJECT_DIR=$PROJECT_DIR" | tee -a "$LOG_FILE"
echo "OUTPUT_ROOT=$OUTPUT_ROOT" | tee -a "$LOG_FILE"

cd "$PROJECT_DIR"
RUN_DIR=""
on_error() {
  status=$?
  if [ "$SEND_TELEGRAM" = "1" ] && [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
    if [ -z "$RUN_DIR" ]; then
      RUN_DIR="$LATEST_DIR"
    fi
    "$PYTHON_BIN" scripts/send_openclaw_telegram.py \
      --mode failure \
      --run-dir "$RUN_DIR" \
      --error-message "run_openclaw_daily.sh 失败，exit_code=$status" 2>&1 | tee -a "$LOG_FILE" || true
  fi
  exit $status
}
trap on_error ERR

CMD=(
  "$PYTHON_BIN"
  scripts/run_openclaw_daily.py
  --config "$CONFIG_PATH"
  --output-root "$OUTPUT_ROOT"
  --latest-dir "$LATEST_DIR"
  --universe-scope "$UNIVERSE_SCOPE"
  --max-symbols "$MAX_SYMBOLS"
  --top "$TOP_N"
  --tag "$RUN_TAG"
  --max-staleness-days "$MAX_STALENESS_DAYS"
)

if [ -n "$SQLITE_DB_PATH" ]; then
  CMD+=(--sqlite-db-path "$SQLITE_DB_PATH")
fi

"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

if [ -f "$LATEST_DIR/manifest.json" ]; then
  RUN_DIR="$LATEST_DIR"
fi

trap - ERR

if [ "$SEND_TELEGRAM" = "1" ] && [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ] && [ -n "$RUN_DIR" ]; then
  "$PYTHON_BIN" scripts/send_openclaw_telegram.py --mode success --run-dir "$RUN_DIR" 2>&1 | tee -a "$LOG_FILE" || \
    echo "Telegram 成功通知发送失败，但主任务已完成" | tee -a "$LOG_FILE"
fi

echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
