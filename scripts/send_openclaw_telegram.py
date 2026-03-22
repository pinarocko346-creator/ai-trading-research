from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib import request


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"缺少环境变量: {name}")
    return value


def _telegram_api(method: str) -> str:
    token = _require_env("TELEGRAM_BOT_TOKEN")
    return f"https://api.telegram.org/bot{token}/{method}"


def _post_json(
    method: str,
    payload: dict[str, object],
    *,
    dry_run: bool,
    retries: int = 3,
    timeout: int = 30,
) -> dict[str, object]:
    if dry_run:
        print(f"[dry-run] {method}: {json.dumps(payload, ensure_ascii=False)}")
        return {"ok": True}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = request.Request(
                _telegram_api(method),
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                raise
            time.sleep(attempt)
    raise RuntimeError(f"发送 Telegram 失败: {last_error}")


def _send_document(path: Path, caption: str, *, dry_run: bool, retries: int = 3) -> None:
    chat_id = _require_env("TELEGRAM_CHAT_ID")
    if dry_run:
        print(f"[dry-run] sendDocument: {path} caption={caption}")
        return
    last_error = ""
    for attempt in range(1, retries + 1):
        cmd = [
            "curl",
            "-sS",
            "-X",
            "POST",
            _telegram_api("sendDocument"),
            "-F",
            f"chat_id={chat_id}",
            "-F",
            f"caption={caption}",
            "-F",
            f"document=@{path}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            payload = json.loads(result.stdout)
            if payload.get("ok"):
                return
            last_error = f"Telegram API 错误: {payload}"
        else:
            last_error = result.stderr.strip()
        if attempt < retries:
            time.sleep(attempt)
    raise RuntimeError(f"发送文件失败: {path.name}: {last_error}")


def _escape_markdown(text: str) -> str:
    for ch in ("_", "*", "[", "]", "(", ")", "~", "`", ">", "#", "+", "-", "=", "|", "{", "}", ".", "!"):
        text = text.replace(ch, f"\\{ch}")
    return text


def _build_success_message(manifest: dict[str, object]) -> str:
    top_candidates = manifest.get("top_candidates", [])
    lines = [
        "A股14买点日报",
        f"运行ID: {manifest.get('run_id', '')}",
        f"状态: {manifest.get('status', '')}",
        f"市场环境: {manifest.get('market_regime', '')}",
        f"股票池数量: {manifest.get('universe_size', 0)}",
        f"扫描信号数: {manifest.get('scan_result_count', 0)}",
        f"日报候选数: {manifest.get('report_signal_count', 0)}",
    ]
    if manifest.get("sqlite_latest_trade_date"):
        lines.append(f"SQLite 最新交易日: {manifest['sqlite_latest_trade_date']}")
    if manifest.get("sqlite_is_stale"):
        lines.append(f"数据状态: warning / {manifest.get('sqlite_staleness_days', '?')}天")
    if top_candidates:
        lines.append("")
        lines.append("前十候选")
        for idx, item in enumerate(top_candidates[:10], 1):
            symbol = str(item.get("symbol", ""))
            signal_name = str(item.get("signal_name", item.get("signal_type", "")))
            score = item.get("score", 0)
            sector = str(item.get("sector_band", ""))
            filter_ok = "Y" if item.get("filter_ok") else "N"
            lines.append(f"{idx}. {symbol} {signal_name} score={score} sector={sector} ok={filter_ok}")
    else:
        lines.append("")
        lines.append("今日无候选。")
    return "\n".join(lines)


def _build_failure_message(run_dir: Path, error_message: str) -> str:
    escaped = _escape_markdown(error_message.strip() or "未知错误")
    return "\n".join(
        [
            "*A股14买点日报失败*",
            f"目录: `{_escape_markdown(str(run_dir))}`",
            f"原因: `{escaped}`",
        ]
    )


def _load_manifest(run_dir: Path) -> dict[str, object]:
    manifest_path = run_dir / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="发送 OpenClaw 日任务结果到 Telegram")
    parser.add_argument("--run-dir", required=True, help="本次运行目录")
    parser.add_argument("--mode", choices=["success", "failure"], default="success", help="发送模式")
    parser.add_argument("--error-message", default="", help="失败模式下的错误消息")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不实际发送")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    chat_id = _require_env("TELEGRAM_CHAT_ID")

    if args.mode == "failure":
        text = _build_failure_message(run_dir, args.error_message)
        response = _post_json(
            "sendMessage",
            {"chat_id": chat_id, "text": text, "parse_mode": "MarkdownV2"},
            dry_run=args.dry_run,
        )
        if not response.get("ok"):
            raise RuntimeError(f"发送 Telegram 失败: {response}")
        return

    manifest = _load_manifest(run_dir)
    text = _build_success_message(manifest)
    response = _post_json(
        "sendMessage",
        {"chat_id": chat_id, "text": text},
        dry_run=args.dry_run,
    )
    if not response.get("ok"):
        raise RuntimeError(f"发送 Telegram 失败: {response}")

    outputs = manifest.get("outputs", {})
    files = [
        (Path(str(outputs.get("daily_candidates_csv", ""))), "每日候选 CSV"),
        (Path(str(outputs.get("daily_report_md", ""))), "日报 Markdown"),
    ]
    for path, caption in files:
        if path.exists():
            try:
                _send_document(path, caption, dry_run=args.dry_run)
            except Exception as exc:
                print(f"⚠️ 附件发送失败: {path.name}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"❌ {exc}", file=sys.stderr)
        raise SystemExit(1)
