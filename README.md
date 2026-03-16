# A股 13 买点 AI 研究系统

这个项目把 `13个买点` 的主观方法论拆成可计算规则、样本标注模板、批量扫描、回测和解释报告。

当前版本已接入 13 个买点的一版程序化规则：

- `抛售高潮`
- `2B结构`
- `假诱空`
- `顺势头肩（右肩简化版）`
- `双突破`
- `强势出现`
- `跳跃小溪`
- `回抽确认`
- `N字突破`
- `支撑压力互换`
- `箱体弹簧`
- `形态突破`
- `趋势急跌后的第一次反弹`

## 目录

- `app/data`：A 股数据拉取、本地缓存、股票池过滤
- `app/features`：K 线、趋势、量价、支撑压力特征
- `app/strategy`：买点定义、规则识别、评分排序
- `app/backtest`：轻量事件回测与绩效统计
- `app/ai`：信号解释、失败归因、LLM 提示词
- `app/report`：Markdown 研究报告
- `data/samples`：标准样本、边界样本、失败样本模板
- `notebooks`：交互式研究入口

## 快速开始

```bash
cd /Users/apple/ai-trading-research
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python scripts/demo_research.py
```

如果本地没有安装 `akshare`，演示脚本会提示先安装依赖。数据拉取失败时，也可以先用本地 CSV 做规则验证。

## 研究流程

1. 使用 `app/data/ingest.py` 下载并缓存 A 股日线。
2. 用 `app/data/universe.py` 过滤 ST、停牌、低流动性个股。
3. 在 `app/features/price_features.py` 上计算趋势、位置和量价特征。
4. 运行 `app/strategy/rules.py` 生成信号。
5. 使用 `app/backtest/engine.py` 与 `app/backtest/metrics.py` 做回测。
6. 通过 `app/ai/explainer.py` 和 `app/report/report_builder.py` 输出研究报告。

## 常用命令

```bash
.venv/bin/python scripts/review_symbol.py 600036 --history --signal double_breakout
.venv/bin/python scripts/batch_scan.py --max-symbols 100
.venv/bin/python scripts/export_annotation_cases.py 600036 000001 --signal 2b_structure
.venv/bin/python scripts/generate_daily_report.py --max-symbols 100 --top 20
.venv/bin/python scripts/run_openclaw_daily.py --max-symbols 0 --top 20 --universe-scope tradeable
```

说明：

- `review_symbol.py`：单票复核，导出图形化信号上下文
- `batch_scan.py`：批量扫描 A 股股票池并输出 CSV
- `export_annotation_cases.py`：批量导出历史案例截图和待标注清单
- `generate_daily_report.py`：生成候选、配图和 AI 解读合并日报
- `run_openclaw_daily.py` / `run_openclaw_daily.sh`：适合 `OpenClaw` 定时执行的统一日任务入口

## OpenClaw 日任务

项目已提供 `OpenClaw` 友好的统一入口：

```bash
cd /Users/apple/ai-trading-research
bash scripts/run_openclaw_daily.sh
```

该流程会输出固定目录结构、`latest/manifest.json`、候选 CSV、日报 Markdown、信号 JSON 和图表目录，便于后续自动读取。

详细说明见 `docs/openclaw_daily.md`。

## 设计原则

- 先规则化，再 AI 化
- 先日线，再多周期
- 先研究闭环，再考虑模拟盘或实盘
- 先可解释，再做复杂建模
