# 甜点策略 / Sweet Spot Strategy

这是一个面向 `美股` 的量化研究仓库，目标是把主观交易方法论拆成可维护、可验证、可扩展的程序化系统。

## 当前重点

当前主干模块：

- `app/us_equities/`
- `config/us_equities_daily.yaml`
- `scripts/run_us_equities_daily.py`

这条主线已经具备：

- 本地 SQLite 全市场扫描
- `MRMC(12,26,9)`
- `NX` 蓝黄梯子
- 日线 / 周线 / 月线状态构建
- 板块共振
- 多策略注册表
- 预留 `4321` 多周期共振接入点

## 项目结构

### 美股主干

- `app/us_equities/config.py`
  配置对象，统一管理数据库、股票池、市场、板块和多周期开关。

- `app/us_equities/database.py`
  本地 SQLite 数据接入层。

- `app/us_equities/daily_logic.py`
  日线 / 周线 / 月线状态构建逻辑。

- `app/us_equities/strategy_registry.py`
  策略注册表，统一维护策略函数与元信息。

- `app/us_equities/sectors.py`
  板块篮子与板块共振统计。

- `app/us_equities/pipeline.py`
  主扫描流水线。

- `app/us_equities/intraday.py`
  多周期数据接口与 `4321` 共振接入点。

## 快速开始

```bash
cd /Users/apple/ai-trading-research
python3 -m venv .venv
.venv/bin/pip install -e .
```

## 运行命令

### 美股日线筛选

```bash
.venv/bin/python scripts/run_us_equities_daily.py --top 30
```

## 关键文档

- `docs/us_equities_project_structure.md`
  美股项目结构说明

- `docs/us_equities_strategy_catalog.md`
  美股策略清单

- `docs/us_futu_mrmc_nx_mapping.md`
  富途 `MRMC + NX` 与项目实现映射

- `docs/us_auto_stock_quant_system.md`
  美股自动选股系统整理稿

## 当前已落地的美股策略

- `daily_bottom_breakout`
- `blue_above_yellow_trend_daily`
- `daily_sweet_spot`
- `weekly_trend_resonance`
- `4321_intraday_resonance`（已接入，默认关闭）

## 多周期说明

当前稳定支持：

- `1d`
- `1w`
- `1mo`

当前已预留但默认关闭：

- `30m`
- `1h`
- `2h`
- `3h`
- `4h`

原因：

- 目前本地主数据仍以日线 SQLite 为主
- `4321` 多周期共振需要分钟级或小时级原始数据

后续只需要补齐分钟库并替换 `app/us_equities/intraday.py` 的数据源实现，即可正式启用多周期策略。

## 设计原则

- 先规则化，再 AI 化
- 先日线主干，再扩多周期
- 先可验证，再追求复杂度
- 先模块化，再扩展自动化执行
