# 美股量化项目结构

## 当前目标

先建立一套可上传 GitHub、可持续迭代的美股量化项目骨架，当前先落地日线版，后续再补齐多时间框架。

## 目录职责

- `app/us_equities/config.py`
  日线项目的配置对象。

- `app/us_equities/database.py`
  本地 SQLite 数据接入层，负责全市场快照和单票历史。

- `app/us_equities/daily_logic.py`
  日线 / 周线 / 月线状态构建逻辑。

- `app/us_equities/strategy_registry.py`
  策略注册表，统一维护策略元信息和策略函数。

- `app/us_equities/sectors.py`
  板块篮子、板块共振统计与板块上下文。

- `app/us_equities/pipeline.py`
  日线主扫描流水线，负责串联数据库、指数过滤、板块评分和候选输出。

- `app/us_equities/intraday.py`
  `4321` 多周期共振的状态构建与策略接入点。

- `app/us_equities/intraday_data.py`
  多周期数据加载层，统一封装 `sqlite / yfinance` 数据来源。

- `config/us_equities_daily.yaml`
  日线版配置文件。

- `scripts/run_us_equities_daily.py`
  日线版运行入口。

## 当前已支持

- 本地 SQLite 全市场扫描
- `MRMC(12,26,9)`
- `NX` 蓝黄梯子
- 日线 / 周线 / 月线重采样
- 大盘过滤
- 板块共振
- 多策略候选输出
- `4321` 多周期共振策略接入点

## 4321 多周期共振

项目已经预留并接入了 `4321` 多周期共振逻辑：

- `30m` 负责右侧突破确认
- `1h / 2h / 3h / 4h` 负责 `MRMC` 抄底共振

当前默认通过配置开关关闭：

- `config/us_equities_daily.yaml`
- `intraday.enabled: false`

原因是当前主数据源仍以本地日线 SQLite 为主，而 `4321` 需要分钟级或小时级原始数据。

等你补齐 timeframe 数据后，只需要：

1. 打开 `intraday.enabled`
2. 配置分钟级数据库或临时数据源

就可以直接进入主流水线，不需要重构日线项目骨架。

## 下一阶段

后续等分钟级数据完善后，再新增：

- `app/us_equities/intraday/`
- `app/us_equities/signals/multi_timeframe.py`
- `app/us_equities/reports/`
- `app/us_equities/backtest/`

这样可以在不破坏日线主干的前提下，逐步补齐：

- `30m / 1h / 2h / 3h / 4h`
- `1234 / 4321`
- 分时甜点时刻
- 更细的板块轮动和执行清单
