# 美股 MRMC + NX 筛选规则说明

## 1. 数据与目标

本项目新增了一套独立于 A 股主流程的美股筛选器，目标是把富途公式里的：

- `MRMC` 买入卖出信号
- `NX` 蓝黄梯子

与实际教程中的右侧交易原则结合起来，输出可直接使用的美股候选清单。

入口脚本：

- `scripts/screen_us_stocks.py`

核心实现：

- `app/us_futu/indicators.py`
- `app/us_futu/data.py`
- `app/us_futu/screener.py`
- `config/us_futu_screener.yaml`

## 2. 富途公式到 Python 的映射

### 2.1 MRMC

项目中按源码翻译了以下核心结构：

- `DIFF = EMA(CLOSE, 12) - EMA(CLOSE, 26)`
- `DEA = EMA(DIFF, 9)`
- `MACD = (DIFF - DEA) * 2`
- `DXDX` 作为 `mrmc_bottom_signal`
- `DBJGXC` 作为 `mrmc_sell_signal`

说明：

- `DXDX` 是源码里真正被标注为“抄底”的条件
- `DBJGXC` 是源码里真正被标注为“卖出”的条件
- 其余中间变量如 `AAA / BBB / CCC / DBBL` 也被保留，便于后续复核

### 2.2 NX 蓝黄梯子

项目中按源码翻译为：

- 蓝梯上沿：`EMA(HIGH, 24)`
- 蓝梯下沿：`EMA(LOW, 23)`
- 黄梯上沿：`EMA(HIGH, 89)`
- 黄梯下沿：`EMA(LOW, 90)`

并派生出常用状态：

- `blue_above_yellow`
- `close_above_blue`
- `close_above_yellow`
- `close_below_blue`
- `close_below_yellow`

## 3. 项目中的组合判断

源码本身只给出信号，不直接告诉你怎么筛股，所以项目额外加入了教程里的执行逻辑。

### 3.1 大盘过滤

用 `SPY / QQQ / IWM` 的日线梯子状态定义市场环境：

- `risk_on`：至少 2 个指数满足 `蓝在黄上 + 价格在蓝梯上方 + 近期无卖出`
- `neutral`：只有 1 个指数满足
- `risk_off`：0 个指数满足

### 3.2 股票池过滤

只扫描配置文件里的熟悉股票白名单，并额外要求：

- 价格不低于设定阈值
- `20日均量` 达标
- `20日均成交额` 达标

### 3.3 策略定义

当前项目输出 4 类候选：

1. `1234_resonance`
   `1h/2h/3h/4h` 都有近期抄底，且 `30m` 右侧突破蓝梯。

2. `tutorial_breakout_1h`
   `1h` 级别近期出现 MRMC 抄底，随后价格快速钻出蓝梯。

3. `blue_above_yellow_trend_daily`
   日线蓝梯稳定在黄梯之上，价格维持在蓝梯之上。

4. `sweet_spot_hourly`
   `1h` 级别先突破蓝黄梯，随后回踩支撑仍站住，视为甜点时刻。

## 4. 运行方式

```bash
python3 scripts/screen_us_stocks.py --top 20
```

输出目录：

- `reports/us_futu/`

会生成：

- 候选 CSV
- 摘要 TXT
- 结构化 JSON

## 5. 当前限制

- 数据源使用 `Yahoo Finance`
- 个别股票的 `60m` 数据可能偶发缺失，项目会自动跳过
- 目前优先支持日线、30分钟、1小时，以及由 1 小时重采样得到的 `2h/3h/4h`
- 这是“规则化筛选器”，不是自动下单系统
