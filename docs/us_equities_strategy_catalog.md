# 美股量化策略清单

## 1. 文档目的

本文档用于整理当前 `us_equities` 项目中已经落地的策略、过滤条件和工程约束，方便后续：

- 对照原始交易文档检查逻辑是否偏移
- 持续修正规则
- 扩展多周期数据
- 上传 GitHub 后长期维护

当前主入口：

- `scripts/run_us_equities_daily.py`

当前主配置：

- `config/us_equities_daily.yaml`

当前主模块：

- `app/us_equities/config.py`
- `app/us_equities/database.py`
- `app/us_equities/daily_logic.py`
- `app/us_equities/strategy_registry.py`
- `app/us_equities/intraday.py`
- `app/us_equities/sectors.py`
- `app/us_equities/pipeline.py`

## 1.1 策略注册表

当前项目已经开始采用“策略注册表”结构：

- 策略定义集中在 `app/us_equities/strategy_registry.py`
- 每个策略都包含统一元信息：
  - `code`
  - `name`
  - `stage`
  - `trigger_timeframe`
  - `required_timeframes`
  - `enabled_by_default`
  - `evaluator`

作用：

- 后续增加策略时，不需要修改主流水线结构
- 可以逐步演进为更完整的配置化策略系统
- 避免策略逻辑散落在多个 `if/else` 中

## 2. 指标定义

### 2.1 MRMC

当前固定参数：

- `12 / 26 / 9`

当前项目中对应关系：

- `mrmc_bottom_signal`：对应富途源码里的 `DXDX`
- `mrmc_sell_signal`：对应富途源码里的 `DBJGXC`

使用原则：

- `MRMC` 只负责提示
- 真正的买入确认必须结合 `NX` 蓝黄梯子和右侧突破结构

### 2.2 NX 蓝黄梯子

当前定义：

- 蓝梯上沿：`EMA(HIGH, 24)`
- 蓝梯下沿：`EMA(LOW, 23)`
- 黄梯上沿：`EMA(HIGH, 89)`
- 黄梯下沿：`EMA(LOW, 90)`

当前项目中的派生状态：

- `blue_above_yellow`
- `close_above_blue`
- `close_above_yellow`
- `breakout_recent`
- `breakout_yellow_recent`
- `blue_cross_yellow_recent`
- `retest_ok`
- `trend_ok`
- `bullish_ok`

## 3. 过滤规则总表

## 3.1 文档原生规则

以下过滤更接近你原始文档中的原则：

- 只做右侧结构，不做左侧猜底
- 只做流动性足够的股票
- 趋势越大，稳定性越高
- 大盘优先
- 板块共振优先
- 买入后以对应级别梯子作为卖出参考

## 3.2 工程保护规则

以下过滤属于工程实现时增加的稳定性保护，不是原文逐字规则：

- 最低价格过滤
- `20日均量` 过滤
- `20日均成交额` 过滤
- 剔除部分美股特殊证券后缀：
  - `W`
  - `WS`
  - `WT`
  - `U`
  - `R`
  - `P`
- 最小历史样本长度要求

说明：

- 这些规则的目的是减少低质量数据、流动性极差标的和不适合策略执行的证券。

## 4. 市场环境逻辑

当前市场环境由以下指数决定：

- `^GSPC`
- `^IXIC`
- `^DJI`

判断标准：

- 指数满足 `蓝在黄上 + 收盘在蓝梯上方 + 近期无卖出`
- 满足个数 `>= 2`：`risk_on`
- 满足个数 `= 0`：`risk_off`
- 其他：`neutral`

作用：

- 作为全局排序增强项
- 不直接替代个股买卖逻辑

## 5. 板块共振逻辑

当前板块共振使用手工篮子：

- `semis`
- `mega_tech`
- `cloud_security`
- `crypto`
- `ev_space`
- `financials`
- `healthcare`

当前板块统计指标：

- `trend_count`：日线趋势股数量
- `breakout_count`：日线突破数量
- `bottom_count`：日线抄底数量
- `weekly_trend_count`：周线趋势数量

当前板块分数公式：

- `score = trend_count * 2.0 + breakout_count * 3.0 + bottom_count * 1.5 + weekly_trend_count * 1.5`

当前板块共振判定：

- `trend_count >= min_resonance_members`
- 或 `breakout_count >= min_breakout_members`

输出字段：

- `sector_name`
- `sector_score`
- `sector_resonance_ok`

## 6. 已落地策略清单

## 6.1 `daily_bottom_breakout`

### 策略定位

- 日线版“教程基础打法”

### 触发条件

- 日线 `MRMC` 近期出现抄底
- 日线近期右侧突破蓝梯
- 当前收盘在蓝梯上方
- 日线近期没有卖出信号

### 增强条件

- 周线趋势有效
- 月线保持偏强
- 大盘处于 `risk_on`
- 板块分数较高

### 卖点参考

- 日线蓝梯下边缘失守

### 所需 timeframe

- `1d`
- `1w`
- `1mo`

### 当前状态

- 已启用

## 6.2 `blue_above_yellow_trend_daily`

### 策略定位

- 日线版“蓝在黄上打法”

### 触发条件

- 日线蓝梯在黄梯之上
- 日线收盘在蓝梯之上
- 日线近期没有卖出信号

### 增强条件

- 周线也处于 `蓝在黄上`
- 周线收盘在蓝梯之上
- 月线蓝梯在黄梯之上
- 板块共振强

### 卖点参考

- 日线蓝梯下边缘失守

### 所需 timeframe

- `1d`
- `1w`
- `1mo`

### 当前状态

- 已启用

## 6.3 `daily_sweet_spot`

### 策略定位

- 日线版“甜点时刻”

### 触发条件

- 日线近期突破黄梯
- 日线出现回踩并确认支撑有效 `retest_ok`
- 收盘仍在蓝梯之上
- 近期没有卖出信号

### 逻辑解释

- 对应“突破蓝黄梯后，阻力变支撑，再回踩确认”的结构

### 卖点参考

- 日线蓝梯下边缘失守

### 所需 timeframe

- `1d`
- `1w`

### 当前状态

- 已启用

## 6.4 `weekly_trend_resonance`

### 策略定位

- 周线趋势共振策略

### 触发条件

- 周线趋势有效
- 周线收盘在蓝梯之上
- 周线近期没有卖出
- 同时日线仍处于右侧强势状态

### 逻辑解释

- 用更高一级别保证稳定性，再由日线维持执行节奏

### 卖点参考

- 周线蓝梯下边缘失守

### 所需 timeframe

- `1d`
- `1w`
- `1mo`

### 当前状态

- 已启用

## 6.5 `4321_intraday_resonance`

### 策略定位

- 多周期共振策略
- 对应文档里的 `1234 / 4321` 打法

### 触发条件

- `1h / 2h / 3h / 4h` 都有近期 `MRMC` 抄底
- `30m` 右侧突破蓝梯
- `1h` 收盘在蓝梯上方
- `1h` 近期没有卖出
- 同时需要更高一级别支持：
  - 日线 `bullish_ok`
  - 或周线 `trend_ok`

### 逻辑解释

- 多个小时级别共振负责提高确定性
- `30m` 负责右侧突破确认

### 卖点参考

- 优先看 `30m` 或 `1h` 蓝梯下边缘

### 所需 timeframe

- `30m`
- `1h`
- `2h`
- `3h`
- `4h`
- `1d / 1w` 作为高周期辅助

### 当前状态

- 逻辑已接入
- 默认关闭

### 当前关闭原因

- 现阶段主数据库只有日线
- 缺少稳定的分钟级或小时级本地数据

### 启用方式

- 打开 `config/us_equities_daily.yaml` 中的 `intraday.enabled: true`
- 再将 `intraday` 数据源切换到后续的本地分钟库

## 7. 时间框架支持矩阵

### 稳定支持

- `1d`
- `1w`
- `1mo`

### 已接入但默认关闭

- `30m`
- `1h`
- `2h`
- `3h`
- `4h`

## 8. 当前配置项说明

当前最关键的开关在 `config/us_equities_daily.yaml`：

- `signal.right_side_only`
  是否强制只保留右侧结构

- `signal.weekly_trend_required`
  是否强制要求周线趋势有效

- `signal.monthly_filter_enabled`
  是否启用月线过滤

- `signal.require_sector_resonance`
  是否强制要求板块共振为真

- `intraday.enabled`
  是否启用 `4321` 多周期共振

## 9. 当前项目边界

当前项目已经具备：

- 全市场扫描
- 指标层
- 过滤层
- 板块层
- 策略层
- 候选输出层

当前还未正式完成：

- 分钟级本地数据库接入
- `4321` 默认启用
- 更完整的行业 / 主题分类
- 回测与绩效归因
- 自动执行清单生成

## 10. 下一步建议

最推荐的后续工作顺序：

1. 接入分钟级本地数据库
2. 正式启用 `4321`
3. 将策略从硬编码改成注册表
4. 扩充板块篮子
5. 增加回测和日报模块
