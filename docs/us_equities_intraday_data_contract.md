# 美股多周期数据接口约定

## 1. 目的

当前 `us_equities` 主干已经预留了 `4321` 多周期共振逻辑，但默认仍以日线 SQLite 为主。

为了后续无痛接入分钟级数据库，约定多周期数据层只需要满足统一的数据接口，不要求和当前 `yfinance` 实现绑定。

## 2. 当前接入点

当前多周期逻辑入口：

- `app/us_equities/intraday.py`
- `app/us_equities/intraday_data.py`

其中 `build_intraday_state()` 负责输出：

- `30m`
- `1h`
- `2h`
- `3h`
- `4h`

对应每个周期的状态快照。

## 3. 输入要求

任何新的分钟级数据库实现，只要能为单个 `symbol` 输出以下字段即可：

- `date`
- `open`
- `high`
- `low`
- `close`
- `volume`

要求：

- 时间戳升序
- 不含重复 bar
- 同一 symbol 的时区口径一致
- 所有价格列为数值
- `volume` 为数值

## 4. 输出要求

在接入 `build_mrmc_nx_indicators()` 之后，系统依赖这些状态：

- `bottom_recent`
- `sell_recent`
- `blue_above_yellow`
- `close_above_blue`
- `close_above_yellow`
- `breakout_recent`
- `breakout_yellow_recent`
- `blue_cross_yellow_recent`
- `retest_ok`
- `trend_ok`
- `bullish_ok`

因此新数据源不需要直接计算策略，只需要稳定输出标准 OHLCV。

## 5. 推荐适配方式

当前已经提供统一接口：

```python
def load_intraday_history(symbol: str, timeframe: str) -> pd.DataFrame:
    ...
```

其中 `timeframe` 至少支持：

- `30m`
- `60m`

项目内部会统一重采样：

- `60m -> 2h / 3h / 4h`

## 5.1 当前支持的数据源

### `yfinance`

适合临时验证：

- `source: yfinance`
- 自动下载 `30m`
- 自动下载 `60m`

### `sqlite`

适合正式分钟库接入：

- `source: sqlite`
- 通过配置指定：
  - 数据库路径
  - `30m` 表名
  - `60m` 表名
  - symbol 字段名
  - datetime 字段名
  - OHLCV 字段名

配置位置：

- `config/us_equities_daily.yaml`
- `intraday.*`
- `strategies.*`
- `config/us_equities_intraday_sqlite.example.yaml`

## 6. 为什么这样设计

这样做的好处是：

- 分钟库到位后，不需要重写策略
- 只替换数据来源，不动策略注册表
- `4321` 能直接进入现有主流水线
- 后续增加分时甜点时刻、盘中监控也更容易扩展

## 7. 启用步骤

当分钟级数据库准备好之后：

1. 准备分钟级数据库
2. 在 `config/us_equities_daily.yaml` 中改为：

```yaml
strategies:
  extra_enabled_codes:
    - "4321_intraday_resonance"

intraday:
  enabled: true
  source: sqlite
  sqlite_db_path: "/path/to/your_intraday.db"
  sqlite_table_by_timeframe:
    "30m": "your_30m_table"
    "60m": "your_60m_table"
```

3. 如果字段名不是标准字段，再继续补这些映射：

```yaml
intraday:
  sqlite_symbol_column: "symbol"
  sqlite_datetime_column: "datetime"
  sqlite_open_column: "open"
  sqlite_high_column: "high"
  sqlite_low_column: "low"
  sqlite_close_column: "close"
  sqlite_volume_column: "volume"
```

4. 重新运行：

```bash
python3 scripts/run_us_equities_daily.py --top 30
```

## 8. 当前状态

- `4321` 逻辑已接入
- 多周期接口已正式抽象
- 当前默认关闭
- 等分钟级数据库到位后即可正式启用
