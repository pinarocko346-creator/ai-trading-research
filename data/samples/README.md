# 样本库说明

样本库用于校准规则，不直接用于训练黑箱模型。

## 三类样本

- `standard_samples.csv`：你认为最标准、最接近原始定义的案例
- `boundary_samples.csv`：看起来像，但存在争议的边界样本
- `failure_samples.csv`：触发后失败或本来就不该触发的样本

## 推荐标注字段

- `sample_id`
- `signal_type`
- `symbol`
- `signal_date`
- `label_bucket`
- `timeframe`
- `is_valid`
- `market_context`
- `notes`

## 使用方式

1. 先从你最熟悉的 `2B结构` 和 `双突破` 开始，每类至少整理 15 个样本。
2. 标出为什么它是标准样本，或者为什么它只是“形似神不似”。
3. 回测表现和人工标注不一致时，优先回看样本定义，不要急着改模型。
