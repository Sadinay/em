# V3 测试结果审核包

生成日期：2026-08-30

## 1. 实验范围

- V3 训练集：30,000 个样本。
- 提前停止验证集：5,000 个样本。
- `core_test`：1,500 个冻结样本，用于与 V2 公平比较。
- `full_test`：Scheme-A 的完整独立测试集，共 14,655 个样本。
- 每种模型使用两个随机种子：20260823、20260824。
- 四种模型：10x10 Mini-Inception、10x10 ResNet20、10x10 SmallCNN、224x224 VGG16。

## 2. 模型输出

每个模型对同一个拓扑同时输出两个连续值：

- `output[:, 0] = Tavg`：平均转矩，单位 N·m；
- `output[:, 1] = DeltaT`：转矩波动，单位 N·m。

两个目标分别标准化，训练损失为两个标准化目标 MSE 的平均值。网络共享前端特征，但使用两个独立回归输出头。

## 3. 目录说明

- `summary_reports/`：总体指标、模型对比图、预测图、残差图、训练曲线、数据划分及 V3 选样分布。
- `per_run_test_results/`：四种模型 × 两个种子的配置、训练历史、核心测试结果、完整测试结果及逐样本预测。
- `disagreement_analysis/`：1,000 个基因的四模型分歧分析原始 JSON。

## 4. 关键文件

- `summary_reports/README.md`：四模型完整测试集 MAE 摘要。
- `summary_reports/aggregate_metrics.json`：跨两个种子汇总的 MAE、RMSE、R²、偏差、误差分位数等。
- `summary_reports/v3_metrics_mean_std.csv`：适合直接查看的均值与标准差表。
- `summary_reports/v2_vs_v3_core_test_mae.png`：V2/V3 在同一 1,500 样本核心测试集上的比较。
- `per_run_test_results/**/test_predictions.csv`：核心测试集逐样本真实值和预测值。
- `per_run_test_results/**/full_test_predictions.csv`：14,655 样本完整测试集逐样本真实值和预测值。
- `per_run_test_results/**/result.json`：核心测试集指标。
- `per_run_test_results/**/full_test_result.json`：完整测试集指标。

逐样本预测 CSV 字段：`sample_index`、`actual_tavg_nm`、`predicted_tavg_nm`、`error_tavg_nm`、`actual_delta_t_nm`、`predicted_delta_t_nm`、`error_delta_t_nm`。其中误差定义为“预测值减真实值”。

## 5. 审核时需要注意

- 四种模型均预测同一对物理量，不是分别只预测一个目标。
- 汇总值是两个随机种子的均值与标准差；逐次运行结果仍保留，便于检查稳定性。
- 核心测试集用于 V2/V3 同口径比较；评价 V3 最终泛化性能时应优先看完整测试集。
- 本审核包不包含 `.pt` 模型权重，以控制文件大小；它包含复核测试统计所需的逐样本预测和真实 FEMM 标签。
