# IPMSM 90° FEM语义图工程（阶段1–3）

本目录把100个材料基因实时映射到 `IPMSM.fem` 的真实90°有限元几何。训练图只存在于内存/显存，不保存17万张PNG，也不会对每个样本调用FEMM。

当前已完成文件审计、数据清洗衔接、几何查找表、实时渲染、固定评估划分、闭环验证、六个V2模型结构/GPU测试和256样本过拟合检查。固定11,700/1,500/1,500子集上的六模型、三随机种子V2公平短测已经全部完成（18/18），最终图表和汇总报告位于 `reports/v2_test_11700/`；尚未开始146,471样本正式全量CNN训练。

## 实际数据口径

- 历史标签对应 `population_all`（结构修正后），不是 `population_noChange_all`。
- 默认监督数据为146,471个无冲突、物理唯一的corrected拓扑。
- `0=Air`、`1=N38 PM`、`2/3=Pure Iron`。
- code 2/3在FEMM中是同一种物理材料，主数据集将其合并；四状态数据仍保留在数据区作审计。
- 目标为 `[Tavg, DeltaT]`，不是Fitvalue，也不是六角度转矩曲线。

## 实时语义通道

默认224×224输入为8通道：

```text
design_air
design_pm_inward
design_pm_outward
design_iron
fixed_air
fixed_iron
winding
geometry_mask
```

边界通道可选。PM在0°–45°径向向内，在45°–90°径向向外。

## 执行命令

在项目根目录执行：

```powershell
cd C:\Users\26096\Desktop\em\02_new_ipmsm_project

# 从已有的一次性FEMM解网格构建128/224查找表
python cnn_zone\semantic90\scripts\build_lookup.py

# 生成少量审计图和3例FEM—内存闭环对照
python cnn_zone\semantic90\scripts\validate_renderer.py

# 建立80/10/10与时间外推两套评估划分
python cnn_zone\semantic90\scripts\create_evaluation_splits.py

# DataLoader批处理冒烟测试
python cnn_zone\semantic90\scripts\smoke_test_dataloader.py

# 全项目测试
python -m pytest -q

# 使用同一批256条训练样本检查两条模型链路能否记忆
python cnn_zone\semantic90\scripts\run_overfit_256.py --epochs 200
```

如果以后只有原始MAT和FEM、没有缓存ANS网格，可先运行一次：

```powershell
python cnn_zone\semantic90\scripts\prepare_reference_ans.py
```

该命令只对参考模型运行一次FEMM，以获得三角网格；之后训练不再调用FEMM。

## 主要输出

- `outputs/lookups/fem90_lookup_128.npz`
- `outputs/lookups/fem90_lookup_224.npz`
- `outputs/audit/semantic_examples/`：12张人工审计图
- `outputs/audit/fem_memory_comparison/`：3例修改后FEM与内存图对照
- `outputs/audit/renderer_validation.json`
- `outputs/splits/`：两套固定评估索引
- `outputs/overfit_256/`：两种输入的过拟合checkpoint、曲线、预测和6张训练拓扑审查图
- `reports/stage1_3_audit.md`
- `reports/overfit_256_report.md`

256样本链路已经通过。六个V2模型已经使用完全相同的固定11,700/1,500/1,500训练、验证和测试子集完成三随机种子公平短测；短测结果只能用于结构筛选，不等同于146,471样本的正式全量训练结果，也不得把过拟合指标当成测试集性能。

## V1与V2模型入口

六个V1/V2模型的逐层结构、张量尺寸、参数量、训练配置和逐项差异，统一记录在上一级 [`cnn_zone/README.md`](../README.md) 中。本目录保留两个版本，V2没有覆盖或删除V1：

- `src/models.py`：V1六模型实现；
- `src/models_v2.py`：V2六模型实现；
- `src/training_v2.py`：V2训练、验证、checkpoint和断点恢复；
- `scripts/run_v2_short_test.py`：运行或续跑三随机种子公平短测；
- `scripts/generate_v2_report.py`：汇总预测、误差图和指标报告。

V2相对V1的核心变化为：

- 两个目标由共享的单层输出改为独立的两层MLP回归头，减少平均转矩与转矩波动之间的梯度干扰；
- 224×224语义模型把BatchNorm改为GroupNorm，并统一保留到4×4末端特征，增强小物理batch训练的稳定性和空间细节保留；
- Mini-Inception加入残差连接；10×10 ResNet的第三阶段不再继续降采样；Small CNN增加仅供转矩波动使用的局部卷积分支；
- 224×224 ResNet缩减通道宽度，显著降低参数量；VGG保留13层卷积骨干，但改用GroupNorm和独立回归头；
- 公平短测统一使用训练集目标标准化后的双目标MSE，而不是SmoothL1；所有模型共用同一数据划分和目标scaler；
- 224×224模型通过梯度累积把有效batch统一为64，并支持完整随机状态和训练状态的checkpoint恢复。

运行或续跑：

```powershell
python cnn_zone\semantic90\scripts\run_v2_short_test.py --resume
```

训练完成后生成汇总报告：

```powershell
python cnn_zone\semantic90\scripts\generate_v2_report.py
```

V2短测结果位于 `reports/v2_test_11700/`。
