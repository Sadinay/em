# 256样本过拟合检查报告

## 目的

本实验只验证输入、目标标准化、GPU渲染、模型、双回归头、反标准化及checkpoint链路是否正常。所有指标都来自用于训练的同一批256条样本，不代表未知拓扑泛化性能。

两种模型使用完全相同的样本索引，目标标准化统计量也只来自这256条训练样本。设备为NVIDIA GeForce RTX 5070。

## 结果

| 输入与模型 | 参数量 | epoch | 时间 | Tavg MAE | Tavg R² | DeltaT MAE | DeltaT R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3×10×10 Mini-Inception | 911,346 | 60 | 7.7 s | 0.01634 N·m | 0.99871 | 0.00832 N·m | 0.99827 |
| 8×224×224 FEM语义 Mini-Inception | 990,706 | 130 | 110.2 s | 0.02078 N·m | 0.99807 | 0.01478 N·m | 0.99556 |

两条链路均达到双目标R²>0.995的记忆门槛。

## 观察

- 10×10模型更容易优化，60 epoch即通过。
- 224×224模型需要约130 epoch，耗时约为10×10的14倍。
- 224×224使用batch size 8，BatchNorm运行统计在前期有明显波动；正式短训练时应重点检查，必要时对比GroupNorm。
- 224×224能够记住样本，证明真实FEM查找表没有丢失决定目标所需的拓扑信息。
- 是否比10×10泛化更好仍完全未知，必须依靠固定验证集和测试集短训练判断。

## 产物

- `outputs/overfit_256/summary.json`
- `outputs/overfit_256/selected_sample_indices.npy`
- `outputs/overfit_256/target_scaler.json`
- `outputs/overfit_256/logical10/checkpoint.pt`
- `outputs/overfit_256/semantic224/checkpoint.pt`
- `outputs/overfit_256/*/prediction_comparison.png`
- `outputs/overfit_256/topology_audit_224/training_topology_01.png`至`06.png`

下一步应选取相同的5,000–10,000条训练样本和独立验证/测试子集，进行短训练对照。届时才报告泛化MAE、RMSE和R²。
