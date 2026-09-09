# 电机拓扑回归模型展示索引

本目录将两个独立的单输出回归任务平行存放。它们读取相同的 1,300 个拓扑、使用完全相同的训练/验证/测试样本，并采用相同的 ResNet-20、MLP 和 Small CNN 架构；区别仅在于监督输出。

| 任务 | 输出 | 最佳模型 | 测试 MAE | 测试 RMSE | 测试 R² | 报告 |
|---|---|---|---:|---:|---:|---|
| 转矩波动预测 | `t_ripple`，无量纲 | Small CNN | 0.0845 | 0.1381 | 0.8370 | [打开报告](torque_ripple/MODEL_TRAINING_REPORT.md) |
| 平均转矩预测 | `t_avg`，Nm | MLP | 0.0316 Nm | 0.0480 Nm | 0.9748 | [打开报告](mean_torque/MODEL_TRAINING_REPORT.md) |

目录结构：

```text
showoutput/
├── README.md
├── torque_ripple/   # 转矩波动率模型、报告与图片
└── mean_torque/     # 平均转矩模型、报告与图片
```

两个任务当前是**两个单独训练的模型集合**，不是一个双输出网络。使用时需要根据目标分别加载对应 checkpoint。

