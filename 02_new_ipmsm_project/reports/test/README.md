# 六模型 GPU 短训练报告

## 1. 本次测试范围

- 运行日期：2026-08-22
- GPU：NVIDIA GeForce RTX 5070
- PyTorch：2.13.0+cu132
- CUDA：13.2
- 原始划分：Scheme A，按性能分层且保持历史修复组隔离
- 固定训练子集：11,700 个样本
- 固定验证子集：1,500 个样本
- 固定测试子集：1,500 个样本
- 六个模型使用完全相同的训练、验证和测试索引
- 目标标准化均值和标准差只由 11,700 个训练样本计算
- 输出 1：平均转矩 `Tavg`，单位 N·m
- 输出 2：转矩波动 `DeltaT`，单位 N·m

三个 10×10 模型训练 10 epochs；三个 224×224 模型训练 2 epochs。每轮后在验证集上评价，并保存验证损失最低的 checkpoint；测试集只用于最终独立评价。

## 2. 独立测试集结果

| 输入 | 模型 | 最佳轮次 | Tavg MAE | Tavg RMSE | Tavg R² | DeltaT MAE | DeltaT RMSE | DeltaT R² | 训练时间 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10×10 | Mini-Inception | 10 | 0.0693 | 0.1044 | 0.9746 | 0.0748 | 0.1308 | 0.8033 | 35.4 s |
| 10×10 | ResNet20 | 8 | 0.0779 | 0.1136 | 0.9699 | 0.0741 | 0.1286 | 0.8100 | 20.4 s |
| 10×10 | SmallCNN | 7 | 0.0645 | 0.0949 | 0.9790 | 0.0776 | 0.1328 | 0.7973 | 7.8 s |
| 224×224 | Mini-Inception | 2 | 0.1301 | 0.1588 | 0.9411 | 0.1031 | 0.1609 | 0.7025 | 189.9 s |
| 224×224 | ResNet18 | 1 | 0.2843 | 0.3353 | 0.7376 | 0.1242 | 0.1810 | 0.6233 | 636.5 s |
| 224×224 | VGG16 | 1 | 0.1546 | 0.2241 | 0.8827 | 0.1701 | 0.2423 | 0.3251 | 289.0 s |

MAE 和 RMSE 均为反标准化后的物理单位 N·m。模型训练总计约 19.7 分钟，不含代码检查和 GPU 冒烟测试时间。

## 3. 初步结论

1. CUDA 训练链路已经验证可用。驱动层连续采样显示 224 ResNet18 训练时 GPU 计算利用率为 97%–100%，约 129 W、54–57°C。任务管理器若显示约 4%，通常显示的是 3D 图形引擎，而不是 CUDA/Compute 引擎。
2. 在本次 10 轮短训练中，10×10 SmallCNN 的平均转矩 MAE 最低；ResNet20 的 DeltaT MAE 和 R²略优。
3. 在只训练 2 轮的 224 模型中，Mini-Inception 当前最好，也明显比 ResNet18 和 VGG16 更快。
4. 不能根据本次表格断言 10×10 输入最终优于 224×224 输入。两类模型训练轮次不同，而且 ResNet18、VGG16 的第 2 轮验证损失反弹，仍处于训练早期。正式比较需要相同的收敛准则、学习率调度和早停策略。
5. 本次结果可以证明六模型的数据读取、GPU 训练、双目标输出、验证选模、反标准化和独立测试流程均已贯通。

## 4. 文件位置

- 模型目录：`cnn_zone/models/test_11700`
- 固定样本索引：`fixed_scheme_a_subsample_11700_1500_1500.npz`
- 训练集目标 scaler：`target_scaler.json`
- 完整机器可读汇总：`run_summary.json`
- 六模型 MAE 汇总图：`six_model_mae_comparison.png`
- 六模型平均转矩有符号绝对误差图：`mean_torque_signed_absolute_error_six_models.png`
- 六模型转矩波动有符号绝对误差图：`torque_ripple_signed_absolute_error_six_models.png`
- 每个模型各有一张双目标真实值—预测值图
- 每个模型目录中均包含：
  - `best_checkpoint.pt`
  - `config.json`
  - `history.json`
  - `test_metrics.json`
  - `test_predictions.csv`

## 5. 复现命令

在项目根目录执行：

```powershell
cd C:\Users\26096\Desktop\em\02_new_ipmsm_project
python cnn_zone\semantic90\scripts\run_six_model_short_test.py
```

该命令默认要求 CUDA 可用，并使用本报告记录的固定随机种子与采样规则。再次运行会重新训练并覆盖同名短测试输出。
