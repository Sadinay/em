# V3 30,000样本训练入口

V3在原有两个模型基础上补充两个运行较快的10x10模型：

- `logical10/mini_inception_v2`：10x10三通道材料矩阵，最大100 epochs；
- `logical10/resnet20_v2`：10x10三通道材料矩阵，最大100 epochs；
- `logical10/small_cnn_v2`：10x10三通道材料矩阵，最大100 epochs；
- `semantic224/vgg16_v2`：224x224八通道FEM语义输入，最大35 epochs。

四个模型均使用随机种子`20260823`和`20260824`，共8个model/seed任务。优化器、学习率、损失函数、scheduler、早停、AMP与有效batch沿用相应V2配置。

## 冻结数据划分

- 训练集：30,000条；
- 训练期验证集：5,000条，用于scheduler、早停和最佳checkpoint选择；
- 核心测试集：1,500条，沿用V2测试索引，用于V2/V3公平比较；
- 完整验证：14,656条，只在训练结束后审计；
- 完整测试：14,655条，只在训练结束后最终评价。

验证集和测试集不会参与训练。目标标准化统计量只从30,000条训练样本计算。

## 推荐运行命令

在项目根目录运行：

```powershell
cd C:\Users\26096\Desktop\em\02_new_ipmsm_project

# 只检查GPU、模型、冻结划分和参数，不启动训练
python cnn_zone\semantic90\scripts\run_v3_30000.py --dry-run

# 正式开始；以后中断恢复也始终使用同一条命令
python cnn_zone\semantic90\scripts\run_v3_30000.py --resume
```

按`Ctrl+C`时，当前尚未完成的microbatch不会单独保存；最近一个完整epoch已经原子化保存。再次执行`--resume`会从该epoch继续，已经完成的model/seed任务和完整评估会自动跳过。

如果只想先跑一个输入分支：

```powershell
python cnn_zone\semantic90\scripts\run_v3_30000.py --resume --models logical10/mini_inception_v2

python cnn_zone\semantic90\scripts\run_v3_30000.py --resume --models semantic224/vgg16_v2
```

不要同时在两个终端运行同一个model/seed，否则会争用同一checkpoint目录。两个分支若需要分终端运行，可以分别使用上面两条互不重叠的`--models`命令，但显存和计算资源会互相竞争，通常顺序运行更稳定。

## 输出位置

```text
cnn_zone/models/v3_30000/
  logical10/mini_inception_v2/seed_*/
  semantic224/vgg16_v2/seed_*/

reports/v3_30000/
  run_summary.json
  target_scaler.json
  fixed_v3_split_30000_5000_1500_full14655.npz
```

每个seed目录保存：

- `last_checkpoint.pt`：断点恢复状态；
- `best_checkpoint.pt`：验证集最优模型；
- `history.json`：逐epoch训练和验证记录；
- `test_predictions.csv`、`result.json`：1,500条核心测试结果；
- `full_validation_predictions.csv`、`full_validation_result.json`；
- `full_test_predictions.csv`、`full_test_result.json`。

全部4个任务完成后生成汇总图：

```powershell
python cnn_zone\semantic90\scripts\generate_v3_report.py
```
