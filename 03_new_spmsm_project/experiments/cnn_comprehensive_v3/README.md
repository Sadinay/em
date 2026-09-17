# CNN 综合对比 v3

本阶段只准备两项内容：

1. 从当前 `.fem`、MAT 几何和基因映射推导 90° gap-aware `6×N` 输入；
2. 由历史 40000 中 F＋LCMD 选出的 20000、上一轮 3299、最新 10000 组成统一候选池，并全局去重、审计标签。

候选池已经按用户确认的 70% / 15% / 15% 冻结划分：训练 23309、验证 4995、测试 4995。历史 20000 与新增 13299 先分别划分，再合并为八个配置共用的一套身份。

## 目录

- `prepare.py`：本阶段唯一主入口；几何推导、候选池整理和报告命令集中在此。
- `train.py`：八个固定配置的集中训练入口；默认只预检，显式 `--train` 才会开始。
- `data/`：本阶段用到的完整冻结数据、清单、映射和缓存。
- `models/`：本阶段涉及的冻结特征模型身份及后续八配置模型位置。
- `reports/`：统计报告和核对图。

目前已冻结的范围：SmallCNN、ResNet20、VGG16、ResNet18；输入只含逻辑 `6×20`、gap-aware `6×N90`、Polar90、Polar360。Mini-Inception、所有 XY224 和其他新增架构均排除。

```powershell
python .\03_new_spmsm_project\experiments\cnn_comprehensive_v3\prepare.py gap90
python .\03_new_spmsm_project\experiments\cnn_comprehensive_v3\prepare.py status
python .\03_new_spmsm_project\experiments\cnn_comprehensive_v3\train.py
```

确认预检通过后，由用户手动顺序启动全部八个配置：

```powershell
python .\03_new_spmsm_project\experiments\cnn_comprehensive_v3\train.py --train --models all
```

中断后恢复并跳过已完成配置：

```powershell
python .\03_new_spmsm_project\experiments\cnn_comprehensive_v3\train.py --train --models all --resume
```

模型均从头训练，检查点只按验证集标准化 MSE 选择；测试集只在每个配置完成后评价一次，不用于归一化或模型选择。
