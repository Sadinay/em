# cnn_shrink_perturb_v1

本轮按用途分为三个目录。

| 目录 | 内容 |
|---|---|
| [data](data/README.md) | 训练数据来源、各模型的数据审计及回放索引 |
| models | 每个模型独立存放配置、权重、日志和源码快照 |
| reports | 本轮报告、指标表和图片 |

## 模型入口

- SmallCNN：[α=0.50 配置](models/small_cnn_v2/alpha_050/config.json) · [α=0.80 配置](models/small_cnn_v2/alpha_080/config.json) · [α=0.95 配置](models/small_cnn_v2/alpha_095/config.json) · [训练代码](models/small_cnn_v2/update.py)

## 报告入口

- [SmallCNN S&P 报告](reports/small_cnn_v2/REPORT.md)

## 使用说明

配置和权重是真实文件，不是快捷方式或硬链接副本。训练参数、数据划分、模型权重和历史结果保持原样；本次只整理路径，不启动训练。旧记录内的路径由 maintenance/experiment_paths.py 按 data/layout_migration.json 解析。旧源码快照保留原始字节，迁移后的活动源码另行校验，因此不会直接取消恢复训练的源码完整性检查。

查看原有命令参数（不启动训练）：`python models/small_cnn_v2/update.py --help`。
