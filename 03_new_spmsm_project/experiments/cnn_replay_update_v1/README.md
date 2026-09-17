# cnn_replay_update_v1

本轮按用途分为三个目录。

| 目录 | 内容 |
|---|---|
| [data](data/README.md) | 训练数据来源、各模型的数据审计及回放索引 |
| models | 每个模型独立存放配置、权重、日志和源码快照 |
| reports | 本轮报告、指标表和图片 |

## 模型入口

- **vgg16**：[配置](models/vgg16/config.json) · [训练代码](models/vgg16/update.py) · [模型结果](models/vgg16/runs/)
- **small_cnn_v2**：[配置](models/small_cnn_v2/config.json) · [训练代码](models/update.py) · [模型结果](models/small_cnn_v2/runs/)
- **resnet20_v2**：[配置](models/resnet20_v2/config.json) · [训练代码](models/update.py) · [模型结果](models/resnet20_v2/runs/)
- **mini_inception_v2**：[配置](models/mini_inception_v2/config.json) · [训练代码](models/update.py) · [模型结果](models/mini_inception_v2/runs/)

三个 6×20 模型共用 models/update.py，各自配置、权重、基线和快照分别放在对应模型目录。VGG16 使用 Polar90 224×224 输入。

## 报告入口

- [VGG16 报告](reports/vgg16/REPORT.md)
- [三个 6×20 模型报告](reports/logical6x20/REPORT.md)

## 使用说明

配置和权重是真实文件，不是快捷方式或硬链接副本。训练参数、数据划分、模型权重和历史结果保持原样；本次只整理路径，不启动训练。旧记录内的路径由 maintenance/experiment_paths.py 按 data/layout_migration.json 解析。旧源码快照保留原始字节，迁移后的活动源码另行校验，因此不会直接取消恢复训练的源码完整性检查。

只查看状态（在本实验目录运行）：`python models/vgg16/update.py status`。
