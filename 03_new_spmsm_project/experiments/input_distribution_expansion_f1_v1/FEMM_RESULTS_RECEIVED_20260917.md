# 一万个基因 FEMM 结果接收记录

接收日期：2026-09-17。

结果保存在 [FEMM 正式结果目录](../../femm_zone/results/FEMM_10000_results_20260917/README.md)，与上一批结果并列。U 盘原件保留，结果包内部未改动。

- 训练 8000、验证 1000、独立测试 1000；合计 10000 基因、60000 角度。
- 复制后完整校验通过：文件和压缩包内哈希、逐角度成功状态、结果身份、清理回执、转矩波形及均值/峰峰值汇总一致。
- 与本实验原始 femm_queue.csv 核对，10000 个基因 ID 和 bits 全部一致。
- 测试标签保持 sealed_test 独立分区；本次仅完整性校验，未做模型评估或训练。
- 成功求解的 FEM/ANS 已由结果生成端清理；包内保留结果、执行记录和清理回执。
- Windows 验证命令需要 UTF-8 模式：`python -X utf8 verify_package.py`。

## 数据入口

- [训练标签](../../femm_zone/results/FEMM_10000_results_20260917/train_dev/train_labeled.csv)
- [验证标签](../../femm_zone/results/FEMM_10000_results_20260917/train_dev/dev_labeled.csv)
- 测试标签位于结果包 sealed_test 分区，后续训练流程不得混入。

STATUS.json 和原有交付报告记录的是选样交付时状态，保留原件；本记录补充外部求解结果已接收的后续状态。
