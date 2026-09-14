# CNN 补样回放更新 v2

本轮只做 G-S、F-S：同一原始冻结 f0 独立热启动；每个物理 batch 为 7 旧＋1 新，累积 8 批，每次有效更新 56 旧＋8 新。固定完成 5,000 次成功优化器更新，取消 v1 的受约束早停。其他参数、数据划分及随机种子沿用 v1。

优先查看 [当前状态](STATUS.md) 和 [结果报告](report/REPORT.md)。所有产物都在本目录；不覆盖 v1。

| 文件/目录 | 用途 |
|---|---|
| [update.py](update.py) | 训练唯一入口：审计、流程核验、训练、恢复、状态 |
| [report_results.py](report_results.py) | 读取已有验证结果，生成 v1/v2 比较报告及两张 PNG |
| `config.json` | 正式冻结配置，保留 f0 路径/哈希、原目标尺度 |
| `audit/` | 数据隔离、v1来源记录、批次/恢复检查、回放索引和最终核验 |
| `baseline/` | f0 第0步旧/新验证结果 |
| `runs/G-S/`、`runs/F-S/` | 各组模型、进度、轨迹和每500步验证结果 |
| `source_snapshot/update.py` | 正式训练开始时的源码快照 |
| `logs/` | 标准输出和异常日志 |
| `report/` | 简洁报告、完整指标CSV、PNG和模型用途清单 |
| `tests/test_update.py` | 批次、损失归一化、测试访问和接受条件检查 |

从仓库根目录 `em` 运行：

```powershell
python ./03_new_spmsm_project/experiments/cnn_replay_update_v2/update.py preflight
python ./03_new_spmsm_project/experiments/cnn_replay_update_v2/update.py train
# 中断后恢复；已完成组自动跳过
python ./03_new_spmsm_project/experiments/cnn_replay_update_v2/update.py train --resume
# 只重新生成报告及PNG，不训练、不推理
python ./03_new_spmsm_project/experiments/cnn_replay_update_v2/update.py report
```

进度见每组 `progress.json`。`last_checkpoint.pt` 每100步保存完整优化器、调度器、AMP与随机状态，完成时就是第5000步最终模型，避免重复存储同一份大文件。`best_unconstrained.pt` 为新验证最优候选；`best_feasible.pt` 只在旧两项MAE≤f0×1.05且新标准化MSE优于f0时存在。实际接受路径及是否回退f0记录在 `result.json` 和 `report/model_registry.json`。

恢复要求源码、配置和数据审计一致。若进程被强制终止留下 `training.lock`，先核实锁内PID已退出再删除该锁；不要同时启动两份训练。异常失败不能标记为完成。

沿用4万旧训练、G/F各1400、新验证200与旧验证6483；新旧最终测试封存。指标Tavg为六角度均值、DeltaT为峰峰差，均为N·m。单随机种子的开发结果不能作为最终盲测或稳定优越性结论。
