# 首轮 CNN 补样回放更新：G-S / F-S

**6×20 小模型扩展：** SmallCNN、Mini-Inception、ResNet20 的 G/F 更新位于 [logical6x20](logical6x20/README.md)。用户已批准 v1 每组10000步、v2每组20000步；原VGG16实验保留。

**已完成：**G-S/F-S各2500次更新，分别用时16.42/15.62分钟，均按预定规则早停。两组新结构误差显著下降，但均未通过旧验证两个目标各自5%的容限，因此没有合格的替代f0模型。先看[结果解释与候选对比](report/interpretation.md)，完整记录见[主报告](report/REPORT.md)。

**本阶段训练、核验、恢复和主报告代码集中在 [update.py](update.py)。** 它复用 `cnn_zone/src` 的原模型、输入渲染、数据接口、AMP损失和验证函数。补充展示无约束候选的短脚本为 [report/diagnostics.py](report/diagnostics.py)，只读取已有验证结果出图，不加载模型。原来的选种子/FEMM入口仍位于 `input_distribution_pilot_v1`。

## 文件位置

| 位置 | 用途 |
|---|---|
| `update.py` | 本实验唯一运行入口与源码 |
| `config.json` | 正式训练前冻结的设置与旧模型配置 |
| `audit/` | 数据隔离、版本、回放索引计划、GPU/恢复检查 |
| `baseline/` | 冻结f0在旧验证和新dev上的第0步评价 |
| `runs/G-S/`、`runs/F-S/` | 两组独立权重、训练轨迹、每500步验证结果 |
| `source_snapshot/` | 正式开始时冻结的本实验源码 |
| `logs/` | 终端运行日志 |
| `report/REPORT.md` | 结果入口，配套CSV与PNG/PDF曲线 |
| `report/diagnostics.py` | 补充候选模型、来源误差与总损失图，不训练/推理 |
| `tests/` | 回放、模型选择边界和测试集访问保护检查 |

`last_checkpoint.pt` 包含优化器、调度器、AMP及随机状态，用于恢复；`best_feasible.pt` 为满足旧分布容限的最优更新检查点，若没有可行更新则不创建；`best_unconstrained.pt` 保存未加约束的新dev最优更新。若可行更新没有优于f0，选用结果回退到f0并明确标注，不能视为训练改善。

新检查点仍用 `model_state`、`target_mean`、`target_std`；架构与输入参数在 `config.baseline_config`，更新设置在 `config`。后续加载复用原 `build_model`，不要把更新配置当成另一种网络。`report/model_registry.json` 明确区分更新候选和按本轮规则接受的模型。

## 运行与恢复

从仓库根目录 `em` 执行：

```powershell
# 只核对数据、来源、测试隔离和配置
python ./03_new_spmsm_project/experiments/cnn_replay_update_v1/update.py audit

# 小量前向/反向与恢复检查，然后重置f0做第0步评价
python ./03_new_spmsm_project/experiments/cnn_replay_update_v1/update.py preflight

# 顺序训练G-S、F-S，完成后生成报告
python ./03_new_spmsm_project/experiments/cnn_replay_update_v1/update.py train

# 中断后继续；已完成的组会跳过
python ./03_new_spmsm_project/experiments/cnn_replay_update_v1/update.py train --resume

# 查看进度；不会训练或预测测试集
python ./03_new_spmsm_project/experiments/cnn_replay_update_v1/update.py status
```

只重生成已有结果的图表（不训练、不预测）：

```powershell
python ./03_new_spmsm_project/experiments/cnn_replay_update_v1/update.py report
python ./03_new_spmsm_project/experiments/cnn_replay_update_v1/report/diagnostics.py
```

每100步写一次完整原子检查点；中断后最多重放100步，日志以检查点中的已完成轨迹为准。`training.lock` 防止同时启动多个训练；强制终止后只有确认锁内PID已经退出才移除锁。恢复检查源码、配置、数据审计及f0身份，不接受不同设置的续训。正常运行自动移除锁。

## 本轮固定方案

两组都从同一份f0独立初始化全网络，新建AdamW与ReduceLROnPlateau。初始学习率1e−5，有效batch64，每批48旧+16新；microbatch8，每次6旧+2新，累积8次。旧40000轮换、新1400循环，两组共用同一抽取计划和种子20260914。目标归一化和双目标平均标准化MSE沿用f0。

每500步分别验证旧6483与共同新200，不合并成一个按人数加权的指标。旧Tavg/DeltaT各自MAE≤f0×1.05，再按新dev双目标平均标准化MSE选优；至少2000步且连续5次无受约束改善时早停，上限10000步。调度器监控新dev标准化MSE，沿用factor0.5、patience3、最低学习率1e−6。

新400测试样本只读 `memberships.csv` 的ID及角色；旧测试只读划分索引和身份列，目标数组仅索引旧train/validation。不调用旧 `train_one()`，因为它在训练后会评价测试集。沿用898组历史标签冲突隔离与初始代异常保留策略，本轮不清洗、改标签、选新种子或运行FEMM。

本轮结束后停在两组和报告，不自动扩展种子、比例、分流或Shrink-and-Perturb。
