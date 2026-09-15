# v1：6×20输入模型补样回放扩展

**完成状态：**6组均完成10,000次更新，接受模型均回退各自f0。先看[本版报告](report/REPORT.md)。

本目录保存 SmallCNN V2、Mini-Inception V2、ResNet20 V2 的 G-S/F-S 更新实验。原VGG16实验保留在上一级，不覆盖其配置、模型或结果。

## 固定设置

- v1：75%旧＋25%新，即每批48旧+16新，固定10000次成功更新。
- v2：87.5%旧＋12.5%新，即每批56旧+8新，固定20000次成功更新。
- 两版每组成功更新对应新数据抽取160000次；旧数据分别480000/1120000次。AMP重试的额外抽取会单独计入实际次数，见预算CSV。种子仍为20260914，打乱后循环，G/F独立从各自网络原始f0初始化。
- 三个网络均保留原物理batch64与BatchNorm，因此不拆成8个microbatch；有效batch仍为64，损失为sum-MSE/(64×2)。
- 更新学习率为各自原初始学习率的0.1倍：SmallCNN/Mini-Inception为1e−4，ResNet20为3e−5。
- 沿用已验收数据、标签、输入编码和冻结目标尺度。40,000旧训练、G/F各1,400；验证为旧6,483、新200（U/L/B/P各50）。新旧测试封存。
- 每500步验证，新建AdamW/调度器/AMP状态；固定预算，不沿用VGG v1的早停。旧两项MAE各自≤自身f0×1.05且新MSE改善，才可接受更新。

## 文件位置

| 位置 | 用途 |
|---|---|
| `update.py` | 本版三个网络的训练、审计、流程检查、恢复和队列入口 |
| `report_results.py` | 读取验证结果生成PNG、指标CSV和报告 |
| `test_update.py` | 对两版比例、损失归一化、接受条件及测试隔离进行检查 |
| `small_cnn_v2/` | SmallCNN的配置、数据来源审计、f0验证及G/F模型 |
| `mini_inception_v2/` | Mini-Inception的同类产物 |
| `resnet20_v2/` | ResNet20的同类产物 |
| `report/REPORT.md` | 三个网络汇总；训练中可能为部分完成状态 |
| `report/model_registry.json` | 候选、最终、接受模型的位置与哈希 |

每个模型目录内：`config.json`、`audit/`、`baseline/`、`runs/G-S/`、`runs/F-S/`、`source_snapshot/`。数据直接引用原已验收位置，避免复制训练标签。`audit/data_audit.json`保存版本、来源路径、允许数据切片哈希、计数和隔离检查；`audit/*replay_indices.npy`保存实际抽样计划。

## 从仓库根目录 em 执行

```powershell
# 顺序执行/恢复全部12个任务；已完成组会跳过
python ./03_new_spmsm_project/experiments/cnn_replay_update_v2/logical6x20/update.py suite --resume

# 单模型恢复（替换网络名即可）
python ./03_new_spmsm_project/experiments/cnn_replay_update_v1/logical6x20/update.py train --architecture small_cnn_v2 --resume

# 只生成本版报告与PNG，不训练或预测测试集
python ./03_new_spmsm_project/experiments/cnn_replay_update_v1/logical6x20/update.py report
```

首次单独启动某网络，先将上述 `train` 换为 `preflight` 并去掉 `--resume`。训练队列会自动完成这一步。完整队列状态在 相对 `experiments/` 的 `cnn_replay_update_v2/logical6x20/suite_progress.json`；逐任务日志在v2的 `logical6x20/logs/`。实际当前步数以各模型 `runs/<组>/progress.json` 为准。

模型文件含义：`best_unconstrained.pt`为新验证最优；`best_feasible.pt`只在存在合格改善时生成；`last_checkpoint.pt`为最终/恢复检查点，含优化器、调度器、AMP和随机状态。接受模型可能回退原f0，以`result.json`为准。

12个任务按顺序执行，不同时争抢GPU。若进程被强制终止，先确认 `experiments/logical6x20_training.lock` 内PID已退出，再删除锁并恢复。不要修改已冻结训练源码或配置后直接续训。

本次小模型预算获用户单独批准，比原VGG实验更长。只生成PNG，不生成PDF；本轮不做分流、Shrink-and-Perturb、架构搜索或新增FEMM标注。

执行布局记录：小模型沿用本次回放入口的channels_last内存布局，并在同一执行布局下重算自身冻结f0作为基线；原模型权重及BatchNorm状态逐张量核验一致。该布局不改变2×6×20编码或网络结构，CUDA/AMP仍可能有数值非确定性。

训练完成后可重新加载候选、最终及合格模型验证完整指标：`python ./03_new_spmsm_project/experiments/cnn_replay_update_v1/logical6x20/report_results.py --verify`。该命令需要GPU，训练期间不要同时运行。

## 单模型性能图

[按网络查看四类性能图](report/性能图索引.md)：新旧基因散点、MAE变化及仅新基因的新老模型对比。绘图源码为 `plot_performance.py`，只读取已保存验证预测；不加载测试集。
