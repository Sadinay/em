# SmallCNN SP＋经验回放首轮系数筛选

**完成状态：**六组均已完成20,000次成功更新，合计约19.46分钟；三组G合格、三组F回退。15份保存检查点重载验证通过。按既定新MSE选择，α=.80、β=.01的G-S（13500步）最有希望。测试集继续封存，本轮已停止。

源代码：`update.py`（数据审计、初始化、检查、训练、恢复）；`report_results.py`（验证结果核验、模型重载、报告与PNG）。训练代码在正式启动前冻结；复用原v2算法的逐函数核对见 `source_reuse_audit.json`。

## 当前范围

仅 Logical6x20SmallCNNV2、原生空气/永磁体one-hot 2×6×20。α=0.95/0.80/0.50，β=0.01；每种系数G-S/F-S各一次，共6次。每组20,000次成功更新，物理batch=64，56旧+8新，每500步完整验证；不提前停止。

原始SmallCNN自身f0独立初始化，不从v1/v2候选叠加。SP只在第一个训练更新前执行一次；恢复检查点时不施加SP。所有可训练参数按名称执行 α×f0+β×原生随机模型，包括BN仿射和偏置。所有缓冲状态从f0完整继承。β是随机参数乘数，不能解释成统一高斯噪声标准差；BN随机scale原生为1，bias原生为0。

训练种子20260914；随机模型使用隔离CPU RNG上下文、种子20261914。原生构造用同一 `tr.build_model`，不同系数和G/F共享随机方向，外部CPU/CUDA训练随机状态不变。AMP失败尝试撤销BN更新，但仍会消耗Dropout随机数，可能导致不同组后续随机序列分化；记录实际抽取和重试。

## 运行（em仓库根目录）

```powershell
python ./03_new_spmsm_project/experiments/cnn_shrink_perturb_v1/update.py suite --resume
```

六组按顺序执行。已完成组跳过，未完成组从每100步原子检查点恢复。单系数恢复：

```powershell
python ./03_new_spmsm_project/experiments/cnn_shrink_perturb_v1/update.py train --alpha 0.80 --resume
```

核验保存模型并生成报告（不训练、不使用测试集）：

```powershell
python ./03_new_spmsm_project/experiments/cnn_shrink_perturb_v1/report_results.py --verify
```

## 文件分类

- `alpha_095/`、`alpha_080/`、`alpha_050/`：各自config、数据版本审计、初始化检查、原f0验证和G/F训练结果。
- 每组 `runs/G-S/` 或 `runs/F-S/`：第0步SP后验证、每500步验证预测、训练轨迹、初始化哈希与逐参数变化、best_feasible（若有）、best_unconstrained、last_checkpoint。
- `logs/`、`STATUS.json`：训练日志、简短进度。
- `report/`：实验内原始报告产物；报告脚本自动将副本集中到主项目 `reports/SPα0.95-0.80-0.50_β0.01_掺杂率12.5pct_v1/`，并更新按时间排列的 `reports/实验索引.md`。

数据始终引用已验收清单：旧训练40000、旧验证6483、G/F各1400、共同新验证200。新旧测试性能标签不读、不推理；没有新增FEMM和数据改写。

接受门槛始终参照原始f0：旧Tavg和DeltaT MAE分别≤1.05×原值，新验证标准化MSE严格优于原f0。无SP G-S合格对照是11500步，而不是15500步无约束最佳。没有合格更新则回退f0，仍完整报告候选/最终表现。

后续仅保留SmallCNN、ResNet20和已有VGG16视图为候选；Mini-Inception停止新增实验，历史保留。本轮不开展β追加筛选、其他架构、GA、分流或测试集评价。

## 新旧基因MSE直接对比

[查看MSE图与结论](report/MSE对比说明.md)。绘图源码 `plot_mse_comparison.py`；读取已保存验证指标，只生成PNG，不训练、不接触测试集。

## 事后容限放宽复核

[6%容限分析](report/6pct容限复核.md)：用户要求小幅放宽后，将5%/6%/7.5%统一应用于已有G/F及无SP验证轨迹。6%下α=.80 F-S的9500步合格；原5%结果保留。源码 `review_tolerance.py`，不重训，不修改模型/数据或原accepted字段。
