# 新旧基因预测性能对比

蓝色：旧验证集 6,483 个基因；橙色：共同新验证集 200 个基因。红色虚线为理想预测 y=x。只使用已保存的验证预测，不读取最终测试集，不重新训练或推理。

- [平均转矩：真实值与预测值](01_平均转矩_真实值与预测值.png)
- [转矩波动：真实值与预测值](02_转矩波动_真实值与预测值.png)
- [相对训练前 f0 的 MAE 变化](03_新旧基因_MAE变化.png)
- [仅新基因：老模型与更新模型叠加对比](04_仅新基因_老模型与更新模型对比.png)：蓝色为老模型，橙色为更新模型；两行分别为平均转矩、转矩波动，两列分别为 G-S、F-S。
- [完整数值](性能指标.csv)，物理单位均为 N·m；图像仅保存 PNG。

展示实际更新后的新验证最优候选：G-S 第 2,000 步、F-S 第 1,500 步（各自的 `best_unconstrained.pt`），不是最后第 2,500 步，也不是按约束回退选用的 f0。两者均未满足旧验证双目标 MAE 不超过 f0 的 1.05 倍，因此不能称为已通过验收的替代模型。

新基因的平均转矩/转矩波动 MAE：G-S 分别降低 78.6%/64.8%，F-S 降低 88.7%/68.6%。旧基因对应 MAE：G-S 增加 45.5%/8.8%，F-S 增加 37.7%/21.7%。这只是单随机种子、验证集上的比较。

源代码集中在 [../diagnostics.py](../diagnostics.py)。从仓库根目录 `em` 重新出图：

```powershell
python ./03_new_spmsm_project/experiments/cnn_replay_update_v1/report/diagnostics.py --overlay
python ./03_new_spmsm_project/experiments/cnn_replay_update_v1/report/diagnostics.py --new-gene-overlay
```

`图表核验.json` 记录本次使用的验证 CSV、绘图源代码及输出哈希。实验根目录 `ARTIFACT_CHECKSUMS.json` 保留上次实验完成时的历史快照，本次绘图扩展不改写该快照。
