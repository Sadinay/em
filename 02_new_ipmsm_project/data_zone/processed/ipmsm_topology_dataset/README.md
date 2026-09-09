# IPMSM topology dataset

该数据集由 `workspace_600.mat` 构建，原 MAT 文件未被修改。

默认监督学习入口是 `training_corrected_physical_three_state/`。其输入是 corrected FEMM 拓扑，并把两个遗传铁状态合并为同一个物理 Iron 类。raw gene 仅用于审计和分组，不被错误地赋予 repaired topology 的 FEMM 标签。

数据规模和划分：

- 可靠物理唯一拓扑：146,471
- train：102,519
- validation：21,977
- test：21,975
- 输出列：`[Tavg, DeltaT]`
- 修复关联组跨集合数量：0
- 拓扑哈希跨集合数量：0

详细依据见 `reports/ipmsm_topology_dataset/`。
