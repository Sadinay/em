# SPMSM去重与平均转矩分带划分

本阶段未启动训练。历史MAT保持只读。

- 历史记录：123,012
- 精确基因去重后：65,702
- 移除重复记录：57,310
- 标签冲突隔离：898
- 最终可监督学习：64,804
- 训练/验证/测试：51,838/6,483/6,483
- 划分依据：仅使用Tavg固定物理分布带；DeltaT保留为标签但不参与当前分层。
- 防泄漏：相同基因先去重，raw→corrected修复关联组不得跨集合。

| Tavg区间 (N·m) | 总数 | 训练 | 验证 | 测试 |
|---|---:|---:|---:|---:|
| <2.0 | 166 | 132 | 17 | 17 |
| 2.0-2.5 | 4,766 | 3,812 | 477 | 477 |
| 2.5-3.0 | 16,615 | 13,291 | 1,662 | 1,662 |
| 3.0-3.25 | 17,256 | 13,804 | 1,726 | 1,726 |
| 3.25-3.5 | 19,576 | 15,660 | 1,958 | 1,958 |
| 3.5-3.75 | 6,176 | 4,940 | 618 | 618 |
| >=3.75 | 249 | 199 | 25 | 25 |

## 训练入口

- 数据：`C:\Users\26096\Desktop\em\03_new_spmsm_project\data_zone\processed\spmsm_topology_dataset\training_corrected_binary`
- 固定索引：`C:\Users\26096\Desktop\em\03_new_spmsm_project\cnn_zone\outputs\splits\scheme_a_tavg_bands_80_10_10.npz`
- `topology_bits.npy`形状为`[N,6,20]`。
- `targets_tavg_delta.npy`两列依次是`Tavg`和历史`DeltaT`。
