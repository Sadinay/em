# improved_dataset_sampling 父本档案试运行

日期：2026-08-07  
Run：`improved_runs/improved_parent_archive_20260807_001`

## 固定配置

```yaml
parent_archive_target: 60
campaign_count: 6
parents_per_campaign: 10
initial_parent_reuse: false
topology_cluster_target: 12
parents_per_cluster: 5
selection_window: 50
hamming_admission_mode: observe_only
```

- `physics_hash=d727b66b38528a60097e12eca1bd52d0304d96537d92ec72cc2407a025792153`
- `sampling_hash=1aa85d5bd1f406a08bc4d86c3ed18e5cbd8f25073c2d4db6c9be85f2ed5a7588`
- `repair_config_hash=61e4dbbcd42fff957c8270da45dc6b2dd6799acaf6a960a534c382dc49afa375`

本阶段只运行纯拓扑proposal、修复、去重和多样性选择，没有启动FEMM。

## 前10个父本结果

- 总proposal：548
- 修复后合法：533（97.26%）
- 合法且唯一：450
- 精确重复：83
- 无需修复即合法：327
- 确定性修复后合法：206
- 超过最大修复距离而拒绝：15
- 父本：10/60
- 用时约29秒

三个变异尺度：

| 尺度 | proposal | 合法 | 原始父子Hamming均值 | 修复Hamming均值 |
|---|---:|---:|---:|---:|
| small | 239 | 239 | 2.46 | 0.87 |
| medium | 194 | 190 | 9.28 | 3.73 |
| large | 115 | 104 | 24.04 | 12.98 |

10个父本两两Hamming距离：最小33、平均85.04、最大144。除种子外，每个新父本加入档案时与已有父本的最近距离依次为：60、51、41、33、40、40、36、40、40。

每个父本均满足当前历史合法规则：存在铜、无悬浮铁、无小于4格的铜岛。需要注意，这里的“合法”严格指现有两项约束，不等于已经通过FEMM网格或电磁性能验证；部分高差异父本只有2个设计域铁格，后续FEMM小试必须观察其网格和性能。

## 持久化验证

- SQLite `integrity_check=ok`；
- 每个proposal保存raw、repair动作、修复结果、特征、距离和谱系；
- 每个proposal后保存完整NumPy RNG状态；
- 相同seed的连续运行与中断恢复逐proposal一致；
- 548个proposal全部重放确定性修复，结果与数据库一致；
- 恢复到已提交的10个父本不会产生新proposal；
- 只读状态命令不修改数据库。

达到60个父本后，程序自动执行12个容量为5的拓扑簇划分，并无重复地分配到6个campaign，每个campaign恰好10个父本。

## 停止点

当前停在10父本快速验证点。继续命令只完成60父本纯拓扑档案和campaign分配，不会自动启动FEMM或1000样本生产。
