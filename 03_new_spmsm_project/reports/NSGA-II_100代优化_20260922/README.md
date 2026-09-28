# NSGA-II 100 代结果摘要

## 主要结果

- 共生成 5,540 个不同候选，其中 601 个完成六角度 FEMM 验证。
- 最大平均转矩：4.1082 → 4.2593 N·m。
- 最小转矩波动：0.0802 → 0.0076 N·m。
- 最终 FEMM 真实帕累托前沿包含 30 个离散候选。

## 每 20 代检查点

| 代数 | 累计 FEMM | 真实前沿点 | 最大平均转矩 / N·m | 最小波动 / N·m |
|---:|---:|---:|---:|---:|
| 1 | 10 | 5 | 4.1082 | 0.0802 |
| 20 | 172 | 11 | 4.2501 | 0.0383 |
| 40 | 297 | 24 | 4.2548 | 0.0142 |
| 60 | 405 | 22 | 4.2593 | 0.0078 |
| 80 | 506 | 25 | 4.2593 | 0.0076 |
| 100 | 601 | 30 | 4.2593 | 0.0076 |

## 图表说明

前五代与每 20 代图可重建累计候选、当代 FEMM 样本和累计真实前沿。运行时未保存每代完整父代列表，因此不能把这些图解释为逐个体父子谱系。

## 图表索引

1. `01_early_generations_1_to_5.png`：前五代候选、FEMM 样本与真实前沿。
2. `02_milestones_every_20_generations.png`：第 1、20、40、60、80、100 代对比。
3. `03_convergence_metrics.png`：100 代性能收敛曲线。
4. `04_verified_pareto_front_evolution.png`：真实帕累托前沿推进。
5. `05_final_verified_pareto.png`：最终 601 个 FEMM 真值和 30 点前沿。
6. `06_representative_genes.png`：最终三个代表性基因。
7. `07_femm_sampling_and_error_by_generation.png`：每代 FEMM 数量及预测误差。
8. `08_early_representative_genes.png`：前五代折中代表结构。
9. `09_milestone_improvement_bars.png`：每 20 代关键指标。
10. `10_cnn_vs_femm_601_genes.png`：601 个基因的 CNN 预测与 FEMM 真值。

`data/` 保存本次搜索的候选、FEMM 标签、选点队列、逐代进度、运行配置和汇总指标；完整六角度 FEMM 模型仍保留在原 GA 运行目录。
