# 数据库与持久化结构

SQLite 文件是每个 run 的权威状态源：`runs/<run_id>/dataset.sqlite`。连接启用 `foreign_keys=ON`、`journal_mode=WAL`、`synchronous=FULL`。

| 表 | 作用 | 关键约束 |
|---|---|---|
| `runs` | 不可变配置哈希、代码/源清单、状态、参考样本与 `T_avg_ref` | `run_id` 主键 |
| `candidates` | 保存每次 CLONALG 出现，包括拒绝、重复、父子谱系、变异位置、ratio 与历史 J | `(run_id,generation,candidate_order)` 唯一 |
| `samples` | 保存唯一物理评价、180 位染色体、18×10 布局、六点汇总、ratio、J、band 与计时 | `(run_id,physical_key_hash)` 唯一 |
| `angle_evaluations` | 每角度状态、转矩、attempt、网格/求解时间、PID 与错误 | `(sample_id,angle_deg)` 主键 |
| `checkpoints` | checkpoint 文件哈希、完整 RNG 状态、种群哈希及代际位置 | 追加记录 |
| `events` | 结构化状态变更审计日志 | 追加记录 |

物理缓存键包含染色体哈希、完整配置哈希、几何模式和角度集合。相同染色体再次出现会创建新的 candidate 谱系记录，但复用既有 sample，不重复调用 FEMM。

每个角度完成后独立事务提交；样本只有在六个角度全部为有限数时才汇总。FEMM 失败、预检查拒绝或不完整角度不会写成 0 适应度，也不会进入 CNN 导出。

`run_config.json`、`run_state.json`、checkpoint 和样本结果使用临时文件 + fsync + 原子替换；关键状态保留 `.bak`。恢复时配置、输入文件和 checkpoint 哈希不一致会拒绝混入原 run。
