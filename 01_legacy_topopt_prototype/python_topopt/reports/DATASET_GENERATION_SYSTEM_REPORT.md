# 可断点恢复 CLONALG + FEMM 数据系统报告

## 结论

系统已完成 SQLite 权威状态、逐角度持久化、三层恢复、物理缓存、历史 J 保存、训练标签导出、单 worker FEMM 隔离、mock 恢复测试和真实六角度/中断/2 代 CLONALG 验证。没有修改 MATLAB 文件，没有交叉操作，没有并行 FEMM，也没有训练 CNN。

入口固定为 `strukturFemm.fem`（未分割参考）和 `seed_bits_fine.mat`（初始染色体）。CLONALG 仍最小化历史 `J`；CNN 未来使用连续 `torque_ratio=T_avg/T_avg_ref`，七个 band 只做覆盖统计。

## 主要新增结构

- `src/dataset_generation/`：配置/哈希、SQLite、状态、checkpoint、逐角度 orchestrator、CLONALG coordinator、mock/FEMM 后端、校验、导出和状态统计。
- `scripts/run_dataset_generation.py`：创建、恢复、参考求解与小规模运行。
- `scripts/show_run_status.py`：严格只读状态。
- `scripts/validate_run.py`：哈希、SQLite、六角度和标量重算校验。
- `scripts/export_dataset.py`：导出 JSONL、CSV、NPZ；包含 ratio 和历史 J。
- `tests/test_dataset_*`：持久化、缓存、恢复、导出及 FEMM job 边界测试。

本阶段实际新增或修改的文件：

- `configs/dataset_merged_v5.json`
- `src/dataset_generation/{__init__,backend,bands,checkpoint,clonalg_runner,config,database,export,femm_backend,hashing,orchestrator,reporting,state,validation}.py`
- `src/femm_runner/{process,worker}.py`
- `scripts/{run_dataset_generation,show_run_status,validate_run,export_dataset}.py`
- `tests/{test_dataset_foundation,test_dataset_orchestrator,test_dataset_clonalg_resume,test_dataset_tools_and_femm_backend}.py`
- `README.md`
- 本报告及 `DATABASE_SCHEMA.md`、`REFERENCE_TORQUE_VALIDATION.md`、`INTERRUPT_RECOVERY_VALIDATION.md`、`PILOT_DATASET_REPORT.md`

## 测试与真实结果

- 自动化测试：67 项通过。
- 参考六角度：`T_avg_ref=0.870740643725 N·m`。
- 4 个不同合法候选连续成功，24/24 候选角度成功。
- 真实中断：前三角度提交后由新进程从第四角度继续，无重算。
- 真实 2 代 CLONALG：40 候选，38 拒绝，1 首次合法，1 缓存重复。
- 两个真实 run 均通过完整校验；结束后无 FEMM/fkn 残留。

## 当前风险与停止点

1. 2 代中除种子外 39 个候选没有新增合法结构，尚不满足 20–50 唯一有效样本试运行门槛。
2. 六角度脉动率只是历史兼容定义，尚无密角度误差验证。
3. `strukturFemm` 参考与 merged-copper 候选几何表示不同，这是用户明确指定的入口；配置和哈希已完整记录，不能与其他 reference 定义混用。
4. 浮铁规则已知仍有 99 条历史漂移，不应反向修改 Python 规则来强行对齐。
5. 第二次 Ctrl+C 的等价清理路径已自动测试，但尚未人工按键强杀真实 FEMM。

当前不建议启动 1000 样本正式生产。应先由用户决定是否授权建立独立的 `improved_dataset_sampling` 模式。
