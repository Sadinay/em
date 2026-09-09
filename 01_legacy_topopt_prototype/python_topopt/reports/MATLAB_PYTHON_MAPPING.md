# MATLAB 到 Python 函数映射

| MATLAB来源 | Python实现 | 状态 |
|---|---|---|
| `CLONALG/clonalg_N_float.m:22-32` 染色体尺寸 | `src/encoding/chromosome.py` | 已实现、单元测试 |
| `reshape(bits,[nt,nr]).'` | `src/encoding/layout.py::matlab_vector_to_grid` | 已实现、往返测试 |
| MATLAB `N×180` 种群矩阵与v7.3 HDF5反序维度 | `src/encoding/layout.py`、`src/data_io/matlab.py` | 已实现、历史MAT测试 |
| `decode_material_bits.m` | `src/encoding/chromosome.py`、`src/topology/historical_inset.py` | 已实现 |
| `compute_turns_per_cell.m` | `src/encoding/turns.py` | 已实现、手工测试 |
| `detect_floating_iron.m` | `src/constraints/connectivity.py::detect_floating_iron` | 按现存源码实现；历史计数部分验证 |
| `detect_floating_copper_small_islands.m` | `src/constraints/connectivity.py::detect_small_copper_islands` | 已实现；历史1000/1000一致 |
| `femm_apply_design_bits_rep6_inset.m` 的材料/镜像/内缩逻辑 | `src/topology/historical_inset.py`、`src/femm_runner/historical_inset_commands.py` | 已生成真实FEMM命令；0°对照通过 |
| `scantorque.m:62-65` | `src/objectives/torque.py` | 标量公式已实现；历史转矩数组缺失 |
| `eval_design_femm_float.m:3-30,52-79` | `src/objectives/torque.py::evaluate_historical_objective` | 硬拒绝、无铜、惩罚和J已实现 |
| `clonalg_N_float.m:142-176` 初始化 | `src/clonalg/operators.py::initialize_population` | 已实现、可重复测试 |
| `clonalg_N_float.m:282-304` 排序和全局最优 | `src/clonalg/engine.py` | 已实现、历史排序测试 |
| `clonalg_N_float.m:315-373` 克隆与超变异 | `src/clonalg/config.py`、`operators.py` | 已实现；无交叉 |
| `clonalg_N_float.m:396-421` 父子择优 | `src/clonalg/operators.py::select_parent_or_best_clone` | 已实现、相同J选克隆测试 |
| `clonalg_N_float.m:423-443` 移民与精英 | `src/clonalg/operators.py::apply_random_immigrants` | 已实现、数量/精英测试 |
| `clonalg_N_float.m:993-1040` 缓存和批内去重 | `src/clonalg/cache.py` | 已实现、调用次数测试 |
| MATLAB未持久保存的完整续跑状态 | `src/clonalg/checkpoint.py`、`history.py` | 新增确定性checkpoint；连续/恢复一致 |
| `eval_design_femm_float.m` 的FEMM评价调用 | `src/evaluator/femm.py`、`src/femm_runner/worker.py` | 单进程隔离worker已实现；单角度真实验证通过 |
| `scantorque.m:18-50` 角度、电流、求解和转矩积分 | `src/femm_runner/worker.py::_scan_torque` | 0°真实验证通过；完整六角度待运行 |
| `femm_apply_design_bits_rep6_inset_merge_cu_island_inset_v5_corner_tri.m` | `src/femm_runner/merged_copper_v5_commands.py` | 独立模式已实现；0°真实A/B通过 |

`merged_copper_v5` 系列文件没有映射进兼容实现；后续只能作为独立实验模式加入。
