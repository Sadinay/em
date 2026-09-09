# MATLAB到Python函数映射草案

| MATLAB源 | 实际职责 | 拟Python模块 | 处理方式 |
|---|---|---|---|
| `clonalg_N_float.m` | 生产CLONALG入口与状态循环 | `ga/clonalg.py`, `scripts/run_ga.py` | 按行为复现，不使用库默认GA |
| 本地 `mutate_trit` | 三值逐基因变异 | `ga/operators.py` | 精确复现“跳到另外两个值之一” |
| 本地 `eval_bits_batch_cached` | 批内去重与运行内缓存 | `data/cache.py` | 扩展为持久缓存并带配置/模板哈希 |
| `stator_design_domain_blank.m` | 设计域边界和mask | `encoding/design_domain.py` | NumPy实现并测试坐标 |
| `prepare_design_domain.m` | MATLAB线性索引、中心/内缩点 | `encoding/design_domain.py` | 显式Fortran order |
| `decode_material_bits.m` | 0/1/2合法性和列向量化 | `encoding/chromosome.py` | 整数dtype和严格范围校验 |
| `compute_turns_per_cell.m` | 每相铜格统计和匝数分配 | `encoding/turns.py` | 纯函数 |
| `detect_floating_iron.m` | 外径锚定的4邻接铁连通 | `encoding/connectivity.py` | 纯函数，历史/改进模式可配置 |
| `detect_floating_copper_small_islands.m` | 小于Amin的4邻接铜岛 | `encoding/connectivity.py` | 纯函数，保存全部分量信息 |
| `femm_apply_design_bits_rep6_inset.m` | 历史生产拓扑写入FEMM | `geometry/historical_inset.py` | 第一优先级兼容实现 |
| `...v5_corner_tri.m` | 合并铜岛/空气带/角三角区 | `geometry/v5_merged_copper.py` | 第二阶段独立模式 |
| `femm_worker_init.m` | worker模板、FEMM COM初始化 | `femm_runner/worker.py` | 改为spawn独立进程、PID监督 |
| `femm_worker_cleanup.m` | FEMM关闭 | `femm_runner/supervisor.py` | finally、超时、仅清理自有进程 |
| `eval_design_femm_float.m` | 预检、建模、求解、目标 | evaluator组合层 | 拆分，不让FEMM耦合GA核心 |
| `scantorque.m` | 角度/电流循环和转矩积分 | `objectives/torque_scan.py` | 参数化角度、电流、极对数 |
| J计算代码 | 脉动率、平均转矩惩罚 | `objectives/objective.py` | 纯函数并保留历史兼容公式 |
| `seed_from_reference_model` | 从已解模型采样材料种子 | `encoding/reference_seed.py` | 后续实现；先修复覆盖错误 |
| `best_result_*.mat` | best与配置历史fixture | `data/mat_reader.py` | SciPy读取、严格shape处理 |
| `trace_all_*.mat` | v7.3完整代际轨迹 | `data/mat_reader.py` | h5py读取并显式转置 |
| `femmfunc_design.m`/`struktur*.m` | 基础电机/模板构建 | 暂保留为参考 | 第一阶段使用既有模板；模板重建另立里程碑 |

