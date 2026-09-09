# Python复现与FEMM加固实施方案（审计后草案）

本文件只描述下一阶段拟新增内容；本轮没有创建这些代码文件，也没有运行FEMM。

## 1. 拟建项目

```text
python_topopt/
├── pyproject.toml
├── README.md
├── configs/
│   ├── historical_compat.yaml
│   ├── smoke_1case.yaml
│   └── stability_2workers.yaml
├── src/motor_topopt/
│   ├── ga/
│   │   ├── clonalg.py
│   │   ├── operators.py
│   │   ├── state.py
│   │   └── checkpoint.py
│   ├── encoding/
│   │   ├── chromosome.py
│   │   ├── design_domain.py
│   │   ├── connectivity.py
│   │   └── turns.py
│   ├── femm_runner/
│   │   ├── client.py
│   │   ├── worker.py
│   │   ├── supervisor.py
│   │   ├── workspace.py
│   │   ├── validation.py
│   │   └── errors.py
│   ├── geometry/
│   │   ├── historical_inset.py
│   │   └── v5_merged_copper.py
│   ├── objectives/
│   │   ├── torque_scan.py
│   │   └── objective.py
│   ├── data/
│   │   ├── mat_reader.py
│   │   ├── cache.py
│   │   └── records.py
│   └── config/
│       └── schema.py
├── scripts/
│   ├── audit_fixture.py
│   ├── evaluate_one.py
│   ├── run_ga.py
│   └── run_stability.py
├── tests/
│   ├── unit/
│   ├── regression/
│   └── integration/
├── fixtures/                 # 只保存小型清单/哈希，原始文件仍从FP只读引用
└── outputs/                  # 运行时生成，不写入FP
```

## 2. 分阶段实施

### 阶段A：纯Python一致性基础

- 配置schema和路径解析。
- 三值染色体、MATLAB列优先映射、设计域坐标。
- 匝数分配、悬浮铁/小铜岛检测。
- 自定义CLONALG完整状态转移，固定随机种子。
- evaluator协议与FakeEvaluator。
- checkpoint、JSONL/SQLite结果记录和内容哈希缓存。
- 用历史MAT验证shape、索引、统计和算法状态，不调用FEMM。

通过条件：所有纯Python单元测试通过；同一seed可重复得到相同历史/变异/选择轨迹。

### 阶段B：FEMM单例执行器

- 在Windows上通过独立worker进程连接 `femm.ActiveFEMM`。
- 实现健康检查、命令日志、PID归属、超时和有限重试。
- 实现唯一run/case目录、原子结果文件和失败包。
- 先实现历史inset几何和6角度扫描。
- 记录 topology/model/mesh/solve/postprocess/total 六类耗时。

通过条件：一个固定模板、一个已知染色体的几何/材料/电路/转矩曲线均合理；重复运行在设定容差内一致；无残留的“本程序启动”FEMM进程。

### 阶段C：历史回归

- 对 `seed_bits_fine.mat` 和最新best做解码结构比较。
- 比较材料计数、坐标、标签、电路、匝数和FEM结构摘要。
- 重新评估若干历史best，与文件名/`bestOverall`比较J。
- 明确FEMM数值容差和无法验证项，不宣称MATLAB完全一致。

### 阶段D：小规模算法测试

严格按用户指定顺序：

1. 一个固定模型；
2. 一个基因拓扑；
3. 五个手工个体；
4. 20个体完整CLONALG小测试；
5. 50个体连续稳定性测试；
6. 两个独立FEMM进程测试；
7. 通过后才讨论1000+样本。

### 阶段E：改进几何与性能

- 以独立配置实现V5合并铜区，和历史模式分别回归。
- 分析模板/固定几何复用；不在未测量前重构求解流程。
- 缓存重复染色体，预计算设计域坐标/邻接/扇区映射。
- 去除批量运行中的PNG和磁密导出；仅成功best/抽样/失败时保留。
- 根据阶段耗时决定是否复用已打开模型或每case重开干净模板。
- 2-worker稳定后才允许更高并发，并设置显式上限。

## 3. FEMM故障状态机

建议每个case至少区分：

```text
PENDING
  -> PRECHECK_REJECTED
  -> BUILD_FAILED
  -> GEOMETRY_INVALID
  -> MESH_FAILED
  -> SOLVE_TIMEOUT
  -> COM_DISCONNECTED
  -> RESULT_EMPTY
  -> RESULT_NONFINITE
  -> RESULT_OUTLIER
  -> RETRYING
  -> SUCCEEDED
  -> FAILED_FINAL
```

重试只用于可能瞬态的问题（COM断开、FEMM崩溃、求解超时一次等）；确定性无效拓扑不重复浪费计算。每次重试先结束该worker拥有的FEMM实例，再创建新COM实例和新case目录。绝不按进程名全局杀死用户手工打开的FEMM。

## 4. 默认记录字段

- run ID、case ID、generation、population index、parent/clone来源；
- 原始染色体、规范化拓扑哈希；
- 模板SHA-256、配置SHA-256、代码版本；
- J、T_avg、T_ripple、penalty、角度数组、转矩数组；
- 悬浮铁/铜统计、材料计数、每相匝数；
- 拓扑生成、FEMM建模、网格、各角度求解、后处理、总耗时；
- worker ID、FEMM PID、尝试次数、最终状态、错误类型与消息；
- `.fem/.ans/log` 路径及是否保留。

## 5. 依赖策略

首选最小依赖：

- Python 3.11+；
- NumPy、SciPy、h5py、PyYAML；
- pydantic用于配置/记录校验；
- pywin32用于Windows COM；
- pytest；
- psutil用于受控进程监督；
- 可选pandas仅用于报告，不进入算法核心。

不建议第一阶段引入DEAP/pymoo来代替自定义CLONALG。若后续需要对照标准GA，可以作为第二个明确算法插件，而不是改变兼容模式。

## 6. 开始编码前需要用户确认的两个决策

1. **兼容基准：** 建议先以3月 `clonalg_N_float + historical_inset` 为必须复现的基准，V5作为后续独立模式。
2. **平均转矩阈值：** 建议历史兼容配置固定0.8；其他0.75/0.6作为单独配置，不从文件名猜测。

