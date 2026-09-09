# 电机拓扑优化 MATLAB→Python 项目审计报告

审计日期：2026-08-06  
审计范围：`C:\Users\26096\Desktop\em`，重点为 `FP\CLONALG` 及其历史输出目录。  
本轮边界：只读审计与实施设计；未启动 FEMM、未改写算法、未修改任何原始 MATLAB/FEM/MAT 文件。

## 1. 结论摘要

1. 当前电机优化算法不是 MATLAB Genetic Algorithm Toolbox 的 `ga`/`gamultiobj`，而是自定义 CLONALG（克隆选择/免疫优化）。全项目未发现有效的 `ga`、`gamultiobj`、`optimoptions`、`selection*`、`crossover*` 或 `mutation*` Toolbox 调用。
2. 能与最新历史结果对应的生产主入口是 `FP/CLONALG/clonalg_N_float.m`。它调用三值染色体、克隆、按排名超变异、父代/最佳克隆择优、随机 repertoire shift 和强制精英保留；没有交叉算子。
3. 历史生产链使用 `femm_apply_design_bits_rep6_inset.m`：每个铜格独立内缩并保留空气带。2026年4月新增的 V5 几何生成器会合并相邻铜格、合并标签和匝数，但只接在串行测试入口 `eval_design_femm.m` 上，尚未接回最新批量优化主入口。这是行为差异，不能直接视为同一算法的“代码整理”。
4. 染色体为180个离散三值基因：`0=Air`、`1=Pure Iron`、`2=Copper`。网格为18个径向单元×10个角向单元，对应0°～15°设计扇区；该扇区按交替镜像方式复制6次，构成0°～90°四分之一模型。
5. 目标为单目标最小化：转矩脉动率加平均转矩不足惩罚。生产配置还在进 FEMM 前硬拒绝悬浮铁和小铜岛。
6. 最新一次所谓“1000gen”结果实际上仍是一次100代续跑；文件名表示累计续跑标签，不是单次1000代。最新结果从第1代到第100代最优值不变。
7. 已有批处理具备“每 worker 私有目录、UUID case 文件、进程内染色体缓存”等良好雏形，但没有超时、重试、COM重连、FEMM进程重启、失败分类、磁盘缓存、checkpoint或阶段耗时。当前直接开10个FEMM实例，不适合未经稳定性验证的大规模运行。
8. 本机已安装 FEMM 4.2（`C:\femm42\bin\femm.exe`），但当前 Python 环境未安装 `femm`、`pywin32` 或 `comtypes`。项目中没有 Lua 文件，也没有现成 Python FEMM 执行器。

## 2. 文件盘点

整个工作区发现：

| 类型 | 数量 | 说明 |
|---|---:|---|
| `.m` | 92 | 65份不同内容；含主链、旧版、实验版、通用测试函数和重复副本 |
| `.mat` | 196 | 优化结果、种子、完整代际轨迹；1个既有审计已判为损坏 |
| `.fem`/`.FEM` | 234 | 模板、历史最优、临时/worker模型和论文模型 |
| `.ans` | 203 | FEMM求解结果 |
| `.lua` | 0 | 当前项目没有Lua自动化层 |
| `.py` | 25 | 数据审计与CNN；没有FEMM自动化或优化算法 |
| YAML | 4 | 现有审计/CNN配置，不是拓扑优化运行配置 |
| requirements 文件 | 3 | 仅服务现有审计/CNN项目 |

`FP/CLONALG` 外的多个 `CLONALG/clonalg.m` 是2018年的通用函数优化副本，不在电机生产调用链。`ackleyfcn.m`、`rastriginfcn.m` 等基准函数也是遗留示例，没有被电机入口调用。

## 3. 实际入口与版本分支

### 3.1 历史生产主入口

**入口：** `FP/CLONALG/clonalg_N_float.m`

判定依据：

- 文件于2026-03-06修改，并明确加载上一阶段 `best_result_0305_groupF_900gen.mat` 作为种子（第121～123行）。
- 输出目录指向 `IA_test_Float`（第465行）。
- `FP/IA_test_Float/best_result_20260306_174158.mat` 与该脚本的 `N=10`、`gen=100`、`beta=0.4`、`d=0.2`、180基因、悬浮岛配置完全吻合。
- 最新结果被复制为 `FP/CLONALG/best_result_0306_groupF_1000gen.mat`；其 `J_hist` 仍只有100行。

### 3.2 4月几何实验/测试入口

**入口：** `FP/CLONALG/test_bits.m`（也有 `test.m`）  
**评估：** `FP/CLONALG/eval_design_femm.m`  
**几何：** `femm_apply_design_bits_rep6_inset_merge_cu_island_inset_v5_corner_tri.m`

这条链是后续几何开发分支，不是3月历史结果的生成链。`test_bits.m` 还使用 `T_min=0.5` 和 `thetaPeriodic=true`，与生产分支的 `T_min=0.8`、`thetaPeriodic=false` 不一致，因此它目前不是可靠的生产回归入口。

### 3.3 其他历史入口

- `clonalg.m`：小规模 Group4 分支，当前文件参数为 `N=4`、`gen=3`。
- `clonalg_N.m`：GroupN 续跑分支，当前文件参数为 `N=10`、`gen=10`。
- `clonalgFine.m`：较早的粗到细尝试，当前参数 `N=5`、`gen=5`，含旧用户绝对路径。
- `femmfunc*.m`、`strukturFemm*.m`：参数化建立基础电机/模板的历史脚本。模板的最终保存步骤和 `blank_18x10.FEM` 的唯一生成来源无法从当前代码闭环确认，可能包含手工FEMM保存/编辑。

## 4. 历史生产完整调用链

```text
clonalg_N_float.m
  ├─ stator_design_domain_blank(cfg)
  ├─ prepare_design_domain(domain0, theta0)
  ├─ load(best_result_0305_groupF_900gen.mat)
  ├─ mutate_trit(...)                      # 初始化与移民
  ├─ femm_worker_init(ctx)                 # 每个 MATLAB worker 一个 FEMM
  │    ├─ get_worker_id()
  │    ├─ copyfile(blank_18x10.fem, worker私有模板)
  │    ├─ openfemm / hidefemm
  │    └─ opendocument(worker私有模板)
  ├─ eval_bits_batch_cached(...)
  │    ├─ 批内染色体去重
  │    ├─ containers.Map 进程内缓存
  │    └─ parfeval(f_wrapper)
  │         └─ eval_design_femm_float(bits, ctx, worker_state)
  │              ├─ detect_floating_iron(...)
  │              ├─ detect_floating_copper_small_islands(...)
  │              ├─ compute_turns_per_cell(...)
  │              │    └─ decode_material_bits(...)
  │              ├─ 重新打开worker私有干净模板
  │              ├─ femm_apply_design_bits_rep6_inset(...)
  │              │    ├─ 将0/1/2映射为空气/铁/铜
  │              │    ├─ 复制并交替镜像6个15°扇区
  │              │    ├─ 合并空气/铁大区域边界
  │              │    └─ 为每个铜格建立独立内缩铜区
  │              ├─ 保存UUID命名的case_*.fem
  │              ├─ scantorque(3.5)
  │              │    ├─ 角度 0,3,6,9,12,15°
  │              │    ├─ 修改滑动气隙边界Inner Angle
  │              │    ├─ 更新六个正负相电路电流
  │              │    ├─ 每个角度 mi_analyze / mi_loadsolution
  │              │    └─ mo_blockintegral(22) 得到转矩
  │              ├─ 计算平均转矩、脉动率和J
  │              └─ 删除成功case的.fem/.ans
  ├─ CLONALG排序、克隆、超变异、择优、移民、精英保留
  ├─ 保存 best_result_*.mat 与 trace_all_*.mat
  └─ 再生成一次 best_*.fem/.ans/.png
```

## 5. 染色体、坐标和材料语义

### 5.1 染色体

- 长度：`Nd=nr×nt=18×10=180`。
- 类型：离散整数/三值基因，不是二进制，也不是连续参数。
- 取值：`{0,1,2}`。
- 含义：`0=Air`、`1=Pure Iron`、`2=Copper`，见 `decode_material_bits.m` 第1～12行及 `clonalg_N_float.m` 第73～76行。
- 代码注释中 `eval_design_femm.m` 的“2*Nd、2bit/单元”已过期，与实际实现冲突。

### 5.2 MATLAB列优先顺序

`stator_design_domain_blank.m` 先建立形状 `[nt,nr]=[10,18]` 的掩膜，`find` 按MATLAB列优先产生基因顺序。反解采用：

```matlab
B = reshape(bits(:), [nt, nr]).';   % 得到 [nr,nt]
```

因此Python必须使用等价列优先处理，不能直接做默认C-order的 `reshape(18,10)`。建议的Python等价式是 `bits.reshape((10,18), order="F").T`，并通过已知180位种子做回归测试。

### 5.3 几何范围和复制

- 设计域：半径31.5～52 mm，角度0°～15°。
- 径向18格、角向10格。
- 6个扇区偏移：0、15、30、45、60、75°。
- 偶数扇区使用镜像，组成0°～90°模型。
- 基础模板是2D planar、静磁、深度36 mm、精度 `1e-8`；固定转子、永磁体、气隙和边界来自 `blank_18x10.FEM`。
- 模板材料定义中名称有重复（Air三份、N40两份）；按名字设置材料是隐式假设，Python实现应在启动时验证材料/电路/边界唯一性与可访问性。

### 5.4 电路和匝数

- 电路名按代码顺序：`A+、A-、B+、B-、C+、C-`。
- 六扇区电路编号：`[4,3,6,5,2,1]`，即依次为 `B-、B+、C-、C+、A-、A+`。
- 每个电路总匝数固定为100。
- `compute_turns_per_cell.m` 统计每相铜格数，并令每铜格匝数为 `100/该相铜格数`；没有铜时返回全0并令 `J=1e6`。

## 6. 实际优化算法

### 6.1 算法类型

单目标最小化的自定义 CLONALG，不是标准遗传算法：

- 有排序、复制、超变异、重选择、随机移民和精英保留。
- **没有交叉。**
- 没有MATLAB GA Toolbox选择器或变异器。
- 不应改用DEAP/pymoo默认GA，否则行为不一致。Python应先逐条复现自定义状态转移。

### 6.2 生产默认参数

来源：`clonalg_N_float.m` 和最新 `trace_all_20260306_174158.mat`。

| 参数 | 值 | 行为 |
|---|---:|---|
| `N` | 10 | 种群大小 |
| `gen` | 100 | 每次续跑代数 |
| `beta` | 0.4 | 最大克隆数 `beta*N=4` |
| `d` | 0.2 | 每代随机重置2个非精英位置 |
| `ns` | 8 | 只克隆排名前 `N-2` 个父代 |
| 克隆数 | `[4,4,3,3,2,2,1,1]` | 每代最多20个克隆候选 |
| 超变异率 | 0.03～0.20 | 随父代排名按平方曲线增加，`gamma=2` |
| 移民变异率 | 0.25 | 50%概率围绕全局最优强变异；否则全随机 |
| 精英 | 1 | 全局最优强制放到下一代第1行 |
| 终止 | 固定100代 | 无早停、容差或预算终止 |

### 6.3 初始化

以900gen历史最优为种子：

- 1个原始种子；
- 5个以0.05逐基因变异概率生成的近邻；
- 2个以0.12生成的中等扰动；
- 1个以0.25生成的远扰动；
- 1个全随机三值个体。

三值变异被触发时，旧值加随机 `1` 或 `2` 后模3，因此一定跳到另一个材料。

### 6.4 选择、克隆与保留

1. 对当前10个体按J升序排序。
2. 前8名按排名生成20个克隆。
3. 每一簇使用对应父代排名的逐基因变异率。
4. 每簇只保留“父代”和“本簇最佳克隆”中的较优者。
5. 后2名先原样继承。
6. 在非精英位置随机选择2个位置进行强变异或全随机替换。
7. 将运行以来全局最优强制写回第1行。

### 6.5 随机性、历史与缓存

- 当前代码使用 `rng('shuffle')`，不是固定种子，无法复现实验随机流。
- `rng_hist(it)=rng` 虽被赋值，但最新分支没有把它保存进结果文件。
- `Ab_hist`、`ind_hist`、`J_hist_all`、每代最佳和悬浮岛统计会保存。
- 染色体按字符串键在一次运行内缓存；批内也去重，但缓存不持久化且只保存J。
- 没有中间checkpoint；程序中断后只能从人工保存的上一轮最终best继续，不能从某一代完整恢复。

## 7. 目标函数、约束和转矩计算

### 7.1 电磁指标

`scantorque.m` 的实际角度是 `0:3:15`，共6点；注释里的“每步1°、0～90°”与代码不一致。

每个机械角度 `th`：

```text
theta_e = 4*th + 90°
Ia = 3.5*cos(theta_e)
Ib = 3.5*cos(theta_e-120°)
Ic = 3.5*cos(theta_e-240°)
```

4是极对数，对应8极模型。正负电路分别写入正负电流。转矩取转子组 `group=1` 上的 `mo_blockintegral(22)`。

```text
T_avg    = mean(T)
T_ripple = (max(T)-min(T)) / max(abs(T_avg), 1e-6)
```

`T_ripple` 是无量纲相对峰峰值，不是Nm；`test_bits.m` 的打印单位标注有误。

### 7.2 平均转矩约束与基础目标

生产配置：`T_min=0.8 Nm`，`penaltyCoef=10`。

```text
penalty = 10*(0.8-T_avg)/0.8, 当 T_avg<0.8
penalty = 0,                    当 T_avg>=0.8
J_em = T_ripple + penalty
```

这是单目标，不是多目标Pareto优化。`pareto.m` 中的少量手工T值只用于绘图，未进入优化主链。

### 7.3 几何快速拒绝

在生产Float分支中，进FEMM前执行：

- 悬浮铁：铁必须通过4邻接连到最外径向一行，否则其单元计为悬浮铁。
- 小铜岛：4邻接铜连通分量面积小于4格时，该分量的所有格计为悬浮铜。
- `thetaPeriodic=false`，0°与15°两侧不在检测中连通。

若任一计数非零：

```text
J = 1e6 + 50*nFloatCu + 50*nFloatFe
T_avg = 0
T_ripple = 0
```

配置中的 `fastReject.AminCu` 实际未被读取；代码读取的是 `ctx.Amin`。两者目前都为4，但这是潜在配置分叉。

### 7.4 Sigmoid悬浮岛惩罚

代码还定义：

```text
pFe = 0.10*sigmoid(50*(nFloatFe-0.5))
pCu = 0.05*sigmoid(30*(nFloatCu-0.5))
J = J_em + pFe + pCu
```

但快速拒绝已经对任何 `nFloat>0` 提前返回，所以正计数时该软惩罚不可达；无悬浮时仍会产生极小的非零偏移。Python一致性模式必须保留这一现状，改进模式再决定是否删除冗余逻辑。

## 8. FEMM模板与执行方式

### 8.1 模板事实

`blank_18x10.FEM`：

- FEMM格式4.0，静磁 `Frequency=0`；
- planar、millimeters、Depth=36；
- Precision=`1e-8`，MinAngle=15；
- 1268节点、1212线段、1146圆弧、12个初始block label；
- 26个边界属性、7个材料定义、6个电路；
- 材料包括 Air、Pure Iron、Copper、N40；
- 电路包括 A+/B+/C+/A-/B-/C-；
- 几何包围盒 `[0,0,74,74]`，即四分之一模型。

### 8.2 生产批处理已有措施

- 每个MATLAB worker复制一份基础模板到 `%TEMP%\femm_topopt_workers\wNN`。
- 每个worker只启动一个FEMM并跨个体复用。
- 每次个体评估重新打开干净的worker模板，避免动态几何累积。
- case使用Java UUID文件名，避免同一目录文件名碰撞。
- 正常完成时删除case `.fem/.ans`。
- 一次运行内缓存重复染色体。

这些设计思想可以保留，但需要用Python进程隔离和更完整的监督器实现。

### 8.3 当前缺失的稳定性能力

- 无单个体或单角度超时；`fetchNext` 可永久阻塞。
- 无有上限重试。
- 无COM健康检查、断线重连或FEMM进程重启。
- 无残留进程PID登记与只清理“自己启动进程”的机制。
- 无网格失败、材料缺失、区域不闭合、空结果、NaN/Inf、异常转矩的错误分类。
- 异常时没有 `finally`，失败case可能残留；成功case反而被删除，不利于回溯。
- 无失败个体模型/基因/日志包。
- 无原子化结果写入。
- 缓存只在内存中，重启丢失，且只存J而不存转矩曲线、版本、配置或耗时。
- 固定开启10个并行worker/10个FEMM实例，没有先做2-worker稳定性验证。
- worker目录名只基于worker ID，多次运行会复用目录；没有run ID层级。
- 路径依赖当前工作目录且主脚本含其他用户的绝对输出路径。

## 9. 历史数据可用于哪些回归测试

### 9.1 强基准

- `seed_bits_fine.mat`：180位已知三值种子，可验证MAT读取、列优先reshape和编码。
- `best_result_20260306_174158.mat`：最新历史best、J历史、ctx/cfg。
- `trace_all_20260306_174158.mat`：完整100代当前种群、排序索引、J和悬浮岛统计。
- `best_J0.0924955_20260306_174158.fem/.ans`：最新历史最优模型及最后一次求解结果。
- 大量 `best_result_*.mat` + `best_J*.fem` 配对，可做解码后FEM结构和J回归。

最新trace确认：

- `N=10`、`gen=100`、`beta=0.4`、`d=0.2`；
- 1000个“代际当前种群位置”中有451个不同染色体；
- 900/1000个J为快速拒绝值 `>=1e6`；
- 最佳 `J=0.09249546785567309`，`genIdx=1`，100代最优值不变；
- “1000gen”不是该MAT内部的1000代。

### 9.2 MAT读取注意事项

- 普通best MAT可用SciPy读取。
- `trace_all_*.mat` 是MATLAB v7.3/HDF5；HDF5看到的 `Ab_hist` 形状为 `(180,10,100)`，MATLAB逻辑形状为 `(100,10,180)`，Python必须显式转置。
- 不能把MATLAB行向量/列向量 squeeze 后的结果直接假定成固定方向。

### 9.3 当前不能证明的内容

历史best MAT没有保存 `T_avg`、`T_ripple` 或6点转矩数组。最终 `.ans` 只对应最后一个角度状态，不能恢复完整转矩曲线。因此在没有MATLAB的情况下：

- 可以验证Python对同一染色体生成的拓扑、材料、匝数、FEM文本/几何摘要以及最终J是否接近历史值；
- 可以依据源代码单元测试平均值和脉动公式；
- 不能声称MATLAB/Python逐步数值“完全一致”；
- 若FEMM版本、网格或几何生成顺序变化，J可能出现数值差异，需要设定可解释容差。

## 10. 可保留与必须移植的部分

### 10.1 可保留

- 原始 `.m/.mat/.fem/.ans` 全部作为只读参考和回归fixture。
- `blank_18x10.FEM` 作为第一阶段历史一致性模板。
- 现有 `motor_dataset_audit` 的FEM/MAT只读解析器和清单数据。
- `cnn_structure_validation/fem_raster.py` 可用于生成结构预览和拓扑哈希辅助检查，但不能替代FEMM物理求解。
- 现有“worker私有模板、唯一case名、染色体去重”的设计原则。

### 10.2 必须移植

- 设计域建立和MATLAB列优先索引映射。
- 三值编码、解码、三值变异。
- CLONALG初始化、排名克隆、超变异、父/克隆择优、移民和精英逻辑。
- 悬浮铁/小铜岛检测与历史惩罚语义。
- 电路/匝数分配。
- 历史 `inset` 拓扑生成器；之后再单独实现V5。
- FEMM COM命令层、模型生命周期、角度扫描、转矩提取和目标函数。
- MAT历史数据加载、运行记录、磁盘缓存、checkpoint和恢复。

## 11. 已确认问题和隐式假设

### 高优先级

1. `eval_design_femm_float.m` 的首函数声明仍叫 `eval_design_femm_worker`，与文件名和调用名不一致。
2. 生产主入口与4月V5几何分支没有集成；旧/新铜区语义不同。
3. `rng('shuffle')` 与“可复现”冲突，且随机状态未保存到最终trace。
4. 无checkpoint、超时、重试、重启、错误分类和失败保存。
5. `clonalg_N_float.m` 输出路径硬编码为其他用户 `C:\Users\Alexander\...`；其他分支还使用 `Administrator`/`lenovo`。
6. 主入口依赖运行时当前目录找到模板、种子和FEMM mfiles；没有统一路径解析。
7. `femm_worker_init.m` 第35行的 `mi_setsegmentprop('', 0, hseg, 0, groupGrid)` 与FEMM包装函数参数顺序不符：`hseg=8` 被放入 `automesh` 位置，疑似网格设置错误。
8. 最新1000个代际个体位置中90%被快速拒绝，搜索效率很低；应在生成/变异阶段引入可配置修复或可行性采样，但历史一致性模式不能擅自改变。

### 中优先级

9. `fastReject.AminCu` 未使用，实际读取 `ctx.Amin`。
10. 软悬浮惩罚被硬拒绝遮蔽，配置含义不清。
11. `scantorque.m` 的角度/单位注释与实现不一致；6点是否足以代表完整周期尚无代码外证据。
12. 电流3.5 A、极对数4、相位90°、边界名 `Air gap`、属性索引10、转子组1均硬编码。
13. `make_seed_bits_from_reference` 中最后的 `seed_bits(rho < 2)=1` 会覆盖此前铜设置，使该未启用辅助函数几乎得到全铁种子。
14. 串行 `eval_design_femm.m` 创建临时模型但不清理、不关闭FEMM；若循环调用会积累文件和进程/COM状态。
15. 正常worker评估只保留失败前残留文件，成功文件全部删除；与“保存失败模型”需求相反。
16. 没有显式几何闭合、材料标签覆盖、非法重叠、薄桥、孤立像素、NaN/Inf或转矩范围检查。
17. `femmfunc_design.m` 首函数名仍为 `femmfunc`，与文件名不一致；模板生成链含手工步骤。

## 12. 目前无法从代码确认

1. 旧 `inset` 还是4月V5应当成为最终生产拓扑语义。
2. “悬浮铜”应指所有非主铜岛，还是仅面积小于4格的岛；项目同时存在两种检测函数。
3. 0～15°的6点扫描是否覆盖了该绕组/边界组合的完整转矩周期，是否需要包含端点两次计权修正。
4. 平均转矩阈值最终应为0.8、0.75、0.6还是按实验配置变化；历史数据中三者都出现过。
5. `blank_18x10.FEM` 是否经过脚本外手工修改，以及这些修改的工程依据。
6. V5几何是否已有可信FEMM输出基准；目前未找到与V5明确绑定的结果MAT。
7. FEMM网格设置的目标值及误差容差。
8. 是否必须与MATLAB随机数流逐位一致；建议只要求固定Python种子和算子统计/状态转换一致。

## 13. 审计后的实施原则

1. 首先实现“历史一致性模式”：旧inset几何、三值CLONALG、生产Float约束和6角度目标。
2. 将算法核心与FEMM evaluator完全解耦；纯Python单元测试不依赖FEMM。
3. FEMM仅在Windows执行进程中运行；每个worker独立进程、独立COM、独立工作目录。
4. 默认1 worker，稳定后只把2 worker作为受控实验；不直接复制当前10实例设置。
5. 所有运行参数进入版本化YAML，路径基于项目根解析，不使用用户绝对路径。
6. 每个评估返回结构化结果：J、T_avg、T_ripple、转矩曲线、阶段耗时、状态、错误类型、重试次数、文件路径和版本哈希。
7. checkpoint与缓存使用原子化写入；缓存键至少包含染色体、模板哈希、几何模式、FEMM/求解配置和目标函数版本。
8. V5作为独立 `geometry_mode=v5_merged_copper`，必须在历史一致性模式通过后再验证，不能静默替换。

