# 电机拓扑优化：纯Python历史兼容层

> 2026-09-09：本项目已精简为结果归档。主1000样本与补充300样本结果保留；旧ANS、生成FEM和运行日志已清理。数值见`runs/<运行名>/result_exports/`，逐样本记录与运行checkpoint见`run_details.zip`，SQLite仍在本机原位置。下文保留历史开发命令，不能把清理后的旧目录直接用于`--resume`；重新求解请新建任务。[清理记录](../../maintenance/cleanup_20260909/README.md)

本目录包含180位三值染色体、MATLAB排列转换、`historical_inset`拓扑、浮铁/小铜岛预检、匝数、标量目标函数，以及不含交叉的自定义CLONALG。除确定性mock外，现在还提供Windows单进程、隔离worker形式的FEMM评价器。

## 环境与测试

在Windows PowerShell中：

```powershell
cd C:\Users\26096\Desktop\em\python_topopt
python -m pip install -e ".[test]"
python -m pytest
```

当前环境依赖已经满足时，不需要重新安装即可执行测试。

需要真实FEMM接入时：

```powershell
python -m pip install -e ".[test,femm]"
```

## 运行mock CLONALG

```powershell
python scripts\run_mock_clonalg.py --generations 5 --random-seed 20260806
```

带checkpoint：

```powershell
python scripts\run_mock_clonalg.py --generations 20 --checkpoint outputs\mock_checkpoint.json
python scripts\run_mock_clonalg.py --resume --checkpoint outputs\mock_checkpoint.json
```

checkpoint包含完整NumPy随机状态、当前种群、全局最优、缓存和代际历史。相同Python seed与配置可重复；不声称复现历史MATLAB `rng('shuffle')` 序列。

## 重建900/1000统计

```powershell
python scripts\analyze_rejections.py
```

输出 `reports/rejection_analysis.json` 和 `reports/REJECTION_ANALYSIS.md`。

## 单个FEMM验证

对已经建好的FEM模型执行单角度探针：

```powershell
python scripts\probe_femm_fixed_model.py "C:\path\model.fem" --angles 0 --timeout 300
```

从历史MAT读取180位染色体、重建 `historical_inset` 并求解：

```powershell
python scripts\run_single_femm_evaluation.py --angles 0 --timeout 360
```

运行独立的优化几何模式：

```powershell
python scripts\run_single_femm_evaluation.py `
  --geometry-mode merged_copper_v5 `
  --angles 0 `
  --timeout 360
```

自动执行两个几何模式的同染色体A/B测试：

```powershell
python scripts\compare_femm_geometry_modes.py --angles 0 --timeout 360
```

完整历史六点可传入 `--angles 0 3 6 9 12 15`，但当前高分辨率模型单角度约184秒，应预留足够超时时间。每次任务使用唯一目录；超时会清理该任务启动的worker、FEMM和fkern进程。

## 关键文件

- `configs/historical_march.yaml`：兼容模式与全部显式配置。
- `reports/ALGORITHM_SPECIFICATION.md`：带MATLAB源文件和行号的算法规格。
- `reports/MATLAB_PYTHON_MAPPING.md`：逐函数迁移映射和验证状态。
- `reports/VALIDATION_MATRIX.md`：已验证、部分验证、无法验证清单。
- `reports/REJECTION_ANALYSIS.md`：拒绝原因、逐代统计和完整示例染色体。
- `reports/MOCK_RUN_REPORT.md`：5代纯Python小规模运行结果。
- `reports/FEMM_INTEGRATION_REPORT.md`：真实FEMM调用、单角度对照和超时清理结果。
- `reports/FEMM_OPTIMIZED_GEOMETRY_REPORT.md`：V5几何重建和同染色体性能A/B结果。
- `reports/NEXT_STAGE_SINGLE_PROCESS_FEMM_DESIGN.md`：下一阶段设计，不含FEMM实现。

## 当前边界

- 没有加入交叉操作或标准GA默认行为。
- FEMM执行层保持单进程；已提供逐角度恢复的历史CLONALG入口，以及读取60父本、运行6个独立campaign的改进采样入口。1000样本长任务尚未在本机完整跑完。
- `merged_copper_v5` 未并入历史兼容版。
- 六角度 `[0,3,6,9,12,15]` 只用于历史公式兼容，不能视为已经证明足以刻画真实转矩脉动。

## 可恢复训练集生成命令

当前生产配置使用 `strukturFemm.fem` 作为未分割参考模型，使用
`seed_bits_fine.mat` 作为初始 180 位染色体。历史 `J` 与连续标签
`torque_ratio` 会同时保存。

创建任务（只初始化，不自动启动长计算）：

```powershell
python scripts\run_dataset_generation.py `
  --new-run my_dataset_run `
  --geometry-mode merged_copper_v5 `
  --target-valid-samples 1000 `
  --config configs\dataset_merged_v5.json
```

先运行六角度参考：

```powershell
python scripts\run_dataset_generation.py `
  --resume runs\my_dataset_run `
  --backend femm `
  --run-reference `
  --reference-only
```

小规模 CLONALG 验证（示例为 2 代，不会被当成整个数据集已完成）：

```powershell
python scripts\run_dataset_generation.py `
  --resume runs\my_dataset_run `
  --backend femm `
  --start `
  --generations 2
```

只读状态、完整校验和导出：

```powershell
python scripts\show_run_status.py --run runs\my_dataset_run
python scripts\validate_run.py --run runs\my_dataset_run
python scripts\export_dataset.py --run runs\my_dataset_run --output exports\my_dataset_run
```

第一次 Ctrl+C 请求在当前角度完成后安全暂停；第二次 Ctrl+C 强制终止当前
worker，并只清理该 worker 记录的 FEMM 进程。已提交角度在恢复后不会重算。

## improved_dataset_sampling 父本档案

该模式与历史CLONALG完全分离。创建并先生成10个父本：

```powershell
python scripts\run_improved_parent_sampling.py `
  --new-run improved_parent_archive_20260807_001 `
  --config configs\improved_dataset_sampling.json `
  --stop-after-parents 10
```

从10个继续生成完整60父本并自动分配6个campaign：

```powershell
python scripts\run_improved_parent_sampling.py `
  --resume improved_runs\improved_parent_archive_20260807_001
```

可以随时按Ctrl+C；程序在当前毫秒级proposal事务结束后保存RNG、窗口和父本状态再退出。查看与验证：

```powershell
python scripts\show_improved_sampling_status.py `
  --run improved_runs\improved_parent_archive_20260807_001

python scripts\validate_improved_sampling.py `
  --run improved_runs\improved_parent_archive_20260807_001
```

父本导出位于run目录的 `exports/parent_archive.jsonl` 和
`exports/parent_archive.csv`。本入口目前只生成父本档案，不启动FEMM。

## 从60父本继续生成约1000个FEMM样本

这一节是正式长任务的完整命令清单。所有命令均在 Windows PowerShell 中执行：

```powershell
cd C:\Users\26096\Desktop\em\python_topopt
```

### 1. 先完成60个父本

当前父本任务已经有10个父本。继续运行到60个，并自动分配为6个campaign、每个10个父本：

```powershell
python scripts\run_improved_parent_sampling.py `
  --resume improved_runs\improved_parent_archive_20260807_001
```

确认父本数量、campaign分配和检查点均正确：

```powershell
python scripts\show_improved_sampling_status.py `
  --run improved_runs\improved_parent_archive_20260807_001

python scripts\validate_improved_sampling.py `
  --run improved_runs\improved_parent_archive_20260807_001
```

父本校验通过后，`exports/campaign_assignments.json` 应包含60条记录。父本阶段不启动FEMM。

### 2. 创建1000样本任务

下面的命令只创建目录、SQLite和检查点，不启动FEMM：

```powershell
python scripts\run_improved_dataset_campaigns.py `
  --new-run improved_dataset_1000_20260807_001 `
  --parent-run improved_runs\improved_parent_archive_20260807_001 `
  --physics-config configs\dataset_merged_v5.json `
  --campaign-config configs\improved_campaign_evolution.json
```

新任务位于 `runs/improved_dataset_1000_20260807_001`。60个父本属于最终1000个样本的一部分，因此正常情况下还需生成约940个合法、唯一的后代样本。

### 3. 单独计算参考模型六角度

建议先单独完成 `strukturFemm.fem` 的六角度参考值：

```powershell
python scripts\run_improved_dataset_campaigns.py `
  --resume runs\improved_dataset_1000_20260807_001 `
  --backend femm `
  --reference-only
```

如果参考计算被中断，原样再次执行该命令即可；已完成角度不会重算。

### 4. 先做10个后代的真实FEMM试运行

初始60个父本也需要进行六角度FEMM。下面的停止阈值是70个物理样本，即60个父本加约10个后代：

```powershell
python scripts\run_improved_dataset_campaigns.py `
  --resume runs\improved_dataset_1000_20260807_001 `
  --backend femm `
  --start `
  --stop-after-samples 70
```

这里的 `--stop-after-samples 70` 只是本次安全暂停点，不会把不可变总目标从1000改成70。

### 5. 继续运行到1000个物理样本

试运行检查正常后，去掉临时停止阈值：

```powershell
python scripts\run_improved_dataset_campaigns.py `
  --resume runs\improved_dataset_1000_20260807_001 `
  --backend femm `
  --workers 3 `
  --start
```

运行逻辑如下：

```text
60个合法父本
→ 6个campaign分别评价初代
→ 二维多尺度变异
→ 确定性修复浮铁和小铜岛
→ 全库哈希去重与Hamming距离准入
→ 合法唯一候选执行六角度FEMM
→ 保存T_avg、ripple、torque_ratio和历史J
→ 每个campaign保留低J精英及差异较大的个体
→ 继续生成，直到物理档案达到1000个唯一有效样本
```

全部完成FEMM的候选都会保留在物理档案中。某个个体即使没有进入下一代，也不会从数据库删除，因此初代、中间代、低性能、中等性能和精英结果都会留下。

这里不是只做很浅的随机扩充。正式配置为每个campaign每代生成5个后代，6个campaign合计每代约30次新FEMM评价；从60个父本扩充到1000个样本通常约经历32代。每次产生后代时，70%的父本选择按历史 `J` 排名施加优化压力，30%用于稀缺转矩带和多样性探索。每个campaign固定保留2个最低 `J` 精英，因此该campaign当前种群的最优 `J` 只能改善或持平；同时还强制至少完成30个完整代际后才允许因样本总数达标而停止。

这能保证程序确实持续进行“选择—变异—评价—精英保留”的优化过程，但不能在FEMM实际计算前保证一定达到某个未经历史代码确认的绝对 `J` 阈值。最终是否获得足够多高性能样本，应通过状态和导出数据中的逐代最优 `J`、band数量及精英改善曲线判断。

### 6. 三个候选并行计算

`--workers 3` 是候选级并行，不是角度级并行：主进程一次准备最多3个不同染色体，每个worker拥有独立FEMM实例、sample目录、`.fem`、`.ans`、结果文件和进程清单；每个候选内部仍按 `0°、3°、6°、9°、12°、15°` 顺序计算。

FEMM窗口默认在COM连接建立后立即最小化并按所属PID隐藏；每个worker还会在整个求解期间监控自己FEMM派生的 `fkn.exe`，新求解器窗口出现后立即执行隐藏。因此FEMM主界面和 `model - fkern` 求解窗口都不会出现在桌面或抢占键盘焦点。隐藏只影响界面，不影响求解和文件保存。如需调试几何，可临时在命令末尾增加 `--show-femm-windows`。

```text
候选A → FEMM实例1 → 六角度顺序求解
候选B → FEMM实例2 → 六角度顺序求解
候选C → FEMM实例3 → 六角度顺序求解
                    ↓
主进程逐角度写入SQLite和checkpoint
```

主进程负责候选生成、随机数、去重、SQLite写入和代际选择；worker不直接写数据库。FEMM COM启动阶段使用进程锁，以便每个worker只记录和清理自己拥有的FEMM PID，实际有限元求解阶段仍然并行。不要另外启动第二个主程序操作同一个run目录。

如果电脑负载或内存压力过大，可恢复为单实例：

```powershell
python scripts\run_improved_dataset_campaigns.py `
  --resume runs\improved_dataset_1000_20260807_001 `
  --backend femm `
  --workers 1 `
  --start
```

worker数量属于执行策略，不改变电流、角度、网格、目标函数或物理缓存键。但并发批次会在同一时刻提出多个候选，稀缺band反馈的读取时点与单进程略有不同，因此从某个checkpoint开始改用不同worker数量后，后续采样轨迹可能不同。每个新proposal会记录当时的 `execution_workers`，同一次可重复性试验应始终使用相同worker数量。

2026-08-07的本机实测中，3个候选共18个角度全部一次成功：没有COM断连、文件冲突、重试或残留FEMM进程。三组内部累计计算时间约599.0秒，实际墙钟301.5秒，并行重叠系数约1.99。由于FEMM主要占用单核但多个实例仍会竞争内存、缓存和其他CPU资源，3个worker不会得到严格3倍加速；当前电脑实测更接近约2倍吞吐提升。

### 7. 中断与恢复

- 第一次按 `Ctrl+C`：停止派发新角度；正在运行的最多3个角度完成后分别写入SQLite和checkpoint，再安全暂停。
- 再次执行第5节的同一条 `--resume --workers 3` 命令：每个候选从自己的未完成角度继续。
- 第二次按 `Ctrl+C`：强制终止当前worker；此前已提交的角度仍然保留。
- 不要同时启动两个主进程操作同一个run目录；并行只由一个主进程内部管理。

### 8. 随时查看状态、校验和导出

下面三条命令都不会启动FEMM：

```powershell
python scripts\show_run_status.py `
  --run runs\improved_dataset_1000_20260807_001

python scripts\validate_run.py `
  --run runs\improved_dataset_1000_20260807_001

python scripts\export_dataset.py `
  --run runs\improved_dataset_1000_20260807_001 `
  --output exports\improved_dataset_1000_20260807_001
```

状态命令适合频繁查看进度；完整校验会复算已保存的 `T_avg`、转矩脉动率、`torque_ratio` 和band，并检查SQLite、checkpoint及重复物理键。导出只包含已经完成六角度并分类的合法唯一物理样本，生成JSONL、CSV和NPZ。

### 9. 相关配置与记录

- `configs/improved_dataset_sampling.json`：60父本的生成、修复和聚类配置。
- `configs/improved_campaign_evolution.json`：campaign后代数量、精英数和Hamming准入配置。
- `configs/dataset_merged_v5.json`：FEMM模型、六角度、材料、目标函数和1000样本上限。
- `improved_campaign_manifest.json`：写入每个数据run，固定父本档案哈希、campaign配置哈希和物理配置哈希。
- `improved_campaign_status.json`：当前代、campaign、proposal数和样本进度。
- `dataset.sqlite` 的 `improved_proposals` 表：保存原始后代、修复后染色体、修复距离、Hamming距离、拒绝原因及对应candidate。

### FEMM封闭空气区域保护

`merged_copper_v5` 的Python执行层已修正MATLAB V5按“格子连通”合并空气带的边界问题。程序现在根据实际铜多边形计算空气带的真实几何连通区域，每个封闭区域严格放置一个Air标签，同时仍删除安全的内部边界。prepared模型记录 `geometry_builder_version`；代码修复后旧缓存会自动失效并重建。

单个候选在达到角度重试上限后会作为物理失败完整留档：初代中的失败父本不再参与繁殖，后代失败则继续生成替代候选，不计入有效样本目标。只有连续3个候选发生FEMM物理失败时才触发保护暂停，避免异常状态下无限消耗计算时间。

## 导师汇报版：从初始拓扑到1000个训练样本的完整流程

### 1. 项目目标与方法边界

本项目的目标是建立一个可断点恢复的“约束感知拓扑搜索 + FEMM物理评价”系统，在不依赖MATLAB许可证的条件下，从已有电机设计种子出发，生成约1000个合法、唯一、具有一定结构差异，并覆盖不同转矩性能水平的真实仿真样本。最终样本用于后续CNN代理模型训练。

当前方案不是把180位染色体直接作为一维信号交给CNN，也不是简单随机生成1000个拓扑。染色体先被解码为二维材料布局，经拓扑合法性检查和多样性准入后，由FEMM完成真实物理计算。搜索过程保留低目标函数 `J` 的优良拓扑，同时将已经完成FEMM的低、中、高性能结果全部存档，以兼顾优化能力和训练数据覆盖度。

本方案分成两个明确阶段：

1. 纯Python父本库生成：不启动FEMM，产生60个合法、唯一且差异较大的初始拓扑。
2. Campaign进化与FEMM评价：将60个父本分配到6条独立搜索线，逐代变异、评价和筛选，直到物理样本库达到1000个唯一有效样本。

这里采用的是面向数据集生成的改进式进化采样流程。它受原CLONALG的“变异—评价—父子择优—精英保留”思想启发，但不冒充历史MATLAB CLONALG的逐随机数复现版本；当前改进模式不使用交叉操作。

### 2. 染色体、设计域与材料编码

每个候选电机拓扑由180位离散染色体表示：

```text
chromosome.shape = (180,)
                 ↓ 按MATLAB兼容顺序转换
material_layout.shape = (10, 18)
```

每个基因对应设计域中的一个材料单元，允许取值为：

| 基因值 | 材料含义 |
|---:|---|
| 0 | Air，空气 |
| 1 | Iron，铁磁材料 |
| 2 | Copper，铜绕组区域 |

初始种子来自 `FP/CLONALG/seed_bits_fine.mat` 中的 `seed_bits`；未分割的参考电机来自 `FP/CLONALG/strukturFemm.fem`。候选拓扑则在 `blank_18x10.FEM` 的固定电机结构和固定设计域内重建。MATLAB矩阵与Python数组之间采用显式排列转换，避免因MATLAB列优先存储、转置或展平顺序不同而改变实际拓扑。

每个染色体在进入后续流程前都会校验长度、取值范围和哈希。哈希用于全库去重、FEMM结果缓存和断点恢复；相同物理配置下，同一个修复后染色体不会重复求解。

### 3. 第一阶段：建立60个合法且多样的初始父本

初始父本配置如下：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `parent_archive_target` | 60 | 初始合法父本总数 |
| `campaign_count` | 6 | 独立进化搜索线数量 |
| `parents_per_campaign` | 10 | 每条搜索线的初始种群大小 |
| `initial_parent_reuse` | false | 父本不在不同campaign之间重复 |
| `topology_cluster_target` | 12 | 目标拓扑簇数量 |
| `parents_per_cluster` | 5 | 每个拓扑簇保留的父本数 |

父本生成只执行快速的纯拓扑操作，不调用FEMM：

```text
seed_bits_fine初始种子
→ 二维多尺度变异
→ 确定性拓扑修复
→ 悬浮铁与小铜岛检查
→ 染色体哈希去重
→ 记录Hamming距离与拓扑特征
→ 聚类为12个拓扑簇
→ 每簇选5个，共60个父本
→ 分配给6个campaign，每个10个且不复用
```

变异在实际存储为 `18×10`（18径向×10角向）的二维设计域中进行，而不是把染色体当作无空间关系的一维串随机翻转。当前候选生成算子包括：

- 连通区域重绘 `connected_region_repaint`；
- 材料边界生长或收缩 `boundary_growth`；
- 局部矩形块替换 `rectangular_patch`。

父本生成时，小、中、大三种变化尺度分别覆盖1–4、5–15和16–40个单元，抽样概率分别为45%、35%和20%。确定性修复会删除悬浮铁和小于4格的铜岛；相同原始染色体和修复配置必须产生完全相同的修复结果。修复最多迭代8次，且修复造成的Hamming距离超过40时拒绝该候选。

### 4. 第二阶段：6个campaign的逐代搜索

`campaign` 可以理解为一条独立的进化搜索线。6个campaign各自从10个不同父本开始演化，既避免所有候选都围绕同一个种子形成，也降低单一路线过早收敛的风险。

一个完整全局代际的计算结构是：

```text
Campaign 1：当前种群 → 产生并完成5个合法唯一后代的FEMM评价
Campaign 2：当前种群 → 产生并完成5个合法唯一后代的FEMM评价
...
Campaign 6：当前种群 → 产生并完成5个合法唯一后代的FEMM评价
                         ↓
一个全局代际最多新增约 6 × 5 = 30 个有效物理样本
```

“5个后代”指最终通过准入并完成FEMM的数量。生成过程中遇到非法拓扑、重复染色体、距离不合格或FEMM终端失败时，该候选不会占用这5个有效名额，程序会继续提出替代候选。实际proposal数量因此通常大于最终物理样本数量。

从60个初始父本扩展到1000个样本，理论上还需要约940个后代：

```text
(1000 - 60) / 30 = 31.33
```

因此通常会进入第32个全局代际，并在总数达到1000后结束；程序同时要求至少完成30个完整代际。若有父本或后代发生不可恢复的FEMM失败，实际proposal数量和运行时间会增加。

### 5. 父本选择、变异和精英保留

每次需要产生后代时，父本选择由两个目标共同控制：

- 70%概率按历史目标函数 `J` 的排名选择，排名压力为2.0，使低 `J` 的高性能个体更容易产生后代；
- 30%概率用于稀缺转矩带和结构多样性探索，避免样本只集中在当前最优拓扑附近。

选中父本后执行小、中或大尺度二维变异，再进行确定性修复、合法性检查、全库去重和Hamming距离准入。正式进化阶段的距离条件为：

| 变异尺度 | 与全库最近样本的最小Hamming距离 | 与直接父本的最大Hamming距离 |
|---|---:|---:|
| small | 1 | 12 |
| medium | 3 | 30 |
| large | 6 | 60 |

每个campaign每代固定保留2个最低 `J` 精英，剩余种群位置按“优先扩大Hamming差异、相同条件下优先较低 `J`”的规则选择。因此：

- 每个campaign当前种群的历史最优 `J` 不会变差，只会改善或持平；
- 当前种群的平均性能长期看通常会提高，但可能因探索样本加入而波动；
- 每代新产生样本的平均性能不保证单调提高，因为系统有意保留一定探索能力；
- 完成FEMM但未进入下一代的样本仍然保留在物理档案中。

这一区分很重要：进化种群负责“寻找更优拓扑”，物理档案负责“积累完整训练数据”，二者不是同一个集合。

### 6. FEMM前的候选准入流程

候选在消耗FEMM时间前按下列顺序处理：

```text
父本选择
→ 二维多尺度变异，得到raw offspring
→ 确定性修复，得到repaired offspring
→ 180位取值与固定区域校验
→ 悬浮铁检查
→ 小铜岛检查
→ 修复距离检查
→ 全库染色体哈希去重
→ Hamming多样性准入
→ 分配唯一candidate ID和工作目录
→ 进入六角度FEMM队列
```

悬浮铁是指没有按历史规则连接到外径向锚定行的铁区域；小铜岛是小于4个四邻接单元的独立铜连通块。当前兼容规则会在FEMM前拒绝仍不合法的结构，因此不会把大量明显无效拓扑送入昂贵求解阶段。

谱系始终记录原始生成关系：

```text
parent → mutation → raw offspring → repair → repaired offspring → candidate
```

即使修复改变了染色体，后代仍指向产生原始候选的父本，便于以后按lineage划分训练集和测试集，避免近亲拓扑同时落入两边造成数据泄漏。

### 7. 六角度FEMM物理评价

每个通过准入的候选在机械角度

```text
[0°, 3°, 6°, 9°, 12°, 15°]
```

分别求解一次。该电机模型具有周期对称性，历史程序用15°机械区间代表一个性能周期，并以3°步长得到6个转矩点。每个角度会同步更新转子位置和三相电流相位，网格划分后调用FEMM求解，并通过转矩块积分提取该角度的电磁转矩。

单个候选的主要计算链为：

```text
载入固定基础模型
→ 按18×10（18径向×10角向）材料布局重建设计域
→ 合并相同材料的安全内部边界
→ 给每个真实封闭空气区域放置Air标签
→ 按历史逻辑分配铜区域匝数和相别
→ 对6个机械角度依次设置激励、划分网格、求解和后处理
→ 保存六点转矩和派生标量
```

运行耗时主要来自FEMM的网格和六次有限元求解。默认仍为1个worker；正式命令可通过 `--workers 3` 启用三个候选级独立FEMM实例。单个FEMM主要占用一个核心，三个实例可以提高总CPU利用率和吞吐量，但会竞争内存与CPU缓存，因此不会严格达到3倍加速。

### 8. 指标和历史目标函数 `J`

设六个机械角度得到的转矩为 `T₁, T₂, …, T₆`。

平均转矩为：

```text
T_avg = (T₁ + T₂ + T₃ + T₄ + T₅ + T₆) / 6
```

历史兼容转矩脉动率为：

```text
ripple = (max(T) - min(T)) / max(|T_avg|, 1×10⁻⁶)
```

当平均转矩低于 `0.8 N·m` 时施加不足惩罚：

```text
P_T = 10 × (0.8 - T_avg) / 0.8,    T_avg < 0.8
P_T = 0,                            T_avg ≥ 0.8
```

完整历史目标函数为：

```text
J = ripple
  + P_T
  + 0.10 × sigmoid[50 × (N_floating_iron - 0.5)]
  + 0.05 × sigmoid[30 × (N_small_copper - 0.5)]
```

其中 `sigmoid(x)=1/(1+exp(-x))`。当前流程在FEMM前已经硬拒绝未修复的悬浮铁和小铜岛，因此能够完成物理评价的合法样本中，这两项通常只剩接近零的历史兼容量。搜索方向为最小化 `J`：低脉动、高于最低平均转矩要求且拓扑合法的个体更优。若预检查直接拒绝，历史无效目标值以 `1,000,000` 为基础并叠加对应规则惩罚，不进入有效物理样本计数。

参考模型平均转矩固定记录为：

```text
T_avg_ref = 0.8707406437252049 N·m
torque_ratio = T_avg / T_avg_ref
```

`torque_ratio` 用于性能分层和后续CNN标签。当前分层为：

| Band | `torque_ratio` 区间 |
|---|---|
| B1 | `[0.0, 0.5)` |
| B2 | `[0.5, 0.6)` |
| B3 | `[0.6, 0.7)` |
| B4 | `[0.7, 0.8)` |
| B5 | `[0.8, 0.9)` |
| B6 | `[0.9, 1.0)` |
| B7 | `[1.0, +∞)` |

负平均转矩样本单独标记，不混入上述正转矩带。

### 9. 为什么既能优化，又能得到差、中、优样本

系统同时维护两个层次的数据：

| 数据集合 | 用途 | 是否会因未入选下一代而删除 |
|---|---|---|
| 当前进化种群 | 选择父本、继续搜索低 `J` 拓扑 | 会更新和替换 |
| 物理结果档案 | 保存全部合法唯一FEMM结果 | 不会删除 |

因此，即使某个后代性能较差、没有进入下一代，它已经产生的六点转矩、`T_avg`、`ripple`、`torque_ratio`、`J` 和完整谱系仍会保留。这使数据库能够包含：

- 初代父本；
- 中间代探索样本；
- 低性能和高脉动样本；
- 中等性能过渡样本；
- 每条campaign逐步形成的精英样本。

转矩带引导只是父本选择的软引导，不是FEMM前的硬配额，因为候选仿真完成前无法知道其真实 `torque_ratio`。达到1000个样本时，各band数量不保证恰好等于预设比例。正确做法是先完成物理档案，再统计分层分布；若某些band明显不足，再启动独立的补充采样，而不是删除已算出的有效结果。

### 10. 数据库、文件与可追溯性

每个候选使用唯一工作目录和文件名。当前默认保存内容包括：

- 原始染色体、修复后染色体和SHA-256哈希；
- 父本ID、campaign、代际、变异算子和变化尺度；
- 修复状态、修复次数、修复Hamming距离和拒绝原因；
- 与父本及全库最近样本的Hamming距离；
- 6个角度各自的状态、转矩、运行时间和重试次数；
- `T_avg`、`ripple`、`torque_ratio`、band和历史 `J`；
- FEMM模型 `.fem`、求解结果 `.ans`、日志和结构化JSON结果；
- 物理配置哈希、采样配置哈希和几何构建器版本。

`.ans` 是FEMM有限元求解后的完整场结果，包含网格、磁矢势以及后处理所需数据，因此明显大于只保存标量的JSON或SQLite记录。按照当前实测量级，完整保留FEMM中间文件约占18 MB/样本，1000个样本预计约18–25 GB；最终CNN训练实际只需导出的染色体、标签和必要拓扑特征，原始求解文件用于追溯和复核。

### 11. 中断恢复与失败保护

系统同时使用SQLite和checkpoint保存运行状态：

- 每完成一个角度就提交结果，不必等六个角度全部完成；
- checkpoint保存Python随机数生成器的完整状态、当前campaign、代际、候选和角度；
- 第一次按 `Ctrl+C` 时停止派发新角度，当前最多3个运行角度完成后安全暂停；
- 使用相同run目录执行 `--resume` 后，已完成角度直接跳过；
- 同一seed、配置和checkpoint恢复路径能够保持Python侧的确定性；
- 单角度最多自动尝试2次，超时上限为360秒；
- FEMM异常后可重启实例，并保存失败类型、模型、染色体和日志；
- 单个后代终端失败后继续生成替代候选；连续3个候选发生物理失败时保护性暂停。

这里保证的是Python随机状态和结构性算法行为可重复。由于历史MATLAB代码使用 `rng('shuffle')`，无法声称逐代复现当年的MATLAB随机序列。

### 12. 当前已解决的FEMM几何稳定性问题

早期 `merged_copper_v5` 逻辑只按照格子连接关系判断空气区域，并可能只给第一个子区域放置Air标签。当材料内部边界被合并后，一个看似连续的空气带可能在真实多边形几何中分裂成多个封闭区域，导致其中一部分没有材料标签，最终触发网格或求解失败。

当前Python实现已改为根据实际多边形连通性识别空气区域，并给每个真实封闭区域放置一个Air标签；同时只删除确认安全的相同材料内部边界。`geometry_builder_version` 会写入prepared模型缓存，几何构建算法更新后旧缓存自动失效，避免继续使用错误模型。

### 13. 预期结果、评价方法和科学局限

运行结束后，至少应从以下角度评价数据集质量，而不能只看样本总数：

- 1000个样本是否均为合法、唯一并完整具有六角度结果；
- 各campaign逐代最佳 `J` 是否改善或持平；
- `T_avg`、`ripple`、`torque_ratio` 和 `J` 的分布；
- B1–B7各转矩带的样本数量；
- 样本之间及父子之间的Hamming距离分布；
- 各变异算子的合法率、修复成功率、重复率和FEMM失败率；
- 不同lineage和拓扑簇是否得到充分覆盖；
- 运行中断恢复后是否出现重复求解或结果缺失。

当前方案仍有以下需要在论文或汇报中明确说明的限制：

1. 六角度扫描是历史兼容定义，并不自动等于科学上充分的转矩脉动测量。后续应选取代表性拓扑做更密角度扫描，量化六点结果的误差。
2. 1000个总样本是物理档案目标，不是各band的硬配额；分层是否均衡必须在仿真后统计。
3. 精英保留保证每个campaign的历史最佳 `J` 不退化，但不保证所有新增样本性能逐代单调提高。
4. 当前单进程FEMM优先保证稳定性和可恢复性，性能主要受单核求解速度限制。并行化必须使用独立进程、独立FEMM实例和独立目录，并在小规模稳定性试验后再启用。
5. `merged_copper_v5` 会改变材料边界、匝数分配或连通结构，因此它是独立实验几何模式，不能与历史兼容模式的结果混为一组而不做同染色体对照。
6. CNN训练集、验证集和测试集应按lineage或拓扑聚类分组切分，不能简单随机打散近亲拓扑，否则会高估模型泛化能力。

### 14. 可直接用于汇报的流程摘要

> 本研究首先将电机设计域编码为180位离散染色体，并按MATLAB兼容顺序还原为18×10（18径向×10角向）的空气、铁和铜材料布局。系统从历史种子出发，通过二维多尺度变异、确定性连通性修复、全库哈希去重和Hamming距离筛选，建立60个合法且差异较大的初始父本，并分配到6个互不复用的campaign中。每条campaign保留10个当前个体，每代产生5个通过准入的后代，6条搜索线每个全局代际最多新增约30个FEMM物理样本。父本选择中70%用于按历史目标函数 `J` 进行性能优化，30%用于稀缺转矩区间和结构多样性探索；每条campaign保留2个最优精英。每个合法唯一拓扑在0°至15°机械区间内以3°步长完成六次FEMM求解，计算平均转矩、转矩脉动率、相对参考模型的转矩比和历史目标函数 `J`。所有完成求解的低、中、高性能样本均进入不可删除的物理档案，而只有筛选后的个体继续参与繁殖。SQLite、逐角度提交和完整随机状态checkpoint保证长时间任务可以安全中断并继续，最终形成约1000个可追溯、可校验的CNN训练候选样本。
# B2–B7 分层补充采样（300 个新物理样本）

补充模式读取已完成的 1000 样本库作为只读父本库和去重库，不修改原始 run，也不重新计算父本。新 run 固定完成 300 个新的六角度 FEMM 候选；所有物理结果均保存，命中 B2–B7 缺口的结果计入 CNN 核心补充统计。

每个完整补充轮次计算 30 个新候选，B2/B3/B4/B5/B6/B7 的预算分别为 `4/3/6/8/3/6`。B7 父本池由已有 B7 和 `torque_ratio >= 0.98` 的 B6 构成，小尺度变异概率为 75%。

第一次只初始化，不启动 FEMM：

```powershell
cd C:\Users\26096\Desktop\em\python_topopt

python scripts\run_band_supplement.py `
  --new-run band_supplement_300_20260808_001 `
  --source-run runs\improved_dataset_1000_20260807_001
```

使用 5 个独立候选级 FEMM worker 启动：

```powershell
python scripts\run_band_supplement.py `
  --resume runs\band_supplement_300_20260808_001 `
  --backend femm `
  --workers 5 `
  --start
```

查看进度：

```powershell
python scripts\show_band_supplement_status.py `
  --run runs\band_supplement_300_20260808_001
```

按一次 `Ctrl+C` 会在当前 FEMM 角度结束后保存并暂停。恢复时重新执行同一条 `--resume ... --start` 命令。补充启动器带独占锁；同一 run 已运行时，第二个终端会直接拒绝启动，避免同时写入 SQLite。

# PyTorch 电机拓扑性能回归

## GPU环境（RTX 5070）

本机训练环境使用官方CUDA版PyTorch。首次安装或误装CPU版后，执行：

```powershell
python -m pip install --force-reinstall --no-deps torch==2.13.0+cu132 `
  --index-url https://download.pytorch.org/whl/cu132
```

训练配置中的 `training.device` 固定为 `cuda`。如果CUDA环境不可用，程序会直接报错，不会静默退回CPU。可在训练前快速确认：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

训练适配器直接读取两次已完成 run 的 `dataset.sqlite`。审计确认数据库中的真实材料矩阵是 `[18,10]`，对应18个径向单元和10个角向单元，因此网络输入固定为 `[B,K,18,10]`，不进行转置、resize或插值。材料 `{0,1,2}` 转换为三通道one-hot。

修正后的ResNet-20尺寸链为：

```text
[B,3,18,10]
→ [B,16,18,10]
→ [B,32,9,5]
→ [B,64,5,3]
→ Flatten(960)
→ Linear(960,64)+ReLU
→ Linear(64,output_dim)
```

先运行只读数据审计：

```powershell
cd C:\Users\26096\Desktop\em\python_topopt

python scripts\train_motor_regression.py audit `
  --config configs\motor_regression_ripple.yaml
```

只训练主模型：

```powershell
python scripts\train_motor_regression.py train `
  --config configs\motor_regression_ripple.yaml `
  --models resnet20
```

使用完全相同的数据划分训练主模型和两个基线：

```powershell
python scripts\train_motor_regression.py train `
  --config configs\motor_regression_ripple.yaml `
  --models resnet20 mlp small_cnn
```

默认输出目录为 `outputs/motor_regression_ripple_1300_rebalanced`。划分在拓扑家族不跨集合的硬约束下，同时平衡B1–B7适应带、torque ripple分位区间和总样本数。根目录保存配置快照、数据审计、统一split和只由训练集拟合的target scaler；每个模型子目录保存最佳checkpoint、训练曲线、MAE/RMSE/R²、预测CSV、真实值-预测值图和残差图。

默认按Hamming拓扑家族进行group split，不允许相同拓扑家族跨集合。配置也支持 `generation`、`parent` 和 `root_parent`。若增加平均转矩多输出，只需在配置的 `data.targets` 中加入：

```yaml
- name: mean_torque
  column: t_avg
```

`output_dim`、目标标准化、反标准化、指标和导出列会自动扩展。
