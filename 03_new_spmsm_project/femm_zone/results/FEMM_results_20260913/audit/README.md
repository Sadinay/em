# 03 下一步：四组输入分布种子

新增代码集中在 [pilot.py](pilot.py)。它复用原来的 `femm_config.py`、基因映射和 CNN 输入转换，不改历史五基因配置。依据为 [四组方案](../../data_zone/SPMSM_Input_Distribution_Pilot_4Groups.md)，种子准备阶段只完成选种子和运行入口，未启动 FEMM 或训练。用户随后授权在新设备实跑 20 个基因测试批次；全量批次由用户手动启动。

种子清单与实际数量见 [selection_audit.md](selection_audit.md)。`train_G.csv` / `train_F.csv` 各 1400；公共 `dev_common.csv` 200、`test_common.csv` 400。G-S/G-E 共用 G，F-S/F-E 共用 F；`memberships.csv` 记录归属，`femm_queue.csv` 是去重计算队列。

本次实际 G/F 交集为 101，去重队列 3299 个基因，共 19794 个角度任务。种子阶段准备了 `pilot20.csv` 的 120 份 FEM 输入，当时全部为 pending。入口检查见 [entry_validation.json](entry_validation.json)：独立重放全部 20600 个输入的单格修正通过；模拟驱动验证了重试、恢复、拒绝损坏缓存和缺角度不生成标签。模拟测试不等于实跑 FEMM；迁移到本机后的实跑结果见 [machine_validation.md](machine_validation.md)。

用户追加执行设置：**默认 6 个 worker**，使用 Windows `spawn` 创建独立 Python 进程。一个任务对应一个基因，每个任务内部顺序算六个角度；各进程使用独立 FEMM 实例与模型目录。授权变更记录见 [execution_override.json](execution_override.json)，并行模拟测试见 [parallel_validation.json](parallel_validation.json)。种子阶段的 `pilot_config.json` 保持原始登记内容，实际 worker 数以 `--workers` 和执行日志为准；未重选种子或修改物理参数。

用户追加的孤立单格修正先于选样：在逻辑 6×20 网格内检查周围 8 邻域，只有所有现存邻居都与中心不同才翻转；边缘不补邻居、不连接两端；同步更新直到稳定。不进行一般多数平滑。训练候选、dev、test 都按同一规则修正，再去重并排除完整已知历史。每行保留 `raw_bits`、`raw_gene_id`、`repair_changed_cells` 和 `repair_passes`。修正可能改变磁体格数，生成器参数中的 m 表示修正前的用量。

## 之后手动运行 FEMM

从 `03_new_spmsm_project` 项目根目录进入本目录。本机 FEMM 环境为根目录下的 `.venv-femm`（Python 3.13.14），系统默认 Python 没有这些依赖。直接指定解释器，无需激活环境：

```powershell
$femmPython = (Resolve-Path .\.venv-femm\Scripts\python.exe).Path
cd .\experiments\input_distribution_pilot_v1
& $femmPython pilot.py
```

默认只显示种子状态。下面的 `prepare` 只写 FEM 文件；只有明确执行 `solve` 才会启动独立 FEMM 实例。

续跑会先检查冻结输入、所有已完成角度的结果/清理凭据，以及待算 FEM 模型；本机在已完成 1646 个基因时，全量预检查实测约 3 分钟。`solve` 现在会先显示校验提示，请等待 `Prepared` 输出后进入调度。若最后一行是 `KeyboardInterrupt`，表示进程收到中断信号（通常为 Ctrl+C/终端停止），不能据栈中显示的 `require(..., "...hash mismatch")` 源码判定哈希不匹配；真正的校验失败会抛出对应 `ValueError`。主进程其他异常现完整保存到 `failure_diagnostics/*.json`，包含 traceback、文件名及系统错误码，以便定位具体操作；该日志改动不改变工况指纹。

```powershell
# 准备预先选定的 20 个小批种子，每个六个角度，不求解
& $femmPython pilot.py prepare --scope pilot

# 你准备好后手动启动小批；成功角度经身份和哈希检查后复用
& $femmPython pilot.py solve --scope pilot --workers 6
& $femmPython pilot.py report

# 检查小批报告后，再手动启动整个去重队列（包含已完成的小批，不重复算）
& $femmPython pilot.py solve --scope all --workers 6
& $femmPython pilot.py report
```

`femm_entry.json` 指向实际运行目录 `femm_runs/<工况指纹>/`。求解前 `<gene_id>/angle_29/model.fem` 等文件可直接打开审查；若希望先准备全队列再审查，使用 `& $femmPython pilot.py prepare --scope all`。只用 FEMM 图形界面点求解不会自动写入本入口的转矩/状态记录；请在终端启动上述 `solve`。

按用户在 2026-09-09 的追加要求，**成功角度只保留结果和校验记录**：先检查工况、模型参数、原始转矩及 FEM/ANS 哈希，保存 `result.json`、`state.json` 和与结果哈希绑定的 `artifact_retention.json`，释放该 FEMM 实例后删除本角度的 FEM/ANS 与已知网格工作文件（含成功重试前的工作副本）。清理中途退出可恢复；无清理凭据却丢失 FEM/ANS，或结果/凭据哈希不符，仍拒绝复用。失败角度保留工作文件和错误记录。原始模板、MAT、历史回归数据不清理。

工况沿用已核验的 03：实际内角 29、32、35、38、41、44°，外角 0°，Min Angle 15°；3.5 A 三相 cos 电流从电角 0°开始，随机械行程正向推进，每转子 1°对应电角 4°。29°只加到内角。六点算术平均为 Tavg，六点最大减最小为 DeltaT，单位均 N·m，无 −2 经验倍率。

已核验的 G2–G5 历史六点结果通过文件哈希和参考值检查后复用，见 `regression_validation.json`；并非本次重新求解。G1 不作为核验标准。20 个新小批在标签产生前固定，包含来源、G/F 归属和用量极端例；全队列入口要求这 20 个都成功。

每角度即时保存原始转矩及 FEM/ANS 哈希；六角度完整才生成有效标签。运行结果为 `labels.csv`、`waveforms.csv`、各基因 `label.json` 和 `report.json`。失败不填 0、不用 CNN 预测代替。每角度最多两次总尝试，重试使用新的隔离 FEMM 实例；连续 3 个基因失败或最近至多 50 个尝试中累计 5 个失败会停止。失败种子不自动替换；若小批失败，先查错误，不扩大计算。

并行时，失败阈值按主进程接收的完成记录判断；达到阈值即停止派发，已在运行的任务完成当前角度、保存结果后暂停。Ctrl+C 同样先停止派发，再等待当前角度保存，因此不是瞬间退出。总队列清单和日志由主进程写入，批次锁阻止两个终端重复启动同一工况。

运行期间可另开终端使用本机环境执行 `pilot.py status` 查看 `progress.json` 中的 worker 数、完成数量和进程 PID；运行中不执行会写汇总表的 `report`。报告中的吞吐与剩余时间估计使用并行批次的实际墙钟耗时，不能用六个进程耗时相加代替。首次实测前不提供耗时承诺。

同一命令可恢复已登记的角度，成功结果须通过哈希校验；已按上述策略清理工作文件的角度不会重新求解。若强制关闭程序留下 `active.lock`，先确认其中 PID 及其 worker 已结束再手动移除锁。修改物理条件或运行实现会形成新的工况目录，避免误用旧标签；仅改变 `--workers` 数量可继续复用同一工况的成功结果。程序不删除旧结果记录，不关闭用户其他 FEMM 实例。不填 `--workers` 时也默认使用 6。

本机首次实跑发现 pyfemm 在一个 worker 中误进入文件通信分支，报 `ifile` 未定义；当前入口先显式导入 Windows COM，再用 `DispatchEx` 创建独立实例并明确启用 COM 通信。失败批次记录保留，未重置失败计数。仅运行 FEMM 所需的依赖固定在 [requirements-femm.txt](requirements-femm.txt)，无需安装 torch；重建种子/CNN 特征仍需完整的训练环境。在另一台机器重建环境可从项目根目录执行：

```powershell
uv venv --python 3.13 .venv-femm
uv pip install --python .venv-femm/Scripts/python.exe -r experiments/input_distribution_pilot_v1/requirements-femm.txt
```

2026-09-10 全量运行在完成 1646 个基因后，主进程因 `PermissionError(13, '拒绝访问。')` 退出；FEMM 基因失败数为 0，另有 4 个基因保留 9 个成功角度。旧日志未记录具体访问路径，文件占用是待确认的原因。JSON/CSV 原子保存现对 `PermissionError` 最多尝试 12 次，退避等待合计最多 7.55 秒；持续拒绝访问仍报错，并记录完整目标路径。重试始终先写临时文件，再替换正式文件，不截断已有结果；其他 I/O 错误不自动重试。本次仅调整保存函数，已核对工况指纹保持不变，原结果无需重算。运行测试含真实 Windows 文件锁，共 15 项通过。恢复检查记录见 [io_recovery_20260910.json](io_recovery_20260910.json)。

## 重建与审查

```powershell
& $femmPython pilot.py selftest       # WL显式核、LCMD对照、孤立单格规则和电流检查
& $femmPython pilot.py parallel-selftest  # 六个真实Python进程执行模拟任务，不启动FEMM
& $femmPython -m unittest test_runtime -v  # COM通信、结果清理、损坏拒绝及清理中断恢复，不启动FEMM
python pilot.py seeds          # 完整种子流程，只有旧CNN推理，不运行FEMM
```

也可分步执行 `generate`、`features`、`select`。同一版本校验冻结输入和缓存；不覆盖原始 MAT、旧数据划分或权重。需要本机 Python 的 numpy、scipy、matplotlib、torch、psutil；FEMM 入口还需要已注册 FEMM、pyfemm、pywin32。模型权重与大型缓存按项目约定只保存在本机。

- `pilot_config.json` / `provenance.json`：试验设置、命名随机流和输入哈希。
- `parent_families.json` / `old_anchors.csv`：P 父代划分和两方法共同 512 个旧 train 锚点。
- `region_graph.json`：492 个真实区域及共享边界邻接，无猜测周期边；区域面积来自三角网格近似。
- `cache/features_*.npy` / `feature_scaler.json`：冻结旧 CNN 两路 256 维 FP32 特征；scaler 仅用旧 train 40000。新 dev/test 不参与此步骤。
- `selection_G.json` / `selection_F.json`：LCMD 选样轨迹、耗时和内存；未按预测转矩过滤。
- `region_previews/`：各表示 10 个展示分区的逻辑网格和真实材料图，编号不能跨 G/F 对应。
- `seed_validation.json`：清单数量、交集、历史排除、家族互斥、编码、孤立单格和工况证据。

有限图与冻结旧模型的特征只是两种选样表示；这一步尚未证明新结构性能或泛化提升。后续四组训练和路由开发按方案另行开展。
