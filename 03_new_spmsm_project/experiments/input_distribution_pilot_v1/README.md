# 03 下一步：四组输入分布种子

**2026-09-13 最新状态：**3299个基因的全量FEMM结果已取回，19794个角度记录和输入哈希全部核验通过；冻结旧CNN基线已完成，最终test未评估。训练兼容数据入口和结果见[post_femm_baseline_20260913](post_femm_baseline_20260913/README.md)。下文pending/未求解是09-09选种子阶段的历史说明，不代表最新标签状态。

导入时发现的 `femm_config.py` 版本差异已修复：现已逐字节恢复为结果包的 **29°初始内角/转矩倍率1** 配置，并重新核验冻结种子入口。详见[配置恢复记录](../../femm_zone/workspaces/config_restore_20260913/README.md)。3299个基因已有有效结果；下方命令保留为运行说明，本次修复未启动重算。

新增代码集中在 [pilot.py](pilot.py)。它复用原来的 `femm_config.py`、基因映射和 CNN 输入转换，不改历史五基因配置。依据为 [四组方案](../../data_zone/SPMSM_Input_Distribution_Pilot_4Groups.md)，本次按用户要求只完成选种子和运行入口，**不启动 FEMM 或训练**。

种子清单与实际数量见 [selection_audit.md](selection_audit.md)。`train_G.csv` / `train_F.csv` 各 1400；公共 `dev_common.csv` 200、`test_common.csv` 400。G-S/G-E 共用 G，F-S/F-E 共用 F；`memberships.csv` 记录归属，`femm_queue.csv` 是去重计算队列。

本次实际 G/F 交集为 101，去重队列 3299 个基因，共 19794 个角度任务。已准备 `pilot20.csv` 的 120 份 FEM 输入，全部为 pending，尚无新标签。入口检查见 [entry_validation.json](entry_validation.json)：独立重放全部 20600 个输入的单格修正通过；模拟驱动验证了重试、恢复、拒绝损坏缓存和缺角度不生成标签。模拟测试不等于已实跑 FEMM。

用户追加执行设置：**默认 6 个 worker**，使用 Windows `spawn` 创建独立 Python 进程。一个任务对应一个基因，每个任务内部顺序算六个角度；各进程使用独立 FEMM 实例与模型目录。授权变更记录见 [execution_override.json](execution_override.json)，并行模拟测试见 [parallel_validation.json](parallel_validation.json)。种子阶段的 `pilot_config.json` 保持原始登记内容，实际 worker 数以 `--workers` 和执行日志为准；未重选种子或修改物理参数。

用户追加的孤立单格修正先于选样：在逻辑 6×20 网格内检查周围 8 邻域，只有所有现存邻居都与中心不同才翻转；边缘不补邻居、不连接两端；同步更新直到稳定。不进行一般多数平滑。训练候选、dev、test 都按同一规则修正，再去重并排除完整已知历史。每行保留 `raw_bits`、`raw_gene_id`、`repair_changed_cells` 和 `repair_passes`。修正可能改变磁体格数，生成器参数中的 m 表示修正前的用量。

## 之后手动运行 FEMM

从项目根目录 `em` 进入本目录（相对路径）：

```powershell
cd ./03_new_spmsm_project/experiments/input_distribution_pilot_v1
python pilot.py
```

默认只显示种子状态。下面的 `prepare` 只写 FEM 文件；只有明确执行 `solve` 才会启动独立 FEMM 实例。

```powershell
# 准备预先选定的 20 个小批种子，每个六个角度，不求解
python pilot.py prepare --scope pilot

# 你准备好后手动启动小批；成功角度经身份和哈希检查后复用
python pilot.py solve --scope pilot --workers 6
python pilot.py report

# 检查小批报告后，再手动启动整个去重队列（包含已完成的小批，不重复算）
python pilot.py solve --scope all --workers 6
python pilot.py report
```

`femm_entry.json` 指向实际运行目录 `femm_runs/<工况指纹>/`，其中 `<gene_id>/angle_29/model.fem` 等文件可直接打开审查。若希望先准备全队列再审查，使用 `python pilot.py prepare --scope all`。只用 FEMM 图形界面点求解不会自动写入本入口的转矩/状态记录；建议由你在终端手动启动上述 `solve`。

工况沿用已核验的 03：实际内角 29、32、35、38、41、44°，外角 0°，Min Angle 15°；3.5 A 三相 cos 电流从电角 0°开始，随机械行程正向推进，每转子 1°对应电角 4°。29°只加到内角。六点算术平均为 Tavg，六点最大减最小为 DeltaT，单位均 N·m，无 −2 经验倍率。

已核验的 G2–G5 历史六点结果通过文件哈希和参考值检查后复用，见 `regression_validation.json`；并非本次重新求解。G1 不作为核验标准。20 个新小批在标签产生前固定，包含来源、G/F 归属和用量极端例；全队列入口要求这 20 个都成功。

每角度即时保存原始转矩及 FEM/ANS 哈希；六角度完整才生成有效标签。运行结果为 `labels.csv`、`waveforms.csv`、各基因 `label.json` 和 `report.json`。失败不填 0、不用 CNN 预测代替。每角度最多两次总尝试，重试使用新的隔离 FEMM 实例；连续 3 个基因失败或最近至多 50 个尝试中累计 5 个失败会停止。失败种子不自动替换；若小批失败，先查错误，不扩大计算。

并行时，失败阈值按主进程接收的完成记录判断；达到阈值即停止派发，已在运行的任务完成当前角度、保存结果后暂停。Ctrl+C 同样先停止派发，再等待当前角度保存，因此不是瞬间退出。总队列清单和日志由主进程写入，批次锁阻止两个终端重复启动同一工况。

运行期间可另开终端执行 `python pilot.py status` 查看 `progress.json` 中的 worker 数、完成数量和进程 PID；运行中不执行会写汇总表的 `report`。报告中的吞吐与剩余时间估计使用并行批次的实际墙钟耗时，不能用六个进程耗时相加代替。首次实测前不提供耗时承诺。

同一命令可恢复已登记的角度，成功结果须通过哈希校验。若强制关闭程序留下 `active.lock`，先确认其中 PID 及其 worker 已结束再手动移除锁。修改物理条件或运行实现会形成新的工况目录，避免误用旧标签；仅改变 `--workers` 数量可继续复用同一工况的成功结果。程序不删除旧结果，不关闭用户其他 FEMM 实例。不填 `--workers` 时也默认使用 6。

## 重建与审查

```powershell
python pilot.py selftest       # WL显式核、LCMD对照、孤立单格规则和电流检查
python pilot.py parallel-selftest  # 六个真实Python进程执行模拟任务，不启动FEMM
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
