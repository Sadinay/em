# 03 FEMM

日常只看两个文件：

- [femm_config.py](femm_config.py)：唯一配置入口。MAT/模型路径、5个基因、内外角、初始电流角、问题定义、电流公式和转矩公式均在这里。
- [run_femm.py](run_femm.py)：历史五基因复现入口，负责准备模型、顺序求解和生成对照图。

下一步四组新种子的批量入口集中在 [pilot.py](../experiments/input_distribution_pilot_v1/pilot.py)，复用本目录的物理配置，不改变这里的历史五基因。手动运行方法见 [试验说明](../experiments/input_distribution_pilot_v1/README.md)。

电流幅值、极对数、深度直接读原始 MAT 的 `inp.Is_amp`、`inp.P`、`inp.Lfe`；当前为3.5 A、4对极、36 mm。几何、绕组和材料BH曲线来自只读FEM模板；基因映射辅助模块保留在 `scripts/spmsm_mapping.py`，供FEMM和CNN共用。

在工作区根目录执行：

```powershell
# 默认只显示配置，不打开FEMM
python 03_new_spmsm_project/femm_zone/run_femm.py

# 只生成可审查的FEM模型；仍不打开FEMM
python 03_new_spmsm_project/femm_zone/run_femm.py prepare --name run01

# 明确执行求解；完成后自动生成summary.json、metrics.csv和逐相位对照图
python 03_new_spmsm_project/femm_zone/run_femm.py solve --name run01

# 仅从完整的已有解重新生成报告
python 03_new_spmsm_project/femm_zone/run_femm.py report --name run01
```

运行目录为 `workspaces/<名称>/`。`prepare`把当前配置冻结到`run.json`，并把全部参数写入每个`model.fem`；`solve`使用这份已审查的输入，不从历史结果继承配置。修改参数后请换一个名称重新`prepare`，程序不覆盖旧目录，也不自动续跑中断任务。求解使用独立FEMM实例，普通Ctrl+C中断会清理该实例。

当前按导师补充代码：`ang_0=29°`，转过的机械角`ANG_R=0:3:15°`，实际内角为29、32、35、38、41、44°；外角0°、Min Angle=15°、电流初相位0°。时间由`ANG_R/wmech`确定，电流使用`Is_amp*cos(omega*t)`及−120°、−240°相移；MAT提供3.5 A、400 Hz及机械角速度。29°来自补充代码，采样范围来自用户要求；原始T/ANG_R生成代码未提供。**没有自动启动整圈筛选**。

本次直接比较FEMM原始气隙积分，不拟合符号或倍率。以前的`−2×气隙积分`没有确认的历史依据，仅在`summary.json`中另存旧换算对照。平均转矩为六点算术平均，MAT的`DeltaT`对应`max(T)−min(T)`，单位N·m；29°复现的G2～G5均逐一符合至1e-12量级。相对波动另存为派生量，不能直接与MAT DeltaT比较。G1的平均与波动未复现，其基因磁体数45与MAT同位置VolumePM=44不一致。

旧的22份试验脚本完整保存在 [archive](archive/README.md)，仅供追溯。原始MAT/FEM及历史`results`未改动。
