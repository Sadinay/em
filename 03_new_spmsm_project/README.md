# 新 SPMSM 遗传优化项目（第三类基因）

本项目按 `02_new_ipmsm_project` 的功能区规格独立组织，用于 `SPMSM_discrete.fem` 与 `workspace_200.mat` 所代表的第三类基因和电机结构。03 与 01、02 的基因编码、FEMM 几何映射、GA 配置、数据库、CNN 模型及实验结果不得混用。

2026-09-13 更新：已接收并核验另一台设备的全部 **3299 个基因 / 19794 个角度**，完成冻结旧 CNN 的基线评估。G/F 各1400及公共dev/test数据入口见[接收与评估说明](experiments/input_distribution_pilot_v1/post_femm_baseline_20260913/README.md)，结论见[基线报告](experiments/input_distribution_pilot_v1/post_femm_baseline_20260913/BASELINE_REPORT.md)。尚未启动补样训练或最终test评估。

导入时发现本机 `femm_zone/femm_config.py` 回退到内角0°起步/转矩−2；现已逐字节恢复为已核验结果包的 **29°起步/倍率1** 配置。求解入口新增检查，旧准备目录的物理设置不一致时拒绝启动 FEMM，需换名称重新 `prepare`。修复及离线验证见[配置恢复记录](femm_zone/workspaces/config_restore_20260913/README.md)。下文部分“种子待算”说明为此前阶段记录。

FEMM配置统一到 [femm_config.py](femm_zone/femm_config.py)，历史五基因复现使用 [run_femm.py](femm_zone/run_femm.py)。默认仅显示配置；准备、求解和出图用法见 [FEMM说明](femm_zone/README.md)。旧试验脚本已归档。

下一步四组输入分布试验使用独立的 [pilot.py](experiments/input_distribution_pilot_v1/pilot.py)，新增选种子、孤立单格修正和批量 FEMM 入口集中在此文件，沿用上述物理配置。种子清单、审计和之后手动启动 FEMM 的命令见 [试验说明](experiments/input_distribution_pilot_v1/README.md)。本次只准备种子，不启动新 FEMM 求解或四组训练。

当前已完成120位基因—FEMM结构映射、数据整理和40000样本旧CNN训练。采用初始内角29°后，历史G2–G5的平均转矩与峰峰转矩差已复现；G1仍有基因与历史记录不一致的问题。早期映射结论见 [基因—FEMM映射与验证报告](reports/MAT_FEM_MAPPING_VALIDATION.md)，当前物理口径见 [FEMM说明](femm_zone/README.md)。

## 目录结构

```text
03_new_spmsm_project/
├── data_zone/
│   ├── raw/                    # 原始 MAT，只读保留
│   │   └── workspace_200.mat
│   ├── processed/              # 解析、清洗和验证后的数据
│   ├── database/               # 03 独立数据库
│   ├── exports/                # CSV、NPZ、MAT 等导出
│   ├── scripts/                # 数据审计与转换脚本
│   └── tests/                  # 数据链路测试
├── femm_zone/
│   ├── models/                 # 原始及版本化 FEM 模型
│   │   └── SPMSM_discrete.fem
│   ├── scripts/                # FEMM/CNN共用的基因映射辅助模块
│   ├── femm_config.py          # 唯一FEMM参数配置
│   ├── run_femm.py             # 唯一运行入口
│   ├── archive/                # 原试验脚本，只供追溯
│   ├── workspaces/             # 单个任务的临时求解目录
│   ├── results/                # FEMM 物理结果
│   ├── logs/                   # 求解日志和失败记录
│   └── tests/                  # FEMM 接口测试
├── ga_zone/
│   ├── scripts/                # 编码、选择、交叉、变异和运行入口
│   ├── configs/                # GA 参数与目标函数配置
│   ├── checkpoints/            # 随机状态和断点恢复
│   └── runs/                   # 分代历史、谱系和运行状态
├── cnn_zone/
│   ├── scripts/                # 数据适配、训练和预测入口
│   ├── configs/                # 网络与训练配置
│   ├── models/                 # checkpoint 和 scaler
│   └── outputs/                # 指标、预测明细和图表
├── outputs/                    # 跨模块分析产物
├── reports/                    # 审计与阶段报告
└── README.md
```

## 数据流

```text
data_zone/raw/workspace_200.mat
        ↓ 审计第三类基因、历史 GA 和物理标签
ga_zone
        ↓ 产生 SPMSM 候选基因
femm_zone
        ↓ 根据独立映射生成结构并计算真实物理结果
data_zone/database
        ↓ 形成可追溯、已验证的数据集
cnn_zone
        ↓ 训练代理模型并输出预测
reports / outputs
```

## 原始输入与校验值

| 文件 | 位置 | SHA-256 |
|---|---|---|
| `workspace_200.mat` | `data_zone/raw/` | `2FBFBE8743CFE0A23C3F3C8DC54378CC82774145845F92D68555FB4EEC1270D8` |
| `SPMSM_discrete.fem` | `femm_zone/models/` | `C6353F71C75E5D743D90717E4405337A5BCE11F2032BEB162B6EC9833D8001F5` |

两个文件已设置为只读，后续处理必须写入其他目录，不得覆盖原件。

## 当前只读检查确认的信息

- 电机模型：SPMSM，原始 FEM 文件名为 `SPMSM_discrete.fem`；
- `BitLength = 120`；
- `popsize = 612`；
- `Generation = Generationnmax = 200`；
- `population` 为 `[612, 120]` 二进制矩阵；
- `population_all` 和 `population_noChange_all` 均为 `[612, 120, 201]`；
- `Tavg_all`、`DeltaT_all`、`Fitvalue_all` 均为 `[612, 201]`；
- `MaterialPosition` 为 `[120, 8]`；
- MAT 中记录 `pcrossover = 0.5`、`pmutation = 0.1`。

以上仅是变量结构检查，不代表已经确认基因到 FEMM 区域的物理映射，也不代表历史标签已经通过重算验证。

## 隔离与后续测试规则

1. 不直接复用 02 的 200 位基因解释、100 区域材料映射或八通道 224×224 渲染器。
2. 必须先审计 120 位编码与 `MaterialPosition`、FEMM block label/几何区域之间的对应关系。
3. 必须抽取代表样本进行 FEMM 重放，确认 `Tavg` 和 `DeltaT` 与历史 MAT 标签一致。
4. 03 使用独立的数据划分、scaler、配置哈希、物理哈希、数据库、模型目录和结果目录。
5. 原始输入只读；临时 FEMM 文件只进入 `femm_zone/workspaces`，正式结果进入 `femm_zone/results` 和独立数据库。
6. 在输入映射和标签审计通过前，不启动正式 CNN 训练或大规模数据生成。
