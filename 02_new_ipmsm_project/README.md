# 新 IPMSM 遗传优化项目

本项目按照功能区组织。每个功能区独立管理自己的代码、配置和结果，不再使用全项目统一的 `src/`、`configs/`、`runs/` 和 `outputs/`。

## 目录结构

```text
02_new_ipmsm_project/
├── data_zone/                  # 数据区
│   ├── raw/                    # 原始MAT输入，只读保留
│   │   └── workspace_600.mat
│   ├── processed/              # 从MAT解析并验证的数据
│   ├── database/               # 新项目SQLite数据库
│   └── exports/                # CSV、NPZ、MAT等数据导出
│
├── femm_zone/                  # FEMM区
│   ├── models/                 # 原始及版本化FEM模型
│   │   └── IPMSM.fem
│   ├── scripts/                # FEMM建模、求解和后处理脚本
│   ├── workspaces/             # 每个个体的临时求解目录
│   ├── results/                # FEMM物理结果
│   └── logs/                   # 求解日志和失败记录
│
├── ga_zone/                    # 遗传算法区
│   ├── scripts/                # 编码、选择、交叉、变异和运行入口
│   ├── configs/                # GA参数与目标函数配置
│   ├── checkpoints/            # 随机状态和断点恢复文件
│   └── runs/                   # 分代历史、谱系和运行状态
│
├── cnn_zone/                   # CNN区
│   ├── scripts/                # 数据适配、训练和预测代码
│   ├── configs/                # 网络与训练参数
│   ├── models/                 # PyTorch checkpoint和scaler
│   └── outputs/                # 指标、预测CSV和图表
│
├── reports/                    # 跨功能区的审计与阶段报告
└── README.md
```

## 各功能区之间的数据流

```text
data_zone/raw/workspace_600.mat
        ↓ 审计新基因和历史GA
ga_zone
        ↓ 产生候选染色体
femm_zone
        ↓ 计算真实物理结果
data_zone/database
        ↓ 形成可追溯训练数据
cnn_zone
        ↓ 训练代理模型并输出预测结果
reports
```

## 两个原始输入

| 文件 | 位置 | 用途 |
|---|---|---|
| `workspace_600.mat` | `data_zone/raw/` | 新遗传算法600代历史、200位种群和物理指标 |
| `IPMSM.fem` | `femm_zone/models/` | 新IPMSM基础电磁模型 |

两个文件均保留原始内容和只读属性。

SHA-256：

```text
workspace_600.mat
26b1dc59b72c4b856b75f9a92358d60d1b26093c9f634a44426df98a82271193

IPMSM.fem
696e68c9f0cbedc7c82ba621f10496fef031b292ff69d4c969b4642ebb515e82
```

## 当前确认的新项目特征

- 染色体记录长度：200；
- 种群规模：504；
- 历史代数：600；
- `population`为`[504,200]`二进制矩阵；
- `Material`为`[504,100]`、材料编号0～3；
- MAT中记录`pcrossover=0.5`、`pmutation=0.1`；
- 保存了`Tavg_all`、`DeltaT_all`和`Fitvalue_all`等历史量。

这些只代表已经观察到的数据结构。新GA的具体编码、选择、交叉、变异、约束和目标函数仍需根据源代码审计，不能直接沿用旧180位项目。

详细信息见 [输入审计报告](reports/INPUT_AUDIT.md)。

## 管理规则

1. 原始输入只放在`data_zone/raw`和`femm_zone/models`，不直接覆盖；
2. GA不直接操作FEMM COM，而是通过`femm_zone`提供的评价接口；
3. FEMM结果首先写入`data_zone/database`，CNN只读取完整且通过验证的数据；
4. 每个功能区的临时文件和最终输出分开保存；
5. 新项目使用独立数据库、配置哈希和物理哈希；
6. 新数据不得直接写入旧项目的1,300样本数据库。

