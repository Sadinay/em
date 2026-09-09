# 电机拓扑优化工作区

本目录已将旧思路验证、IPMSM 和第三类 SPMSM 项目完全分开。各项目的基因编码、遗传算法、FEMM 模型和数据库不得混用。

```text
em/
├── 01_legacy_topopt_prototype/   # 旧180位拓扑优化与CNN思路验证项目
├── 02_new_ipmsm_project/         # 新IPMSM模型、200位编码与正式实验
└── 03_new_spmsm_project/         # SPMSM模型、第三类120位基因独立项目
```

## 项目入口

| 项目 | 用途 | 入口说明 |
|---|---|---|
| [01_legacy_topopt_prototype](01_legacy_topopt_prototype/README.md) | 旧 MATLAB/Python 源码、参考模型、1,300 样本数据和实验结果 | 已精简；批量求解文件已清理，结果保留 |
| [02_new_ipmsm_project](02_new_ipmsm_project/README.md) | 基于 `IPMSM.fem` 和 `workspace_600.mat` 建立新基因、新遗传算法和新数据库 | 后续开发主目录 |
| [03_new_spmsm_project](03_new_spmsm_project/README.md) | 基于 `SPMSM_discrete.fem` 和 `workspace_200.mat` 的120位基因、CNN及FEMM复现 | 初始内角29°设置已核验；FEMM入口见项目内说明 |

## 隔离原则

1. 新项目不得直接写入旧项目的 `runs/`、`outputs/` 或 SQLite 数据库。
2. 新项目重新审计 `workspace_600.mat` 的编码、目标函数和 GA 调用链，不能默认沿用旧项目的180位染色体与 CLONALG 定义。
3. 新项目使用独立配置哈希、物理哈希、运行目录和数据库。
4. 原始输入和汇总结果保留；经确认可再生成的批量求解文件可清理，清理记录见 `maintenance/`。
5. 03 不得默认复用 02 的200位基因映射、IPMSM几何、CNN输入或训练数据。

## Git 中保存什么

源码、配置、原始MAT、参考FEM模型、汇总CSV、报告和图表进入Git。FEMM求解中间文件、训练权重、大型派生数据、SQLite、论文资料及重复的报告ZIP保留本地并由`.gitignore`排除；01已删除的ANS不再保留。`data_zone/processed`和`data_zone/exports`仅纳入说明和JSON清单。

01的逐角度数值可直接查看 `python_topopt/runs/<运行名>/result_exports/`；运行细节保存在同目录 `run_details.zip`。仅从Git克隆后不包含本机训练权重或SQLite，不能直接恢复旧求解任务。

整理范围和保留结果说明见 [清理记录](maintenance/cleanup_20260909/README.md)。
