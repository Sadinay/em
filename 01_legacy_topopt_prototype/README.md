# 旧拓扑优化思路验证项目归档

该目录保存此前围绕180位染色体、10×18设计域、CLONALG/FEMM数据生成和CNN回归开展的思路验证结果。2026-09-09按用户要求清理旧求解文件，改为精简结果归档。

## 目录结构

```text
01_legacy_topopt_prototype/
├── FP/                       # 原始MATLAB、FEMM、论文和历史数据
├── python_topopt/            # Python移植、FEMM执行器、数据集和神经网络
├── audit_reports/            # Python移植前的项目审计报告
└── early_experiments/        # 早期CNN和数据审计输出
```

## 主要成果

- MATLAB CLONALG 调用链和历史算法规格审计；
- 180位染色体、18×10材料布局、拓扑预检查和目标函数的 Python 实现；
- 可恢复的 FEMM 批处理与改进式数据采样；
- 1,000个主样本和300个适应度带补充样本；
- 转矩波动率和平均转矩的 ResNet-20、MLP、Small CNN 回归实验；
- SQLite、训练模型、预测CSV和展示报告；原始参考模型保留，批量生成的FEM及ANS已清理；逐角度JSON和运行checkpoint压缩保存在`run_details.zip`。

Python 项目入口：

```powershell
cd C:\Users\26096\Desktop\em\01_legacy_topopt_prototype\python_topopt
```

模型展示入口：

```text
python_topopt/showoutput/README.md
```

## 归档状态

清理前包含70,277个文件、35.13 GB数据；清理后约0.31 GB，具体核验数字见[清理记录](../maintenance/cleanup_20260909/README.md)。

- 8,075份ANS和9,388份生成FEM已删除；36份参考模型/原始几何快照保留。
- 主1000样本和补充300样本的数据库、最终模型、图表保留。逐角度数值导出到`python_topopt/runs/<运行名>/result_exports/`。
- 31,483份逐样本/逐角度/运行checkpoint文件合并为每个run的`run_details.zip`，文件内容逐项通过SHA-256校验。
- 明确废弃的6×1 Conv1D程序`early_experiments/cnn_flow_validation`已删除，其`cnn_flow_output`实验结果仍保留。
- 空的已废弃初始化任务已移出`runs`；其结果档案在`python_topopt/outputs/archived_runs/`。
- `FP/CLONALG`原有嵌套Git历史移到工作区的`maintenance/cleanup_20260909/local_only/clonalg_git`本地保留，源码仍在原位置。

该项目现在定位为“思路验证与历史证据归档”。后续新电机项目不在这里创建新数据库或继续追加训练样本。

## 路径兼容性

旧配置、历史 manifest 和报告中保存了原绝对路径 `C:\Users\26096\Desktop\em\FP` 与 `C:\Users\26096\Desktop\em\python_topopt`。为了保持配置哈希和历史记录不变，本次没有批量改写这些文件。

数据库、图表、预测结果和源码仍可查看。旧FEMM任务不再作为可直接resume的运行目录；若重新计算，应使用保留的模板和代码创建新任务，并先阅读 [PATH_MIGRATION.md](PATH_MIGRATION.md)。
