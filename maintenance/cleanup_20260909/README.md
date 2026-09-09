# 2026-09-09：01精简与Git准备

按用户要求，01以保留实验结果为主，删除旧FEMM求解文件和废弃组件；02、03的原始数据、当前代码和求解结果未删除，仅清理Python缓存并调整Git忽略规则。

## 清理结果

- 01由35.13 GB缩减到310,914,921字节（约0.31 GB），文件数由70,277降到1,252，体积减少99.11%。
- 删除8,075份ANS（33.83 GB）、9,388份生成FEM（0.913 GB）；保留36份原始参考模型或几何快照。
- 删除旧worker日志、存在当前副本的`.bak`、过期运行锁和Python缓存。
- 已废弃的`01/early_experiments/cnn_flow_validation`（6×1 Conv1D基线）程序已移除，其实验结果`cnn_flow_output`保留。
- 31,483个逐样本、逐角度与运行checkpoint文件合并为ZIP，逐项验证内容后移除散文件；并未把ANS放入ZIP。
- 主清理共移除68,883个文件、34,811,215,516字节。此数包含归档后的原文件；压缩档案和结果CSV另占少量空间。

## 结果在哪里

| 结果 | 保留位置 |
|---|---|
| 1000个主样本、300个补充样本及其他验证的完整数据库 | `01_legacy_topopt_prototype/python_topopt/runs/<运行名>/dataset.sqlite`，留在本机 |
| 基因、样本指标、逐角度转矩CSV | 上述运行目录的`result_exports/`，进入Git |
| 原始逐角度JSON、作业信息和运行checkpoint | 上述运行目录的`run_details.zip`，进入Git |
| CNN训练权重、预测表、图表与报告 | 原`outputs/`、`showoutput/`及早期结果目录；权重保留本机 |
| MATLAB实验MAT、FIG、PNG与源代码 | 原`01/FP/`位置保留；论文资料留本机 |
| 废弃的空初始化任务 | `01/python_topopt/outputs/archived_runs/discarded_initialization_20260808.zip`；原目录移到本地保留区 |

导出5个数据库时完整性检查均为`ok`，共保存7,854条带转矩值的逐角度记录。主样本库的1,001条样本记录含1条参考，补充库的301条也含1条参考。废弃初始化库没有样本或角度结果，已归档移出正常运行目录。

1,231份直接保留的文件以及31,483份进入ZIP的文件，在修改说明文档前逐项通过SHA-256核验。保留的是MAT/数据库/标量结果/训练成果；已删除ANS中的完整场解不再可用。旧运行目录不支持直接恢复FEMM求解，重算应使用参考模型和源码创建新任务。

## Git整理

- `.gitignore`排除求解场文件、网格中间文件、生成模型、缓存、权重、SQLite、大型数组与派生数据；原始MAT、参考FEM、源码、配置、报告、图表和结果CSV保留。
- `.gitattributes`明确保留MAT和FEM的原始字节，避免换行转换破坏记录的输入哈希。
- 先移除暂存区中已被忽略的旧文件，再整理本次待提交内容。没有提交、推送或配置远程仓库。
- 核验时Git待提交内容约367 MB，最大单文件约24 MB；6份关键MAT/FEM在Git中的内容与本机原文件逐字节一致。02、03共6,422份非缓存文件的大小与修改时间保持不变。
- 已回收Git中不再被引用的旧暂存对象；整理后的Git对象包约128 MiB。当前暂存对象完整性通过`git fsck --cache --no-dangling`检查。根仓库尚无提交，旧CLONALG历史单独保留。
- 原`01/FP/CLONALG/.git`历史没有删除，移到了本机`maintenance/cleanup_20260909/local_only/clonalg_git`，避免以嵌套仓库形式漏传源码。可通过`git --git-dir=<该目录> log`查看；其未提交源码仍保留在原CLONALG目录。
- 从Git重新克隆不包含本机SQLite、训练权重或派生大数据；CSV和ZIP可供检查结果，训练或重新求解需要另行准备对应数据。

## 核查文件

- `cleanup_plan.json`：删除分类与数量。
- `result_archives.json`：结果导出及ZIP校验汇总。
- `validation_summary.json`：结果保留和数据库核验。
- `discarded_run_archive.json`：空初始化任务归档记录。
- `final_summary.json`：最终工作区与Git暂存区统计。

逐文件删除清单、原始哈希、旧Git索引与详细库存留在本机，已排除出Git。`prepare_cleanup.py`与`archive_results.py`是本次维护脚本，`execute_cleanup.ps1`仅执行经过检查的清单，不是日常运行入口；清单为一次性记录，不应重新执行。
