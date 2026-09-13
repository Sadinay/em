# FEMM 全量结果交付（2026-09-13）

3299 个基因、19794 个角度全部完成，最终失败基因数为 0。当前状态以 `package_summary.json` 和 `tables/report.json` 为准。

最终报告中的 `attempted_genes = 3303` 包含 3299 条成功任务记录和 4 条中途暂停记录；暂停任务随后已恢复完成，去重基因总数仍为 3299。

## 主机验证与使用

- `tables/dataset_all.csv`：全部 3299 个基因的 120 位编码、来源/分组、磁体信息、Tavg、DeltaT 和工况指纹，适合直接核验或导入。
- `tables/train_G_labeled.csv`、`train_F_labeled.csv`：各 1400 行；`dev_common_labeled.csv` 200 行、`test_common_labeled.csv` 400 行。G/F 训练集有 101 个共同基因；四组映射见 `manifests/memberships.csv`。
- `tables/labels.csv`：原始标签汇总；`tables/waveforms.csv`：19794 行原始转矩、角度、电流和 FEM/ANS 哈希。
- `full_run_records.zip`：当前最终工况目录的全部原始文件，含每基因 `label.json`、每角度 `result.json` / `state.json` / `artifact_retention.json`、`contract.json`、运行日志和中断后续跑记录。解压后层级与原工况目录相同。
- `manifests/`：原始冻结清单与候选/选样资料。这里 `femm_queue.csv` 的 pending 是选样时的历史登记值；实际完成状态请读 `tables/`。
- `audit/`：选样、试跑、回归和停机恢复资料；其中 20 个测试结果、1646 个完成结果等数字是历史快照。
- `source_snapshot/`：导出时的运行代码、物理配置、映射、原始 MAT 和 FEM 模板，以及方案文档。CNN 权重和大型特征缓存不在此交付包中。
- `region_previews/`：原始选样区域预览。

标签单位均为 N·m。实际内角为 29、32、35、38、41、44°；Tavg 是六点均值，DeltaT 是六点最大值减最小值。

## 拷贝完整性检查

把整个文件夹复制到主机后，在本文件夹打开终端运行：

```powershell
python verify_package.py
```

只需 Python 3.9+ 标准库，不启动 FEMM，也不需要 NumPy、torch 或解压原始记录。程序检查所有交付文件和压缩包内每份原始记录的 SHA256、基因编码、分组、六角度完整性、Tavg/DeltaT，以及结果与清理凭据的相互关联。最后应输出 `status: passed`、`genes: 3299`、`angles: 19794`。

`checksums.json` 覆盖除其自身外的全部交付文件；归档内部 `RUN_CHECKSUMS.json` 覆盖每份原始文件。成功的 FEM/ANS 工作文件已按要求清理，保留的输入模板、配置与哈希记录用于追溯；本包不会把数值一致性检查宣称为重新运行 FEMM。
