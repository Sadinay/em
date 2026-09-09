# 旧项目路径迁移说明

2026-09-09补充：已按用户要求删除旧ANS和生成FEM；运行细节已合并到ZIP，数据库和指标结果保留。以下路径说明用于追溯和新建复现实验，不代表旧运行目录仍可直接恢复求解。嵌套CLONALG Git元数据已移到工作区`maintenance/cleanup_20260909/local_only/clonalg_git`，未删除其历史。

旧位置：

```text
C:\Users\26096\Desktop\em\FP
C:\Users\26096\Desktop\em\python_topopt
```

新位置：

```text
C:\Users\26096\Desktop\em\01_legacy_topopt_prototype\FP
C:\Users\26096\Desktop\em\01_legacy_topopt_prototype\python_topopt
```

## 为什么没有自动替换全部路径

旧项目的 JSON、YAML、SQLite、checkpoint 和运行 manifest 中存在配置哈希、物理哈希与源文件 SHA-256。直接批量替换路径可能导致：

- 已有 run 的不可变配置校验失败；
- `--resume` 认为物理配置发生变化；
- 历史报告失去当时的路径证据；
- 新旧 FEMM 缓存被错误混用。

因此本次只整理目录，不修改旧运行记录。

## 重新运行旧项目时需要检查

主要硬编码位置包括：

```text
python_topopt/configs/dataset_merged_v5.json
python_topopt/configs/dataset_merged_v5_supplement_300.json
python_topopt/configs/improved_dataset_sampling.json
python_topopt/configs/historical_march.yaml
python_topopt/tests/test_improved_sampling.py
```

如果未来确实需要恢复旧运行，应为该次恢复建立新配置副本和新 run ID，不要覆盖旧配置或旧数据库。新路径应指向本归档目录中的 `FP/CLONALG` 和 `python_topopt`。

仅重新训练已有1,300样本的神经网络时，从新的 `python_topopt` 目录运行即可，因为训练配置使用项目内相对 `runs/` 路径。
