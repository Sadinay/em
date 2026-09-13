# FEMM 配置恢复与基线兼容性

本机 `femm_zone/femm_config.py` 已恢复为结果包 `source_snapshot` 中的已验证原文件，SHA-256：

`52459cf11c89ee1589f4d3a11a7e86e92450ac14ea082037a94a6c86e8a7835f`

工况为：内角 29°、32°、35°、38°、41°、44°，外角固定 0°；3.5 A 三相电流从电角 0° 正向推进，电角增量为机械行程的四倍，29° 初始机械偏置不加入电流角；Min Angle=15°；转矩倍率 1；平均转矩为六点算术平均，波动为六点最大值减最小值，单位 N·m。

首次导入时的本机物理文件 SHA-256 为 `17bcee0c96180595f323f698d65891d844f1beff7c24069dae294574fd279264`，属于 0°/−2 的旧配置。原导入按结果包冻结工况重建输入，并临时覆盖本机旧电流公式，核对了 19,794 份求解前模型哈希。因此本次恢复配置不改变已导入标签，也不要求重新推理旧 CNN。

此次只调整基线入口的兼容性：

- `prepared_fem()` 直接调用唯一的 `current_physics.configure_fem()`，删除额外修改三相电流的临时补丁。电流核验调用同一模块，同时用导师公式作独立比较。
- 核验器仅接受上述已验证物理文件哈希，且要求它同时等于包内快照、原 contract 和原清单的冻结预期。其他冻结输入仍必须与首次导入时哈希一致，不允许任意修改。
- `verify` 只读取既有标签、预测和指标，另写 `physics_fix_20260913_verification.json`；不调用 FEMM、不运行推理，不覆盖历史 `output_verification.json`。
- 当前 README 和将来生成报告的文字改为修复后状态。

原 `import_manifest.json`、`data_audit.md/.json`、`BASELINE_REPORT.md`、`received/femm_config.diff`、模型清单、预测、指标、图表、可视化核验以及 `OUTPUT_CHECKSUMS.json` 保留原样。这些文件中关于“当前 0°/−2、尚未修复”的表述属于首次导入时的历史记录。历史输出校验清单中的 `baseline.py` 与 README 哈希也仍指向修复前版本；当前两个入口文件的哈希及其与历史版本的差别记录在新的核验状态文件中。

从 em 根目录运行日常复核：

```powershell
python ./03_new_spmsm_project/experiments/input_distribution_pilot_v1/post_femm_baseline_20260913/baseline.py verify
```

`audit`、`evaluate`、`report` 和 `all` 会重新生成对应输出；本次修复未执行这些命令。当前验证结果见 [独立修复核验状态](physics_fix_20260913_verification.json)。

修复后已通过 `verify`：既有训练视图、公共 dev、9,382 条预测及指标、test 隔离和原包/冻结输入校验全部通过，历史输出仅当前入口 `baseline.py` 与 README 发生预期变化。另离线重建先前抽查的两个基因共 12 个角度，其求解前 FEM 哈希和三相电流与收到的记录逐点完全一致；注入未授权物理文件或其他冻结输入哈希变化时，核验器均拒绝通过。该复核没有重新求解或推理。
