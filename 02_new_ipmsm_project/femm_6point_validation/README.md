# 历史六角度 FEMM 复现验证

> **2026-08-21重要更正：** 修复前模型有PM磁化方向缺失问题，旧 `validation_results.csv` 和 `extended_30deg` 数值无效。请以 `corrected_pm_direction/README.md` 为当前结论。修正后最佳配置的Tavg误差为0.186%，DeltaT误差为3.020%。

本目录只用于审计 `workspace_600.mat` 与 `IPMSM.fem`，并验证能否从历史染色体重建模型、按六个机械角度求解后复现历史标量。原始 MAT 和 FEM 始终只读；所有模型均由模板复制生成。

## 当前停止点

- 已从 302,904 条历史记录中按规则固定选出 8 个候选，见 `selected_candidates.csv`。
- 仅运行了 `candidate_001`、`candidate_002` 两个 smoke 样本。
- 每个 smoke 样本统一枚举 `air0`、`air2`、`air3` 三种材料映射；每种映射求解 0、3、6、9、12、15°，共 36 个角度任务。
- 36 个任务全部成功，剩余 6 个候选未运行。
- 当前未达到历史误差阈值，所以不得直接启动剩余候选或把某种材料映射视为已确认。

结论和数值见 [validation_report.md](validation_report.md)。完整公式枚举见 `validation_results.csv`，统一配置排名见 `global_configuration_ranking.csv`。

## 目录内容

- `selected_candidates.csv`：8 个候选的位置、历史标签、200 位修正前后染色体、哈希和运行阶段。
- `checkpoint.json`：逐个候选/映射/角度的状态和转矩，可断点恢复。
- `generated_fem/<candidate>/<mapping>/base/`：从参考 FEM 复制并重建的候选基础模型、基因、10×10 网格、四副本映射、预览图和 manifest。
- `generated_fem/<candidate>/<mapping>/angle_NNN/`：独立的 `.fem`、`.ans` 和 `result.json`。
- `torque_curves/`：每个候选和映射的六点原始转矩及相电流。
- `logs/`：每个角度、每次尝试的命令、耗时、stdout/stderr 和失败信息。
- `analysis_summary.json`：当前统一配置比较结论。
- `scripts/`：候选选择、拓扑重建调用、单角度求解、断点运行和结果分析。

## 可复现命令

在项目根目录执行：

```powershell
cd C:\Users\26096\Desktop\em\02_new_ipmsm_project

# 重新生成同一份候选清单（不启动 FEMM）
python femm_6point_validation\scripts\select_candidates.py

# 两例 smoke；最多三个相互隔离的 FEMM 实例，已完成角度自动跳过
python femm_6point_validation\scripts\run_parallel_smoke.py `
  --workers 3 `
  --timeout-seconds 600 `
  --maximum-attempts 2

# 只从已有六点曲线重新枚举公式并生成排名，不启动 FEMM
python femm_6point_validation\scripts\analyze_smoke_results.py

# 纯 Python 测试
python -m pytest data_zone\tests femm_zone\tests femm_6point_validation\tests -q
```

Ctrl+C 后保留已完成的 `result.json` 与 checkpoint；再次运行会跳过完整角度。`.validation.lock` 使用操作系统文件锁，防止两个终端同时写入同一验证目录。单角度超时会记录日志，并在 Windows 上终止该 worker 的进程树，避免残留 FEMM 后直接重试。

## 暂不执行的命令

`run_validation.py --stage all` 会运行候选清单中的剩余样本。当前 smoke 尚未复现历史标签，因此按任务要求停止，不执行这一阶段。只有先补齐或确认历史 MATLAB 的材料映射、转子零位/角度更新和转矩后处理定义后，才应继续。

## 固定求解设置

- 机械角：`[0, 3, 6, 9, 12, 15]°`
- 极对数：`P=4`
- 电流幅值：`Is_amp=3.5 A`
- 电角：`theta_e=P*theta_m`
- 三相电流：余弦相位 `0°/-120°/+120°`
- 转子角：`mi_modifyboundprop("sliding_airgap", 10, theta_m)`
- 转矩：`mo_gapintegral("sliding_airgap", 0)`
- 每个角度从同一个干净候选基础模型复制，不累积旋转。

材料候选都固定代码 1 为永磁体，分别令 0、2 或 3 为空气，其余两个代码为铁。它们是当前两个源文件允许枚举的三种物理候选，不是三种可混用的样本级参数。
