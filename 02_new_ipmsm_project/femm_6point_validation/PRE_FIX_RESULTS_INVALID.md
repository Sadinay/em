# 修复前结果无效通知

2026-08-21发现：旧拓扑生成器在把原Air/Iron标签改为永磁体时，没有同步设置磁化方向。`candidate_002/air0` 的128个PM标签中有76个方向错误。因此以下结果只保留作故障追踪，不能继续用于历史参数判断：

- 根目录旧 `validation_results.csv`；
- `generated_fem/` 中修复前模型及其结果；
- `extended_30deg/` 的3–30°结果；
- 旧版三种材料映射排名。

有效结果请使用 `corrected_pm_direction/README.md` 及其子目录。
