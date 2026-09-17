# f1输入分布扩样

2026-09-17：外部 FEMM 结果已接收并完整校验，10000 基因 / 60000 角度。见[结果接收记录与数据入口](FEMM_RESULTS_RECEIVED_20260917.md)。

选样源码集中在[expansion.py](expansion.py)，交付与审计在[deliver.py](deliver.py)，FEMM仅用[femm_entry.py](femm_entry.py)。

[结果报告](report/REPORT.md) · [FEMM运行说明](FEMM运行说明.md) · [模型登记](../model_registry/README.md) · [状态](STATUS.json)

按顺序可恢复：`python expansion.py reference`、`generate`、`features`、`select`；正式清单已冻结，日常只使用FEMM入口。阶段均写缓存；更换模型/源码/参数会拒绝混用特征或选择状态。生成按来源块恢复，特征每512个保存，LCMD每100中心保存。
