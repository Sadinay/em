# 03 FEMM 配置恢复

本机统一配置已从 `FEMM_results_20260913/source_snapshot/femm_zone/femm_config.py` 逐字节恢复，解决旧的0°初始内角/转矩−2与已验证结果包不一致的问题。

| 设置 | 修复后 |
|---|---|
| 机械行程 `ANG_R` | 0、3、6、9、12、15° |
| 滑动气隙内角 | `29° + ANG_R`，即29、32、35、38、41、44° |
| 外角 | 固定0° |
| 电流 | MAT中的3.5 A；三相cos，初始电角0°，电角增量为机械行程的4倍 |
| 初始29°偏置 | 只加入气隙内角，不加入电流角 |
| 问题定义 | 静磁、平面、mm、深度36、精度1e−8、Min Angle=15°、Smart Mesh开启 |
| 转矩 | 原始气隙积分×1；六点算术平均；波动为max−min，单位N·m |

29°来自导师补充的代码；电流幅值、频率、机械角速度、极对数及深度从原始MAT读取。

## 防止旧设置被继续使用

`run_femm.solve()` 在启动 FEMM 前比较准备目录与当前统一配置的电流、问题定义、气隙、行程、初相位和转矩倍率。不一致便拒绝求解，并要求换名称重新 `prepare`。旧运行记录保留原样。四组种子的 `pilot.py` 沿用原有冻结参数和来源哈希检查，现已重新通过。

基线导入入口也已改为直接调用统一配置，移除了导入期间临时覆盖电流的代码。原始标签、CNN预测、指标和历史导入报告保持原样，兼容性记录见[基线配置修复说明](../../../experiments/input_distribution_pilot_v1/post_femm_baseline_20260913/physics_fix_20260913.md)。Git属性为这一配置文件固定LF换行，防止Windows检出时改变已登记的配置哈希。

## 离线验证

- FEMM工作流20项测试通过，包括9类旧配置或缺字段在导入FEMM/COM前被拒绝。
- 17份冻结种子文件与来源校验通过，去重队列仍为3299个基因。
- 使用修复后的统一配置重新生成此前跨设备抽查的2个基因、共12个角度的输入；求解前FEM哈希及三相电流逐点等于收到的结果记录。
- 复核保留的G2～G5共24组FEM/ANS哈希，重新汇总既有原始转矩；均值和峰峰差与MAT参考值的最大绝对差为1.252e−12 N·m。此项是既有结果回归检查。
- 没有启动新FEMM求解、CNN推理或训练。

详细数据见[verification.json](verification.json)。修复前备份为[before_femm_config.py.txt](before_femm_config.py.txt)，恢复来源记录见[restoration.json](restoration.json)。

修复前SHA-256：`17bcee0c96180595f323f698d65891d844f1beff7c24069dae294574fd279264`。

修复后SHA-256：`52459cf11c89ee1589f4d3a11a7e86e92450ac14ea082037a94a6c86e8a7835f`，与已核验快照相同。

从 `em` 根目录查看配置或运行离线测试：

```powershell
python ./03_new_spmsm_project/femm_zone/run_femm.py
python -m pytest ./03_new_spmsm_project/femm_zone/tests/test_femm_workflow.py -q
```
