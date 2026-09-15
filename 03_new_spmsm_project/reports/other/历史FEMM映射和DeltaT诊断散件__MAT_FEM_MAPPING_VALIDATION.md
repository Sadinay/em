# SPMSM 120位基因—FEMM结构映射与历史重算验证

日期：2026-09-02

> **2026-09-02 诊断更正：** 本报告首次把 MAT 中的 `DeltaT` 当作单位为 N·m 的绝对峰峰值比较。后续检查发现，本地同一台 8极12槽 SPMSM 的 MATLAB 扫描代码把转矩脉动定义为 `(max(T)-min(T))/abs(Tavg)`。按相对脉动率重算后，6个样本的平均误差由158.77%降至34.20%，但仍未完全复现。完整排查和已排除项见 [DELTAT_MISMATCH_DIAGNOSIS.md](<历史FEMM映射和DeltaT诊断散件__DELTAT_MISMATCH_DIAGNOSIS.md>)。本报告中所有“重算峰峰值”仍是绝对量，仅保留作审计记录，不应再视为与历史 `DeltaT` 同口径。

## 结论摘要

1. `workspace_200.mat` 中的染色体是120位二进制向量，不是02项目的200位双比特材料编码。
2. 120位组成 `6径向 × 20角向` 的设计域：

   ```text
   gene_index_0based = angular_index * 6 + radial_index
   ```

   每个角向位置内，6个径向单元由内向外排列；角向索引增加时，第一对称区域的物理极角递减。
3. `MaterialPosition[120,8]` 每行包含四组 `(x,y)`，对应同一基因在FEM四个对称区域中的四个block label。全部480个坐标均在 `SPMSM_discrete.fem` 中精确找到，最大误差为 `1.52×10⁻¹⁴ mm`，且一一指向 `c1…c120`。
4. 材料语义为：

   ```text
   gene=0 → Air
   gene=1 → N38 permanent magnet
   ```

   直接证据是末代612个个体中，`VolumePM_all` 与染色体1的数量逐个完全一致（612/612）；完整历史中为122,474/123,012一致。FEM模板同时提供Air、N38和逐单元 `c1…c120` 属性。
5. 生成新拓扑时只替换 `cN` 的材料参数，不修改原始几何、block label位置、四个副本的磁化方向、绕组、边界或网格设置。
6. 对6个代表基因、11个不同机械角共完成66次FEMM求解，66/66成功。FEMM原始转矩乘以 `−2` 后，历史平均转矩与重算值高度一致：相关系数 `r=0.999568`、过原点比例斜率 `1.003779`。
7. 3°六点方案下，6个样本的平均转矩平均相对误差为1.275%，中位误差0.431%；其中5/6低于0.615%，只有初始代极低转矩样本为6.044%。这构成基因—结构映射正确的强验证证据。
8. 历史 `DeltaT` 尚未复现。3°方案平均相对误差158.77%，2.5°六个不重复点方案仍为148.35%。因此目前只能确认结构映射和平均转矩链路，不能宣称已经确认历史 `DeltaT` 的具体后处理公式或全部求解设置。

## 原始输入

| 文件 | SHA-256 |
|---|---|
| `workspace_200.mat` | `2FBFBE8743CFE0A23C3F3C8DC54378CC82774145845F92D68555FB4EEC1270D8` |
| `SPMSM_discrete.fem` | `C6353F71C75E5D743D90717E4405337A5BCE11F2032BEB162B6EC9833D8001F5` |

原件保持只读，没有被覆盖。

## FEM与设计域静态审计

FEM文件包含：

- 平面静磁问题，轴向深度36 mm；
- 639个节点、554条线段、571条圆弧；
- 124个材料属性，其中包括 `Air`、`Copper`、`N38`、`Pure Iron` 和 `c1…c120`；
- 492个block label，其中480个属于120个基因的四个对称副本；
- A/B/C三相回路；
- `sliding_airgap` 滑动气隙边界。

径向中心为：

```text
28.05, 28.55, 29.05, 29.55, 30.05, 30.55 mm
```

第一对称区域的20个角向中心从85.637463°递减至67.965063°，步长约−0.930126°。

完整的480行映射位于 `femm_zone/results/gene_to_fem_labels.csv`，空间关系图位于 `reports/gene_to_structure_mapping.png`。

## 历史基因来源

- 推荐FEMM输入使用 `population_all`；
- `population_noChange_all` 被保留用于修正前对照；
- 完整历史123,012条记录中有117,651条修正前后完全一致；
- 本次6个验证样本全部限定为修正前后完全一致，因此本轮结论不依赖对修正顺序的猜测。

## FEMM重算配置

从MAT中读取：`P=4`、`Is_amp=3.5 A`、`steps=6`。本次使用：

```text
theta_e = 4 * theta_m
Ia = 3.5*cos(theta_e)
Ib = 3.5*cos(theta_e - 120°)
Ic = 3.5*cos(theta_e + 120°)
rotor angle = sliding_airgap inner angle
raw torque = mo_gapintegral("sliding_airgap", 0)
historical torque scale = -2 * raw FEMM torque
```

验证了两套六点取样：

- 3°含周期端点：`0,3,6,9,12,15°`；Tavg为梯形平均；
- 2.5°不重复周期点：`0,2.5,5,7.5,10,12.5°`；Tavg为算术平均。

## 3°方案逐样本结果

| 样本 | 状态/个体（0-based） | PM格数 | 历史Tavg | 重算Tavg | Tavg误差 | 历史DeltaT | 重算峰峰值 | DeltaT误差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| low_tavg | 0 / 338 | 45 | 1.617231 | 1.714984 | 6.044% | 0.350749 | 0.635642 | 81.224% |
| median_tavg | 31 / 190 | 80 | 3.183133 | 3.169296 | 0.435% | 0.454533 | 0.462370 | 1.724% |
| high_tavg | 193 / 258 | 100 | 3.918658 | 3.919595 | 0.024% | 0.213292 | 0.530058 | 148.513% |
| final_low_tavg | 200 / 491 | 82 | 3.155746 | 3.152366 | 0.107% | 0.184943 | 0.848540 | 358.813% |
| final_median_tavg | 200 / 395 | 91 | 3.525754 | 3.547435 | 0.615% | 0.474898 | 1.024468 | 115.724% |
| final_high_tavg | 200 / 33 | 99 | 3.850157 | 3.866614 | 0.427% | 0.494777 | 1.714897 | 246.600% |

平均转矩统计：平均相对误差1.275%，中位相对误差0.431%，最大6.044%。去除唯一的初始代极低转矩样本后，其余5个最大误差为0.615%。

## 2.5°不重复周期点对照

该方案的平均转矩平均相对误差为1.190%、中位误差0.248%，与3°方案同样支持结构映射；但 `DeltaT` 平均相对误差仍为148.35%，所以“3°方案重复计算15°端点”不是波动误差的主要原因。

## 对结果的判断

### 已验证

- 120位基因的空间顺序；
- 每位基因到四个FEM标签的坐标对应；
- `0=Air、1=N38 PM`；
- FEM拓扑生成、建网格、求解和滑动气隙转矩读取链路；
- 历史平均转矩使用的整体方向/尺度等价于当前FEMM原始结果乘以 `−2`；
- 至少对成熟种群样本，历史平均转矩可高精度复现。

### 尚未验证

- 历史 `DeltaT` 是峰峰值、最大偏差、标准差还是其他后处理量；
- 历史程序是否还使用未保存的转角零点、网格、材料版本或转矩波动滤波；
- 少数 `population_all` 与 `population_noChange_all` 不同记录的确切修正调用顺序；
- 初始代低转矩样本6.044%偏差的来源。

因此，03后续可以采用当前映射生成SPMSM结构，并可用平均转矩链路进行下一阶段测试；在获得历史MATLAB求解/后处理代码或原始六点转矩曲线前，不应把当前重算峰峰值与历史 `DeltaT` 直接混合作为同口径标签。

## 主要产物

- `femm_zone/scripts/spmsm_mapping.py`：120位基因生成FEM拓扑；
- `femm_zone/scripts/solve_one_angle.py`：单角度独立求解；
- `femm_zone/scripts/validate_history_replay.py`：3°历史重放；
- `femm_zone/scripts/validate_alternative_2p5deg_grid.py`：2.5°方案对照；
- `femm_zone/results/spmsm_structure_analysis.json`：结构审计数据；
- `femm_zone/results/gene_to_fem_labels.csv`：480个标签的完整对应；
- `femm_zone/results/history_replay_validation/analysis_summary.json`：3°方案详细结果；
- `femm_zone/results/history_replay_validation/alternative_2p5deg_summary.json`：2.5°方案详细结果；
- `femm_zone/results/history_replay_validation/expected_configuration_comparison.csv`：3°逐样本对照；
- `reports/gene_to_structure_mapping.png`：基因编号与四对称区域映射图。
