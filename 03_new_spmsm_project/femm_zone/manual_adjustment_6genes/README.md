# 6个SPMSM基因的可调FEMM模型

这里包含6个彼此不同的120位基因，每个基因对应一个独立 `.fem` 文件。文件名中的 `g` 是 MAT 历史代数，`row` 是 `population_all` 的种群行号，均为0-based。

## 文件和初始参数

| FEM文件 | 代数 | 行号 | PM格数 | 历史Tavg | 历史DeltaT |
|---|---:|---:|---:|---:|---:|
| `gene_g000_row338.fem` | 0 | 338 | 45 | 1.617231 | 0.350749 |
| `gene_g031_row190.fem` | 31 | 190 | 80 | 3.183133 | 0.454533 |
| `gene_g193_row258.fem` | 193 | 258 | 100 | 3.918658 | 0.213292 |
| `gene_g200_row491.fem` | 200 | 491 | 82 | 3.155746 | 0.184943 |
| `gene_g200_row395.fem` | 200 | 395 | 91 | 3.525754 | 0.474898 |
| `gene_g200_row033.fem` | 200 | 33 | 99 | 3.850157 | 0.494777 |

6个模型当前统一设置为：

- 机械角：`0°`；
- 极对数：`4`；
- 相电流幅值：`3.5 A`；
- A相：`3.5 A`；
- B相：`-1.75 A`；
- C相：`-1.75 A`；
- 滑动气隙边界：`sliding_airgap`，Inner Angle=`0°`；
- `gene=0`：Air；
- `gene=1`：N38永磁体；
- 轴向深度：`36 mm`；
- FEMM精度：`1e-8`。

这些参数以及每个基因完整的120位编码也记录在 `model_parameters_and_genes.csv` 中。

## 手动改变角度

如果机械角改为 `theta_m`，当前验证采用的同步电流为：

```text
theta_e = 4 * theta_m
Ia = 3.5*cos(theta_e)
Ib = 3.5*cos(theta_e - 120°)
Ic = 3.5*cos(theta_e + 120°)
```

同时把 `sliding_airgap` 的 `Inner Angle` 改成 `theta_m`。只改气隙角度而不改三相电流，会变成不同的电流相角工况。

## 当前转矩读取口径

自动验证使用：

```text
raw torque = mo_gapintegral("sliding_airgap", 0)
reported torque = -2 * raw torque
```

其中 `-2` 是依据历史平均转矩拟合得到的整体方向和尺度。历史 `DeltaT` 的确切公式仍未完全确认，所以手动试验时建议同时保存每个角度的原始转矩，不要只保存最终峰峰值。

本目录中的文件是可编辑副本，原始 `SPMSM_discrete.fem` 和历史验证结果没有被修改。

