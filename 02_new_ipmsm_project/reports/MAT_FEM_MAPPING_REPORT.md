# workspace_600.mat 转换与基因—FEM 结构映射报告

审计日期：2026-08-19

## 结论摘要

1. `workspace_600.mat` 是 MATLAB 5.0 MAT 文件，已经能够由 Python/Scipy 完整读取，不需要 MATLAB 许可证。
2. 染色体长度为 200 位。相邻两位编码一个材料单元，因此每条染色体对应 100 个材料单元：

   ```text
   material[i] = 2 * bit[2*i] + bit[2*i+1]

   00 -> 0
   01 -> 1
   10 -> 2
   11 -> 3
   ```

   该公式对第 600 代全部 504 个个体逐元素验证通过。
3. 100 个材料单元组成 10×10 设计域。MATLAB 向量首先按角度分组，每组包含由内到外的 10 个半径位置。CNN 文件已转换为：

   ```text
   material_grid[sample, radial_index, angular_index]
   ```

4. `MaterialPosition` 的形状为 `[100, 8]`。每一行的 8 个数是四组 `(x,y)` 坐标，对应同一基因单元在 FEM 四个对称区域中的四个标签位置。
5. 100 行 `MaterialPosition` 的 400 个坐标，全部在 `IPMSM.fem` 的 block label 中精确找到；没有缺失，且同一单元的四个副本物理材料一致。因此“基因索引到 FEM 空间位置”的关系已经验证。
6. `IPMSM.fem` 示例的 100 个设计单元包含 62 个铁、20 个永磁体和 18 个空气单元。
7. 该 FEM 示例不是 MAT 历史种群中可以直接找到的某一个个体。对全部 302,904 条历史记录搜索后，在保持编码 1 为永磁体的前提下，所有简单的 `0/1/2/3 -> 空气/磁钢/铁` 映射均没有得到完全一致的结构；最佳情况仍有 32 个单元不同。因此不能把 FEM 示例本身当作某条历史染色体的标签。

## 推荐给 CNN 使用的文件

### `data_zone/processed/current_generation.npz`

第 600 代的 504 条记录。主要字段：

| 字段 | 形状 | 含义 |
|---|---:|---|
| `genome_bits_raw` | `[504,200]` | 原始二进制染色体 |
| `genome_bits_corrected` | `[504,200]` | 约束/材料修正后的染色体 |
| `material_grid_raw` | `[504,10,10]` | 原始材料编码网格 |
| `material_grid_corrected` | `[504,10,10]` | 推荐的 CNN 结构输入 |
| `t_avg_nm` | `[504]` | MAT 中保存的平均转矩标量 |
| `delta_t_nm` | `[504]` | MAT 中保存的 `DeltaT` 标量 |
| `fitness` | `[504]` | MAT 中保存的适应度 |
| `volume_pm_cells` | `[504]` | 永磁体单元计数 |

当前代有 504 条记录、442 个唯一的修正后拓扑。目标范围：

| 目标 | 最小值 | 中位数 | 最大值 | 平均值 |
|---|---:|---:|---:|---:|
| `Tavg` (Nm) | 1.5773 | 2.5022 | 3.0394 | 2.5081 |
| `DeltaT` (Nm) | 0.4793 | 0.8068 | 1.9218 | 0.8486 |
| `Fitvalue` | 4.0221 | 4.9791 | 5.0805 | 4.9328 |

### `data_zone/processed/generation_best.npz`

第 1 到 600 代，每代一条最高适应度个体，共 600 条。保存的 `Bestpopulation` 与每代最高适应度行对应的“修正后染色体”600/600 完全一致。

### `data_zone/processed/full_history.npz`

全部 601 个状态（初始状态加 600 代）×504 个体，共 302,904 条记录。数组主要形状为：

```text
material_grid_corrected: [601, 504, 10, 10]
t_avg_nm:                [601, 504]
delta_t_nm:              [601, 504]
fitness:                 [601, 504]
```

其中只有 170,251 个唯一修正后染色体，重复记录为 132,653 条。因此训练前必须按染色体哈希去重或分组，不能随机逐行划分，否则相同结构会跨入训练集与测试集。

同一修正后拓扑在少量历史位置存在不同的物理标量：`Tavg`/`DeltaT` 有 334 个重复组出现差异，`Fitvalue` 有 347 个重复组出现差异。可能原因包括历史数组的时序、修正前后评价顺序或 FEM 求解状态；在获得配套 MATLAB 源码前，完整历史应作为“待清洗原始数据”，不能直接宣称 302,904 条均为可靠独立监督样本。

## Python/PyTorch 读取示例

```python
import numpy as np
import torch

data = np.load("data_zone/processed/current_generation.npz")

# 离散材料图：[N, 10, 10]
grid = data["material_grid_corrected"].astype(np.int64)

# 四类 one-hot：[N, 4, 10, 10]
x = torch.nn.functional.one_hot(
    torch.from_numpy(grid), num_classes=4
).permute(0, 3, 1, 2).float()

# 可选择一个或多个回归目标
y_tavg = torch.from_numpy(data["t_avg_nm"]).float().unsqueeze(1)
y_delta_t = torch.from_numpy(data["delta_t_nm"]).float().unsqueeze(1)
```

不建议把 10×10 网格插值成自然图像尺寸。它本身就是规则的离散设计域，直接 one-hot 后进行二维卷积即可。

## 从基因到结构的准确顺序

```text
200 bit chromosome
        ↓ 相邻两位解码
100 material-code vector
        ↓ cell = angular_index*10 + radial_index
10 radial positions × 10 angular positions
        ↓ MaterialPosition[cell, :]
four (x,y) FEM block labels per cell
        ↓ 同一编码赋给四个对称副本
IPMSM quarter-machine design domain
```

空间关系图见 `reports/gene_to_structure_mapping.png`。左图每格数字为 1-based 基因单元编号；右图显示每个单元在 FEM 中的四个对称标签位置。

## 材料编码能确认到什么程度

- 编码 `1` 可以确认与永磁体计数有关：第 600 代原始材料矩阵中，编码 1 的数量与 `VolumePM` 对全部 504 个体完全相等。
- `IPMSM.fem` 物理上只观察到空气、永磁体和铁三类设计域材料，而染色体存在四个编码值。
- 缺少 MAT 中函数句柄指向的配套源码，例如原路径中的 `fitness_GA5.m` 和材料赋值主程序。因此目前不能严谨确认编码 `0`、`2`、`3` 分别如何归并成空气或铁，也不能排除其中包含冗余编码或约束修正状态。
- CNN 可以把 0/1/2/3 当作四个离散类别进行 one-hot 学习；但在重新调用 FEMM 前，必须先取得源代码或用一组已知染色体—FEM 配对做实验验证，不能凭猜测写死物理材料名。

## 其他已读出的数据

- 种群规模：504
- 记录代数：600；历史数组含初始状态，共 601 列
- `pcrossover = 0.5`
- `pmutation = 0.1`
- 转矩相关参数：`steps = 6`、`stps = 72`
- 电机主要尺寸：转子外径 61.6 mm、定子内径 63 mm、轴向深度 36 mm
- FEM：FEMM 4.0、静磁场、平面问题、精度 `1e-8`、411 个 block label

`DeltaT` 的准确公式和 `Fitvalue` 的完整目标函数不能只靠工作区变量唯一还原。MAT 内保存的匿名函数只显示：

```matlab
@(x) 1./(1+exp(-50*(x-0.99)))
```

这只是适应度计算中的 sigmoid 函数，不等于完整目标函数。

## 可复现命令

在项目根目录运行：

```powershell
python data_zone\scripts\convert_workspace.py --include-full-history
python femm_zone\scripts\analyze_ipmsm_structure.py
```

两个脚本均只读原始 `.mat` 和 `.fem`，不会覆盖原始文件。
