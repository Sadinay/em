# FEMM区

该区域负责从候选基因建立IPMSM模型、执行FEMM求解并提取物理结果。

- `models/`：原始模型和经过版本管理的基础模型；
- `scripts/`：建模、角度扫描、求解、重试和后处理代码；
- `workspaces/`：每个个体的唯一临时工作目录；
- `results/`：经验证的物理结果文件；
- `logs/`：FEMM、fkern、超时、重试和失败日志。

后续批处理必须继续采用独立工作目录、超时、有限重试和断点恢复。

## 200位基因还原为FEM拓扑

入口脚本：`scripts/build_femm_topology.py`

当前采用参考 `models/IPMSM.fem` 审计得到的三材料映射：

```text
00 -> 编码0 -> Air
01 -> 编码1 -> Permanent magnet
10 -> 编码2 -> Pure Iron
11 -> 编码3 -> Pure Iron
```

100个材料单元按下面的顺序排列：

```text
cell_index = angular_index * 10 + radial_index
```

每个单元通过 `workspace_600.mat/MaterialPosition[cell]` 中的四组坐标，写入参考FEM的四个对称block label。固定几何、绕组、边界条件、材料参数、磁化方向和FEM求解设置均沿用参考文件。

### 从MAT历史生成一个模型

生成第600代最高适应度个体，并执行隐藏窗口的打开和建网格检查：

```powershell
python femm_zone\scripts\build_femm_topology.py `
  --state-index 600 `
  --output-dir femm_zone\results\generated_examples\best_generation_600 `
  --mesh-check
```

指定种群行，例如第10行（Python从0开始）：

```powershell
python femm_zone\scripts\build_femm_topology.py `
  --state-index 600 `
  --population-row 10 `
  --output-dir femm_zone\results\generated_examples\state600_row10
```

### 从外部基因文件生成

支持包含单条200位基因的JSON、NPZ、TXT或CSV：

```powershell
python femm_zone\scripts\build_femm_topology.py `
  --gene-file data_zone\exports\readable_example_best_current.json `
  --output-dir femm_zone\results\generated_examples\external_gene `
  --mesh-check
```

也可以直接传入200字符的01字符串：

```powershell
python femm_zone\scripts\build_femm_topology.py `
  --gene "这里放200位01字符串" `
  --output-dir femm_zone\results\generated_examples\direct_gene
```

### 每个输出目录

- `model.fem`：可由FEMM直接打开的完整模型；
- `manifest.json`：来源、模板哈希、映射版本、校验结果及物理拓扑哈希；
- `gene_bits.txt`：输入的原始200位基因；
- `canonical_physical_bits.txt`：将编码3统一归并为铁编码2后的规范基因；
- `material_codes_10x10.csv`：原始0/1/2/3网格；
- `physical_classes_10x10.csv`：空气0、磁钢1、铁2的物理网格；
- `gene_to_fem_mapping.csv`：100个单元到400个FEM标签的逐项映射。

脚本同时计算两个缓存键：

- `genotype_sha256`：区分原始编码2和3；
- `physical_topology_sha256`：将2和3视为相同的铁，可避免物理重复求解。

`--mesh-check` 只进行隐藏窗口的FEMM打开和网格生成，不执行磁场求解。完整角度扫描和优化评价仍应在后续独立runner中实现。
