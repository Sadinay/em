# 数据区

本目录保存 `workspace_600.mat` 的只读原始文件、可追溯转换结果和 CNN 清洗数据集。

## 已确认的数据语义

- 染色体长度为 200 bit，相邻两位按高位在前解码：`code = 2*b[2i] + b[2i+1]`。
- 100 个材料单元按角度优先排列；转成 CNN 网格后形状为 `[radius=10, angle=10]`。
- `population_noChange_all` 是结构修正前的 raw gene。
- `population_all` 是结构修正后、真正送入历史 FEMM 并对应 `Tavg/DeltaT` 的 corrected gene。
- code 0 = Air，code 1 = N38 永磁体，code 2/3 = Pure Iron。
- 默认 CNN 数据将 code 2/3 合并为一个 Iron 类，输入动态转成 3 通道 one-hot。

raw 拓扑只用于修复审计和防泄漏分组。不能把 corrected 拓扑的 FEMM 标签复制给不同的 raw 拓扑，否则会制造错误的训练标签。

## 重建清洗数据集

在项目根目录执行：

```powershell
cd C:\Users\26096\Desktop\em\02_new_ipmsm_project
python data_zone\scripts\build_ipmsm_cnn_dataset.py
```

默认训练数据位于：

```text
data_zone\processed\ipmsm_topology_dataset\training_corrected_physical_three_state
```

核心文件：

- `topology_codes.npy`：`[146471,10,10] uint8`，值为 Air/PM/Iron 三类；支持 `mmap_mode="r"`。
- `targets_tavg_delta.npy`：`[146471,2]`，两列依次为 `Tavg` 和 `DeltaT`。
- `split_codes.npy`：0/1/2 分别代表 train/validation/test。
- `split_indices.npz`：三个集合的固定样本索引。
- `split_manifest.csv`：拓扑哈希、修复关联组、性能层和固定划分。
- `metadata.csv`：代数、重复次数、目标值、修复程度等审计字段。

全量 raw、四状态 corrected、三状态物理 corrected 及冲突隔离表保存在同级其他子目录。`combined_audit_four_state` 仅作 raw/corrected 并集审计，不是默认训练集。

## 通用 MAT 转换

```powershell
python data_zone\scripts\convert_workspace.py --include-full-history
```

它生成 `current_generation.npz`、`generation_best.npz` 和 `full_history.npz`，同时保留 raw 与 corrected 两个版本。

## 验证

```powershell
python cnn_zone\scripts\smoke_test_ipmsm_dataloader.py
python -m pytest -q
```

审计报告位于 `reports\ipmsm_topology_dataset`。

