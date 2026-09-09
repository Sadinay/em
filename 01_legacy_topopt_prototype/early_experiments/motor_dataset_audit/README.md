# Motor Dataset Audit

用于只读审计杂乱的 FEMM/MATLAB 电机仿真数据目录。工具不会修改、移动、删除或重命名输入目录中的任何文件；所有结果写入单独输出目录。

## 环境

- Python 3.10+
- Windows / Linux

安装：

```bash
python -m pip install -r requirements.txt
```

## 完整运行

在本项目目录执行：

```bash
python run_audit.py --input "C:\path\to\raw_data" --output "C:\path\to\dataset_audit_output"
```

主要阶段：

1. 扫描文件、快速哈希和精确重复检测。
2. 安全解析 MATLAB v5–v7.2 与 v7.3/HDF5 文件、FEMM 文本文件。
3. 依据文件名、运行标签、目录和时间生成透明配对分数。
4. 生成几何/仿真标识和 CNN 数据集清单草案。
5. 生成 Markdown 报告、低分辨率预览和 HTML 人工复核页。

## 断点续跑

已有阶段 CSV 会作为缓存读取：

```bash
python run_audit.py --input "..." --output "..." --from-stage inspect
python run_audit.py --input "..." --output "..." --from-stage match
```

用 `--force` 强制重算。日志写入输出目录的 `audit.log`。

## 检查样本

```bash
python inspect_sample.py --output "C:\path\to\dataset_audit_output" --sample-id SAMPLE_0001
```

也可以直接打开输出目录中的 `sample_review.html`，按状态或关键词筛选。

## 输出

工具生成需求中指定的扫描、MAT、FEM、配对、manifest 与报告文件，另含：

- `audit.log`：运行日志。
- `audit_config_used.yaml`：本次运行配置快照。
- `design_group_summary.csv`：按精确几何签名聚合的仿真数量。

CSV 使用 UTF-8 BOM，便于 Windows Excel 直接打开。JSON 和 Markdown 使用 UTF-8。

## 重要限制

- 几何预览中的直线段是准确的，圆弧目前用端点虚线近似，仅供核对解析结果。
- 匹配分数是候选证据，不替代人工确认。
- `geometry_id` 是严格内容签名；几何仅有数值微小差异时会被视作不同设计。
- 从文件名提取的电流、转速和角度只是初步元数据，必须结合脚本/MAT 内容复核。
