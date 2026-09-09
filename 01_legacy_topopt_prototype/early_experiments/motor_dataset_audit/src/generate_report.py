from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import keyword_matches, read_csv


def generate_report(root: Path, output: Path, inventory: list[dict[str, Any]], fem: list[dict[str, Any]], mat: list[dict[str, Any]], variables: list[dict[str, Any]], pairs: list[dict[str, Any]], manifest: list[dict[str, Any]], logger) -> None:
    ext = Counter(str(r["extension"]) or "<none>" for r in inventory)
    total_bytes = sum(int(r["size_bytes"]) for r in inventory)
    fem_ok = sum(r.get("read_status") == "ok" for r in fem)
    mat_ok = sum(r.get("read_status") == "ok" for r in mat)
    statuses = Counter(str(r["match_status"]) for r in pairs)
    materials = Counter()
    for row in fem:
        for name in str(row.get("material_names", "")).split("|"):
            if name:
                materials[name] += 1
    candidate_rows = read_csv(output / "candidate_output_variables.csv")
    candidate_names = Counter(str(r["variable_name"]) for r in candidate_rows)
    physical_terms = ("torque", "t_avg", "t_ripple", "cogging", "flux", "bemf", "emf", "iron_loss", "copper_loss", "efficiency", "power")
    physical_candidates = [
        row for row in candidate_rows
        if keyword_matches(str(row["variable_path"]), physical_terms)
    ]
    duplicate_rows = read_csv(output / "duplicate_files.csv")
    duplicate_groups = len({r["duplicate_group"] for r in duplicate_rows})
    empty_files = sum(int(r["size_bytes"]) == 0 for r in inventory)
    largest = sorted(inventory, key=lambda r: int(r["size_bytes"]), reverse=True)[:5]
    geometry_ids = {str(r["geometry_id"]) for r in manifest if r.get("geometry_id")}
    operation_known = sum(bool(r.get("operating_point_id")) for r in manifest)
    high_pairs = statuses["confirmed"] + statuses["probable"]
    ready = high_pairs >= 20 and operation_known >= max(1, high_pairs // 2) and bool(physical_candidates)
    report = f"""# 电机仿真数据集审计报告

生成目录：`{output}`  
原始数据目录（只读）：`{root}`

## 1. 总体情况

- 文件：{len(inventory)} 个；总大小：{total_bytes / 1e9:.3f} GB。
- 目录：见 `directory_tree.txt`。
- FEM：{len(fem)} 个，可解析 {fem_ok} 个（{percent(fem_ok, len(fem))}）。
- MAT：{len(mat)} 个，可解析 {mat_ok} 个（{percent(mat_ok, len(mat))}）。
- 精确几何签名：{len(geometry_ids)} 个。该值是基于 FEM 节点、线段、圆弧、材料标签与深度生成的保守估计。
- 重复文件：{duplicate_groups} 组（完整 SHA-256 确认）。
- 重复文件记录：{len(duplicate_rows)} 个；空文件：{empty_files} 个；扫描不可读文件：{len(read_csv(output / "scan_errors.csv"))} 个。

最大的 5 个文件：

{markdown_table(["路径", "字节"], [(r["relative_path"], r["size_bytes"]) for r in largest])}

## 2. 文件类型分布

{markdown_table(["扩展名", "数量"], [(k, v) for k, v in ext.most_common()])}

## 3. FEM 解析结果

解析器读取 FEMM 头部设置、材料/边界/电路定义、节点、线段、圆弧和 block label，并生成少量低分辨率几何预览。发现的材料：

{markdown_table(["材料", "出现文件数"], materials.most_common())}

材料名称保留原始字符串，未擅自合并同义材料。预览只用于人工验证，圆弧当前以端点虚线近似显示，因此不应作为 CNN 输入。

## 4. MAT 解析结果与候选输出

共生成 {len(variables)} 条变量/嵌套结构记录和 {len(candidate_rows)} 条候选输出记录，其中明确物理名称候选 {len(physical_candidates)} 条。候选基于变量完整路径的大小写无关关键词匹配；候选不是最终标签。当前 MAT 主要保存优化状态和设计变量，常见候选是 `bestOverall`、`J_hist`、`best_bits` 和 `design_mask`，未发现以 torque/loss/flux 等明确物理名称保存的大规模标签。MATLAB 源码显示 `bestOverall` 是目标函数 J，而部分评估函数定义 `J = T_ripple + penalty`；`T_avg` 与 `T_ripple` 虽在计算时存在，但没有作为统一字段保存在这些结果 MAT 中。因此不能把 `bestOverall` 不加解释地视作转矩标签。

{markdown_table(["候选变量名", "出现次数"], candidate_names.most_common(30))}

对数值数组记录 shape、dtype、范围、NaN/Inf 和短预览；没有把完整大数组写入日志或 CSV。

## 5. FEM–MAT 配对

{markdown_table(["状态", "数量"], [(k, statuses.get(k, 0)) for k in ["confirmed", "probable", "uncertain", "conflict", "unmatched"]])}

高置信/较高置信配对共 {high_pairs} 个。评分依据和阈值详见 `pairing_rules.md`。工具不按目录顺序强制配对；存在多个近分候选时标为 `conflict` 或 `uncertain`。

## 6. 独立设计、工况与泄漏风险

- 基于内容的精确几何签名估计独立几何数为 {len(geometry_ids)}。
- 可从文件名初步提取工况的记录为 {operation_known}/{len(manifest)}。
- 多个仿真可能共享相同几何，且优化过程包含 trace、seed、best result、中间 FEM/ANS 等不同阶段文件。
- 训练/验证/测试必须按 `geometry_id` 分组划分，不能按单个文件随机划分；否则同一几何或同一优化运行可能跨集合，产生严重数据泄漏。
- 在人工确认配对、明确目标变量语义与单位、可靠提取电流/速度/角度前，不应把同目录文件默认视为独立样本。

## 7. CNN 可行性结论

当前是否已具备直接训练条件：**{"初步具备，但仍需人工确认标签和工况" if ready else "尚未具备"}**。

CNN 输入可从 FEM 构建二维规则栅格：每个像素表示材料/区域；还可增加边界、绕组相别、永磁体磁化方向等通道。推荐使用“离散材料 ID + one-hot 通道”，而非把材料编号当连续数值；空气、铁磁材料、永磁体、铜/铝、轴及未知材料应分通道，具体同义材料合并必须在查看材料物性后决定。

CNN 可能预测的 MAT 输出只能从 `candidate_output_variables.csv` 中选择。目前关键词候选涵盖转矩/目标函数/损耗/磁链/反电势等类别，但必须逐项确认单位、shape 和物理含义。若同一几何在不同电流、转速、转子角度下有不同结果，应将这些工况作为额外数值输入（或条件通道）；不能只给几何图。

数据集应按 `geometry_id` 分组，再尽量按优化运行/设计族分层划分训练、验证和测试。当前仍缺少或需要确认的信息包括：目标变量的物理定义与单位、MAT 中优化历史与最终标签的区别、转子角度/电流/转速的可靠来源、FEM 与 MAT 的人工抽查结果，以及材料物性同义映射。

## 8. 数据质量问题与下一步

1. 使用 `sample_review.html` 或 `python inspect_sample.py --output "{output}" --sample-id SAMPLE_XXXX` 抽查 confirmed/probable/conflict 样本。
2. 将人工结论写入外部校验表（不要修改原始数据），再冻结配对清单。
3. 明确候选目标变量及其单位、标量化/序列化方式。
4. 解析脚本中的参数与结果保存逻辑，补充电流、转速、角度和设计参数。
5. 完成材料同义词与物性映射后，再设计统一栅格分辨率和 CNN 架构。
"""
    (output / "DATASET_AUDIT_REPORT.md").write_text(report, encoding="utf-8")
    generate_review_html(output, manifest)
    logger.info("Generated Markdown report and sample review HTML")


def generate_review_html(output: Path, manifest: list[dict[str, Any]]) -> None:
    previews = list((output / "fem_previews").glob("*.png"))
    payload = []
    for row in manifest:
        nested = row.get("_nested_metadata", {})
        fem_name = Path(str(row["fem_path"])).stem
        preview = next((p.name for p in previews if fem_name[:50] in p.name), "")
        payload.append({
            "sample_id": row["sample_id"], "fem_path": row["fem_path"], "mat_path": row["mat_path"],
            "status": row["match_status"], "score": row["match_score"], "materials": row["material_names"],
            "targets": row["candidate_target_names"], "evidence": nested.get("pairing_evidence", ""),
            "conflict": nested.get("conflicting_evidence", ""), "preview": f"fem_previews/{preview}" if preview else "",
        })
    escaped_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    page = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>样本复核</title>
<style>body{font:14px system-ui;margin:24px;color:#18202a}input,select{padding:7px;margin-right:8px}.card{border:1px solid #ccd3db;border-radius:8px;padding:14px;margin:12px 0;display:grid;grid-template-columns:220px 1fr;gap:16px}.card img{max-width:210px;max-height:210px}.muted{color:#667085}.pill{padding:3px 8px;border-radius:12px;background:#e9eef5}code{word-break:break-all}h3{margin-top:0}</style></head>
<body><h1>FEM–MAT 样本复核</h1><p>这是只读审计视图；不会修改原始数据。</p>
<input id="q" placeholder="搜索路径/样本/变量" size="42"><select id="status"><option value="">全部状态</option><option>confirmed</option><option>probable</option><option>uncertain</option><option>conflict</option><option>unmatched</option></select><span id="count"></span><div id="cards"></div>
<script>const data=PAYLOAD;const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
function render(){const q=document.querySelector("#q").value.toLowerCase(),st=document.querySelector("#status").value;const rows=data.filter(x=>(!st||x.status===st)&&JSON.stringify(x).toLowerCase().includes(q));document.querySelector("#count").textContent=`${rows.length} / ${data.length}`;document.querySelector("#cards").innerHTML=rows.map(x=>`<section class="card"><div>${x.preview?`<img src="${esc(x.preview)}">`:"无预览"}</div><div><h3>${esc(x.sample_id)} <span class="pill">${esc(x.status)} · ${esc(x.score)}</span></h3><p><b>FEM</b> <code>${esc(x.fem_path)}</code><br><b>MAT</b> <code>${esc(x.mat_path)}</code></p><p><b>材料</b> ${esc(x.materials)||"—"}<br><b>候选输出</b> ${esc(x.targets)||"—"}</p><p><b>证据</b> ${esc(x.evidence)||"—"}<br><b>冲突</b> ${esc(x.conflict)||"—"}</p></div></section>`).join("")}document.querySelector("#q").oninput=render;document.querySelector("#status").onchange=render;render();</script></body></html>""".replace("PAYLOAD", escaped_json)
    (output / "sample_review.html").write_text(page, encoding="utf-8")


def percent(numerator: int, denominator: int) -> str:
    return f"{(100 * numerator / denominator):.1f}%" if denominator else "n/a"


def markdown_table(headers: list[str], rows: list[tuple[Any, ...] | list[Any]]) -> str:
    if not rows:
        return "（无）"
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |")
    return "\n".join(lines)
