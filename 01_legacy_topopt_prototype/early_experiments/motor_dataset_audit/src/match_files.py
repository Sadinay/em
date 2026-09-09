from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .common import canonical_stem, numeric_tokens, write_csv


PAIR_FIELDS = [
    "sample_id", "fem_path", "mat_path", "match_score", "match_status",
    "matching_evidence", "conflicting_evidence",
]


def parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def tag_tokens(stem: str) -> set[str]:
    lower = stem.casefold()
    tokens = set(re.findall(r"\d{8}[_-]?\d{6}|\d{4}[_-]?\d{4}|\d{4,}", lower))
    # Common optimization run tags such as 0103_groupB_5gen or 20260109_165414.
    group = re.search(r"(\d{4}[_-]group[a-z0-9]+[_-]\d+gen)", lower)
    if group:
        tokens.add(group.group(1).replace("-", "_"))
    return tokens


def score_pair(fem: dict[str, Any], mat: dict[str, Any], suffixes: list[str], close_seconds: float) -> tuple[int, list[str], list[str]]:
    fem_path = Path(str(fem["relative_path"]))
    mat_path = Path(str(mat["relative_path"]))
    fem_stem = fem_path.stem.casefold()
    mat_stem = mat_path.stem.casefold()
    evidence: list[str] = []
    conflict: list[str] = []
    score = 0
    if fem_stem == mat_stem:
        score += 80
        evidence.append("文件 stem 完全相同")
    fem_clean = canonical_stem(fem_path.stem, suffixes)
    mat_clean = canonical_stem(mat_path.stem, suffixes)
    if fem_clean and fem_clean == mat_clean and fem_stem != mat_stem:
        score += 65
        evidence.append(f"清理后 stem 相同: {fem_clean}")
    ftags, mtags = tag_tokens(fem_stem), tag_tokens(mat_stem)
    shared_tags = sorted(ftags & mtags)
    if shared_tags:
        score += min(55, 35 + 5 * len(shared_tags))
        evidence.append("共享运行/时间标签: " + ", ".join(shared_tags))
    fnumbers, mnumbers = numeric_tokens(fem_stem), numeric_tokens(mat_stem)
    shared_numbers = sorted(fnumbers & mnumbers)
    if shared_numbers:
        score += min(20, 5 * len(shared_numbers))
        evidence.append("共享数字标识: " + ", ".join(shared_numbers))
    if fem_path.parent == mat_path.parent:
        score += 15
        evidence.append("位于同一目录")
    elif fem_path.parent in mat_path.parents or mat_path.parent in fem_path.parents:
        score += 8
        evidence.append("位于父子目录")
    elif fem_path.parent.parent == mat_path.parent.parent:
        score += 4
        evidence.append("位于相邻目录")
    ftime = parse_iso(str(fem.get("modified_time", "")))
    mtime = parse_iso(str(mat.get("modified_time", "")))
    if ftime and mtime:
        seconds = abs((ftime - mtime).total_seconds())
        if seconds <= 120:
            score += 12
            evidence.append(f"修改时间相差 {seconds:.0f} 秒")
        elif seconds <= 3600:
            score += 8
            evidence.append(f"修改时间相差 {seconds / 60:.1f} 分钟")
        elif seconds <= close_seconds:
            score += 3
            evidence.append(f"修改时间相差 {seconds / 3600:.1f} 小时")
        elif score < 40:
            conflict.append(f"修改时间相差 {seconds / 86400:.1f} 天")
    return min(100, score), evidence, conflict


def match_files(inventory: list[dict[str, Any]], output: Path, config: dict[str, Any], logger) -> list[dict[str, Any]]:
    match_cfg = config.get("matching", {})
    suffixes = [str(x).casefold() for x in match_cfg.get("removable_suffixes", [])]
    confirmed_threshold = int(match_cfg.get("confirmed_score", 85))
    probable_threshold = int(match_cfg.get("probable_score", 60))
    margin = int(match_cfg.get("ambiguity_margin", 8))
    close_seconds = float(match_cfg.get("time_close_seconds", 86400))
    fems = [r for r in inventory if str(r["extension"]).casefold() == ".fem"]
    mats = [r for r in inventory if str(r["extension"]).casefold() == ".mat"]
    pairs: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    matched_mat_paths: set[str] = set()
    for index, fem in enumerate(sorted(fems, key=lambda x: str(x["relative_path"]).casefold()), 1):
        ranked = []
        for mat in mats:
            score, evidence, conflict = score_pair(fem, mat, suffixes, close_seconds)
            if score > 0:
                ranked.append((score, str(mat["relative_path"]), evidence, conflict))
        ranked.sort(key=lambda item: (-item[0], item[1].casefold()))
        top = ranked[0] if ranked else None
        second = ranked[1] if len(ranked) > 1 else None
        if top is None or top[0] < 25:
            status, mat_path, score = "unmatched", "", top[0] if top else 0
            evidence = top[2] if top else []
            conflict = (top[3] if top else []) + (["最高候选分低于最低证据阈值"] if top else ["没有发现候选"])
        else:
            score, mat_path, evidence, conflict = top
            gap = score - second[0] if second else 100
            if second and gap < margin:
                status = "conflict" if score >= probable_threshold else "uncertain"
                conflict = conflict + [f"次优候选 {second[1]} 得分 {second[0]}，分差仅 {gap}"]
                for candidate in ranked[:5]:
                    if candidate[0] >= max(25, score - margin):
                        ambiguous.append({
                            "fem_path": fem["relative_path"], "candidate_mat_path": candidate[1],
                            "score": candidate[0], "matching_evidence": "；".join(candidate[2]),
                            "conflicting_evidence": "；".join(candidate[3]),
                        })
            elif score >= confirmed_threshold:
                status = "confirmed"
            elif score >= probable_threshold:
                status = "probable"
            else:
                status = "uncertain"
            matched_mat_paths.add(mat_path)
        pairs.append({
            "sample_id": f"SAMPLE_{index:04d}",
            "fem_path": fem["relative_path"],
            "mat_path": mat_path,
            "match_score": score,
            "match_status": status,
            "matching_evidence": "；".join(evidence),
            "conflicting_evidence": "；".join(conflict),
        })
    unmatched_fems = [p for p in pairs if p["match_status"] == "unmatched"]
    unmatched_mats = [
        {"mat_path": row["relative_path"]}
        for row in mats if str(row["relative_path"]) not in matched_mat_paths
    ]
    write_csv(output / "fem_mat_pairs.csv", pairs, PAIR_FIELDS)
    write_csv(output / "unmatched_fem_files.csv", unmatched_fems, PAIR_FIELDS)
    write_csv(output / "unmatched_mat_files.csv", unmatched_mats, ["mat_path"])
    write_csv(output / "ambiguous_pairs.csv", ambiguous, ["fem_path", "candidate_mat_path", "score", "matching_evidence", "conflicting_evidence"])
    write_pairing_rules(output / "pairing_rules.md", config)
    logger.info("Pairing: %s", dict(status_counts(pairs)))
    return pairs


def status_counts(pairs: list[dict[str, Any]]) -> list[tuple[str, int]]:
    statuses = ["confirmed", "probable", "uncertain", "conflict", "unmatched"]
    return [(status, sum(p["match_status"] == status for p in pairs)) for status in statuses]


def write_pairing_rules(path: Path, config: dict[str, Any]) -> None:
    cfg = config.get("matching", {})
    path.write_text(
        """# FEM–MAT 配对规则

本工具只提出候选，不把目录顺序当作配对依据。每个 FEM 对全部 MAT 计算透明分数：

| 证据 | 分值 |
|---|---:|
| stem 完全相同 | +80 |
| 去除 result/results/output/analysis/ans/solution/data/fem 后缀后相同 | +65 |
| 共享时间戳或优化运行标签 | +35～55 |
| 共享数字 ID | 每个 +5，最多 +20 |
| 同目录 | +15 |
| 父子目录 | +8 |
| 相邻目录 | +4 |
| 修改时间差 ≤2 分钟 / ≤1 小时 / ≤配置窗口 | +12 / +8 / +3 |

得分封顶 100。默认分类：

- `confirmed`：得分 ≥ {confirmed}，且领先次优候选至少 {margin} 分。
- `probable`：得分 ≥ {probable}，且不存在近分候选。
- `uncertain`：有候选但证据不足。
- `conflict`：存在分差小于 {margin} 的多个强候选。
- `unmatched`：没有候选，或最高候选分低于 25。

时间差很大且缺少其他强证据时记入冲突证据。评分不使用 MAT 内容参数相等这一尚未可靠提取的信号，因此结果偏保守，人工复核页应作为最终确认入口。
""".format(
            confirmed=cfg.get("confirmed_score", 85),
            probable=cfg.get("probable_score", 60),
            margin=cfg.get("ambiguity_margin", 8),
        ),
        encoding="utf-8",
    )
