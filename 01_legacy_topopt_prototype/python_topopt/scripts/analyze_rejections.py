from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from constraints.connectivity import historical_precheck  # noqa: E402
from data_io.matlab import load_trace_all  # noqa: E402


TRACE = (
    WORKSPACE_ROOT
    / "FP"
    / "IA_test_Float"
    / "trace_all_20260306_174158.mat"
)
REPORT_DIR = PROJECT_ROOT / "reports"


def _genes(array: np.ndarray) -> list[int]:
    return [int(value) for value in array.tolist()]


def _example(
    trace,
    generation: int,
    member: int,
    label: str,
) -> dict[str, object]:
    genes = trace.populations[generation, member]
    return {
        "label": label,
        "generation_matlab": generation + 1,
        "member_matlab": member + 1,
        "objective": float(trace.objectives[generation, member]),
        "stored_floating_iron_cells": int(trace.floating_iron_cells[generation, member]),
        "stored_small_copper_cells": int(trace.small_copper_cells[generation, member]),
        "genes": _genes(genes),
        "cache_key": "".join(str(int(value)) for value in genes),
    }


def build_analysis() -> dict[str, object]:
    trace = load_trace_all(TRACE)
    shape = trace.objectives.shape
    current_iron = np.zeros(shape, dtype=int)
    current_copper = np.zeros(shape, dtype=int)
    for generation in range(trace.generations):
        for member in range(trace.population_size):
            check = historical_precheck(trace.populations[generation, member])
            current_iron[generation, member] = check.iron.n_floating
            current_copper[generation, member] = check.copper.n_floating

    stored_fe = trace.floating_iron_cells > 0
    stored_cu = trace.small_copper_cells > 0
    rejected = stored_fe | stored_cu
    both = stored_fe & stored_cu
    fe_only = stored_fe & ~stored_cu
    cu_only = stored_cu & ~stored_fe
    neither = ~rejected

    unique_all = {
        tuple(row) for row in trace.populations.reshape(-1, trace.populations.shape[-1])
    }
    unique_rejected = {
        tuple(trace.populations[generation, member])
        for generation, member in np.argwhere(rejected)
    }
    unique_valid = unique_all - unique_rejected

    per_generation = []
    for generation in range(trace.generations):
        per_generation.append(
            {
                "generation_matlab": generation + 1,
                "rejected": int(np.count_nonzero(rejected[generation])),
                "floating_iron_any": int(np.count_nonzero(stored_fe[generation])),
                "small_copper_any": int(np.count_nonzero(stored_cu[generation])),
                "both": int(np.count_nonzero(both[generation])),
                "iron_only": int(np.count_nonzero(fe_only[generation])),
                "copper_only": int(np.count_nonzero(cu_only[generation])),
                "valid": int(np.count_nonzero(neither[generation])),
            }
        )

    # The trace lacks operator provenance.  Hamming distance to the previous
    # generation is therefore only a diagnostic grouping, not exact attribution.
    transition_bands = {
        "exact_previous": [0, 0],
        "local_1_45": [1, 45],
        "middle_46_70": [46, 70],
        "random_like_gt_70": [71, 180],
    }
    transition_stats = {
        name: {"positions": 0, "rejected": 0}
        for name in transition_bands
    }
    for generation in range(1, trace.generations):
        previous = trace.populations[generation - 1]
        for member in range(trace.population_size):
            distance = int(
                np.min(np.count_nonzero(previous != trace.populations[generation, member], axis=1))
            )
            for name, (low, high) in transition_bands.items():
                if low <= distance <= high:
                    transition_stats[name]["positions"] += 1
                    transition_stats[name]["rejected"] += int(rejected[generation, member])
                    break
    for values in transition_stats.values():
        values["rejection_rate"] = (
            values["rejected"] / values["positions"] if values["positions"] else None
        )

    historical_trace_files = sorted(
        (WORKSPACE_ROOT / "FP").rglob("trace_all_*.mat")
    )
    no_copper_count = 0
    readable_trace_count = 0
    for path in historical_trace_files:
        try:
            older = load_trace_all(path)
        except (KeyError, OSError, ValueError):
            continue
        readable_trace_count += 1
        no_copper_count += int(
            np.count_nonzero(np.all(older.populations != 2, axis=2))
        )

    first_cu_only = tuple(int(x) for x in np.argwhere(cu_only)[0])
    first_both = tuple(int(x) for x in np.argwhere(both)[0])
    first_valid = tuple(int(x) for x in np.argwhere(neither)[0])
    duplicate_valid = (1, 0)

    iron_matches = current_iron == trace.floating_iron_cells
    mismatch_keys = {
        tuple(trace.populations[generation, member])
        for generation, member in np.argwhere(~iron_matches)
    }
    current_rejected = (current_iron > 0) | (current_copper > 0)
    current_objective = 1_000_000 + 50 * current_iron + 50 * current_copper
    historical_rejected = trace.objectives >= 1_000_000

    return {
        "source_trace": str(TRACE),
        "population_positions": int(rejected.size),
        "unique_chromosomes": len(unique_all),
        "unique_rejected_chromosomes": len(unique_rejected),
        "unique_valid_chromosomes": len(unique_valid),
        "stored_rejection_counts": {
            "rejected": int(np.count_nonzero(rejected)),
            "floating_iron_any": int(np.count_nonzero(stored_fe)),
            "small_copper_any": int(np.count_nonzero(stored_cu)),
            "both": int(np.count_nonzero(both)),
            "iron_only": int(np.count_nonzero(fe_only)),
            "copper_only": int(np.count_nonzero(cu_only)),
            "valid": int(np.count_nonzero(neither)),
        },
        "per_generation": per_generation,
        "transition_distance_diagnostic": transition_stats,
        "current_source_regression": {
            "copper_count_matches": int(
                np.count_nonzero(current_copper == trace.small_copper_cells)
            ),
            "iron_count_matches": int(np.count_nonzero(iron_matches)),
            "iron_count_mismatches": int(np.count_nonzero(~iron_matches)),
            "unique_iron_mismatch_chromosomes": len(mismatch_keys),
            "rejection_decision_matches": int(
                np.count_nonzero(current_rejected == historical_rejected)
            ),
            "invalid_objective_matches_using_current_source": int(
                np.count_nonzero(
                    current_objective[historical_rejected]
                    == trace.objectives[historical_rejected]
                )
            ),
            "invalid_objective_matches_using_stored_counters": int(
                np.count_nonzero(
                    (
                        1_000_000
                        + 50 * trace.floating_iron_cells[historical_rejected]
                        + 50 * trace.small_copper_cells[historical_rejected]
                    )
                    == trace.objectives[historical_rejected]
                )
            ),
        },
        "historical_category_availability": {
            "readable_trace_files_scanned": readable_trace_count,
            "no_copper_positions_found": no_copper_count,
            "low_average_torque_identifiable": False,
            "reason": "trace MAT files do not store T_avg, T_ripple, or six torque samples",
        },
        "examples": [
            _example(trace, *first_valid, "valid/high-performance and elite"),
            _example(trace, *duplicate_valid, "duplicate of elite in next generation"),
            _example(trace, *first_cu_only, "small copper only by stored counters"),
            _example(trace, *first_both, "floating iron plus small copper"),
        ],
    }


def write_reports(analysis: dict[str, object]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "rejection_analysis.json"
    json_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    counts = analysis["stored_rejection_counts"]
    regression = analysis["current_source_regression"]
    availability = analysis["historical_category_availability"]
    transitions = analysis["transition_distance_diagnostic"]
    lines = [
        "# 900/1000 无效个体原因分析",
        "",
        f"数据源：`{analysis['source_trace']}`。统计单位是保存的种群位置；重复染色体仍按其在每代出现的位置计数。",
        "",
        "## 结论",
        "",
        f"- 1000 个位置中，{counts['rejected']} 个被拒绝，{counts['valid']} 个有效。",
        f"- 小铜岛触发 {counts['small_copper_any']} 个，浮铁触发 {counts['floating_iron_any']} 个。",
        f"- 同时触发两条规则 {counts['both']} 个；仅小铜岛 {counts['copper_only']} 个；仅浮铁 {counts['iron_only']} 个。",
        "- 因而在这份 trace 中，铜规则参与了全部 900 次拒绝；不能据此断言铜编码在其他种群中必然是唯一主因。",
        f"- 共 {analysis['unique_chromosomes']} 个唯一染色体；其中 {analysis['unique_rejected_chromosomes']} 个唯一无效、{analysis['unique_valid_chromosomes']} 个唯一有效。",
        "- 每一代均为 9/10 被拒绝。100 个有效位置实际都是同一个精英染色体的重复保留。",
        "",
        "## 当前源码与历史输出的回归边界",
        "",
        f"- 小铜岛单元数：{regression['copper_count_matches']}/1000 精确一致。",
        f"- 浮铁单元数：{regression['iron_count_matches']}/1000 精确一致；{regression['iron_count_mismatches']} 个位置、{regression['unique_iron_mismatch_chromosomes']} 个唯一染色体不一致。",
        f"- 最终拒绝布尔判定：{regression['rejection_decision_matches']}/1000 一致，因为上述不一致位置仍被小铜岛规则拒绝。",
        f"- 用历史 MAT 内保存的两种计数代入 `1e6 + 50*nFloatCu + 50*nFloatFe`：{regression['invalid_objective_matches_using_stored_counters']}/900 精确一致。",
        f"- 用当前源码规则重新计算完整无效 J：{regression['invalid_objective_matches_using_current_source']}/900 一致。",
        "",
        "这说明目标公式和铜岛实现已被历史数据验证；现存 `detect_floating_iron.m` 不能解释全部历史浮铁计数。文件内没有算法版本号或运行时源码快照，因此该差异记录为源码/输出版本漂移，不能通过擅改 Python 规则来掩盖。",
        "",
        "## 各代统计",
        "",
        "| 代 | 拒绝 | 浮铁 | 小铜岛 | 同时 | 仅铁 | 仅铜 | 有效 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["per_generation"]:
        lines.append(
            f"| {row['generation_matlab']} | {row['rejected']} | {row['floating_iron_any']} | "
            f"{row['small_copper_any']} | {row['both']} | {row['iron_only']} | "
            f"{row['copper_only']} | {row['valid']} |"
        )
    lines.extend(
        [
            "",
            "## 随机移民与超变异诊断",
            "",
            "历史 trace 没有保存个体来源标签或随机状态，因此不能把某行精确归因为克隆或移民。下面只按它与上一代所有个体的最小 Hamming 距离分组：",
            "",
            "| 距离诊断组 | 位置数 | 拒绝数 | 拒绝率 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, values in transitions.items():
        rate = values["rejection_rate"]
        lines.append(
            f"| `{name}` | {values['positions']} | {values['rejected']} | {rate:.1%} |"
        )
    lines.extend(
        [
            "",
            "除与上一代某个体完全相同的行外，所有 441 个新结构位置均被拒绝；88 个距离大于 70 的随机型位置也全部被拒绝。这支持“初始化/变异分布与可行域严重不匹配”，但不是精确的操作来源归因。兼容模式不据此改变规则。",
            "",
            "## 代表性染色体",
            "",
            "以下均为 MATLAB 线性顺序的完整 180 位数组。",
            "",
        ]
    )
    for example in analysis["examples"]:
        lines.extend(
            [
                f"### {example['label']}",
                "",
                f"代 {example['generation_matlab']}，个体 {example['member_matlab']}，"
                f"J={example['objective']}，历史浮铁={example['stored_floating_iron_cells']}，"
                f"历史小铜岛={example['stored_small_copper_cells']}。",
                "",
                "```text",
                str(example["genes"]),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## 不存在或不可识别的历史类别",
            "",
            f"- 扫描 {availability['readable_trace_files_scanned']} 个可读 `trace_all_*.mat`，未找到不含铜的历史位置（计数 {availability['no_copper_positions_found']}）。合法铁/空气结构只能用合成测试覆盖，不能冒充历史样本。",
            "- 历史 trace 未保存 `T_avg`、`T_ripple` 或六点转矩数组，因此不能从这些文件识别“平均转矩低于 0.8 Nm”的历史个体。该分支只做手工数值单元测试。",
            "- 当前 trace 没有“仅浮铁”样本；对应规则由合成连通区域测试验证。",
            "",
        ]
    )
    (REPORT_DIR / "REJECTION_ANALYSIS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    result = build_analysis()
    write_reports(result)
    print(json.dumps(result["stored_rejection_counts"], ensure_ascii=False))
