"""Create compact, publication-style diagnostics for a completed NSGA-II run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


MILESTONES = (1, 20, 40, 60, 80, 100)
EARLY = (1, 2, 3, 4, 5)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def truth_rows(rows: list[dict[str, str]], generation: int) -> list[dict[str, str]]:
    return [row for row in rows if row["femm_generation"] and int(float(row["femm_generation"])) <= generation]


def candidate_rows(rows: list[dict[str, str]], generation: int) -> list[dict[str, str]]:
    return [row for row in rows if int(row["first_generation"]) <= generation]


def xy(rows: list[dict[str, str]], truth: bool) -> np.ndarray:
    prefix = "femm" if truth else "pred"
    if not rows:
        return np.empty((0, 2))
    return np.asarray([[float(row[f"{prefix}_tavg_nm"]), float(row[f"{prefix}_delta_t_nm"])]
                       for row in rows], dtype=float)


def front(points: np.ndarray) -> np.ndarray:
    if not len(points):
        return np.empty((0, 2))
    indices = NonDominatedSorting().do(np.column_stack((-points[:, 0], points[:, 1])),
                                       only_non_dominated_front=True)
    result = points[indices]
    return result[np.argsort(result[:, 0])]


def row_front(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    points = xy(rows, True)
    if not len(points):
        return []
    indices = NonDominatedSorting().do(np.column_stack((-points[:, 0], points[:, 1])),
                                       only_non_dominated_front=True)
    return [rows[int(index)] for index in indices]


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "#fbfcfe",
        "axes.grid": True,
        "grid.alpha": .18,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_generation_panel(ax, rows, queue, generation: int, title: str) -> None:
    candidates = candidate_rows(rows, generation)
    predicted = xy(candidates, False)
    verified_rows = truth_rows(rows, generation)
    verified = xy(verified_rows, True)
    current_ids = {row["gene_id"] for row in queue if int(row["generation"]) == generation}
    current = xy([row for row in verified_rows if row["gene_id"] in current_ids], True)
    true_front = front(verified)
    ax.scatter(predicted[:, 0], predicted[:, 1], s=7, color="#aab2bd", alpha=.20,
               linewidths=0, label="累计 CNN 候选")
    if len(current):
        ax.scatter(current[:, 0], current[:, 1], s=30, color="#f39c12", alpha=.85,
                   edgecolors="white", linewidths=.4, label="本代 FEMM")
    if len(true_front):
        ax.plot(true_front[:, 0], true_front[:, 1], "o-", color="#1565c0", ms=4,
                lw=1.3, label="累计 FEMM 前沿")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("平均转矩 / N·m")
    ax.set_ylabel("转矩波动 ΔT / N·m（对数轴）")
    ax.text(.03, .97, f"候选 {len(candidates)}  |  FEMM {len(verified_rows)}  |  前沿 {len(true_front)}",
            transform=ax.transAxes, va="top", fontsize=8, color="#34495e")


def save_early(rows, queue, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    for ax, generation in zip(axes.flat, EARLY):
        plot_generation_panel(ax, rows, queue, generation, f"第 {generation} 代")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .055), ncol=3, frameon=False)
    fig.suptitle("前五代完整候选生成与 FEMM 验证过程", y=.995, fontsize=17, fontweight="bold")
    fig.text(.5, .018, "灰点为截至该代所有新候选的代理预测；橙点为当代真实求解；蓝线为累计真实帕累托前沿。",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, .92))
    fig.savefig(output, dpi=190)
    plt.close(fig)


def save_milestones(rows, queue, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    for ax, generation in zip(axes.flat, MILESTONES):
        plot_generation_panel(ax, rows, queue, generation, f"第 {generation} 代")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .018), ncol=3, frameon=False)
    fig.suptitle("每 20 代的搜索空间扩展与真实帕累托前沿", y=.995, fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, .065, 1, .91))
    fig.savefig(output, dpi=190)
    plt.close(fig)


def normalized_hypervolume(points: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> float:
    lower, upper = bounds
    span = np.maximum(upper - lower, 1e-12)
    normalized = (points - lower) / span
    objectives = np.column_stack((1.0 - normalized[:, 0], normalized[:, 1]))
    return float(HV(ref_point=np.array([1.05, 1.05]))(objectives))


def generation_metrics(rows: list[dict[str, str]]) -> tuple[list[dict], tuple[np.ndarray, np.ndarray]]:
    all_truth = xy([row for row in rows if row["femm_generation"]], True)
    lower, upper = all_truth.min(axis=0), all_truth.max(axis=0)
    metrics = []
    for generation in range(1, 101):
        verified_rows = truth_rows(rows, generation)
        points = xy(verified_rows, True)
        current_front = front(points)
        normalized = (current_front - lower) / np.maximum(upper - lower, 1e-12)
        distance = np.sqrt((1 - normalized[:, 0]) ** 2 + normalized[:, 1] ** 2)
        knee = current_front[int(np.argmin(distance))]
        metrics.append({
            "generation": generation,
            "verified": len(points),
            "pareto": len(current_front),
            "best_tavg_nm": float(points[:, 0].max()),
            "lowest_delta_t_nm": float(points[:, 1].min()),
            "knee_tavg_nm": float(knee[0]),
            "knee_delta_t_nm": float(knee[1]),
            "normalized_hypervolume": normalized_hypervolume(current_front, (lower, upper)),
        })
    return metrics, (lower, upper)


def save_convergence(metrics: list[dict], output: Path) -> None:
    g = np.asarray([row["generation"] for row in metrics])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    axes[0, 0].plot(g, [row["best_tavg_nm"] for row in metrics], color="#c0392b", lw=2)
    axes[0, 0].set_ylabel("最大 FEMM 平均转矩 / N·m")
    axes[0, 1].plot(g, [row["lowest_delta_t_nm"] for row in metrics], color="#16a085", lw=2)
    axes[0, 1].set_ylabel("最小 FEMM 转矩波动 / N·m")
    axes[1, 0].plot(g, [row["normalized_hypervolume"] for row in metrics], color="#6c5ce7", lw=2)
    axes[1, 0].set_ylabel("归一化超体积（越大越好）")
    axes[1, 1].plot(g, [row["pareto"] for row in metrics], color="#2980b9", lw=2, label="真实前沿点")
    axes[1, 1].plot(g, [row["verified"] for row in metrics], color="#95a5a6", lw=1.5, label="累计 FEMM")
    axes[1, 1].set_ylabel("样本数量")
    axes[1, 1].legend(frameon=False)
    for ax in axes.flat:
        ax.set_xlabel("代数")
        for milestone in MILESTONES:
            ax.axvline(milestone, color="#bdc3c7", lw=.7, ls="--", alpha=.6)
    fig.suptitle("100 代真实性能收敛曲线", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output, dpi=190)
    plt.close(fig)


def save_front_overlay(rows, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.viridis(np.linspace(.05, .95, len(MILESTONES)))
    for generation, color in zip(MILESTONES, colors):
        current = front(xy(truth_rows(rows, generation), True))
        ax.plot(current[:, 0], current[:, 1], "o-", ms=4, lw=1.3, color=color,
                label=f"第 {generation} 代（{len(current)} 点）")
    ax.set_yscale("log")
    ax.set_xlabel("平均转矩 / N·m")
    ax.set_ylabel("转矩波动 ΔT / N·m（对数轴）")
    ax.set_title("FEMM 真实帕累托前沿的推进", fontsize=16, fontweight="bold")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def representative_rows(rows: list[dict[str, str]]) -> list[tuple[str, dict[str, str]]]:
    verified = [row for row in rows if row["femm_generation"]]
    final = row_front(verified)
    torque = max(final, key=lambda row: float(row["femm_tavg_nm"]))
    ripple = min(final, key=lambda row: float(row["femm_delta_t_nm"]))
    points = xy(final, True)
    lower, upper = points.min(axis=0), points.max(axis=0)
    normalized = (points - lower) / np.maximum(upper - lower, 1e-12)
    knee_index = int(np.argmin(np.sqrt((1 - normalized[:, 0]) ** 2 + normalized[:, 1] ** 2)))
    knee = final[knee_index]
    result = [("最低波动", ripple), ("折中拐点", knee), ("最高转矩", torque)]
    unique = []
    used = set()
    for label, row in result:
        if row["gene_id"] not in used:
            unique.append((label, row)); used.add(row["gene_id"])
    return unique


def save_final_front(rows, output: Path) -> list[tuple[str, dict[str, str]]]:
    verified = [row for row in rows if row["femm_generation"]]
    points = xy(verified, True)
    final_rows = row_front(verified)
    final_points = xy(final_rows, True)
    representatives = representative_rows(rows)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(points[:, 0], points[:, 1], s=15, color="#b0bec5", alpha=.38, label="全部 FEMM 真值")
    ordered = final_points[np.argsort(final_points[:, 0])]
    ax.plot(ordered[:, 0], ordered[:, 1], "o-", color="#1565c0", ms=5, lw=1.5,
            label=f"最终真实前沿（{len(final_rows)} 点）")
    colors = ("#00897b", "#f9a825", "#c62828")
    for (label, row), color in zip(representatives, colors):
        x, y = float(row["femm_tavg_nm"]), float(row["femm_delta_t_nm"])
        ax.scatter([x], [y], s=110, marker="*", color=color, edgecolors="white", zorder=5)
        ax.annotate(f"{label}\nT={x:.4f}, ΔT={y:.4f}", (x, y), xytext=(8, 8),
                    textcoords="offset points", fontsize=9, color=color)
    ax.set_yscale("log")
    ax.set_xlabel("平均转矩 / N·m")
    ax.set_ylabel("转矩波动 ΔT / N·m（对数轴）")
    ax.set_title("第 100 代：FEMM 真值与最终帕累托前沿", fontsize=16, fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return representatives


def save_genes(representatives, output: Path) -> None:
    cmap = ListedColormap(["#f4f6f7", "#c62828"])
    fig, axes = plt.subplots(1, len(representatives), figsize=(5 * len(representatives), 4.7), squeeze=False)
    for ax, (label, row) in zip(axes.flat, representatives):
        grid = np.fromiter((int(value) for value in row["bits"]), dtype=np.uint8).reshape(20, 6).T
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, interpolation="nearest", aspect="equal")
        ax.set_xticks(np.arange(-.5, 20, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=.55)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.set_title(f"{label}\nT={float(row['femm_tavg_nm']):.4f} N·m\nΔT={float(row['femm_delta_t_nm']):.4f} N·m\n磁体格 {row['magnet_cells']}",
                     fontsize=11)
    fig.suptitle("最终真实前沿的代表性基因（红色为磁体格）", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .90))
    fig.savefig(output, dpi=210)
    plt.close(fig)


def cumulative_knee(rows: list[dict[str, str]], generation: int,
                    bounds: tuple[np.ndarray, np.ndarray]) -> dict[str, str]:
    current = row_front(truth_rows(rows, generation))
    points = xy(current, True)
    lower, upper = bounds
    normalized = (points - lower) / np.maximum(upper - lower, 1e-12)
    index = int(np.argmin(np.sqrt((1 - normalized[:, 0]) ** 2 + normalized[:, 1] ** 2)))
    return current[index]


def save_early_genes(rows, bounds, output: Path) -> None:
    cmap = ListedColormap(["#f4f6f7", "#c62828"])
    fig, axes = plt.subplots(5, 1, figsize=(12, 12))
    for ax, generation in zip(axes, EARLY):
        row = cumulative_knee(rows, generation, bounds)
        grid = np.fromiter((int(value) for value in row["bits"]), dtype=np.uint8).reshape(20, 6).T
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, interpolation="nearest", aspect="equal")
        ax.set_xticks(np.arange(-.5, 20, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=.5)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.set_ylabel(f"第 {generation} 代", rotation=0, labelpad=40, va="center", fontweight="bold")
        ax.set_title(f"累计折中代表：T={float(row['femm_tavg_nm']):.4f} N·m，"
                     f"ΔT={float(row['femm_delta_t_nm']):.4f} N·m，磁体格={row['magnet_cells']}", fontsize=10)
    fig.suptitle("前五代累计真实前沿中的折中代表基因", y=.995, fontsize=16, fontweight="bold")
    fig.text(.5, .01, "这是每代截至当时的代表方案，不表示五个方案构成一条已记录的直接父子谱系。",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(.03, .035, 1, .96))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def save_milestone_bars(metrics: list[dict], output: Path) -> None:
    selected = [metrics[generation - 1] for generation in MILESTONES]
    labels = [str(row["generation"]) for row in selected]
    torque = np.asarray([row["best_tavg_nm"] for row in selected])
    ripple = np.asarray([row["lowest_delta_t_nm"] for row in selected])
    front_count = np.asarray([row["pareto"] for row in selected])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].bar(labels, torque, color="#c0392b")
    axes[0].set_ylim(max(0, torque.min() - .03), torque.max() + .015)
    axes[0].set_ylabel("最大 FEMM 平均转矩 / N·m")
    axes[1].bar(labels, ripple, color="#16a085")
    axes[1].set_ylabel("最小 FEMM 转矩波动 / N·m")
    axes[2].bar(labels, front_count, color="#2980b9")
    axes[2].set_ylabel("累计真实帕累托前沿点数")
    for ax, values in zip(axes, (torque, ripple, front_count)):
        ax.set_xlabel("代数")
        for index, value in enumerate(values):
            text = f"{value:.4f}" if isinstance(value, np.floating) else str(value)
            ax.text(index, value, text, ha="center", va="bottom", fontsize=8)
    fig.suptitle("第 1 代及每 20 代的关键提升", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(output, dpi=190)
    plt.close(fig)


def save_prediction_check(rows, output: Path) -> None:
    verified = [row for row in rows if row["femm_generation"]]
    prediction = xy(verified, False)
    truth = xy(verified, True)
    generations = np.asarray([int(float(row["femm_generation"])) for row in verified])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    labels = (("平均转矩", 0), ("转矩波动", 1))
    scatter = None
    for ax, (label, index) in zip(axes, labels):
        lo = min(prediction[:, index].min(), truth[:, index].min())
        hi = max(prediction[:, index].max(), truth[:, index].max())
        margin = max((hi - lo) * .04, 1e-4)
        scatter = ax.scatter(truth[:, index], prediction[:, index], c=generations, cmap="viridis",
                             vmin=1, vmax=100, s=19, alpha=.60, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "--", color="#455a64", lw=1)
        error = np.abs(prediction[:, index] - truth[:, index])
        signed = prediction[:, index] - truth[:, index]
        rmse = float(np.sqrt(np.mean(signed ** 2)))
        denominator = float(np.sum((truth[:, index] - truth[:, index].mean()) ** 2))
        r_squared = 1.0 - float(np.sum(signed ** 2)) / denominator
        p95 = float(np.quantile(error, .95))
        ax.set(xlabel=f"FEMM {label} / N·m", ylabel=f"CNN {label} / N·m",
               title=label, xlim=(lo - margin, hi + margin), ylim=(lo - margin, hi + margin))
        ax.set_aspect("equal", adjustable="box")
        ax.text(.04, .96, f"n = {len(verified)}\nMAE = {error.mean():.4f} N·m\n"
                f"RMSE = {rmse:.4f} N·m\nR² = {r_squared:.4f}\n95%误差 ≤ {p95:.4f} N·m",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox={"boxstyle": "round,pad=.35", "facecolor": "white", "alpha": .86, "edgecolor": "#cfd8dc"})
    fig.colorbar(scatter, ax=axes, fraction=.035, pad=.04, label="获得 FEMM 真值的代数")
    fig.suptitle("601 个遗传候选：CNN 预测值与 FEMM 真值", fontsize=16, fontweight="bold")
    fig.text(.5, .015, "虚线为理想预测 y=x；点越贴近虚线，预测越准确。颜色表示该真值来自哪一代。",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.07, right=.90, bottom=.12, top=.86, wspace=.24)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def save_sampling_and_error_trends(rows, output: Path) -> None:
    verified = [row for row in rows if row["femm_generation"]]
    generations = np.arange(1, 101)
    counts, t_mae, d_mae, cumulative_t, cumulative_d = [], [], [], [], []
    for generation in generations:
        current = [row for row in verified if int(float(row["femm_generation"])) == generation]
        cumulative = [row for row in verified if int(float(row["femm_generation"])) <= generation]
        counts.append(len(current))
        current_pred, current_true = xy(current, False), xy(current, True)
        cumulative_pred, cumulative_true = xy(cumulative, False), xy(cumulative, True)
        t_mae.append(float(np.mean(np.abs(current_pred[:, 0] - current_true[:, 0]))) if current else np.nan)
        d_mae.append(float(np.mean(np.abs(current_pred[:, 1] - current_true[:, 1]))) if current else np.nan)
        cumulative_t.append(float(np.mean(np.abs(cumulative_pred[:, 0] - cumulative_true[:, 0]))))
        cumulative_d.append(float(np.mean(np.abs(cumulative_pred[:, 1] - cumulative_true[:, 1]))))
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    axes[0].bar(generations, counts, color="#607d8b", width=.85)
    axes[0].set_ylabel("当代 FEMM 基因数")
    axes[0].set_title("每代真实求解数量")
    axes[1].plot(generations, t_mae, color="#ef6c00", alpha=.45, lw=1, label="当代 MAE")
    axes[1].plot(generations, cumulative_t, color="#c62828", lw=2, label="累计 MAE")
    axes[1].set_ylabel("平均转矩 MAE / N·m")
    axes[1].legend(frameon=False)
    axes[2].plot(generations, d_mae, color="#26a69a", alpha=.45, lw=1, label="当代 MAE")
    axes[2].plot(generations, cumulative_d, color="#00695c", lw=2, label="累计 MAE")
    axes[2].set_ylabel("转矩波动 MAE / N·m")
    axes[2].set_xlabel("代数")
    axes[2].legend(frameon=False)
    for ax in axes:
        for milestone in MILESTONES:
            ax.axvline(milestone, color="#cfd8dc", lw=.7, ls="--")
    fig.suptitle("FEMM 采样数量与 CNN 预测误差随代数变化", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output, dpi=190)
    plt.close(fig)


def write_summary(output: Path, metrics: list[dict], representatives) -> None:
    milestones = [metrics[generation - 1] for generation in MILESTONES]
    with (output / "milestone_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(milestones[0]))
        writer.writeheader(); writer.writerows(milestones)
    first, final = milestones[0], milestones[-1]
    summary = {
        "generation_1": first,
        "generation_100": final,
        "change": {
            "best_tavg_nm": final["best_tavg_nm"] - first["best_tavg_nm"],
            "lowest_delta_t_nm": final["lowest_delta_t_nm"] - first["lowest_delta_t_nm"],
            "normalized_hypervolume": final["normalized_hypervolume"] - first["normalized_hypervolume"],
        },
        "representatives": [{
            "label": label, "gene_id": row["gene_id"], "bits": row["bits"],
            "magnet_cells": int(row["magnet_cells"]),
            "femm_generation": int(float(row["femm_generation"])),
            "tavg_nm": float(row["femm_tavg_nm"]),
            "delta_t_nm": float(row["femm_delta_t_nm"]),
        } for label, row in representatives],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# NSGA-II 100 代结果摘要", "",
        "## 主要结果", "",
        f"- 共生成 5,540 个不同候选，其中 601 个完成六角度 FEMM 验证。",
        f"- 最大平均转矩：{first['best_tavg_nm']:.4f} → {final['best_tavg_nm']:.4f} N·m。",
        f"- 最小转矩波动：{first['lowest_delta_t_nm']:.4f} → {final['lowest_delta_t_nm']:.4f} N·m。",
        f"- 最终 FEMM 真实帕累托前沿包含 {final['pareto']} 个离散候选。", "",
        "## 每 20 代检查点", "",
        "| 代数 | 累计 FEMM | 真实前沿点 | 最大平均转矩 / N·m | 最小波动 / N·m |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in milestones:
        lines.append(f"| {row['generation']} | {row['verified']} | {row['pareto']} | "
                     f"{row['best_tavg_nm']:.4f} | {row['lowest_delta_t_nm']:.4f} |")
    lines += ["", "## 图表说明", "",
              "前五代与每 20 代图可重建累计候选、当代 FEMM 样本和累计真实前沿。运行时未保存每代完整父代列表，因此不能把这些图解释为逐个体父子谱系。", "",
              "## 图表索引", "",
              "1. `01_early_generations_1_to_5.png`：前五代候选、FEMM 样本与真实前沿。",
              "2. `02_milestones_every_20_generations.png`：第 1、20、40、60、80、100 代对比。",
              "3. `03_convergence_metrics.png`：100 代性能收敛曲线。",
              "4. `04_verified_pareto_front_evolution.png`：真实帕累托前沿推进。",
              "5. `05_final_verified_pareto.png`：最终 601 个 FEMM 真值和 30 点前沿。",
              "6. `06_representative_genes.png`：最终三个代表性基因。",
              "7. `07_femm_sampling_and_error_by_generation.png`：每代 FEMM 数量及预测误差。",
              "8. `08_early_representative_genes.png`：前五代折中代表结构。",
              "9. `09_milestone_improvement_bars.png`：每 20 代关键指标。",
              "10. `10_cnn_vs_femm_601_genes.png`：601 个基因的 CNN 预测与 FEMM 真值。", "",
              "`data/` 保存本次搜索的候选、FEMM 标签、选点队列、逐代进度、运行配置和汇总指标；完整六角度 FEMM 模型仍保留在原 GA 运行目录。"]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_report_data(run: Path, output: Path) -> None:
    data = output / "data"
    data.mkdir(exist_ok=True)
    for name in ("candidates.csv", "femm_labels.csv", "femm_queue.csv", "progress.jsonl",
                 "run.json", "state.json", "seeds.csv"):
        shutil.copy2(run / name, data / name)
    for name in ("milestone_metrics.csv", "summary.json"):
        shutil.copy2(output / name, data / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    state = json.loads((run / "state.json").read_text(encoding="utf-8"))
    if state.get("stage") != "complete" or state.get("generation_completed") != 100:
        raise ValueError("analysis requires the completed 100-generation run")
    rows, queue = read_csv(run / "candidates.csv"), read_csv(run / "femm_queue.csv")
    output = (args.output.resolve() if args.output else run / "analysis")
    output.mkdir(parents=True, exist_ok=True)
    setup_style()
    save_early(rows, queue, output / "01_early_generations_1_to_5.png")
    save_milestones(rows, queue, output / "02_milestones_every_20_generations.png")
    metrics, bounds = generation_metrics(rows)
    save_convergence(metrics, output / "03_convergence_metrics.png")
    save_front_overlay(rows, output / "04_verified_pareto_front_evolution.png")
    representatives = save_final_front(rows, output / "05_final_verified_pareto.png")
    save_genes(representatives, output / "06_representative_genes.png")
    save_sampling_and_error_trends(rows, output / "07_femm_sampling_and_error_by_generation.png")
    save_early_genes(rows, bounds, output / "08_early_representative_genes.png")
    save_milestone_bars(metrics, output / "09_milestone_improvement_bars.png")
    save_prediction_check(rows, output / "10_cnn_vs_femm_601_genes.png")
    write_summary(output, metrics, representatives)
    copy_report_data(run, output)
    print(output)


if __name__ == "__main__":
    main()
