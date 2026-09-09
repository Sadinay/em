from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_JSON = (
    PROJECT_ROOT
    / "femm_zone"
    / "results"
    / "history_replay_validation_minangle25"
    / "dense_1deg_summary.json"
)
OUTPUT_PNG = PROJECT_ROOT / "reports" / "6genes_torque_waveforms_minangle25.png"


def main() -> None:
    with INPUT_JSON.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    candidates = payload["candidates"]
    angles = np.arange(16, dtype=float)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
        }
    )

    fig, axes = plt.subplots(3, 2, figsize=(15, 13), constrained_layout=True)
    fig.suptitle(
        "Min Angle = 25°：6个基因的FEMM转矩波形（0–15°，步长1°）",
        fontsize=17,
        fontweight="bold",
    )

    for axis, item in zip(axes.flat, candidates, strict=True):
        torque = np.asarray(item["full_machine_sign_corrected_torques_nm"], dtype=float)
        mean_torque = float(torque.mean())
        t_min = float(torque.min())
        t_max = float(torque.max())
        abs_ripple = t_max - t_min
        calculated_ripple = abs_ripple / abs(mean_torque)
        reference_ripple = float(item["historical_delta_t"])
        relative_difference = abs(calculated_ripple - reference_ripple) / abs(reference_ripple)

        state = int(item["state_index_0based"])
        row = int(item["population_row_0based"])

        axis.plot(
            angles,
            torque,
            marker="o",
            markersize=4.5,
            linewidth=2.0,
            label="FEMM转矩",
        )
        axis.axhline(
            mean_torque,
            linestyle="--",
            linewidth=1.2,
            color="0.35",
            label=f"16点均值 = {mean_torque:.4f} N·m",
        )
        axis.scatter(
            [int(np.argmin(torque)), int(np.argmax(torque))],
            [t_min, t_max],
            marker="s",
            s=44,
            zorder=4,
            label="最小值/最大值",
        )

        axis.set_title(
            f"第{state}代 / 第{row}行\n"
            f"计算相对波动={calculated_ripple:.4f}，MAT参考={reference_ripple:.4f}，"
            f"相对差异={relative_difference:.1%}",
            fontsize=11,
        )
        axis.set_xlabel("机械角 (°)")
        axis.set_ylabel("转矩 (N·m)")
        axis.set_xticks(np.arange(0, 16, 1))
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", fontsize=8)

        y_span = max(t_max - t_min, 0.05)
        axis.set_ylim(t_min - 0.18 * y_span, t_max + 0.23 * y_span)
        axis.text(
            0.02,
            0.03,
            f"Tmin={t_min:.4f}，Tmax={t_max:.4f}\n峰峰值={abs_ripple:.4f} N·m",
            transform=axis.transAxes,
            fontsize=8.5,
            verticalalignment="bottom",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "alpha": 0.78, "edgecolor": "0.75"},
        )

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
