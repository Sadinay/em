from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORT_SOURCE = ROOT / "reports" / "spmsm_topology_dataset"
OUTPUT = ROOT / "reports" / "V3" / "01_结果图" / "08_最终训练验证测试样本分布.png"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


allocation = read_rows(REPORT_SOURCE / "train40000_tavg_band_allocation.csv")
base_split = read_rows(REPORT_SOURCE / "tavg_band_split_distribution.csv")

labels = [row["tavg_band"] for row in allocation]
train = np.asarray([int(row["selected_train"]) for row in allocation])
validation = np.asarray([int(row["validation"]) for row in base_split])
test = np.asarray([int(row["test"]) for row in base_split])

series = [
    ("Train (40,000)", train, "#2f6db5"),
    ("Validation (6,483)", validation, "#e6862f"),
    ("Test (6,483)", test, "#4a9b63"),
]

x = np.arange(len(labels))
width = 0.24
fig, ax = plt.subplots(figsize=(14, 7.5), constrained_layout=True)

for index, (name, counts, color) in enumerate(series):
    percentages = counts / counts.sum() * 100.0
    bars = ax.bar(x + (index - 1) * width, percentages, width, label=name, color=color)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.45,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

ax.set_title("Final V3 split distribution by mean-torque band", fontsize=17)
ax.set_xlabel("Mean torque Tavg band (N m)", fontsize=12)
ax.set_ylabel("Share within each split (%)", fontsize=12)
ax.set_xticks(x, labels)
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=True)
ax.set_ylim(0, max(train / train.sum() * 100.0) + 7)
ax.text(
    0.01,
    0.98,
    "Numbers above bars are sample counts. Validation and test remain unchanged.",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=10,
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=220)
plt.close(fig)
print(OUTPUT)
