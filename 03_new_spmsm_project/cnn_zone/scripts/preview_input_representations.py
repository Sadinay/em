"""Create a visual audit of logical, Cartesian and polar SPMSM inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CNN_ROOT = ROOT / "cnn_zone"
if str(CNN_ROOT) not in sys.path:
    sys.path.insert(0, str(CNN_ROOT))
from src.topology_renderer import SemanticRenderer  # noqa: E402


DATASET = ROOT / "data_zone" / "processed" / "spmsm_topology_dataset" / "training_corrected_binary"
OUTPUT = ROOT / "reports" / "spmsm_inputs" / "input_representation_preview.png"


def class_image(rendered: np.ndarray) -> np.ndarray:
    result = np.zeros(rendered.shape[1:], dtype=np.uint8)
    for channel in range(7):
        result[rendered[channel] > 0.5] = channel + 1
    return result


def main() -> None:
    bits_all = np.load(DATASET / "topology_bits.npy")
    targets = np.load(DATASET / "targets_tavg_delta.npy")
    sample = int(np.argsort(targets[:, 0])[len(targets) // 2])
    bits = bits_all[sample].reshape(-1)
    logical = bits.reshape(20, 6).T
    xy = SemanticRenderer(CNN_ROOT / "outputs" / "lookups" / "spmsm_xy224_full_motor.npz").render_numpy(bits)
    polar = SemanticRenderer(CNN_ROOT / "outputs" / "lookups" / "spmsm_polar90_224_full_motor.npz").render_numpy(bits)

    colors = ["#ffffff", "#dceeff", "#bf1d39", "#ef4444", "#cfe8ff", "#777777", "#e5962d", "#9a59b5"]
    cmap = ListedColormap(colors)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), constrained_layout=True)
    axes[0].imshow(logical, origin="lower", aspect="auto", cmap=ListedColormap(["#dceeff", "#ef4444"]), vmin=0, vmax=1)
    axes[0].set_title("Logical 6x20 gene")
    axes[0].set_xlabel("angular gene index (20)")
    axes[0].set_ylabel("radial gene index (6)")
    axes[1].imshow(class_image(xy), origin="lower", cmap=cmap, vmin=0, vmax=7)
    axes[1].set_title("xy224: complete FEM quadrant")
    axes[1].set_xlabel("physical x pixel")
    axes[1].set_ylabel("physical y pixel")
    axes[2].imshow(class_image(polar), origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=7)
    axes[2].set_title("polar90_224: complete FEM quadrant")
    axes[2].set_xlabel("physical angle 0-90 deg")
    axes[2].set_ylabel("radius 0-64 mm")
    fig.suptitle(f"SPMSM input audit, sample={sample}, Tavg={targets[sample,0]:.4f}, DeltaT={targets[sample,1]:.4f}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
