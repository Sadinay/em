"""Export same-gene FEM-like 90-degree and full-circle 224x224 previews."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CNN_ROOT = ROOT / "cnn_zone"
if str(CNN_ROOT) not in sys.path:
    sys.path.insert(0, str(CNN_ROOT))
from src.topology_renderer import SemanticRenderer  # noqa: E402


DATASET = ROOT / "data_zone" / "processed" / "spmsm_topology_dataset" / "training_corrected_binary"
OUTPUT = ROOT / "reports" / "spmsm_inputs"
COLORS = np.asarray(
    [
        [0.73, 0.87, 0.98],
        [0.70, 0.06, 0.22],
        [0.98, 0.22, 0.24],
        [0.82, 0.91, 0.98],
        [0.50, 0.50, 0.50],
        [0.93, 0.57, 0.13],
        [0.60, 0.35, 0.72],
    ],
    dtype=np.float32,
)


def composite(tensor: np.ndarray) -> np.ndarray:
    material = tensor[:7]
    rgb = np.einsum("chw,ck->hwk", material, COLORS)
    rgb += np.maximum(0.0, 1.0 - material.sum(axis=0))[..., None]
    return np.clip(rgb, 0.0, 1.0)


def hard_composite(tensor: np.ndarray) -> np.ndarray:
    material = tensor[:7]
    dominant = np.argmax(material, axis=0)
    rgb = COLORS[dominant]
    rgb[material.sum(axis=0) <= 0] = 1.0
    return rgb


def main() -> None:
    bits_all = np.load(DATASET / "topology_bits.npy")
    targets = np.load(DATASET / "targets_tavg_delta.npy")
    sample = int(np.argsort(targets[:, 0])[len(targets) // 2])
    bits = bits_all[sample].reshape(-1)
    specifications = (
        ("spmsm_xy224_full_motor.npz", "xy90_224_input.png"),
        ("spmsm_xy360_224_full_motor_ss4.npz", "xy360_224_input.png"),
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for lookup, filename in specifications:
        tensor = SemanticRenderer(CNN_ROOT / "outputs" / "lookups" / lookup).render_numpy(bits)
        preview = hard_composite(tensor)
        plt.imsave(OUTPUT / filename, preview, origin="lower")
        saved = plt.imread(OUTPUT / filename)
        if saved.shape[:2] != (224, 224):
            raise AssertionError(f"Expected 224x224, got {saved.shape}")
        if filename == "xy360_224_input.png":
            plt.imsave(OUTPUT / "xy360_224_fractional_composite.png", composite(tensor), origin="lower")
            enlarged = np.repeat(np.repeat(preview, 4, axis=0), 4, axis=1)
            plt.imsave(OUTPUT / "xy360_224_input_nearest4x.png", enlarged, origin="lower")
        print(f"{OUTPUT / filename} | sample={sample} Tavg={targets[sample,0]:.6f} DeltaT={targets[sample,1]:.6f}")


if __name__ == "__main__":
    main()
