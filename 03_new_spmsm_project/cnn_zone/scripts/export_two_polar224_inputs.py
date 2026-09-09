"""Export exact 224x224 RGB previews of the polar90 and polar360 tensors."""

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
        [0.73, 0.87, 0.98],  # design air
        [0.70, 0.06, 0.22],  # PM inward
        [0.98, 0.22, 0.24],  # PM outward
        [0.82, 0.91, 0.98],  # fixed air
        [0.50, 0.50, 0.50],  # fixed iron
        [0.93, 0.57, 0.13],  # winding
        [0.60, 0.35, 0.72],  # other
    ],
    dtype=np.float32,
)


def composite(tensor: np.ndarray) -> np.ndarray:
    material = tensor[:7]
    rgb = np.einsum("chw,ck->hwk", material, COLORS)
    coverage = material.sum(axis=0)
    rgb += np.maximum(0.0, 1.0 - coverage)[..., None]
    return np.clip(rgb, 0.0, 1.0)


def main() -> None:
    bits_all = np.load(DATASET / "topology_bits.npy")
    targets = np.load(DATASET / "targets_tavg_delta.npy")
    sample = int(np.argsort(targets[:, 0])[len(targets) // 2])
    bits = bits_all[sample].reshape(-1)
    specifications = (
        ("polar90", "spmsm_polar90_224_full_motor.npz", "polar90_224_input.png"),
        ("polar360", "spmsm_polar360_224_full_motor_ss4.npz", "polar360_224_input.png"),
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for _, lookup, filename in specifications:
        rendered = SemanticRenderer(CNN_ROOT / "outputs" / "lookups" / lookup).render_numpy(bits)
        plt.imsave(OUTPUT / filename, composite(rendered), origin="lower")
        image = plt.imread(OUTPUT / filename)
        if image.shape[:2] != (224, 224):
            raise AssertionError(f"Expected an exact 224x224 PNG, got {image.shape}")
        print(f"{OUTPUT / filename} | sample={sample} Tavg={targets[sample,0]:.6f} DeltaT={targets[sample,1]:.6f}")


if __name__ == "__main__":
    main()
