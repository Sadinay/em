"""Build both full-motor SPMSM 224x224 coordinate representations."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CNN_ROOT = ROOT / "cnn_zone"
if str(CNN_ROOT) not in sys.path:
    sys.path.insert(0, str(CNN_ROOT))

from src.fem_mesh import build_lookup, build_supersampled_xy360_lookup  # noqa: E402


FEM = ROOT / "femm_zone" / "models" / "SPMSM_discrete.fem"
MAT = ROOT / "data_zone" / "raw" / "workspace_200.mat"
ANS = (
    ROOT
    / "femm_zone"
    / "results"
    / "history_replay_validation_minangle25"
    / "final_high_tavg"
    / "angle_0"
    / "model.ans"
)
OUTPUT = CNN_ROOT / "outputs" / "lookups"


def main() -> None:
    summaries = {}
    for coordinate_system, filename in (
        ("xy", "spmsm_xy224_full_motor.npz"),
        ("polar90", "spmsm_polar90_224_full_motor.npz"),
    ):
        summaries[coordinate_system] = build_lookup(
            fem_path=FEM,
            mat_path=MAT,
            ans_path=ANS,
            image_size=224,
            coordinate_system=coordinate_system,
            output_path=OUTPUT / filename,
        )
    summaries["xy360"] = build_supersampled_xy360_lookup(
        fem_path=FEM,
        mat_path=MAT,
        ans_path=ANS,
        image_size=224,
        supersample=4,
        output_path=OUTPUT / "spmsm_xy360_224_full_motor_ss4.npz",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
