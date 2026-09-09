"""Create 128x128 and 224x224 lookup tables from one cached FEMM solution mesh."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[3]
SEMANTIC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from cnn_zone.semantic90.src.fem_mesh import build_lookup  # noqa: E402


DEFAULT_FEM = PROJECT / "femm_zone" / "models" / "IPMSM.fem"
DEFAULT_MAT = PROJECT / "data_zone" / "raw" / "workspace_600.mat"
DEFAULT_ANS = PROJECT / "femm_6point_validation" / "corrected_pm_direction" / "angles_3deg" / "angle_000" / "model.ans"
DEFAULT_OUTPUT = SEMANTIC_ROOT / "outputs" / "lookups"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fem", type=Path, default=DEFAULT_FEM)
    parser.add_argument("--mat", type=Path, default=DEFAULT_MAT)
    parser.add_argument("--ans", type=Path, default=DEFAULT_ANS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 224])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for size in args.sizes:
        summary = build_lookup(args.fem, args.mat, args.ans, size, args.output / f"fem90_lookup_{size}.npz")
        summaries.append(summary)
        print(f"Built {size}x{size}: design_pixels={summary['design_pixels']}, mesh={summary['mesh_triangles']} triangles")
    (args.output / "lookup_build_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

