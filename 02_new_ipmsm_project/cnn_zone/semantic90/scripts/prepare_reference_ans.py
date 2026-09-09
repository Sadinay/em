"""Optionally create one cached reference ANS mesh from IPMSM.fem using FEMM."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_FEM = PROJECT / "femm_zone" / "models" / "IPMSM.fem"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "reference_mesh"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fem", type=Path, default=DEFAULT_FEM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = args.output / "reference.fem"
    shutil.copy2(args.fem, model)
    import femm

    opened = False
    try:
        femm.openfemm(1)
        opened = True
        femm.opendocument(str(model.resolve()))
        femm.mi_saveas(str(model.resolve()))
        femm.mi_analyze(1)
        answer = model.with_suffix(".ans")
        if not answer.exists():
            raise FileNotFoundError("FEMM finished without creating reference.ans")
        print(f"Reference mesh solution: {answer}")
    finally:
        if opened:
            try:
                femm.mi_close()
            except Exception:
                pass
            try:
                femm.closefemm()
            except Exception:
                pass


if __name__ == "__main__":
    main()

