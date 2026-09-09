"""Instantiate all nine V3 models and record parameter/output-shape checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
CNN_ROOT = ROOT / "cnn_zone"
if str(CNN_ROOT) not in sys.path:
    sys.path.insert(0, str(CNN_ROOT))
from src.model_registry import MODEL_CLASSES  # noqa: E402
from src.models_v2 import parameter_count  # noqa: E402


OUTPUT = ROOT / "reports" / "spmsm_inputs" / "model_matrix_audit.json"


def main() -> None:
    torch.manual_seed(20260903)
    rows = []
    for name, model_class in MODEL_CLASSES.items():
        model = model_class().eval()
        shape = (1, 2, 6, 20) if name.startswith("logical6x20/") else (1, 8, 224, 224)
        with torch.inference_mode():
            output = model(torch.zeros(shape, dtype=torch.float32))
        rows.append(
            {
                "experiment": name,
                "input_shape": list(shape),
                "output_shape": list(output.shape),
                "trainable_parameters": parameter_count(model),
                "predicts": ["Tavg", "DeltaT"],
                "forward_check": bool(tuple(output.shape) == (1, 2)),
            }
        )
        del model
    summary = {
        "status": "all_forward_checks_passed" if all(row["forward_check"] for row in rows) else "failed",
        "experiment_count": len(rows),
        "v2_weight_transfer_allowed": False,
        "logical_adaptation": "input changed from V2 3x10x10 to binary 2x6x20; adaptive feature grid is 3x10",
        "models": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
