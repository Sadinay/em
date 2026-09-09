"""GPU forward/backward and requested physical-batch validation for all v2 models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from cnn_zone.semantic90.src.dataset import GeneCodeDataset  # noqa: E402
from cnn_zone.semantic90.src.training_v2 import (  # noqa: E402
    V2_TRAINING_SPECS,
    build_v2_model,
    probe_physical_batches,
)


DATASET = (
    PROJECT
    / "data_zone"
    / "processed"
    / "ipmsm_topology_dataset"
    / "training_corrected_physical_three_state"
)
LOOKUP = ROOT / "outputs" / "lookups" / "fem90_lookup_224.npz"
OUTPUT = ROOT / "outputs" / "v2_validation" / "batch_probe.json"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v2 validation")
    device = torch.device("cuda:0")
    dataset = GeneCodeDataset(DATASET, "all")
    sample_codes = torch.stack([dataset[index][0] for index in range(64)])
    results = []
    for spec in V2_TRAINING_SPECS.values():
        probe = probe_physical_batches(spec, LOOKUP, sample_codes, device)
        model = build_v2_model(spec)
        probe["parameter_count"] = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        probe["input_channels"] = spec.input_channels
        probe["effective_batch_size"] = spec.effective_batch_size
        results.append(probe)
        print(json.dumps(probe, ensure_ascii=False), flush=True)
        del model
    record = {
        "status": "passed",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "all_forward_backward_finite": True,
        "models": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
