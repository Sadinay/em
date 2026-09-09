"""Run the same fixed 256-sample memorization diagnostic for all six v2 models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from cnn_zone.semantic90.src.dataset import GeneCodeDataset  # noqa: E402
from cnn_zone.semantic90.src.training_v2 import (  # noqa: E402
    V2_TRAINING_SPECS,
    overfit_spec,
    target_scaler,
    train_v2_model,
)


DATASET = (
    PROJECT
    / "data_zone"
    / "processed"
    / "ipmsm_topology_dataset"
    / "training_corrected_physical_three_state"
)
LOOKUP = ROOT / "outputs" / "lookups" / "fem90_lookup_224.npz"
BATCH_PROBE = ROOT / "outputs" / "v2_validation" / "batch_probe.json"
LEGACY_SELECTED = ROOT / "outputs" / "overfit_256" / "selected_sample_indices.npy"
MODEL_ROOT = PROJECT / "cnn_zone" / "models" / "v2_overfit_256"
REPORT_ROOT = PROJECT / "reports" / "v2_overfit_256"
SEED = 20260822


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    base_dataset = GeneCodeDataset(DATASET, "all")
    selected = np.load(LEGACY_SELECTED).astype(np.int64)
    if len(selected) != 256 or len(np.unique(selected)) != 256:
        raise RuntimeError("Expected the legacy fixed 256 unique sample indices")
    targets = np.asarray(base_dataset.targets)
    mean, std = target_scaler(targets[selected])
    probe = json.loads(BATCH_PROBE.read_text(encoding="utf-8"))
    batches = {item["model_id"]: int(item["selected"]) for item in probe["models"]}
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    np.save(REPORT_ROOT / "selected_sample_indices.npy", selected)
    (REPORT_ROOT / "target_scaler.json").write_text(
        json.dumps(
            {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "source": "same fixed legacy 256 training samples only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = {
        "status": "running",
        "purpose": "memorization/pipeline diagnostic only; not generalization",
        "samples": 256,
        "seed": SEED,
        "pass_rule": "both same-sample target R2 values >= 0.97 and all values finite",
        "results": [],
    }
    summary_path = REPORT_ROOT / "summary.json"
    for spec in V2_TRAINING_SPECS.values():
        family, name = spec.model_id.split("/")
        output = MODEL_ROOT / family / name
        result_path = output / "result.json"
        if args.resume and result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            print(f"SKIP completed {spec.model_id}", flush=True)
        else:
            diagnostic_spec = overfit_spec(spec, args.max_epochs)
            result = train_v2_model(
                diagnostic_spec,
                base_dataset,
                selected,
                selected,
                selected,
                mean,
                std,
                min(batches[spec.model_id], diagnostic_spec.effective_batch_size),
                LOOKUP,
                output,
                device,
                SEED,
                stop_r2=0.985,
                disable_dropout_during_training=True,
            )
        metrics = result["metrics"]["overall"]
        values = [value for target in metrics.values() for value in target.values()]
        result["overfit_pass"] = bool(
            metrics["tavg"]["r2"] >= 0.97
            and metrics["delta_t"]["r2"] >= 0.97
            and all(np.isfinite(value) for value in values)
        )
        manifest["results"].append(result)
        summary_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["all_six_pass"] = all(item["overfit_pass"] for item in manifest["results"])
    manifest["status"] = "passed" if manifest["all_six_pass"] else "failed"
    summary_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    if not manifest["all_six_pass"]:
        raise SystemExit("One or more v2 models did not pass the 256-sample diagnostic")


if __name__ == "__main__":
    main()
