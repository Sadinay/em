"""Read one batch from each fixed split without training a neural network."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cnn_zone.dataset import DEFAULT_DATASET, IPMSMTopologyDataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--coordinate-channels", action="store_true")
    args = parser.parse_args()
    for split in ("train", "validation", "test"):
        dataset = IPMSMTopologyDataset(args.dataset, split, args.coordinate_channels)
        inputs, targets = next(iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False)))
        if not torch.isfinite(inputs).all() or not torch.isfinite(targets).all():
            raise AssertionError("DataLoader returned NaN or Inf")
        expected_channels = 5 if args.coordinate_channels else 3
        if inputs.shape[1:] != (expected_channels, 10, 10) or targets.shape[1:] != (2,):
            raise AssertionError(f"Unexpected batch shapes: {inputs.shape}, {targets.shape}")
        print(f"{split}: samples={len(dataset):,}, x={tuple(inputs.shape)}, y={tuple(targets.shape)}")


if __name__ == "__main__":
    main()
