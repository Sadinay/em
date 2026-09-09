"""Verify batched gene loading and in-memory FEM semantic rendering."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT = Path(__file__).resolve().parents[3]
SEMANTIC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from cnn_zone.semantic90.src.dataset import GeneCodeDataset  # noqa: E402
from cnn_zone.semantic90.src.topology_renderer import FEMSemanticRenderer  # noqa: E402


DEFAULT_LOOKUP = SEMANTIC_ROOT / "outputs" / "lookups" / "fem90_lookup_224.npz"
DEFAULT_DATASET = PROJECT / "data_zone" / "processed" / "ipmsm_topology_dataset" / "training_corrected_physical_three_state"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookup", type=Path, default=DEFAULT_LOOKUP)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    renderer = FEMSemanticRenderer(args.lookup)
    for split in ("train", "validation", "test"):
        dataset = GeneCodeDataset(args.dataset, split)
        codes, targets = next(iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False)))
        with torch.inference_mode():
            images = renderer(codes)
        assert images.shape == (len(codes), 8, 224, 224)
        assert targets.shape == (len(codes), 2)
        assert torch.isfinite(images).all() and torch.isfinite(targets).all()
        print(f"{split}: genes={tuple(codes.shape)}, images={tuple(images.shape)}, targets={tuple(targets.shape)}")


if __name__ == "__main__":
    main()
