"""Memory-mapped SPMSM gene/target dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SPMSMGeneDataset(Dataset):
    def __init__(self, dataset_dir: str | Path) -> None:
        directory = Path(dataset_dir)
        grids = np.load(directory / "topology_bits.npy", mmap_mode="r")
        if grids.shape[1:] != (6, 20):
            raise ValueError(f"Expected [N,6,20] topology bits, got {grids.shape}")
        # The stored grid is already [radial, angular]; flatten using the audited
        # gene order angular*6+radial.
        self.bits = grids.swapaxes(1, 2).reshape(len(grids), 120)
        self.targets = np.load(directory / "targets_tavg_delta.npy", mmap_mode="r")
        if self.targets.shape != (len(grids), 2):
            raise ValueError("Target array must be [N,2]")

    def __len__(self) -> int:
        return len(self.bits)

    def __getitem__(self, index: int):
        bits = torch.from_numpy(np.array(self.bits[index], dtype=np.uint8, copy=True))
        target = torch.from_numpy(np.array(self.targets[index], dtype=np.float32, copy=True))
        return torch.tensor(index, dtype=torch.int64), bits, target


class IndexedSubset(Dataset):
    def __init__(self, dataset: Dataset, indices) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        return self.dataset[int(self.indices[item])]
