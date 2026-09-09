"""Gene-code dataset for batch-time FEM semantic rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class GeneCodeDataset(Dataset):
    """Return `[100]` material codes and `[Tavg,DeltaT]`; render after batching."""

    def __init__(self, cleaned_dir: str | Path, split: str = "train") -> None:
        cleaned_dir = Path(cleaned_dir)
        grids = np.load(cleaned_dir / "topology_codes.npy", mmap_mode="r")
        self.codes = grids.swapaxes(1, 2).reshape(len(grids), 100)
        self.targets = np.load(cleaned_dir / "targets_tavg_delta.npy", mmap_mode="r")
        split_codes = np.load(cleaned_dir / "split_codes.npy", mmap_mode="r")
        split_value = {"train": 0, "validation": 1, "test": 2}.get(split)
        if split == "all":
            self.indices = np.arange(len(grids), dtype=np.int64)
        elif split_value is not None:
            self.indices = np.flatnonzero(split_codes == split_value)
        else:
            raise ValueError(f"Unknown split {split!r}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        codes = torch.from_numpy(np.array(self.codes[index], dtype=np.int64, copy=True))
        target = torch.from_numpy(np.array(self.targets[index], dtype=np.float32, copy=True))
        return codes, target

