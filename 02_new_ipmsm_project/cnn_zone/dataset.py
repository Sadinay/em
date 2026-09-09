"""PyTorch adapter for the cleaned 10x10 IPMSM topology dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    ROOT
    / "data_zone"
    / "processed"
    / "ipmsm_topology_dataset"
    / "training_corrected_physical_three_state"
)
Split = Literal["train", "validation", "test", "all"]


class IPMSMTopologyDataset(Dataset):
    """Memory-mapped categorical topology data with dynamic one-hot encoding.

    Inputs are returned as ``[C, radius=10, angle=10]``.  The default three
    channels are Air, permanent magnet and Pure Iron.  Optional coordinate
    channels are fixed in [-1, 1] and are intended only for an ablation study.
    """

    def __init__(
        self,
        dataset_dir: str | Path = DEFAULT_DATASET,
        split: Split = "train",
        add_coordinate_channels: bool = False,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.topologies = np.load(self.dataset_dir / "topology_codes.npy", mmap_mode="r")
        self.targets = np.load(self.dataset_dir / "targets_tavg_delta.npy", mmap_mode="r")
        self.split_codes = np.load(self.dataset_dir / "split_codes.npy", mmap_mode="r")
        if self.topologies.ndim != 3 or self.topologies.shape[1:] != (10, 10):
            raise ValueError(f"Unexpected topology shape: {self.topologies.shape}")
        if self.targets.shape != (len(self.topologies), 2):
            raise ValueError(f"Unexpected target shape: {self.targets.shape}")
        if np.any(self.topologies > 2):
            raise ValueError("Default physical dataset must contain only codes 0, 1 and 2")
        split_lookup = {"train": 0, "validation": 1, "test": 2}
        if split == "all":
            self.indices = np.arange(len(self.topologies), dtype=np.int64)
        elif split in split_lookup:
            self.indices = np.flatnonzero(self.split_codes == split_lookup[split])
        else:
            raise ValueError(f"Unknown split: {split}")
        self.split = split
        self.add_coordinate_channels = add_coordinate_channels
        radius = torch.linspace(-1.0, 1.0, 10).view(1, 10, 1).expand(1, 10, 10)
        angle = torch.linspace(-1.0, 1.0, 10).view(1, 1, 10).expand(1, 10, 10)
        self.coordinate_channels = torch.cat((radius, angle), dim=0)

    @property
    def input_channels(self) -> int:
        return 5 if self.add_coordinate_channels else 3

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_index = int(self.indices[item])
        grid = torch.from_numpy(np.array(self.topologies[sample_index], dtype=np.int64, copy=True))
        inputs = functional.one_hot(grid, num_classes=3).permute(2, 0, 1).to(torch.float32)
        if self.add_coordinate_channels:
            inputs = torch.cat((inputs, self.coordinate_channels), dim=0)
        target = torch.from_numpy(np.array(self.targets[sample_index], dtype=np.float32, copy=True))
        return inputs, target

