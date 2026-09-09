"""GPU-side input conversion for logical and FEM semantic representations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn


def logical6x20(bits: torch.Tensor) -> torch.Tensor:
    grid = bits.reshape(-1, 20, 6).transpose(1, 2).long()
    return functional.one_hot(grid, num_classes=2).permute(0, 3, 1, 2).float()


class TorchSemanticRenderer(nn.Module):
    """Render `[B,120]` binary genes into `[B,8,224,224]` on the GPU."""

    def __init__(self, lookup_path: str | Path) -> None:
        super().__init__()
        with np.load(lookup_path) as lookup:
            supersampled = "subpixel_gene_id" in lookup.files
            if supersampled:
                gene = np.asarray(lookup["subpixel_gene_id"], dtype=np.int64)
                polarity = np.asarray(lookup["subpixel_magnet_polarity_map"], dtype=np.int8)
                fixed = np.asarray(lookup["subpixel_fixed_material_map"], dtype=np.int64)
                geometry = np.asarray(lookup["subpixel_geometry_mask"], dtype=np.float32)
            else:
                gene = np.asarray(lookup["pixel_gene_id"], dtype=np.int64)[..., None]
                polarity = np.asarray(lookup["magnet_polarity_map"], dtype=np.int8)[..., None]
                fixed = np.asarray(lookup["fixed_material_map"], dtype=np.int64)[..., None]
                geometry = np.asarray(lookup["geometry_mask"], dtype=np.float32)[..., None]
            coordinate_system = str(lookup["coordinate_system"].item())
        self.register_buffer("gene", torch.from_numpy(np.maximum(gene, 0)), persistent=False)
        self.register_buffer("design", torch.from_numpy(gene >= 0), persistent=False)
        self.register_buffer("polarity", torch.from_numpy(polarity), persistent=False)
        fixed_channels = np.stack([(fixed == value).mean(axis=-1) for value in (1, 2, 3, 4)]).astype(np.float32)
        self.register_buffer("fixed_channels", torch.from_numpy(fixed_channels), persistent=False)
        self.register_buffer("geometry", torch.from_numpy(geometry.mean(axis=-1)), persistent=False)
        self.sample_count = gene.shape[-1]
        self.coordinate_system = coordinate_system

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        if bits.ndim != 2 or bits.shape[1] != 120:
            raise ValueError(f"Expected [B,120], got {tuple(bits.shape)}")
        states = bits.long()[:, self.gene]
        design = self.design.unsqueeze(0)
        polarity = self.polarity.unsqueeze(0)
        air = (design & (states == 0)).float().mean(dim=-1)
        inward = (design & (states == 1) & (polarity < 0)).float().mean(dim=-1)
        outward = (design & (states == 1) & (polarity > 0)).float().mean(dim=-1)
        batch = len(bits)
        fixed = self.fixed_channels.unsqueeze(0).expand(batch, -1, -1, -1)
        geometry = self.geometry.unsqueeze(0).unsqueeze(0).expand(batch, -1, -1, -1)
        return torch.cat((air[:, None], inward[:, None], outward[:, None], fixed, geometry), dim=1)


class AngularCircularConv(nn.Module):
    """Use circular padding only along theta (width), zero padding radially."""

    def __init__(self, convolution: nn.Conv2d) -> None:
        super().__init__()
        self.convolution = convolution
        padding = convolution.padding if isinstance(convolution.padding, tuple) else (convolution.padding, convolution.padding)
        self.radial_padding, self.angular_padding = padding
        convolution.padding = (0, 0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.angular_padding:
            inputs = functional.pad(inputs, (self.angular_padding, self.angular_padding, 0, 0), mode="circular")
        if self.radial_padding:
            inputs = functional.pad(inputs, (0, 0, self.radial_padding, self.radial_padding))
        return self.convolution(inputs)


def enable_angular_circular_padding(module: nn.Module) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d) and any(child.padding):
            setattr(module, name, AngularCircularConv(child))
        else:
            enable_angular_circular_padding(child)
    return module
