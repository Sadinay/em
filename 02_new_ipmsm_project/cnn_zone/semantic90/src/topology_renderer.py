"""Vectorized in-memory rendering of genes onto a real FEMM geometry lookup."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


PHYSICAL_CHANNELS = (
    "design_air",
    "design_pm_inward",
    "design_pm_outward",
    "design_iron",
    "fixed_air",
    "fixed_iron",
    "winding",
    "geometry_mask",
)


class FEMSemanticRenderer(nn.Module):
    """Render `[B,100]` codes to `[B,C,H,W]` without writing images."""

    def __init__(self, lookup_path: str | Path, include_boundary: bool = False) -> None:
        super().__init__()
        with np.load(lookup_path) as lookup:
            gene_id = np.asarray(lookup["pixel_gene_id"], dtype=np.int64)
            self.register_buffer("gene_id", torch.from_numpy(np.maximum(gene_id, 0)), persistent=False)
            self.register_buffer("design_mask", torch.from_numpy(gene_id >= 0), persistent=False)
            self.register_buffer(
                "polarity", torch.from_numpy(np.asarray(lookup["magnet_polarity_map"], dtype=np.int8)), persistent=False
            )
            self.register_buffer(
                "fixed", torch.from_numpy(np.asarray(lookup["fixed_material_map"], dtype=np.int64)), persistent=False
            )
            self.register_buffer(
                "geometry", torch.from_numpy(np.asarray(lookup["geometry_mask"], dtype=np.bool_)), persistent=False
            )
            self.register_buffer(
                "boundary", torch.from_numpy(np.asarray(lookup["boundary_map"], dtype=np.bool_)), persistent=False
            )
        self.include_boundary = include_boundary

    @property
    def channel_names(self) -> tuple[str, ...]:
        return PHYSICAL_CHANNELS + (("boundary",) if self.include_boundary else ())

    def forward(self, gene_codes: torch.Tensor) -> torch.Tensor:
        if gene_codes.ndim != 2 or gene_codes.shape[1] != 100:
            raise ValueError(f"Expected [B,100] genes, got {tuple(gene_codes.shape)}")
        if torch.any((gene_codes < 0) | (gene_codes > 3)):
            raise ValueError("Gene states must be in {0,1,2,3}")
        states = gene_codes.to(torch.long)[:, self.gene_id]
        design = self.design_mask.unsqueeze(0)
        channels = [
            design & (states == 0),
            design & (states == 1) & (self.polarity.unsqueeze(0) < 0),
            design & (states == 1) & (self.polarity.unsqueeze(0) > 0),
            design & ((states == 2) | (states == 3)),
            (self.fixed == 1).unsqueeze(0).expand(len(gene_codes), -1, -1),
            (self.fixed == 2).unsqueeze(0).expand(len(gene_codes), -1, -1),
            (self.fixed == 3).unsqueeze(0).expand(len(gene_codes), -1, -1),
            self.geometry.unsqueeze(0).expand(len(gene_codes), -1, -1),
        ]
        if self.include_boundary:
            channels.append(self.boundary.unsqueeze(0).expand(len(gene_codes), -1, -1))
        return torch.stack(channels, dim=1).to(torch.float32)

