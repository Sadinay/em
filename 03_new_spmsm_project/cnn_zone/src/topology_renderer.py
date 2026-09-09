"""Render binary SPMSM genes into FEM-aware 8-channel semantic tensors."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # Lookups can still be built without PyTorch.
    torch = None

from .fem_mesh import FIXED_AIR, FIXED_IRON, FIXED_OTHER, FIXED_WINDING


CHANNEL_NAMES = (
    "design_air",
    "design_pm_radially_inward",
    "design_pm_radially_outward",
    "fixed_air",
    "fixed_iron",
    "winding",
    "fixed_other",
    "geometry_mask",
)


def logical_one_hot(bits):
    """Convert [...,120] binary genes to [...,2,6,20] Air/PM tensors."""
    if torch is not None and isinstance(bits, torch.Tensor):
        flat = bits.reshape(*bits.shape[:-1], 20, 6).transpose(-1, -2)
        return torch.stack(((flat == 0), (flat == 1)), dim=-3).to(dtype=torch.float32)
    values = np.asarray(bits, dtype=np.uint8)
    grid = values.reshape(*values.shape[:-1], 20, 6).swapaxes(-1, -2)
    return np.stack((grid == 0, grid == 1), axis=-3).astype(np.float32)


class SemanticRenderer:
    """Small lookup-table renderer; no 64k-image dataset is stored on disk."""

    def __init__(self, lookup_path: Path | str) -> None:
        lookup = np.load(Path(lookup_path))
        self.supersampled = "subpixel_gene_id" in lookup.files
        if self.supersampled:
            self.pixel_gene_id = np.asarray(lookup["subpixel_gene_id"], dtype=np.int64)
            self.fixed = np.asarray(lookup["subpixel_fixed_material_map"], dtype=np.uint8)
            self.geometry = np.asarray(lookup["subpixel_geometry_mask"], dtype=bool)
            self.polarity = np.asarray(lookup["subpixel_magnet_polarity_map"], dtype=np.int8)
        else:
            self.pixel_gene_id = np.asarray(lookup["pixel_gene_id"], dtype=np.int64)
            self.fixed = np.asarray(lookup["fixed_material_map"], dtype=np.uint8)
            self.geometry = np.asarray(lookup["geometry_mask"], dtype=bool)
            self.polarity = np.asarray(lookup["magnet_polarity_map"], dtype=np.int8)
        self.design = self.pixel_gene_id >= 0
        self.safe_gene_id = np.maximum(self.pixel_gene_id, 0)
        self.coordinate_system = str(lookup["coordinate_system"].item())

    def render_numpy(self, bits: np.ndarray) -> np.ndarray:
        values = np.asarray(bits, dtype=np.uint8)
        if values.shape[-1] != 120 or np.any((values != 0) & (values != 1)):
            raise ValueError("Expected binary genes with final dimension 120")
        batch_shape = values.shape[:-1]
        states = values[..., self.safe_gene_id]
        design = self.design.reshape((1,) * len(batch_shape) + self.design.shape)
        polarity = self.polarity.reshape((1,) * len(batch_shape) + self.polarity.shape)
        image_shape = self.design.shape[:2]
        channels = np.zeros(batch_shape + (8,) + image_shape, dtype=np.float32)
        reduce = (lambda array: array.mean(axis=-1)) if self.supersampled else (lambda array: array)
        channels[..., 0, :, :] = reduce(design & (states == 0))
        channels[..., 1, :, :] = reduce(design & (states == 1) & (polarity < 0))
        channels[..., 2, :, :] = reduce(design & (states == 1) & (polarity > 0))
        channels[..., 3, :, :] = reduce(self.fixed == FIXED_AIR)
        channels[..., 4, :, :] = reduce(self.fixed == FIXED_IRON)
        channels[..., 5, :, :] = reduce(self.fixed == FIXED_WINDING)
        channels[..., 6, :, :] = reduce(self.fixed == FIXED_OTHER)
        channels[..., 7, :, :] = reduce(self.geometry)
        return channels

    def render_torch(self, bits):
        if torch is None:
            raise RuntimeError("PyTorch is not installed")
        result = self.render_numpy(bits.detach().cpu().numpy() if isinstance(bits, torch.Tensor) else bits)
        device = bits.device if isinstance(bits, torch.Tensor) else None
        return torch.as_tensor(result, dtype=torch.float32, device=device)
