from __future__ import annotations

import torch
from torch import nn


class StructureCNN(nn.Module):
    """Conv2D over FEM material raster, with T_min as a separate scalar input."""

    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        # Adaptive pooling gives 16×4×4=256 image features, plus T_min.
        self.regressor = nn.Sequential(
            nn.Linear(16 * 4 * 4 + 1, 384),
            nn.ReLU(),
            nn.Linear(384, 1),
        )

    def forward(self, image: torch.Tensor, t_min: torch.Tensor) -> torch.Tensor:
        encoded = self.image_encoder(image)
        combined = torch.cat((encoded, t_min), dim=1)
        return self.regressor(combined)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
