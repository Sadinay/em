from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class SpatialConv2d(nn.Module):
    """Conv2d with optional circular padding along width only."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        circular_width: bool = False,
    ) -> None:
        super().__init__()
        self.padding = int(padding)
        self.circular_width = bool(circular_width)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0 if self.circular_width else self.padding,
            bias=bias,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.circular_width and self.padding:
            inputs = F.pad(inputs, (self.padding, self.padding, 0, 0), mode="circular")
            inputs = F.pad(inputs, (0, 0, self.padding, self.padding), mode="constant", value=0.0)
        return self.conv(inputs)


def conv3x3(
    in_channels: int,
    out_channels: int,
    *,
    stride: int = 1,
    circular_width: bool = False,
) -> SpatialConv2d:
    return SpatialConv2d(
        in_channels,
        out_channels,
        3,
        stride=stride,
        padding=1,
        bias=False,
        circular_width=circular_width,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        circular_width: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(
            in_channels, out_channels, stride=stride, circular_width=circular_width
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, circular_width=circular_width)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(inputs)
        output = self.relu(self.bn1(self.conv1(inputs)))
        output = self.bn2(self.conv2(output))
        return self.relu(output + identity)


class CifarResNet20Regression(nn.Module):
    def __init__(self, input_channels: int, output_dim: int, *, circular_width: bool = False) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            conv3x3(input_channels, 16, circular_width=circular_width),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._stage(16, 16, blocks=3, stride=1, circular_width=circular_width)
        self.stage2 = self._stage(16, 32, blocks=3, stride=2, circular_width=circular_width)
        self.stage3 = self._stage(32, 64, blocks=3, stride=2, circular_width=circular_width)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 5, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim),
        )
        self.apply(_initialize)

    @staticmethod
    def _stage(
        in_channels: int,
        out_channels: int,
        *,
        blocks: int,
        stride: int,
        circular_width: bool,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            BasicBlock(
                in_channels,
                out_channels,
                stride=stride,
                circular_width=circular_width,
            )
        ]
        layers.extend(
            BasicBlock(out_channels, out_channels, circular_width=circular_width)
            for _ in range(blocks - 1)
        )
        return nn.Sequential(*layers)

    def feature_shapes(self, inputs: torch.Tensor) -> dict[str, tuple[int, ...]]:
        with torch.no_grad():
            stem = self.stem(inputs)
            stage1 = self.stage1(stem)
            stage2 = self.stage2(stage1)
            stage3 = self.stage3(stage2)
        return {
            "input": tuple(inputs.shape),
            "stem": tuple(stem.shape),
            "stage1": tuple(stage1.shape),
            "stage2": tuple(stage2.shape),
            "stage3": tuple(stage3.shape),
            "flatten": (inputs.shape[0], int(stage3[0].numel())),
        }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.stage3(self.stage2(self.stage1(self.stem(inputs))))
        if output.shape[1:] != (64, 5, 3):
            raise ValueError(f"unexpected ResNet feature shape {tuple(output.shape)}")
        return self.regressor(output)


class MLPRegression(nn.Module):
    def __init__(self, input_channels: int, output_dim: int, **_: Any) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channels * 18 * 10, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim),
        )
        self.apply(_initialize)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class SmallCNNRegression(nn.Module):
    def __init__(self, input_channels: int, output_dim: int, *, circular_width: bool = False) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv3x3(input_channels, 32, circular_width=circular_width),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            conv3x3(32, 64, stride=2, circular_width=circular_width),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv3x3(64, 64, stride=2, circular_width=circular_width),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 5, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim),
        )
        self.apply(_initialize)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.features(inputs))


def _initialize(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "resnet20": CifarResNet20Regression,
    "mlp": MLPRegression,
    "small_cnn": SmallCNNRegression,
}


def build_model(
    name: str,
    *,
    input_channels: int,
    output_dim: int,
    circular_width: bool = False,
) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {name}; choose from {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](
        input_channels=input_channels,
        output_dim=output_dim,
        circular_width=circular_width,
    )
