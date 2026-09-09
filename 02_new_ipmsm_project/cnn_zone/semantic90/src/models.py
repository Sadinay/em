"""CNN regressors for logical 10x10 and real-FEM 224x224 inputs."""

from __future__ import annotations

import torch
from torch import nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int, kernel_size, padding=0) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )


class InceptionBlock(nn.Module):
    """Parallel point, local, wider, radial and angular feature branches."""

    def __init__(self, input_channels: int, branch_channels: int) -> None:
        super().__init__()
        self.branch_1x1 = ConvBNReLU(input_channels, branch_channels, 1)
        self.branch_3x3 = nn.Sequential(
            ConvBNReLU(input_channels, branch_channels, 1),
            ConvBNReLU(branch_channels, branch_channels, 3, padding=1),
        )
        self.branch_5x5 = nn.Sequential(
            ConvBNReLU(input_channels, branch_channels, 1),
            ConvBNReLU(branch_channels, branch_channels, 3, padding=1),
            ConvBNReLU(branch_channels, branch_channels, 3, padding=1),
        )
        self.branch_directional = nn.Sequential(
            ConvBNReLU(input_channels, branch_channels, 1),
            ConvBNReLU(branch_channels, branch_channels, (1, 5), padding=(0, 2)),
            ConvBNReLU(branch_channels, branch_channels, (5, 1), padding=(2, 0)),
        )

    @property
    def output_channels(self) -> int:
        return 4 * self.branch_1x1[0].out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                self.branch_1x1(inputs),
                self.branch_3x3(inputs),
                self.branch_5x5(inputs),
                self.branch_directional(inputs),
            ),
            dim=1,
        )


class TwoHeadRegressor(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True))
        self.head_tavg = nn.Linear(hidden_dim, 1)
        self.head_delta_t = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        shared = self.shared(features)
        return torch.cat((self.head_tavg(shared), self.head_delta_t(shared)), dim=1)


class Logical10MiniInception(nn.Module):
    """Three-channel 10x10 baseline with only one spatial downsampling."""

    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        self.stem = ConvBNReLU(input_channels, 32, 3, padding=1)
        self.block1 = InceptionBlock(32, 16)  # 64 x 10 x 10
        self.block2 = InceptionBlock(64, 24)  # 96 x 10 x 10
        self.pool = nn.MaxPool2d(2, 2)
        self.block3 = InceptionBlock(96, 32)  # 128 x 5 x 5
        self.regressor = TwoHeadRegressor(128 * 5 * 5, 256)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.block3(self.pool(self.block2(self.block1(self.stem(inputs)))))
        return self.regressor(torch.flatten(features, 1))


class Semantic90MiniInception(nn.Module):
    """Low-early-pooling model for 128/224 FEM semantic images."""

    def __init__(self, input_channels: int = 8) -> None:
        super().__init__()
        self.stem = ConvBNReLU(input_channels, 32, 3, padding=1)
        self.block1 = InceptionBlock(32, 16)  # 64, full resolution
        self.pool1 = nn.MaxPool2d(2, 2)
        self.block2 = InceptionBlock(64, 24)  # 96
        self.pool2 = nn.MaxPool2d(2, 2)
        self.block3 = InceptionBlock(96, 32)  # 128
        self.pool3 = nn.MaxPool2d(2, 2)
        self.block4 = InceptionBlock(128, 48)  # 192
        self.adaptive = nn.AdaptiveAvgPool2d((4, 4))
        self.regressor = TwoHeadRegressor(192 * 4 * 4, 256)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.block1(self.stem(inputs))
        features = self.block2(self.pool1(features))
        features = self.block3(self.pool2(features))
        features = self.block4(self.pool3(features))
        return self.regressor(torch.flatten(self.adaptive(features), 1))


class ResidualBasicBlock(nn.Module):
    """Two-convolution residual block with an optional projection shortcut."""

    expansion = 1

    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        features = self.relu(self.bn1(self.conv1(inputs)))
        features = self.bn2(self.conv2(features))
        return self.relu(features + residual)


def _residual_stage(
    input_channels: int, output_channels: int, block_count: int, first_stride: int
) -> nn.Sequential:
    blocks = [ResidualBasicBlock(input_channels, output_channels, first_stride)]
    blocks.extend(ResidualBasicBlock(output_channels, output_channels) for _ in range(block_count - 1))
    return nn.Sequential(*blocks)


class Logical10ResNet20(nn.Module):
    """CIFAR-style ResNet-20 adapted to a three-channel 10x10 regression input."""

    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        self.stem = ConvBNReLU(input_channels, 16, 3, padding=1)
        self.stage1 = _residual_stage(16, 16, block_count=3, first_stride=1)
        self.stage2 = _residual_stage(16, 32, block_count=3, first_stride=2)
        self.stage3 = _residual_stage(32, 64, block_count=3, first_stride=2)
        self.regressor = TwoHeadRegressor(64 * 3 * 3, 128)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stage3(self.stage2(self.stage1(self.stem(inputs))))
        return self.regressor(torch.flatten(features, 1))


class Logical10SmallCNN(nn.Module):
    """Plain small-CNN control model for the three-channel 10x10 topology."""

    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(input_channels, 32, 3, padding=1),
            ConvBNReLU(32, 64, 3, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBNReLU(64, 128, 3, padding=1),
        )
        self.regressor = TwoHeadRegressor(128 * 5 * 5, 128)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.regressor(torch.flatten(self.features(inputs), 1))


class Semantic90ResNet18(nn.Module):
    """ResNet-18 body with a stride-one 3x3 stem and no initial max-pool."""

    def __init__(self, input_channels: int = 8) -> None:
        super().__init__()
        self.stem = ConvBNReLU(input_channels, 64, 3, padding=1)
        self.stage1 = _residual_stage(64, 64, block_count=2, first_stride=1)
        self.stage2 = _residual_stage(64, 128, block_count=2, first_stride=2)
        self.stage3 = _residual_stage(128, 256, block_count=2, first_stride=2)
        self.stage4 = _residual_stage(256, 512, block_count=2, first_stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = TwoHeadRegressor(512, 256)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stage4(self.stage3(self.stage2(self.stage1(self.stem(inputs)))))
        return self.regressor(torch.flatten(self.pool(features), 1))


def _vgg_block(input_channels: int, output_channels: int, convolution_count: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index in range(convolution_count):
        layers.append(ConvBNReLU(input_channels if index == 0 else output_channels, output_channels, 3, padding=1))
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class Semantic90VGG16(nn.Module):
    """VGG16 convolutional body adapted to eight semantic channels and regression."""

    def __init__(self, input_channels: int = 8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _vgg_block(input_channels, 64, 2),
            _vgg_block(64, 128, 2),
            _vgg_block(128, 256, 3),
            _vgg_block(256, 512, 3),
            _vgg_block(512, 512, 3),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.regressor = TwoHeadRegressor(512 * 4 * 4, 512)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.features(inputs))
        return self.regressor(torch.flatten(features, 1))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
