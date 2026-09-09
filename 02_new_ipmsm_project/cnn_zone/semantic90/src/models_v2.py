"""Optimized v2 regressors; legacy model classes remain unchanged in models.py."""

from __future__ import annotations

import torch
from torch import nn


def group_count(channels: int, preferred: int = 8) -> int:
    """Return the largest sensible group count not exceeding ``preferred``."""
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBNReLU(nn.Sequential):
    def __init__(
        self, input_channels: int, output_channels: int, kernel_size, stride: int = 1, padding=0
    ) -> None:
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )


class ConvGNReLU(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size,
        stride: int = 1,
        padding=0,
        preferred_groups: int = 8,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(group_count(output_channels, preferred_groups), output_channels),
            nn.ReLU(inplace=True),
        )


class IndependentRegressionHeads(nn.Module):
    """Two fully independent two-layer MLP heads."""

    def __init__(
        self, feature_dim: int, hidden_dim: int = 128, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.head_tavg = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.head_delta_t = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.head_tavg(features), self.head_delta_t(features)), dim=1)


class _InceptionBranches(nn.Module):
    def __init__(self, input_channels: int, branch_channels: int, conv_factory) -> None:
        super().__init__()
        self.branch_1x1 = conv_factory(input_channels, branch_channels, 1)
        self.branch_3x3 = nn.Sequential(
            conv_factory(input_channels, branch_channels, 1),
            conv_factory(branch_channels, branch_channels, 3, padding=1),
        )
        self.branch_5x5 = nn.Sequential(
            conv_factory(input_channels, branch_channels, 1),
            conv_factory(branch_channels, branch_channels, 3, padding=1),
            conv_factory(branch_channels, branch_channels, 3, padding=1),
        )
        self.branch_directional = nn.Sequential(
            conv_factory(input_channels, branch_channels, 1),
            conv_factory(branch_channels, branch_channels, (1, 5), padding=(0, 2)),
            conv_factory(branch_channels, branch_channels, (5, 1), padding=(2, 0)),
        )

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


class ResidualInceptionBlock(nn.Module):
    """Batch-normalized four-branch Inception block with a projection residual."""

    def __init__(self, input_channels: int, branch_channels: int) -> None:
        super().__init__()
        output_channels = 4 * branch_channels
        self.branches = _InceptionBranches(input_channels, branch_channels, ConvBNReLU)
        if input_channels == output_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, bias=False),
                nn.BatchNorm2d(output_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.relu(self.branches(inputs) + self.shortcut(inputs))


class ResidualInceptionGNBlock(nn.Module):
    """Group-normalized four-branch Inception block with a projection residual."""

    def __init__(self, input_channels: int, branch_channels: int) -> None:
        super().__init__()
        output_channels = 4 * branch_channels
        self.branches = _InceptionBranches(input_channels, branch_channels, ConvGNReLU)
        if input_channels == output_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, bias=False),
                nn.GroupNorm(group_count(output_channels, 8), output_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.relu(self.branches(inputs) + self.shortcut(inputs))


class Logical10MiniInceptionV2(nn.Module):
    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        self.stem = ConvBNReLU(input_channels, 32, 3, padding=1)
        self.block1 = ResidualInceptionBlock(32, 16)
        self.block2 = ResidualInceptionBlock(64, 24)
        self.pool = nn.MaxPool2d(2, 2)
        self.block3 = ResidualInceptionBlock(96, 32)
        self.regressor = IndependentRegressionHeads(128 * 5 * 5, 128, 0.1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.block3(self.pool(self.block2(self.block1(self.stem(inputs)))))
        return self.regressor(torch.flatten(features, 1))


class ResidualBasicBlockBN(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, 3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(output_channels)
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(inputs)
        features = self.relu(self.norm1(self.conv1(inputs)))
        features = self.norm2(self.conv2(features))
        return self.relu(features + shortcut)


class ResidualBasicBlockGN(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, 3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(group_count(output_channels, 8), output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(group_count(output_channels, 8), output_channels)
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(group_count(output_channels, 8), output_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(inputs)
        features = self.relu(self.norm1(self.conv1(inputs)))
        features = self.norm2(self.conv2(features))
        return self.relu(features + shortcut)


def _residual_stage(block_type, input_channels: int, output_channels: int, count: int, stride: int):
    blocks = [block_type(input_channels, output_channels, stride)]
    blocks.extend(block_type(output_channels, output_channels, 1) for _ in range(count - 1))
    return nn.Sequential(*blocks)


class Logical10ResNet20V2(nn.Module):
    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        self.stem = ConvBNReLU(input_channels, 16, 3, padding=1)
        self.stage1 = _residual_stage(ResidualBasicBlockBN, 16, 16, 3, 1)
        self.stage2 = _residual_stage(ResidualBasicBlockBN, 16, 32, 3, 2)
        self.stage3 = _residual_stage(ResidualBasicBlockBN, 32, 64, 3, 1)
        self.regressor = IndependentRegressionHeads(64 * 5 * 5, 128, 0.1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stage3(self.stage2(self.stage1(self.stem(inputs))))
        return self.regressor(torch.flatten(features, 1))


class Logical10SmallCNNV2(nn.Module):
    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(input_channels, 32, 3, padding=1),
            ConvBNReLU(32, 64, 3, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBNReLU(64, 128, 3, padding=1),
        )
        self.delta_features = ConvBNReLU(128, 128, 3, padding=1)
        self.head_tavg = nn.Sequential(
            nn.Linear(128 * 5 * 5, 128), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(128, 1)
        )
        self.head_delta_t = nn.Sequential(
            nn.Linear(128 * 5 * 5, 128), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(128, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shared = self.features(inputs)
        tavg = self.head_tavg(torch.flatten(shared, 1))
        delta_t = self.head_delta_t(torch.flatten(self.delta_features(shared), 1))
        return torch.cat((tavg, delta_t), dim=1)


class Semantic90MiniInceptionV2(nn.Module):
    def __init__(self, input_channels: int = 8) -> None:
        super().__init__()
        self.stem = ConvGNReLU(input_channels, 32, 3, padding=1)
        self.block1 = ResidualInceptionGNBlock(32, 16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.block2 = ResidualInceptionGNBlock(64, 24)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.block3 = ResidualInceptionGNBlock(96, 32)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.block4 = ResidualInceptionGNBlock(128, 48)
        self.adaptive = nn.AdaptiveAvgPool2d((4, 4))
        self.regressor = IndependentRegressionHeads(192 * 4 * 4, 128, 0.1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.block1(self.stem(inputs))
        features = self.block2(self.pool1(features))
        features = self.block3(self.pool2(features))
        features = self.block4(self.pool3(features))
        return self.regressor(torch.flatten(self.adaptive(features), 1))


class Semantic90ResNet18V2(nn.Module):
    """Width-reduced Semantic ResNet18 with [2,2,2,2] residual depth."""

    def __init__(self, input_channels: int = 8) -> None:
        super().__init__()
        self.stem = ConvGNReLU(input_channels, 32, 3, padding=1)
        self.stage1 = _residual_stage(ResidualBasicBlockGN, 32, 32, 2, 1)
        self.stage2 = _residual_stage(ResidualBasicBlockGN, 32, 64, 2, 2)
        self.stage3 = _residual_stage(ResidualBasicBlockGN, 64, 128, 2, 2)
        self.stage4 = _residual_stage(ResidualBasicBlockGN, 128, 256, 2, 2)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.regressor = IndependentRegressionHeads(256 * 4 * 4, 128, 0.1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.stage4(self.stage3(self.stage2(self.stage1(self.stem(inputs)))))
        return self.regressor(torch.flatten(self.pool(features), 1))


def _vgg_gn_block(
    input_channels: int, output_channels: int, convolution_count: int
) -> nn.Sequential:
    preferred_groups = 8 if output_channels <= 128 else 16
    layers: list[nn.Module] = []
    for index in range(convolution_count):
        layers.append(
            ConvGNReLU(
                input_channels if index == 0 else output_channels,
                output_channels,
                3,
                padding=1,
                preferred_groups=preferred_groups,
            )
        )
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class Semantic90VGG16V2(nn.Module):
    def __init__(self, input_channels: int = 8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _vgg_gn_block(input_channels, 64, 2),
            _vgg_gn_block(64, 128, 2),
            _vgg_gn_block(128, 256, 3),
            _vgg_gn_block(256, 512, 3),
            _vgg_gn_block(512, 512, 3),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.regressor = IndependentRegressionHeads(512 * 4 * 4, 256, 0.2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.features(inputs))
        return self.regressor(torch.flatten(features, 1))


V2_MODEL_CLASSES = {
    "logical10/mini_inception_v2": Logical10MiniInceptionV2,
    "logical10/resnet20_v2": Logical10ResNet20V2,
    "logical10/small_cnn_v2": Logical10SmallCNNV2,
    "semantic224/mini_inception_v2": Semantic90MiniInceptionV2,
    "semantic224/resnet18_v2": Semantic90ResNet18V2,
    "semantic224/vgg16_v2": Semantic90VGG16V2,
}


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
