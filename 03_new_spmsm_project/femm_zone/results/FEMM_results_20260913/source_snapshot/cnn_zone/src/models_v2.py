"""V2 architecture families adapted to SPMSM 6x20 and dual 224 inputs."""

from __future__ import annotations

import torch
from torch import nn


def group_count(channels: int, preferred: int = 8) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin: int, cout: int, kernel_size, stride: int = 1, padding=0) -> None:
        super().__init__(
            nn.Conv2d(cin, cout, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class ConvGNReLU(nn.Sequential):
    def __init__(self, cin: int, cout: int, kernel_size, stride: int = 1, padding=0, preferred_groups: int = 8) -> None:
        super().__init__(
            nn.Conv2d(cin, cout, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(group_count(cout, preferred_groups), cout),
            nn.ReLU(inplace=True),
        )


class IndependentRegressionHeads(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.head_tavg = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.head_delta_t = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.head_tavg(features), self.head_delta_t(features)), dim=1)


class _InceptionBranches(nn.Module):
    def __init__(self, cin: int, branch: int, conv_factory) -> None:
        super().__init__()
        self.b1 = conv_factory(cin, branch, 1)
        self.b3 = nn.Sequential(conv_factory(cin, branch, 1), conv_factory(branch, branch, 3, padding=1))
        self.b5 = nn.Sequential(conv_factory(cin, branch, 1), conv_factory(branch, branch, 3, padding=1), conv_factory(branch, branch, 3, padding=1))
        self.directional = nn.Sequential(
            conv_factory(cin, branch, 1),
            conv_factory(branch, branch, (1, 5), padding=(0, 2)),
            conv_factory(branch, branch, (5, 1), padding=(2, 0)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.b1(x), self.b3(x), self.b5(x), self.directional(x)), dim=1)


class ResidualInceptionBlock(nn.Module):
    def __init__(self, cin: int, branch: int, group_norm: bool = False) -> None:
        super().__init__()
        cout = 4 * branch
        factory = ConvGNReLU if group_norm else ConvBNReLU
        self.branches = _InceptionBranches(cin, branch, factory)
        if cin == cout:
            self.shortcut = nn.Identity()
        elif group_norm:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.GroupNorm(group_count(cout), cout))
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.branches(x) + self.shortcut(x))


class ResidualBasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1, group_norm: bool = False) -> None:
        super().__init__()
        norm = (lambda c: nn.GroupNorm(group_count(c), c)) if group_norm else (lambda c: nn.BatchNorm2d(c))
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.norm1 = norm(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.norm2 = norm(cout)
        self.shortcut = nn.Identity() if stride == 1 and cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), norm(cout))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)
        x = self.relu(self.norm1(self.conv1(x)))
        return self.relu(self.norm2(self.conv2(x)) + shortcut)


def residual_stage(cin: int, cout: int, count: int, stride: int, group_norm: bool = False) -> nn.Sequential:
    layers = [ResidualBasicBlock(cin, cout, stride, group_norm)]
    layers.extend(ResidualBasicBlock(cout, cout, 1, group_norm) for _ in range(count - 1))
    return nn.Sequential(*layers)


class Logical6x20MiniInceptionV2(nn.Module):
    """V2 Mini-Inception family with a shape-safe 3x10 feature grid."""

    def __init__(self, input_channels: int = 2) -> None:
        super().__init__()
        self.stem = ConvBNReLU(input_channels, 32, 3, padding=1)
        self.block1 = ResidualInceptionBlock(32, 16)
        self.block2 = ResidualInceptionBlock(64, 24)
        self.pool = nn.MaxPool2d(2, 2)
        self.block3 = ResidualInceptionBlock(96, 32)
        self.adaptive = nn.AdaptiveAvgPool2d((3, 10))
        self.regressor = IndependentRegressionHeads(128 * 3 * 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block3(self.pool(self.block2(self.block1(self.stem(x)))))
        return self.regressor(torch.flatten(self.adaptive(x), 1))


class Logical6x20ResNet20V2(nn.Module):
    def __init__(self, input_channels: int = 2) -> None:
        super().__init__()
        self.stem = ConvBNReLU(input_channels, 16, 3, padding=1)
        self.stage1 = residual_stage(16, 16, 3, 1)
        self.stage2 = residual_stage(16, 32, 3, 2)
        self.stage3 = residual_stage(32, 64, 3, 1)
        self.adaptive = nn.AdaptiveAvgPool2d((3, 10))
        self.regressor = IndependentRegressionHeads(64 * 3 * 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage3(self.stage2(self.stage1(self.stem(x))))
        return self.regressor(torch.flatten(self.adaptive(x), 1))


class Logical6x20SmallCNNV2(nn.Module):
    def __init__(self, input_channels: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(input_channels, 32, 3, padding=1),
            ConvBNReLU(32, 64, 3, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBNReLU(64, 128, 3, padding=1),
        )
        self.delta_features = ConvBNReLU(128, 128, 3, padding=1)
        self.adaptive = nn.AdaptiveAvgPool2d((3, 10))
        self.head_tavg = nn.Sequential(nn.Linear(128 * 3 * 10, 128), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(128, 1))
        self.head_delta_t = nn.Sequential(nn.Linear(128 * 3 * 10, 128), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.features(x)
        tavg = self.head_tavg(torch.flatten(self.adaptive(shared), 1))
        delta = self.head_delta_t(torch.flatten(self.adaptive(self.delta_features(shared)), 1))
        return torch.cat((tavg, delta), dim=1)


class Semantic224MiniInceptionV2(nn.Module):
    def __init__(self, input_channels: int = 8) -> None:
        super().__init__()
        self.stem = ConvGNReLU(input_channels, 32, 3, padding=1)
        self.block1 = ResidualInceptionBlock(32, 16, True)
        self.block2 = ResidualInceptionBlock(64, 24, True)
        self.block3 = ResidualInceptionBlock(96, 32, True)
        self.block4 = ResidualInceptionBlock(128, 48, True)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive = nn.AdaptiveAvgPool2d((4, 4))
        self.regressor = IndependentRegressionHeads(192 * 4 * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(self.stem(x))
        x = self.block2(self.pool(x))
        x = self.block3(self.pool(x))
        x = self.block4(self.pool(x))
        return self.regressor(torch.flatten(self.adaptive(x), 1))


class Semantic224ResNet18V2(nn.Module):
    def __init__(self, input_channels: int = 8) -> None:
        super().__init__()
        self.stem = ConvGNReLU(input_channels, 32, 3, padding=1)
        self.stage1 = residual_stage(32, 32, 2, 1, True)
        self.stage2 = residual_stage(32, 64, 2, 2, True)
        self.stage3 = residual_stage(64, 128, 2, 2, True)
        self.stage4 = residual_stage(128, 256, 2, 2, True)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.regressor = IndependentRegressionHeads(256 * 4 * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage4(self.stage3(self.stage2(self.stage1(self.stem(x)))))
        return self.regressor(torch.flatten(self.pool(x), 1))


def vgg_block(cin: int, cout: int, count: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index in range(count):
        layers.append(ConvGNReLU(cin if index == 0 else cout, cout, 3, padding=1, preferred_groups=8 if cout <= 128 else 16))
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class Semantic224VGG16V2(nn.Module):
    def __init__(self, input_channels: int = 8) -> None:
        super().__init__()
        self.features = nn.Sequential(
            vgg_block(input_channels, 64, 2),
            vgg_block(64, 128, 2),
            vgg_block(128, 256, 3),
            vgg_block(256, 512, 3),
            vgg_block(512, 512, 3),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.regressor = IndependentRegressionHeads(512 * 4 * 4, 256, 0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(torch.flatten(self.pool(self.features(x)), 1))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
