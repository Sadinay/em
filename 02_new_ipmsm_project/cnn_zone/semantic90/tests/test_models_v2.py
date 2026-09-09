from __future__ import annotations

import torch
from torch import nn

from cnn_zone.semantic90.src.models_v2 import (
    Logical10MiniInceptionV2,
    Logical10ResNet20V2,
    Logical10SmallCNNV2,
    Semantic90MiniInceptionV2,
    Semantic90ResNet18V2,
    Semantic90VGG16V2,
    V2_MODEL_CLASSES,
)


def test_all_v2_forward_shapes() -> None:
    with torch.inference_mode():
        for model in (
            Logical10MiniInceptionV2(),
            Logical10ResNet20V2(),
            Logical10SmallCNNV2(),
        ):
            assert model.eval()(torch.zeros(2, 3, 10, 10)).shape == (2, 2)
        for model in (
            Semantic90MiniInceptionV2(),
            Semantic90ResNet18V2(),
            Semantic90VGG16V2(),
        ):
            assert model.eval()(torch.zeros(1, 8, 224, 224)).shape == (1, 2)


def test_logical_v2_spatial_shapes() -> None:
    with torch.inference_mode():
        inception = Logical10MiniInceptionV2().eval()
        x = inception.block2(inception.block1(inception.stem(torch.zeros(2, 3, 10, 10))))
        assert x.shape == (2, 96, 10, 10)
        assert inception.block3(inception.pool(x)).shape == (2, 128, 5, 5)

        resnet = Logical10ResNet20V2().eval()
        x = resnet.stage1(resnet.stem(torch.zeros(2, 3, 10, 10)))
        assert x.shape == (2, 16, 10, 10)
        x = resnet.stage2(x)
        assert x.shape == (2, 32, 5, 5)
        assert resnet.stage3(x).shape == (2, 64, 5, 5)

        small = Logical10SmallCNNV2().eval()
        shared = small.features(torch.zeros(2, 3, 10, 10))
        assert shared.shape == (2, 128, 5, 5)
        assert small.delta_features(shared).shape == (2, 128, 5, 5)


def test_semantic_v2_final_spatial_shapes() -> None:
    with torch.inference_mode():
        inception = Semantic90MiniInceptionV2().eval()
        x = inception.block1(inception.stem(torch.zeros(1, 8, 224, 224)))
        x = inception.block2(inception.pool1(x))
        x = inception.block3(inception.pool2(x))
        x = inception.block4(inception.pool3(x))
        assert x.shape == (1, 192, 28, 28)
        assert inception.adaptive(x).shape == (1, 192, 4, 4)

        resnet = Semantic90ResNet18V2().eval()
        x = resnet.stage1(resnet.stem(torch.zeros(1, 8, 224, 224)))
        x = resnet.stage2(x)
        x = resnet.stage3(x)
        x = resnet.stage4(x)
        assert x.shape == (1, 256, 28, 28)
        assert resnet.pool(x).shape == (1, 256, 4, 4)

        vgg = Semantic90VGG16V2().eval()
        x = vgg.features(torch.zeros(1, 8, 224, 224))
        assert x.shape == (1, 512, 7, 7)
        assert vgg.pool(x).shape == (1, 512, 4, 4)


def test_group_norm_divisibility() -> None:
    for factory in (
        Semantic90MiniInceptionV2,
        Semantic90ResNet18V2,
        Semantic90VGG16V2,
    ):
        model = factory()
        norms = [module for module in model.modules() if isinstance(module, nn.GroupNorm)]
        assert norms
        assert all(module.num_channels % module.num_groups == 0 for module in norms)


def test_two_output_heads_have_no_shared_parameters() -> None:
    for factory in V2_MODEL_CLASSES.values():
        model = factory()
        if hasattr(model, "regressor"):
            tavg_parameters = {id(parameter) for parameter in model.regressor.head_tavg.parameters()}
            delta_parameters = {id(parameter) for parameter in model.regressor.head_delta_t.parameters()}
        else:
            tavg_parameters = {id(parameter) for parameter in model.head_tavg.parameters()}
            delta_parameters = {id(parameter) for parameter in model.head_delta_t.parameters()}
        assert tavg_parameters
        assert delta_parameters
        assert tavg_parameters.isdisjoint(delta_parameters)


def test_v2_input_channels_are_parameterized() -> None:
    logical = Logical10SmallCNNV2(input_channels=4)
    semantic = Semantic90ResNet18V2(input_channels=9)
    assert logical.features[0][0].in_channels == 4
    assert semantic.stem[0].in_channels == 9
