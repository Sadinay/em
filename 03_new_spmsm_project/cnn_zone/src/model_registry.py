"""Nine V3 experiments: three logical plus two 224 representations x three backbones."""

from __future__ import annotations

from .models_v2 import (
    Logical6x20MiniInceptionV2,
    Logical6x20ResNet20V2,
    Logical6x20SmallCNNV2,
    Semantic224MiniInceptionV2,
    Semantic224ResNet18V2,
    Semantic224VGG16V2,
)


MODEL_CLASSES = {
    "logical6x20/mini_inception_v2": Logical6x20MiniInceptionV2,
    "logical6x20/resnet20_v2": Logical6x20ResNet20V2,
    "logical6x20/small_cnn_v2": Logical6x20SmallCNNV2,
    "xy224/mini_inception_v2": Semantic224MiniInceptionV2,
    "xy224/resnet18_v2": Semantic224ResNet18V2,
    "xy224/vgg16_v2": Semantic224VGG16V2,
    "polar90_224/mini_inception_v2": Semantic224MiniInceptionV2,
    "polar90_224/resnet18_v2": Semantic224ResNet18V2,
    "polar90_224/vgg16_v2": Semantic224VGG16V2,
}


def create_model(name: str):
    try:
        return MODEL_CLASSES[name]()
    except KeyError as exc:
        raise KeyError(f"Unknown model {name!r}; choose one of {sorted(MODEL_CLASSES)}") from exc
