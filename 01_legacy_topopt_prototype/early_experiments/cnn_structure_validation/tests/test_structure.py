import torch

from model import StructureCNN, parameter_count


def test_structure_model_shape() -> None:
    model = StructureCNN()
    output = model(torch.zeros(4, 5, 96, 96), torch.zeros(4, 1))
    assert output.shape == (4, 1)
    assert parameter_count(model) == 100193
