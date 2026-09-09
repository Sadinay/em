from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cnn_zone.semantic90.src.dataset import GeneCodeDataset
from cnn_zone.semantic90.src.models import (
    Logical10MiniInception,
    Logical10ResNet20,
    Logical10SmallCNN,
    Semantic90MiniInception,
    Semantic90ResNet18,
    Semantic90VGG16,
)
from cnn_zone.semantic90.src.topology_renderer import FEMSemanticRenderer


PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT / "cnn_zone" / "semantic90"
LOOKUP_128 = ROOT / "outputs" / "lookups" / "fem90_lookup_128.npz"
LOOKUP_224 = ROOT / "outputs" / "lookups" / "fem90_lookup_224.npz"
DATASET = PROJECT / "data_zone" / "processed" / "ipmsm_topology_dataset" / "training_corrected_physical_three_state"


def test_lookup_shapes_and_all_gene_regions() -> None:
    for size, path in ((128, LOOKUP_128), (224, LOOKUP_224)):
        with np.load(path) as lookup:
            gene = lookup["pixel_gene_id"]
            replica = lookup["pixel_replica_id"]
            assert gene.shape == (size, size)
            assert set(np.unique(gene[gene >= 0]).tolist()) == set(range(100))
            assert set(np.unique(replica[replica >= 0]).tolist()) == {0, 1, 2, 3}
            label_to_gene = lookup["label_to_gene"]
            assert np.all(np.bincount(label_to_gene[label_to_gene >= 0], minlength=100) == 4)


def test_renderer_is_deterministic_and_preserves_pm_polarity() -> None:
    renderer = FEMSemanticRenderer(LOOKUP_224)
    codes = torch.zeros((2, 100), dtype=torch.long)
    codes[0, :] = 1
    codes[1, :] = 2
    first = renderer(codes)
    second = renderer(codes)
    assert first.shape == (2, 8, 224, 224)
    assert torch.equal(first, second)
    assert first[0, 1].sum() > 0
    assert first[0, 2].sum() > 0
    assert first[1, 3].sum() > 0
    assert not torch.any((first[:, 0] + first[:, 1] + first[:, 2] + first[:, 3]) > 1)


def test_dataloader_renders_after_batching() -> None:
    dataset = GeneCodeDataset(DATASET, "test")
    codes, targets = next(iter(DataLoader(dataset, batch_size=8, shuffle=False)))
    renderer = FEMSemanticRenderer(LOOKUP_224)
    images = renderer(codes)
    assert codes.shape == (8, 100)
    assert images.shape == (8, 8, 224, 224)
    assert targets.shape == (8, 2)
    assert torch.isfinite(images).all() and torch.isfinite(targets).all()


def test_three_fem_memory_comparisons_pass() -> None:
    result = json.loads((ROOT / "outputs" / "audit" / "renderer_validation.json").read_text(encoding="utf-8"))
    assert result["all_three_fem_memory_comparisons_pass"]
    assert result["repeat_render_pixel_exact"]
    assert all(item["fem_design_label_mismatches"] == 0 for item in result["comparison_samples"])
    assert all(item["maximum_pm_direction_error_deg"] == 0 for item in result["comparison_samples"])


def test_evaluation_split_files_have_no_component_leakage() -> None:
    split_root = ROOT / "outputs" / "splits"
    for filename in ("scheme_a_performance_80_10_10.npz", "scheme_b_temporal_extrapolation.npz"):
        with np.load(split_root / filename) as split:
            codes = split["split_code"]
            assert len(codes) == 146471
            assert set(np.unique(codes).tolist()) == {0, 1, 2}
            assert len(np.intersect1d(split["train"], split["validation"])) == 0
            assert len(np.intersect1d(split["train"], split["test"])) == 0
            assert len(np.intersect1d(split["validation"], split["test"])) == 0


def test_overfit_model_tensor_shapes() -> None:
    with torch.inference_mode():
        for model in (Logical10MiniInception(), Logical10ResNet20(), Logical10SmallCNN()):
            assert model.eval()(torch.zeros(2, 3, 10, 10)).shape == (2, 2)
        for model in (Semantic90MiniInception(), Semantic90ResNet18(), Semantic90VGG16()):
            assert model.eval()(torch.zeros(1, 8, 224, 224)).shape == (1, 2)


def test_resnet_stage_tensor_shapes() -> None:
    logical = Logical10ResNet20().eval()
    semantic = Semantic90ResNet18().eval()
    with torch.inference_mode():
        logical_stem = logical.stem(torch.zeros(2, 3, 10, 10))
        logical_stage1 = logical.stage1(logical_stem)
        logical_stage2 = logical.stage2(logical_stage1)
        logical_stage3 = logical.stage3(logical_stage2)
        assert logical_stem.shape == (2, 16, 10, 10)
        assert logical_stage1.shape == (2, 16, 10, 10)
        assert logical_stage2.shape == (2, 32, 5, 5)
        assert logical_stage3.shape == (2, 64, 3, 3)
        semantic_stem = semantic.stem(torch.zeros(1, 8, 224, 224))
        assert semantic_stem.shape == (1, 64, 224, 224)
        assert semantic.stage1(semantic_stem).shape == (1, 64, 224, 224)
