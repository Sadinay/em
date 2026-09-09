from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CNN_ROOT = ROOT / "cnn_zone"
if str(CNN_ROOT) not in sys.path:
    sys.path.insert(0, str(CNN_ROOT))

from src.topology_renderer import SemanticRenderer, logical_one_hot  # noqa: E402


DATASET = ROOT / "data_zone" / "processed" / "spmsm_topology_dataset" / "training_corrected_binary"
LOOKUPS = CNN_ROOT / "outputs" / "lookups"
SPLIT = CNN_ROOT / "outputs" / "splits" / "scheme_a_tavg_bands_train40000_val6483_test6483.npz"


def test_train40000_keeps_eval_sets_disjoint() -> None:
    split = np.load(SPLIT)
    assert len(split["train"]) == 40_000
    assert len(split["validation"]) == 6_483
    assert len(split["test"]) == 6_483
    assert len(split["unused_train_pool"]) == 11_838
    assert len(np.intersect1d(split["train"], split["validation"])) == 0
    assert len(np.intersect1d(split["train"], split["test"])) == 0


def test_both_224_lookups_cover_full_motor_and_all_genes() -> None:
    for filename, system in (
        ("spmsm_xy224_full_motor.npz", "xy"),
        ("spmsm_polar90_224_full_motor.npz", "polar90"),
    ):
        lookup = np.load(LOOKUPS / filename)
        assert lookup["pixel_gene_id"].shape == (224, 224)
        assert str(lookup["coordinate_system"].item()) == system
        represented = np.unique(lookup["pixel_gene_id"][lookup["pixel_gene_id"] >= 0])
        np.testing.assert_array_equal(represented, np.arange(120))
        assert set(np.unique(lookup["fixed_material_map"])) >= {0, 1, 2, 3}
    polar = np.load(LOOKUPS / "spmsm_polar90_224_full_motor.npz")
    assert polar["radius_axis_mm"][0] > 0
    assert polar["radius_axis_mm"][-1] < 64.0
    assert polar["theta_axis_deg"][0] > 0
    assert polar["theta_axis_deg"][-1] < 90.0


def test_logical_binary_channels_and_semantic_rendering() -> None:
    bits = np.load(DATASET / "topology_bits.npy")[:2].reshape(2, 120)
    logical = logical_one_hot(bits)
    assert logical.shape == (2, 2, 6, 20)
    np.testing.assert_array_equal(logical.sum(axis=1), np.ones((2, 6, 20)))
    for filename in ("spmsm_xy224_full_motor.npz", "spmsm_polar90_224_full_motor.npz"):
        rendered = SemanticRenderer(LOOKUPS / filename).render_numpy(bits)
        assert rendered.shape == (2, 8, 224, 224)
        assert rendered.dtype == np.float32
        assert np.isfinite(rendered).all()
