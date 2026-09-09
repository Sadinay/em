from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "data_zone" / "scripts" / "build_ipmsm_cnn_dataset.py"
MAT = ROOT / "data_zone" / "raw" / "workspace_600.mat"
DATASET = ROOT / "data_zone" / "processed" / "ipmsm_topology_dataset"


def load_script():
    spec = importlib.util.spec_from_file_location("build_ipmsm_cnn_dataset", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = load_script()
sys.path.insert(0, str(ROOT))
from cnn_zone.dataset import IPMSMTopologyDataset  # noqa: E402


def test_bit_decode_and_round_trip() -> None:
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, size=(17, 200), dtype=np.uint8)
    codes = builder.decode_bit_pairs(bits)
    recovered = builder.encode_material_codes(codes)
    assert np.array_equal(bits, recovered)


def test_grid_orientation_and_round_trip() -> None:
    vector = np.arange(100, dtype=np.uint8).reshape(1, 100)
    grid = builder.vectors_to_grids(vector)
    assert grid.shape == (1, 10, 10)
    assert grid[0, 7, 3] == vector[0, 3 * 10 + 7]
    assert np.array_equal(builder.grids_to_vectors(grid), vector)


def test_material_position_proves_orientation() -> None:
    data = loadmat(MAT, variable_names=["MaterialPosition"], squeeze_me=True)
    result = builder._validate_material_position(data["MaterialPosition"])
    assert result["verified_vector_index_formula"] == "index = angular_index * 10 + radial_index"
    assert np.allclose(result["angle_centres_deg"], 1.125 + 2.25 * np.arange(10))


def test_final_state_alignment_and_corrected_semantics() -> None:
    data = loadmat(
        MAT,
        variable_names=["population", "population_all", "population_noChange_all", "Material", "Material_change", "VolumePM"],
        squeeze_me=True,
    )
    corrected = np.asarray(data["population_all"], dtype=np.uint8)[:, :, -1]
    raw = np.asarray(data["population_noChange_all"], dtype=np.uint8)[:, :, -1]
    assert np.array_equal(data["population"], corrected)
    assert np.array_equal(builder.decode_bit_pairs(raw), data["Material"])
    corrected_codes = builder.decode_bit_pairs(corrected)
    assert np.array_equal(corrected_codes, data["Material_change"])
    assert np.array_equal(np.sum(corrected_codes == 1, axis=1), data["VolumePM"])


def test_hash_is_stable_and_order_sensitive() -> None:
    first = np.arange(100, dtype=np.uint8) % 4
    second = first.copy()
    second[[0, 1]] = second[[1, 0]]
    assert builder.topology_hash(first) == builder.topology_hash(first.copy())
    assert builder.topology_hash(first) != builder.topology_hash(second)


def test_default_training_dataset_and_no_split_leakage() -> None:
    training = DATASET / "training_corrected_physical_three_state"
    topology = np.load(training / "topology_codes.npy", mmap_mode="r")
    targets = np.load(training / "targets_tavg_delta.npy", mmap_mode="r")
    split = np.load(training / "split_codes.npy", mmap_mode="r")
    assert topology.shape == (146471, 10, 10)
    assert targets.shape == (146471, 2)
    assert set(np.unique(topology).tolist()) == {0, 1, 2}
    assert np.isfinite(targets).all()
    hashes: dict[int, set[str]] = {0: set(), 1: set(), 2: set()}
    components: dict[int, set[int]] = {}
    with (training / "split_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            split_code = {"train": 0, "validation": 1, "test": 2}[row["split"]]
            hashes[split_code].add(row["physical_topology_hash"])
            components.setdefault(int(row["repair_component_id"]), set()).add(split_code)
    assert hashes[0].isdisjoint(hashes[1])
    assert hashes[0].isdisjoint(hashes[2])
    assert hashes[1].isdisjoint(hashes[2])
    assert all(len(values) == 1 for values in components.values())
    assert np.array_equal(np.unique(split), np.array([0, 1, 2], dtype=np.int8))


def test_dataloader_smoke() -> None:
    training = DATASET / "training_corrected_physical_three_state"
    dataset = IPMSMTopologyDataset(training, split="validation")
    inputs, targets = next(iter(DataLoader(dataset, batch_size=16, shuffle=False)))
    assert inputs.shape == (16, 3, 10, 10)
    assert targets.shape == (16, 2)
    assert np.allclose(inputs.sum(dim=1).numpy(), 1.0)
    coordinate_dataset = IPMSMTopologyDataset(training, split="validation", add_coordinate_channels=True)
    inputs_with_coordinates, _ = next(iter(DataLoader(coordinate_dataset, batch_size=4, shuffle=False)))
    assert inputs_with_coordinates.shape == (4, 5, 10, 10)

