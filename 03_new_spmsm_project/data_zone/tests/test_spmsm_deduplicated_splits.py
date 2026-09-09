from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT / "data_zone" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_spmsm_deduplicated_splits import TAVG_EDGES, assign_tavg_bands, vectors_to_grids


DATASET = PROJECT / "data_zone" / "processed" / "spmsm_topology_dataset" / "training_corrected_binary"
SPLIT = PROJECT / "cnn_zone" / "outputs" / "splits" / "scheme_a_tavg_bands_80_10_10.npz"


def test_gene_grid_order_and_round_trip() -> None:
    vector = np.arange(120, dtype=np.uint8)[None, :]
    grid = vectors_to_grids(vector)
    assert grid.shape == (1, 6, 20)
    assert np.array_equal(grid[0, :, 0], np.arange(6))
    assert np.array_equal(grid[0, :, 1], np.arange(6, 12))
    assert np.array_equal(grid.swapaxes(1, 2).reshape(1, 120), vector)


def test_tavg_band_boundaries() -> None:
    values = np.asarray([1.9, 2.0, 2.5, 3.0, 3.25, 3.5, 3.75, 4.0])
    assert np.array_equal(assign_tavg_bands(values), np.asarray([0, 1, 2, 3, 4, 5, 6, 6]))
    assert len(TAVG_EDGES) == 8


def test_generated_dataset_and_splits_are_complete_and_disjoint() -> None:
    topology = np.load(DATASET / "topology_bits.npy", mmap_mode="r")
    targets = np.load(DATASET / "targets_tavg_delta.npy", mmap_mode="r")
    split_codes = np.load(DATASET / "split_codes.npy", mmap_mode="r")
    with np.load(SPLIT) as split:
        train = np.asarray(split["train"])
        validation = np.asarray(split["validation"])
        test = np.asarray(split["test"])

    assert topology.shape == (64804, 6, 20)
    assert targets.shape == (64804, 2)
    assert split_codes.shape == (64804,)
    assert np.array_equal(np.unique(topology), np.asarray([0, 1], dtype=np.uint8))
    assert np.isfinite(targets).all()
    assert not np.intersect1d(train, validation).size
    assert not np.intersect1d(train, test).size
    assert not np.intersect1d(validation, test).size
    assert np.array_equal(np.sort(np.r_[train, validation, test]), np.arange(len(topology)))


def test_no_hash_or_repair_component_crosses_splits() -> None:
    hashes: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    components: dict[int, set[str]] = {}
    with (DATASET / "split_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            split = row["split"]
            hashes[split].add(row["topology_hash"])
            components.setdefault(int(row["repair_component_id"]), set()).add(split)
    assert not (hashes["train"] & hashes["validation"])
    assert not (hashes["train"] & hashes["test"])
    assert not (hashes["validation"] & hashes["test"])
    assert all(len(values) == 1 for values in components.values())
