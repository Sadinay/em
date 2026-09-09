from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from motor_regression.config import load_config, resolved_run_paths
from motor_regression.data import (
    MotorTopologyDataset,
    SampleRecord,
    TargetScaler,
    assign_topology_families,
    group_value,
    grouped_split,
    load_records,
)
from motor_regression.evaluation import regression_metrics
from motor_regression.models import BasicBlock, CifarResNet20Regression, build_model
from motor_regression.training import load_reused_split


ROOT = Path(__file__).resolve().parents[1]


def _record(index: int, root: str, value: int = 0) -> SampleRecord:
    return SampleRecord(
        key=f"run:S{index:06d}",
        run_id="run",
        sample_id=f"S{index:06d}",
        candidate_id=f"C{index:06d}",
        matrix_18x10=np.full((18, 10), value, dtype=np.int64),
        chromosome_hash=f"hash-{index}",
        fitness_band=f"B{1 + index % 7}",
        targets=np.asarray([float(index)], dtype=np.float64),
        generation=index // 2,
        direct_parent=f"parent-{root}",
        root_parent=root,
        lineage_id=root,
        topology_family=f"family-{root}",
    )


def test_resnet20_actual_data_shape_chain() -> None:
    model = CifarResNet20Regression(input_channels=3, output_dim=1)
    inputs = torch.randn(4, 3, 18, 10)
    shapes = model.feature_shapes(inputs)
    assert shapes == {
        "input": (4, 3, 18, 10),
        "stem": (4, 16, 18, 10),
        "stage1": (4, 16, 18, 10),
        "stage2": (4, 32, 9, 5),
        "stage3": (4, 64, 5, 3),
        "flatten": (4, 960),
    }
    assert model(inputs).shape == (4, 1)


def test_basic_block_projection_and_circular_width_keep_expected_shape() -> None:
    block = BasicBlock(16, 32, stride=2, circular_width=True)
    assert block(torch.randn(2, 16, 18, 10)).shape == (2, 32, 9, 5)


def test_all_models_support_multioutput_without_changing_input() -> None:
    inputs = torch.randn(2, 3, 18, 10)
    for name in ("resnet20", "mlp", "small_cnn"):
        model = build_model(name, input_channels=3, output_dim=2)
        assert model(inputs).shape == (2, 2)


def test_one_hot_dataset_and_training_only_target_scaler() -> None:
    records = [_record(0, "a", 0), _record(1, "b", 1), _record(2, "c", 2)]
    scaler = TargetScaler.fit(np.stack([records[0].targets, records[1].targets]), ["target"])
    dataset = MotorTopologyDataset(records, [2], categories=[0, 1, 2], scaler=scaler)
    inputs, target = dataset[0]
    assert inputs.shape == (3, 18, 10)
    assert torch.all(inputs.sum(dim=0) == 1)
    assert torch.all(inputs[2] == 1)
    assert np.isclose(scaler.mean[0], 0.5)
    assert np.allclose(scaler.inverse_transform(target.numpy().reshape(1, -1)), [[2.0]])


def test_group_split_is_deterministic_and_has_no_group_leakage() -> None:
    records = [_record(index, f"root-{index // 3}", index % 3) for index in range(60)]
    split1 = grouped_split(
        records, group_mode="root_parent", fractions=(0.7, 0.15, 0.15), seed=42
    )
    split2 = grouped_split(
        records, group_mode="root_parent", fractions=(0.7, 0.15, 0.15), seed=42
    )
    assert split1 == split2
    groups = {
        name: {group_value(records[index], "root_parent") for index in indices}
        for name, indices in split1.items()
    }
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])


def test_reused_split_preserves_sample_membership(tmp_path: Path) -> None:
    records = [_record(index, f"root-{index // 2}", index % 3) for index in range(6)]
    payload = {
        "samples": {
            "train": [records[0].key, records[1].key],
            "val": [records[2].key, records[3].key],
            "test": [records[4].key, records[5].key],
        }
    }
    path = tmp_path / "data_split.json"
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    split = load_reused_split(records, path, group_mode="root_parent")
    assert split == {"train": [0, 1], "val": [2, 3], "test": [4, 5]}


def test_topology_family_clustering_and_metrics() -> None:
    records = [_record(0, "a", 0), _record(1, "b", 0), _record(2, "c", 2)]
    clustered = assign_topology_families(records, threshold=0)
    assert clustered[0].topology_family == clustered[1].topology_family
    assert clustered[0].topology_family != clustered[2].topology_family
    metrics = regression_metrics(np.asarray([[0.0], [2.0]]), np.asarray([[0.0], [1.0]]), ["y"])
    assert np.isclose(metrics["y"]["mae"], 0.5)
    assert np.isclose(metrics["y"]["rmse"], np.sqrt(0.5))


def test_actual_sqlite_adapter_reads_1300_audited_matrices() -> None:
    config = load_config(ROOT / "configs" / "motor_regression_ripple.yaml")
    paths = resolved_run_paths(config, ROOT)
    if not all((path / "dataset.sqlite").is_file() for path in paths):
        return
    records = load_records(paths, target_specs=config["data"]["targets"])
    assert len(records) == 1300
    assert all(record.matrix_18x10.shape == (18, 10) for record in records)
    assert all(np.isfinite(record.targets).all() for record in records)
    assert len({record.chromosome_hash for record in records}) == 1300
    records = assign_topology_families(records, threshold=12)
    split = grouped_split(
        records,
        group_mode="topology_family",
        fractions=(0.70, 0.15, 0.15),
        seed=20260809,
        band_balance_weight=4.0,
        target_balance_weight=1.0,
        search_restarts=128,
    )
    groups = {
        name: {records[index].topology_family for index in indices}
        for name, indices in split.items()
    }
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])
    assert all(
        {records[index].fitness_band for index in indices} == {f"B{i}" for i in range(1, 8)}
        for indices in split.values()
    )
