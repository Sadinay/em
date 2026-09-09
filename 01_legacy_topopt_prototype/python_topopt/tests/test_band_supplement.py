from __future__ import annotations

import copy
import json
from pathlib import Path

from improved_sampling.band_supplement_config import load_band_supplement_config
from improved_sampling.band_supplement_runner import BandSupplementRunner, _physics_identity


ROOT = Path(__file__).resolve().parents[1]


def test_supplement_config_allocates_exactly_300_over_ten_generations() -> None:
    config = load_band_supplement_config(ROOT / "configs" / "band_supplement_300.json")
    per_generation = sum(
        int(value) for value in config.data["offspring_per_band_generation"].values()
    )
    assert per_generation == 30
    assert per_generation * 10 == config.target
    assert config.data["scale_probabilities"]["B7"]["small"] == 0.75


def test_physics_identity_ignores_run_limits_but_not_solver_physics() -> None:
    data = json.loads((ROOT / "configs" / "dataset_merged_v5.json").read_text(encoding="utf-8"))
    changed_limits = copy.deepcopy(data)
    changed_limits["limits"]["target_valid_unique_samples"] = 300
    assert _physics_identity(data) == _physics_identity(changed_limits)
    changed_current = copy.deepcopy(data)
    changed_current["physics"]["stator_current_peak_a"] = 4.0
    assert _physics_identity(data) != _physics_identity(changed_current)


def test_b7_parent_sort_prefers_measured_b7_then_highest_b6() -> None:
    rows = [
        {"fitness_band": "B6", "torque_ratio": 0.999, "historical_j": 0.1, "chromosome_hash": "a"},
        {"fitness_band": "B7", "torque_ratio": 1.01, "historical_j": 0.2, "chromosome_hash": "b"},
        {"fitness_band": "B6", "torque_ratio": 0.981, "historical_j": 0.05, "chromosome_hash": "c"},
    ]
    ordered = sorted(rows, key=lambda row: BandSupplementRunner._parent_sort_key(row, "B7"))
    assert [row["chromosome_hash"] for row in ordered] == ["b", "a", "c"]
