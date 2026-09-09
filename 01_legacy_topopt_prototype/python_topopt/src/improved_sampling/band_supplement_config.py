from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from dataset_generation.hashing import json_sha256


TARGET_BANDS = tuple(f"B{index}" for index in range(2, 8))


@dataclass(frozen=True, slots=True)
class BandSupplementConfig:
    path: Path
    data: dict[str, Any]
    config_hash: str

    @property
    def random_seed(self) -> int:
        return int(self.data["random_seed"])

    @property
    def target(self) -> int:
        return int(self.data["target_new_physics_samples"])


def band_supplement_config_from_data(
    data: dict[str, Any], path: Path
) -> BandSupplementConfig:
    if data.get("schema_version") != 1 or data.get("mode") != "band_supplement_v1":
        raise ValueError("unsupported band supplement configuration")
    required = {
        "random_seed",
        "target_new_physics_samples",
        "population_per_band",
        "offspring_per_band_generation",
        "elite_count",
        "maximum_generations",
        "maximum_proposals",
        "total_dataset_band_targets",
        "eligible_parent_bands",
        "scale_probabilities",
        "hamming_admission",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(f"band supplement configuration missing {sorted(missing)}")
    if int(data["target_new_physics_samples"]) <= 0:
        raise ValueError("target_new_physics_samples must be positive")
    population = int(data["population_per_band"])
    offspring_spec = data["offspring_per_band_generation"]
    if not isinstance(offspring_spec, dict) or set(offspring_spec) != set(TARGET_BANDS):
        raise ValueError("offspring_per_band_generation must define B2 through B7")
    offspring_values = [int(value) for value in offspring_spec.values()]
    elite = int(data["elite_count"])
    if population <= 0 or any(value <= 0 for value in offspring_values) or not 1 <= elite <= population:
        raise ValueError("invalid supplement population, offspring or elite count")
    if int(data["maximum_generations"]) <= 0 or int(data["maximum_proposals"]) <= 0:
        raise ValueError("supplement limits must be positive")
    for key in ("total_dataset_band_targets", "eligible_parent_bands", "scale_probabilities"):
        if set(data[key]) != set(TARGET_BANDS):
            raise ValueError(f"{key} must define B2 through B7")
    for band, value in data["total_dataset_band_targets"].items():
        if int(value) < 0:
            raise ValueError(f"negative target for {band}")
    allowed = {f"B{index}" for index in range(1, 8)}
    for band, parents in data["eligible_parent_bands"].items():
        if not parents or not set(parents) <= allowed:
            raise ValueError(f"invalid eligible parent bands for {band}")
    for band, probabilities in data["scale_probabilities"].items():
        if set(probabilities) != {"small", "medium", "large"}:
            raise ValueError(f"invalid scale probabilities for {band}")
        values = [float(value) for value in probabilities.values()]
        if any(value < 0 for value in values) or sum(values) <= 0:
            raise ValueError(f"invalid scale probability values for {band}")
    if set(data["hamming_admission"]) != {"small", "medium", "large"}:
        raise ValueError("hamming_admission must define small, medium and large")
    for scale, spec in data["hamming_admission"].items():
        low = int(spec["minimum_archive_distance"])
        high = int(spec["maximum_parent_distance"])
        if low < 0 or high < low or high > 180:
            raise ValueError(f"invalid Hamming limits for {scale}")
    threshold = float(data.get("b7_parent_minimum_torque_ratio", 0.98))
    if not 0.9 <= threshold <= 1.0:
        raise ValueError("b7_parent_minimum_torque_ratio must be within [0.9, 1.0]")
    return BandSupplementConfig(Path(path).resolve(), data, json_sha256(data))


def load_band_supplement_config(path: Path) -> BandSupplementConfig:
    resolved = Path(path).resolve()
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("band supplement configuration must be a JSON object")
    return band_supplement_config_from_data(data, resolved)
