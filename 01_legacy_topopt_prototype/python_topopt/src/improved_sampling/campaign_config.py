from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from dataset_generation.hashing import json_sha256


@dataclass(frozen=True, slots=True)
class CampaignEvolutionConfig:
    path: Path
    data: dict[str, Any]
    config_hash: str

    @property
    def random_seed(self) -> int:
        return int(self.data["random_seed"])

    @property
    def target(self) -> int:
        return int(self.data["target_valid_unique_samples"])


def campaign_config_from_data(data: dict[str, Any], path: Path) -> CampaignEvolutionConfig:
    if data.get("schema_version") != 1 or data.get("mode") != "improved_dataset_campaigns":
        raise ValueError("unsupported improved campaign configuration")
    for key in (
        "random_seed", "target_valid_unique_samples", "offspring_per_campaign_generation",
        "elite_count", "minimum_completed_generations", "maximum_generations",
        "maximum_proposals", "hamming_admission",
    ):
        if key not in data:
            raise ValueError(f"campaign configuration missing {key}")
    if int(data["target_valid_unique_samples"]) <= 0:
        raise ValueError("target_valid_unique_samples must be positive")
    offspring = int(data["offspring_per_campaign_generation"])
    elite = int(data["elite_count"])
    if offspring <= 0 or not 1 <= elite <= offspring:
        raise ValueError("elite_count must be between 1 and offspring count")
    minimum_generations = int(data["minimum_completed_generations"])
    if minimum_generations <= 0 or minimum_generations > int(data["maximum_generations"]):
        raise ValueError("minimum_completed_generations is outside the configured limits")
    selection = data.get("parent_selection", {})
    exploitation = float(selection.get("objective_exploitation_probability", 0.0))
    pressure = float(selection.get("objective_rank_pressure", 0.0))
    if not 0.0 <= exploitation <= 1.0 or pressure < 0.0:
        raise ValueError("invalid objective parent-selection parameters")
    if set(data["hamming_admission"]) != {"small", "medium", "large"}:
        raise ValueError("hamming_admission must define small, medium and large")
    for scale, spec in data["hamming_admission"].items():
        low = int(spec["minimum_archive_distance"])
        high = int(spec["maximum_parent_distance"])
        if low < 0 or high < low or high > 180:
            raise ValueError(f"invalid Hamming limits for {scale}")
    return CampaignEvolutionConfig(Path(path).resolve(), data, json_sha256(data))


def load_campaign_config(path: Path) -> CampaignEvolutionConfig:
    resolved = Path(path).resolve()
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("campaign configuration must be a JSON object")
    return campaign_config_from_data(data, resolved)
