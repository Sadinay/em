from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from dataset_generation.hashing import file_sha256, json_sha256


@dataclass(frozen=True, slots=True)
class ImprovedSamplingConfig:
    path: Path
    data: dict[str, Any]
    config_hash: str
    physics_hash: str
    sampling_hash: str
    repair_config_hash: str

    @property
    def parent_target(self) -> int:
        return int(self.data["parent_archive_target"])

    @property
    def random_seed(self) -> int:
        return int(self.data["random_seed"])


def _validate(data: dict[str, Any]) -> None:
    if data.get("mode") != "improved_dataset_sampling":
        raise ValueError("mode must be improved_dataset_sampling")
    target = int(data["parent_archive_target"])
    campaigns = int(data["campaign_count"])
    per_campaign = int(data["parents_per_campaign"])
    clusters = int(data["topology_cluster_target"])
    per_cluster = int(data["parents_per_cluster"])
    if target != campaigns * per_campaign:
        raise ValueError("parent target must equal campaign_count * parents_per_campaign")
    if target != clusters * per_cluster:
        raise ValueError("parent target must equal topology_cluster_target * parents_per_cluster")
    if data.get("initial_parent_reuse") is not False:
        raise ValueError("the approved design requires initial_parent_reuse=false")
    if data["proposal"].get("hamming_admission_mode") not in {"observe_only", "enforce"}:
        raise ValueError("unsupported Hamming admission mode")
    weights = data["proposal"]["operator_weights"]
    if not weights or abs(sum(float(x) for x in weights.values()) - 1.0) > 1e-12:
        raise ValueError("operator weights must sum to one")
    scale_probability = sum(
        float(spec["probability"]) for spec in data["proposal"]["scales"].values()
    )
    if abs(scale_probability - 1.0) > 1e-12:
        raise ValueError("scale probabilities must sum to one")
    for name, spec in data["proposal"]["scales"].items():
        if not 1 <= int(spec["minimum_cells"]) <= int(spec["maximum_cells"]) <= 180:
            raise ValueError(f"invalid cell range for scale {name}")
    fixed = data["constraints"].get("fixed_cells", {})
    for key, value in fixed.items():
        if not 0 <= int(key) < 180 or int(value) not in (0, 1, 2):
            raise ValueError(f"invalid fixed cell {key}:{value}")
    if int(data["repair"]["maximum_repair_hamming"]) < 0:
        raise ValueError("maximum_repair_hamming cannot be negative")


def improved_sampling_config_from_data(
    data: dict[str, Any], *, source_path: Path
) -> ImprovedSamplingConfig:
    resolved = Path(source_path).resolve()
    _validate(data)
    seed = data["seed"]
    if file_sha256(Path(seed["mat_path"])) != seed["mat_sha256"]:
        raise ValueError("seed MAT hash mismatch")
    source = Path(data["physics"]["source_config_path"])
    physics_source = json.loads(source.read_text(encoding="utf-8"))
    if json_sha256(physics_source) != data["physics"]["source_config_sha256"]:
        raise ValueError("physics source config hash mismatch")
    physics_material = {
        "reference_model": physics_source["reference_model"],
        "candidate_model": physics_source["candidate_model"],
        "physics": physics_source["physics"],
        "constraints": physics_source["constraints"],
        "historical_objective": physics_source["historical_objective"],
        "reference_t_avg_nm": data["physics"]["reference_t_avg_nm"],
        "formula_version": data["physics"]["formula_version"],
    }
    sampling_material = {key: value for key, value in data.items() if key != "physics"}
    return ImprovedSamplingConfig(
        path=resolved,
        data=data,
        config_hash=json_sha256(data),
        physics_hash=json_sha256(physics_material),
        sampling_hash=json_sha256(sampling_material),
        repair_config_hash=json_sha256(data["repair"]),
    )


def load_improved_sampling_config(path: Path) -> ImprovedSamplingConfig:
    resolved = Path(path).resolve()
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("improved sampling config must be a JSON object")
    return improved_sampling_config_from_data(data, source_path=resolved)
