from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from .hashing import file_sha256, json_sha256


REQUIRED_ANGLES = (0.0, 3.0, 6.0, 9.0, 12.0, 15.0)


@dataclass(frozen=True, slots=True)
class DatasetRunConfig:
    path: Path
    data: dict[str, Any]
    config_hash: str

    @property
    def geometry_mode(self) -> str:
        return str(self.data["candidate_model"]["geometry_mode"])

    @property
    def angles_deg(self) -> tuple[float, ...]:
        return tuple(float(x) for x in self.data["physics"]["torque_angles_deg"])

    @property
    def random_seed(self) -> int:
        return int(self.data["clonalg"]["random_seed"])

    @property
    def target_valid_samples(self) -> int:
        return int(self.data["limits"]["target_valid_unique_samples"])

    def verify_input_files(self) -> None:
        checks = (
            ("seed", "mat_path", "mat_sha256"),
            ("reference_model", "fem_path", "fem_sha256"),
            ("candidate_model", "base_fem_path", "base_fem_sha256"),
        )
        for section, path_key, hash_key in checks:
            path = Path(self.data[section][path_key]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            expected = self.data[section].get(hash_key)
            if expected and file_sha256(path) != expected:
                raise ValueError(f"{section} file hash changed: {path}")


def _validate_config(data: dict[str, Any]) -> None:
    required_sections = {
        "schema_version",
        "compatibility_mode",
        "selection",
        "training_label",
        "seed",
        "reference_model",
        "candidate_model",
        "physics",
        "constraints",
        "historical_objective",
        "clonalg",
        "fitness_bands",
        "limits",
        "recovery",
    }
    missing = required_sections - set(data)
    if missing:
        raise ValueError(f"dataset config missing sections: {sorted(missing)}")
    if data["compatibility_mode"] != "historical_compatibility":
        raise ValueError("this stage only supports historical_compatibility")
    if data["candidate_model"]["geometry_mode"] not in {
        "historical_inset",
        "merged_copper_v5",
    }:
        raise ValueError("unsupported candidate geometry mode")
    angles = tuple(float(x) for x in data["physics"]["torque_angles_deg"])
    if angles != REQUIRED_ANGLES:
        raise ValueError(f"historical compatibility angles must be {REQUIRED_ANGLES}")
    if len(set(angles)) != len(angles):
        raise ValueError("torque angles must be unique")
    if data["selection"] != {"metric": "historical_j", "direction": "minimize"}:
        raise ValueError("historical compatibility must minimize historical_j")
    if data["training_label"] != "torque_ratio":
        raise ValueError("the continuous training label must be torque_ratio")
    targets = data["fitness_bands"]["targets"]
    if set(targets) != {f"B{i}" for i in range(1, 8)}:
        raise ValueError("fitness band targets must define B1 through B7")
    if any(int(value) < 0 for value in targets.values()):
        raise ValueError("fitness band targets cannot be negative")
    limits = data["limits"]
    for key in (
        "target_valid_unique_samples",
        "maximum_candidates",
        "maximum_femm_evaluations",
        "maximum_generations",
    ):
        if int(limits[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if not math.isfinite(float(limits["maximum_runtime_hours"])) or float(
        limits["maximum_runtime_hours"]
    ) <= 0:
        raise ValueError("maximum_runtime_hours must be finite and positive")


def load_dataset_config(path: Path) -> DatasetRunConfig:
    resolved = Path(path).resolve()
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("dataset config must be a JSON object")
    _validate_config(data)
    config = DatasetRunConfig(path=resolved, data=data, config_hash=json_sha256(data))
    config.verify_input_files()
    return config
