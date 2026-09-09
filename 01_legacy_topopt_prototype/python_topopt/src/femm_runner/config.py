from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FemmRunConfig:
    base_model: Path
    work_root: Path
    torque_angles_deg: tuple[float, ...] = (0.0, 3.0, 6.0, 9.0, 12.0, 15.0)
    stator_current_a: float = 3.5
    rotor_group: int = 1
    air_gap_boundary_name: str = "Air gap"
    timeout_seconds: float = 900.0
    maximum_retries: int = 1
    failure_objective: float = 10_000_000.0
    require_clean_process_state: bool = True
    geometry_mode: str = "historical_inset"

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_model", Path(self.base_model).resolve())
        object.__setattr__(self, "work_root", Path(self.work_root).resolve())
        if not self.base_model.is_file():
            raise FileNotFoundError(self.base_model)
        if not self.torque_angles_deg:
            raise ValueError("at least one torque angle is required")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.maximum_retries < 0:
            raise ValueError("maximum_retries cannot be negative")
        if self.geometry_mode not in {"historical_inset", "merged_copper_v5"}:
            raise ValueError(f"unsupported geometry_mode {self.geometry_mode!r}")
