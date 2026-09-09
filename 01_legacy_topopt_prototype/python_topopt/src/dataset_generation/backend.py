from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from encoding.chromosome import Chromosome
from encoding.layout import matlab_vector_to_grid


@dataclass(frozen=True, slots=True)
class AngleEvaluationResult:
    torque_nm: float
    build_time: float = 0.0
    mesh_time: float = 0.0
    solve_time: float = 0.0
    postprocess_time: float = 0.0
    total_time: float = 0.0
    mesh_nodes: int | None = None
    mesh_elements: int | None = None
    worker_pid: int | None = None
    femm_pids: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def metrics(self) -> dict[str, Any]:
        return {
            "build_time": self.build_time,
            "mesh_time": self.mesh_time,
            "solve_time": self.solve_time,
            "postprocess_time": self.postprocess_time,
            "total_time": self.total_time,
            "mesh_nodes": self.mesh_nodes,
            "mesh_elements": self.mesh_elements,
            "worker_pid": self.worker_pid,
            "femm_pids": list(self.femm_pids),
            **self.metadata,
        }


class AngleEvaluationBackend(Protocol):
    def evaluate_angle(
        self,
        *,
        sample_id: str,
        sample_kind: str,
        chromosome: Chromosome | None,
        angle_deg: float,
        work_directory: Path,
    ) -> AngleEvaluationResult:
        ...

    def close(self) -> None:
        ...


@dataclass(slots=True)
class DeterministicAngleMockBackend:
    """Deterministic per-angle backend used to validate durable orchestration."""

    fail_once_at: set[tuple[str, float]] = field(default_factory=set)
    timeout_once_at: set[tuple[str, float]] = field(default_factory=set)
    calls: list[tuple[str, float]] = field(default_factory=list)
    _failed: set[tuple[str, float]] = field(default_factory=set)
    _timed_out: set[tuple[str, float]] = field(default_factory=set)
    max_workers: int = 1

    @staticmethod
    def _shape(angle_deg: float) -> float:
        values = {0.0: -1.0, 3.0: -0.4, 6.0: 0.2, 9.0: 1.0, 12.0: 0.4, 15.0: -0.2}
        if float(angle_deg) not in values:
            raise ValueError(f"unsupported mock angle {angle_deg}")
        return values[float(angle_deg)]

    def evaluate_angle(
        self,
        *,
        sample_id: str,
        sample_kind: str,
        chromosome: Chromosome | None,
        angle_deg: float,
        work_directory: Path,
    ) -> AngleEvaluationResult:
        key = (sample_id, float(angle_deg))
        self.calls.append(key)
        if key in self.timeout_once_at and key not in self._timed_out:
            self._timed_out.add(key)
            raise TimeoutError(f"mock timeout for {sample_id} at {angle_deg} degrees")
        if key in self.fail_once_at and key not in self._failed:
            self._failed.add(key)
            raise RuntimeError(f"mock FEMM failure for {sample_id} at {angle_deg} degrees")
        if sample_kind == "reference":
            mean = 1.0
            amplitude = 0.02
            nodes = 1000
            elements = 1900
        else:
            if chromosome is None:
                raise ValueError("candidate mock evaluation requires a chromosome")
            grid = matlab_vector_to_grid(chromosome)
            iron = int(np.count_nonzero(grid == 1))
            copper = int(np.count_nonzero(grid == 2))
            transitions = int(
                np.count_nonzero(grid[1:, :] != grid[:-1, :])
                + np.count_nonzero(grid[:, 1:] != grid[:, :-1])
            )
            mean = 0.45 + 0.0045 * iron + 0.004 * copper
            amplitude = 0.01 + 0.00025 * transitions
            nodes = 2000 + transitions
            elements = 2 * nodes
        torque = mean + amplitude * self._shape(float(angle_deg))
        return AngleEvaluationResult(
            torque_nm=float(torque),
            build_time=0.01,
            mesh_time=0.02,
            solve_time=0.03,
            postprocess_time=0.001,
            total_time=0.061,
            mesh_nodes=nodes,
            mesh_elements=elements,
            metadata={"mock": True, "work_directory": str(work_directory)},
        )

    def close(self) -> None:
        return None
