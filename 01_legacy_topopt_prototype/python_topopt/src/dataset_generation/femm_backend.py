from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any

from encoding.chromosome import Chromosome
from femm_runner.process import run_isolated_worker

from .backend import AngleEvaluationResult
from .config import DatasetRunConfig
from .state import atomic_write_json


COUNT_PATTERN = re.compile(
    r"^\[(NumPoints|NumSegments|NumArcSegments|NumBlockLabels)\]\s*=\s*(\d+)",
    re.MULTILINE,
)

GEOMETRY_BUILDER_VERSIONS = {
    "historical_inset": "historical_inset_v1",
    "merged_copper_v5": "merged_copper_v5_air_regions_v5",
}


def _geometry_counts(model: Path) -> dict[str, int]:
    text = Path(model).read_text(encoding="utf-8", errors="replace")
    raw = {name: int(value) for name, value in COUNT_PATTERN.findall(text)}
    return {
        "geometry_points": raw.get("NumPoints", 0),
        "geometry_segments": raw.get("NumSegments", 0) + raw.get("NumArcSegments", 0),
        "material_labels": raw.get("NumBlockLabels", 0),
    }


@dataclass(slots=True)
class IsolatedFemmAngleBackend:
    """One isolated FEMM worker per angle with a reusable prepared candidate FEM."""

    config: DatasetRunConfig
    max_workers: int = 1
    hide_windows: bool = True

    def __post_init__(self) -> None:
        if not 1 <= int(self.max_workers) <= 6:
            raise ValueError("FEMM worker count must be between 1 and 6")

    def _common_job(self) -> dict[str, Any]:
        physics = self.config.data["physics"]
        return {
            "stator_current_a": float(physics["stator_current_peak_a"]),
            "rotor_group": int(physics["rotor_group"]),
            "air_gap_boundary_name": physics["air_gap_boundary_name"],
            "electrical_angle_multiplier": float(physics["electrical_angle_multiplier"]),
            "electrical_phase_offset_deg": float(physics["electrical_phase_offset_deg"]),
            "hide_femm_window": bool(self.hide_windows),
        }

    def _run_job(self, job_path: Path, result_path: Path, manifest_path: Path) -> dict[str, Any]:
        recovery = self.config.data["recovery"]
        outcome = run_isolated_worker(
            job_path,
            timeout_seconds=float(recovery["angle_timeout_seconds"]),
            require_clean_process_state=int(self.max_workers) == 1,
            process_manifest_path=manifest_path,
        )
        (job_path.parent / "worker.stdout.log").write_text(outcome.stdout, encoding="utf-8")
        (job_path.parent / "worker.stderr.log").write_text(outcome.stderr, encoding="utf-8")
        if outcome.timed_out:
            raise TimeoutError(
                f"FEMM worker exceeded {recovery['angle_timeout_seconds']} seconds; "
                f"terminated owned PIDs {outcome.terminated_pids}"
            )
        if not result_path.is_file():
            raise RuntimeError(f"FEMM worker returned {outcome.return_code} without result.json")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("status") != "SUCCEEDED":
            raise RuntimeError(
                f"FEMM {payload.get('error_type', 'failure')}: {payload.get('error', '')}"
            )
        payload["worker_pid"] = outcome.worker_pid
        payload["terminated_pids"] = list(outcome.terminated_pids)
        return payload

    def _prepare_candidate(
        self,
        *,
        chromosome: Chromosome,
        femm_root: Path,
    ) -> tuple[Path, float, dict[str, int]]:
        prepared = femm_root / "prepared"
        model = prepared / "model.fem"
        result = prepared / "result.json"
        if model.is_file() and result.is_file():
            payload = json.loads(result.read_text(encoding="utf-8"))
            if (
                payload.get("status") == "SUCCEEDED"
                and payload.get("geometry_builder_version")
                == GEOMETRY_BUILDER_VERSIONS[self.config.geometry_mode]
            ):
                return model, 0.0, _geometry_counts(model)
        prepared.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(self.config.data["candidate_model"]["base_fem_path"]), model)
        job_path = prepared / "job.json"
        manifest = prepared / "process_manifest.json"
        job = {
            **self._common_job(),
            "operation": "prepare_candidate",
            "model_path": str(model),
            "result_path": str(result),
            "process_manifest_path": str(manifest),
            "genes": list(chromosome.genes),
            "geometry_mode": self.config.geometry_mode,
            "geometry_builder_version": GEOMETRY_BUILDER_VERSIONS[
                self.config.geometry_mode
            ],
            "torque_angles_deg": [],
            "circuit_mode": self.config.data["candidate_model"]["circuit_mode"],
        }
        atomic_write_json(job_path, job)
        payload = self._run_job(job_path, result, manifest)
        topology_seconds = float(payload.get("timings", {}).get("topology_seconds", 0.0))
        return model, topology_seconds, _geometry_counts(model)

    def evaluate_angle(
        self,
        *,
        sample_id: str,
        sample_kind: str,
        chromosome: Chromosome | None,
        angle_deg: float,
        work_directory: Path,
    ) -> AngleEvaluationResult:
        work_directory = Path(work_directory)
        work_directory.mkdir(parents=True, exist_ok=True)
        if sample_kind == "reference":
            source_model = Path(self.config.data["reference_model"]["fem_path"])
            circuit_mode = self.config.data["reference_model"]["circuit_mode"]
            build_time = 0.0
            counts = _geometry_counts(source_model)
        else:
            if chromosome is None:
                raise ValueError("candidate FEMM evaluation requires a chromosome")
            source_model, build_time, counts = self._prepare_candidate(
                chromosome=chromosome,
                femm_root=work_directory.parent,
            )
            circuit_mode = self.config.data["candidate_model"]["circuit_mode"]
        model = work_directory / "model.fem"
        result_path = work_directory / "result.json"
        manifest = work_directory / "process_manifest.json"
        shutil.copy2(source_model, model)
        job_path = work_directory / "job.json"
        job = {
            **self._common_job(),
            "operation": "solve_angle",
            "model_path": str(model),
            "result_path": str(result_path),
            "process_manifest_path": str(manifest),
            "torque_angles_deg": [float(angle_deg)],
            "circuit_mode": circuit_mode,
            "explicit_mesh_before_solve": True,
        }
        atomic_write_json(job_path, job)
        payload = self._run_job(job_path, result_path, manifest)
        torques = payload.get("torque_values_nm", [])
        if len(torques) != 1 or not math.isfinite(float(torques[0])):
            raise ValueError(f"invalid single-angle FEMM torque result: {torques!r}")
        timing = payload["timings"]["angles"][0]
        return AngleEvaluationResult(
            torque_nm=float(torques[0]),
            build_time=build_time,
            mesh_time=float(timing.get("mesh_seconds") or 0.0),
            solve_time=float(timing.get("solve_seconds") or 0.0),
            postprocess_time=float(timing.get("postprocess_seconds") or 0.0),
            total_time=float(timing.get("total_angle_seconds") or 0.0) + build_time,
            mesh_nodes=int(timing["mesh_nodes"]),
            mesh_elements=int(timing["mesh_elements"]),
            worker_pid=int(payload["worker_pid"]),
            femm_pids=tuple(int(pid) for pid in payload.get("terminated_pids", ())),
            metadata=counts,
        )

    def close(self) -> None:
        return None
