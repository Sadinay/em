from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import time
import uuid

from constraints.connectivity import historical_precheck
from encoding.chromosome import Chromosome
from femm_runner.config import FemmRunConfig
from femm_runner.process import run_isolated_worker
from objectives.torque import ObjectiveConfig, evaluate_historical_objective

from .base import EvaluationResult


@dataclass(slots=True)
class IsolatedFemmEvaluator:
    """Single-individual FEMM evaluator isolated behind a worker subprocess."""

    run_config: FemmRunConfig
    objective_config: ObjectiveConfig = ObjectiveConfig()
    calls: int = 0

    def evaluate(self, chromosome: Chromosome) -> EvaluationResult:
        self.calls += 1
        precheck = historical_precheck(chromosome)
        if precheck.rejected or not precheck.has_any_copper:
            result = evaluate_historical_objective(
                None, precheck=precheck, config=self.objective_config
            )
            return EvaluationResult(
                objective=result.objective,
                status=result.status,
                average_torque_nm=result.average_torque_nm,
                torque_ripple_ratio=result.torque_ripple_ratio,
                rejection_reasons=result.rejection_reasons,
                metadata={"femm_started": False},
            )

        failures: list[dict] = []
        for attempt in range(self.run_config.maximum_retries + 1):
            case_id = f"case_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}"
            case_dir = self.run_config.work_root / case_id
            case_dir.mkdir(parents=True, exist_ok=False)
            model_path = case_dir / "model.fem"
            result_path = case_dir / "result.json"
            job_path = case_dir / "job.json"
            shutil.copy2(self.run_config.base_model, model_path)
            job = {
                "case_id": case_id,
                "model_path": str(model_path),
                "result_path": str(result_path),
                "genes": list(chromosome.genes),
                "geometry_mode": self.run_config.geometry_mode,
                "torque_angles_deg": list(self.run_config.torque_angles_deg),
                "stator_current_a": self.run_config.stator_current_a,
                "rotor_group": self.run_config.rotor_group,
                "air_gap_boundary_name": self.run_config.air_gap_boundary_name,
            }
            job_path.write_text(
                json.dumps(job, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            try:
                outcome = run_isolated_worker(
                    job_path,
                    timeout_seconds=self.run_config.timeout_seconds,
                    require_clean_process_state=(
                        self.run_config.require_clean_process_state
                    ),
                )
            except Exception as exc:
                failures.append(
                    {
                        "attempt": attempt + 1,
                        "case_dir": str(case_dir),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                continue
            (case_dir / "worker.stdout.log").write_text(
                outcome.stdout, encoding="utf-8"
            )
            (case_dir / "worker.stderr.log").write_text(
                outcome.stderr, encoding="utf-8"
            )
            if outcome.timed_out:
                failures.append(
                    {
                        "attempt": attempt + 1,
                        "case_dir": str(case_dir),
                        "error_type": "Timeout",
                        "error": f"exceeded {self.run_config.timeout_seconds} seconds",
                        "terminated_pids": list(outcome.terminated_pids),
                    }
                )
                continue
            if result_path.is_file():
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                payload = {
                    "status": "FAILED",
                    "error_type": "MissingResult",
                    "error": f"worker return code {outcome.return_code}",
                }
            if payload.get("status") != "SUCCEEDED":
                failures.append(
                    {
                        "attempt": attempt + 1,
                        "case_dir": str(case_dir),
                        **payload,
                    }
                )
                continue

            torques = tuple(float(value) for value in payload["torque_values_nm"])
            result = evaluate_historical_objective(
                torques, precheck=precheck, config=self.objective_config
            )
            return EvaluationResult(
                objective=result.objective,
                status=result.status,
                average_torque_nm=result.average_torque_nm,
                torque_ripple_ratio=result.torque_ripple_ratio,
                torque_values_nm=torques,
                metadata={
                    "femm_started": True,
                    "case_dir": str(case_dir),
                    "attempt": attempt + 1,
                    "timings": payload.get("timings", {}),
                },
            )

        return EvaluationResult(
            objective=float(self.run_config.failure_objective),
            status="FEMM_FAILED",
            rejection_reasons=("FEMM_FAILURE",),
            metadata={
                "femm_started": True,
                "failures": failures,
                "failure_objective_is_operational_not_historical": True,
            },
        )
