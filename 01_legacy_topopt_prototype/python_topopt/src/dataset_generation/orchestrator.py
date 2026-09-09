from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
from typing import Any, Callable, Sequence
from datetime import datetime, timezone

from constraints.connectivity import historical_precheck
from data_io.matlab import load_seed
from encoding.chromosome import Chromosome
from encoding.layout import matlab_vector_to_grid
from objectives.torque import ObjectiveConfig, evaluate_historical_objective

from .backend import AngleEvaluationBackend
from .checkpoint import write_checkpoint
from .config import DatasetRunConfig, load_dataset_config
from .database import DatasetDatabase, utc_now
from .hashing import file_sha256, json_sha256, physical_sample_hash, source_manifest_hash
from .state import (
    AngleStatus,
    CandidateStatus,
    RunStatus,
    SampleStatus,
    atomic_write_json,
    atomic_write_bytes,
)


CommitCallback = Callable[[], None]
StopRequested = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class RunLayout:
    root: Path

    @property
    def config(self) -> Path:
        return self.root / "run_config.json"

    @property
    def state(self) -> Path:
        return self.root / "run_state.json"

    @property
    def database(self) -> Path:
        return self.root / "dataset.sqlite"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def latest_checkpoint(self) -> Path:
        return self.checkpoints / "latest.json"

    @property
    def samples(self) -> Path:
        return self.root / "samples"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def create(self) -> None:
        self.root.mkdir(parents=True, exist_ok=False)
        for folder in (self.checkpoints, self.samples, self.logs, self.reports):
            folder.mkdir()


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    candidate_id: str
    sample_id: str | None
    status: str
    torque_ratio: float | None
    historical_j: float | None
    fitness_band: str | None
    rejection_reasons: tuple[str, ...] = ()
    duplicate: bool = False


class DatasetRunManager:
    def __init__(self, layout: RunLayout, config: DatasetRunConfig, run_id: str) -> None:
        self.layout = layout
        self.config = config
        self.run_id = run_id
        self.database = DatasetDatabase(layout.database)
        objective = config.data["historical_objective"]
        self.objective_config = ObjectiveConfig(
            minimum_average_torque_nm=float(objective["minimum_average_torque_nm"]),
            penalty_coefficient=float(objective["average_torque_penalty_coefficient"]),
            ripple_denominator_floor_nm=float(objective["ripple_denominator_floor_nm"]),
            invalid_base_objective=float(objective["invalid_base_objective"]),
            floating_iron_sigmoid_k=float(objective["floating_iron_sigmoid_k"]),
            floating_iron_sigmoid_weight=float(objective["floating_iron_sigmoid_weight"]),
            floating_copper_sigmoid_k=float(objective["floating_copper_sigmoid_k"]),
            floating_copper_sigmoid_weight=float(objective["floating_copper_sigmoid_weight"]),
        )
        self.current: dict[str, Any] = {
            "generation": 0,
            "candidate_order": 0,
            "current_sample_id": None,
            "current_angle_deg": None,
        }
        self.progress_callback: Callable[[dict[str, Any]], None] | None = None

    @classmethod
    def create(
        cls,
        *,
        project_root: Path,
        run_name: str,
        config_path: Path,
    ) -> "DatasetRunManager":
        project = Path(project_root).resolve()
        if not run_name or any(char in run_name for char in "\\/:*?\"<>|"):
            raise ValueError("run name is empty or contains invalid path characters")
        layout = RunLayout(project / "runs" / run_name)
        layout.create()
        config = load_dataset_config(config_path)
        envelope = {"config_hash": config.config_hash, "config": config.data}
        atomic_write_json(layout.config, envelope)
        database = DatasetDatabase(layout.database)
        database.initialize()
        source_hash = source_manifest_hash(project)
        database.create_run(
            run_id=run_name,
            config=config.data,
            config_hash=config.config_hash,
            source_hash=source_hash,
            code_version=f"source-manifest:{source_hash}",
        )
        manager = cls(layout, config, run_name)
        manager.write_checkpoint({"kind": "initial", "generation": 0})
        manager.write_state()
        return manager

    @classmethod
    def resume(cls, run_directory: Path) -> "DatasetRunManager":
        layout = RunLayout(Path(run_directory).resolve())
        envelope = json.loads(layout.config.read_text(encoding="utf-8"))
        config_data = envelope["config"]
        if json_sha256(config_data) != envelope.get("config_hash"):
            raise ValueError("run_config.json hash mismatch")
        # Validate through the same strict loader without changing the immutable file.
        temp_config = DatasetRunConfig(
            path=layout.config,
            data=config_data,
            config_hash=envelope["config_hash"],
        )
        temp_config.verify_input_files()
        database = DatasetDatabase(layout.database)
        rows = database.query_all("SELECT run_id,config_hash FROM runs")
        if len(rows) != 1:
            raise ValueError("run database must contain exactly one run")
        if rows[0]["config_hash"] != temp_config.config_hash:
            raise ValueError("database and immutable config hashes differ")
        manager = cls(layout, temp_config, str(rows[0]["run_id"]))
        manager.database.interrupt_running_angles(manager.run_id)
        manager.write_state()
        return manager

    @property
    def seed_chromosome(self) -> Chromosome:
        seed = self.config.data["seed"]
        chromosome = load_seed(Path(seed["mat_path"]), variable=seed["mat_variable"])
        if chromosome.sha256() != seed["chromosome_sha256"]:
            raise ValueError("decoded seed chromosome hash changed")
        return chromosome

    def sample_directory(self, sample_id: str) -> Path:
        folder = self.layout.samples / sample_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "femm").mkdir(exist_ok=True)
        return folder

    def write_checkpoint(self, algorithm_state: dict[str, Any]) -> None:
        payload = {
            "run_id": self.run_id,
            "config_hash": self.config.config_hash,
            "current": self.current,
            "algorithm_state": algorithm_state,
            "created_at": utc_now(),
        }
        envelope = write_checkpoint(self.layout.latest_checkpoint, payload)
        generation = int(algorithm_state.get("generation", self.current.get("generation", 0)))
        generation_path = self.layout.checkpoints / f"generation_{generation:06d}.json"
        if not generation_path.exists() or algorithm_state.get("generation_completed"):
            write_checkpoint(generation_path, payload)
        population = algorithm_state.get("population", [])
        self.database.record_checkpoint(
            run_id=self.run_id,
            generation=generation,
            candidate_order=(
                int(algorithm_state["candidate_order"])
                if algorithm_state.get("candidate_order") is not None
                else None
            ),
            checkpoint_path=str(self.layout.latest_checkpoint),
            checkpoint_hash=str(envelope["payload_hash"]),
            rng_state=algorithm_state.get("rng_state"),
            population_hash=json_sha256(population),
            completed_generation=bool(algorithm_state.get("generation_completed", False)),
        )

    def write_state(self) -> dict[str, Any]:
        summary = self.database.progress_summary(self.run_id)
        current_sample = self.current.get("current_sample_id")
        completed: list[float] = []
        pending: list[float] = []
        if current_sample:
            for row in self.database.angle_rows(current_sample):
                target = completed if row["status"] == AngleStatus.COMPLETED else pending
                target.append(float(row["angle_deg"]))
        state = {
            **summary,
            **self.current,
            "completed_angles_deg": completed,
            "pending_angles_deg": pending,
            "updated_at": utc_now(),
        }
        atomic_write_json(self.layout.state, state, keep_backup=True)
        with (self.layout.logs / "run.log").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                f"{state['updated_at']} run={self.run_id} status={state['status']} "
                f"generation={state.get('generation')} sample={state.get('current_sample_id')} "
                f"angle={state.get('current_angle_deg')} valid={state['valid_unique_samples']}\n"
            )
            handle.flush()
        if self.progress_callback is not None:
            self.progress_callback(state)
        return state

    def _refresh_rejected_candidates_csv(self) -> None:
        rows = self.database.query_all(
            """SELECT candidate_id,generation,candidate_order,chromosome_hash,
                      rejection_reasons_json,chromosome_json,created_at
               FROM candidates WHERE run_id=? AND validity_status='rejected'
               ORDER BY candidate_pk""",
            (self.run_id,),
        )
        lines = [
            "candidate_id,generation,candidate_order,chromosome_hash,rejection_reasons,chromosome_180,created_at\n"
        ]
        import csv
        import io

        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        for row in rows:
            writer.writerow(
                (
                    row["candidate_id"],
                    row["generation"],
                    row["candidate_order"],
                    row["chromosome_hash"],
                    row["rejection_reasons_json"],
                    row["chromosome_json"],
                    row["created_at"],
                )
            )
        lines.append(stream.getvalue())
        atomic_write_bytes(
            self.layout.logs / "rejected_candidates.csv",
            "".join(lines).encode("utf-8"),
        )

    def stop_condition(self) -> str | None:
        """Return a configured terminal/guard condition at a safe boundary."""

        summary = self.database.progress_summary(self.run_id)
        limits = self.config.data["limits"]
        minimum_band = int(limits["minimum_samples_per_band"])
        target_met = int(summary["valid_unique_samples"]) >= int(
            limits["target_valid_unique_samples"]
        ) and all(int(value) >= minimum_band for value in summary["fitness_bands"].values())
        if target_met:
            return "target_reached"
        if int(summary["candidate_total"]) >= int(limits["maximum_candidates"]):
            return "maximum_candidates"
        femm_samples = self.database.query_all(
            """SELECT COUNT(*) n FROM samples s WHERE s.run_id=? AND s.sample_kind='candidate'
               AND EXISTS(SELECT 1 FROM angle_evaluations a
                          WHERE a.sample_id=s.sample_id AND a.attempt>0)""",
            (self.run_id,),
        )[0]["n"]
        if int(femm_samples) >= int(limits["maximum_femm_evaluations"]):
            return "maximum_femm_evaluations"
        created = datetime.fromisoformat(str(self.database.get_run(self.run_id)["created_at"]))
        elapsed_hours = (datetime.now(timezone.utc).astimezone() - created).total_seconds() / 3600
        if elapsed_hours >= float(limits["maximum_runtime_hours"]):
            return "maximum_runtime_hours"
        return None

    def create_reference_sample(self) -> str:
        reference = self.config.data["reference_model"]
        physical_key = json_sha256(
            {
                "kind": "reference",
                "fem_sha256": reference["fem_sha256"],
                "angles_deg": list(self.config.angles_deg),
                "physics": self.config.data["physics"],
            }
        )
        existing = self.database.find_sample(self.run_id, physical_key)
        if existing is not None:
            return str(existing["sample_id"])
        sample_id = self.database.create_sample(
            run_id=self.run_id,
            candidate_id=None,
            sample_kind="reference",
            physical_key_hash=physical_key,
            chromosome=None,
            material_matrix=None,
            chromosome_hash=None,
            config_hash=self.config.config_hash,
            geometry_mode=reference["geometry_mode"],
            angles_deg=self.config.angles_deg,
            femm_version=self.config.data["physics"]["femm_version"],
        )
        folder = self.sample_directory(sample_id)
        atomic_write_json(
            folder / "model_source.json",
            {
                "sample_id": sample_id,
                "kind": "reference",
                "fem_path": reference["fem_path"],
                "fem_sha256": reference["fem_sha256"],
                "circuit_mode": reference["circuit_mode"],
            },
        )
        return sample_id

    def register_candidate(
        self,
        chromosome: Chromosome,
        *,
        generation: int,
        candidate_order: int,
        source: str,
        parent_candidate_id: str | None = None,
        parent_rank: int | None = None,
        clone_index: int | None = None,
        mutation_operator: str | None = None,
        mutation_strength: float | None = None,
        mutation_indices: Sequence[int] = (),
        hamming_distance_to_parent: int | None = None,
        lineage_id: str | None = None,
    ) -> tuple[str, str | None, tuple[str, ...], bool]:
        precheck = historical_precheck(
            chromosome,
            minimum_copper_island_cells=int(
                self.config.data["constraints"]["minimum_copper_island_cells"]
            ),
        )
        reasons = precheck.reasons
        if not precheck.has_any_copper and not reasons:
            reasons = ("NO_COPPER",)
        rejected_historical_j = (
            float(precheck.objective_if_rejected)
            if precheck.objective_if_rejected is not None
            else (
                float(self.objective_config.invalid_base_objective)
                if not precheck.has_any_copper
                else None
            )
        )
        chromosome_hash = chromosome.sha256()
        physical_key = physical_sample_hash(
            chromosome_hash=chromosome_hash,
            config_hash=self.config.config_hash,
            geometry_mode=self.config.geometry_mode,
            angles_deg=self.config.angles_deg,
        )
        existing = self.database.find_sample(self.run_id, physical_key)
        rejected = bool(reasons)
        candidate_id = self.database.insert_candidate(
            {
                "run_id": self.run_id,
                "generation": generation,
                "candidate_order": candidate_order,
                "chromosome": list(chromosome.genes),
                "chromosome_hash": chromosome_hash,
                "physical_key_hash": physical_key,
                "source": source,
                "parent_candidate_id": parent_candidate_id,
                "parent_rank": parent_rank,
                "clone_index": clone_index,
                "mutation_operator": mutation_operator,
                "mutation_strength": mutation_strength,
                "mutation_indices": list(mutation_indices),
                "hamming_distance_to_parent": hamming_distance_to_parent,
                "lineage_id": lineage_id or f"L-{chromosome_hash[:16]}",
                "status": CandidateStatus.REJECTED
                if rejected
                else (CandidateStatus.DUPLICATE if existing is not None else CandidateStatus.VALID),
                "validity_status": "rejected"
                if rejected
                else ("duplicate" if existing is not None else "valid"),
                "rejection_reasons": list(reasons),
                "duplicate_of_sample_id": str(existing["sample_id"]) if existing else None,
            }
        )
        if rejected:
            self.database.update_candidate_result(
                candidate_id,
                status=CandidateStatus.REJECTED,
                torque_ratio=None,
                historical_j=rejected_historical_j,
                fitness_band=None,
            )
            self._refresh_rejected_candidates_csv()
            return candidate_id, None, tuple(reasons), False
        if existing is not None:
            self.database.mark_candidate_duplicate(candidate_id, str(existing["sample_id"]))
            return candidate_id, str(existing["sample_id"]), (), True
        grid = matlab_vector_to_grid(chromosome).astype(int).tolist()
        sample_id = self.database.create_sample(
            run_id=self.run_id,
            candidate_id=candidate_id,
            sample_kind="candidate",
            physical_key_hash=physical_key,
            chromosome=chromosome.genes,
            material_matrix=grid,
            chromosome_hash=chromosome_hash,
            config_hash=self.config.config_hash,
            geometry_mode=self.config.geometry_mode,
            angles_deg=self.config.angles_deg,
            femm_version=self.config.data["physics"]["femm_version"],
        )
        folder = self.sample_directory(sample_id)
        atomic_write_json(
            folder / "chromosome.json",
            {
                "sample_id": sample_id,
                "candidate_id": candidate_id,
                "chromosome_180": list(chromosome.genes),
                "material_matrix_18x10": grid,
                "chromosome_hash": chromosome_hash,
                "physical_key_hash": physical_key,
                "geometry_mode": self.config.geometry_mode,
            },
        )
        return candidate_id, sample_id, (), False

    def _historical_j(self, chromosome: Chromosome, torques: Sequence[float]) -> float:
        precheck = historical_precheck(chromosome)
        return evaluate_historical_objective(
            torques,
            precheck=precheck,
            config=self.objective_config,
        ).objective

    def _write_torque_snapshot(self, sample_id: str) -> None:
        folder = self.sample_directory(sample_id)
        atomic_write_json(
            folder / "torques.json",
            {
                "sample_id": sample_id,
                "angles": [
                    {
                        "angle_deg": float(item["angle_deg"]),
                        "status": item["status"],
                        "torque_nm": item["torque"],
                        "attempt": int(item["attempt"]),
                    }
                    for item in self.database.angle_rows(sample_id)
                ],
            },
            keep_backup=True,
        )

    def _finalize_completed_sample(
        self, sample_id: str, chromosome: Chromosome | None
    ) -> None:
        sample = self.database.get_sample(sample_id)
        if sample["status"] == SampleStatus.CLASSIFIED:
            return
        rows = self.database.angle_rows(sample_id)
        if not rows or any(row["status"] != AngleStatus.COMPLETED for row in rows):
            raise ValueError(f"cannot finalize incomplete sample {sample_id}")
        torques = [float(row["torque"]) for row in rows]
        if sample["sample_kind"] == "reference":
            result = self.database.finalize_sample(
                run_id=self.run_id,
                sample_id=sample_id,
                t_avg_ref=None,
                historical_j=None,
            )
            self.database.set_reference(
                self.run_id,
                sample_id=sample_id,
                t_avg_ref=float(result["T_avg"]),
            )
        else:
            if chromosome is None:
                raise ValueError("candidate finalization requires chromosome")
            run = self.database.get_run(self.run_id)
            if run["t_avg_ref"] is None:
                raise RuntimeError("reference torque must be committed before candidate evaluation")
            historical_j = self._historical_j(chromosome, torques)
            result = self.database.finalize_sample(
                run_id=self.run_id,
                sample_id=sample_id,
                t_avg_ref=float(run["t_avg_ref"]),
                historical_j=historical_j,
            )
        atomic_write_json(
            self.sample_directory(sample_id) / "result.json",
            {"sample_id": sample_id, **result},
        )

    def evaluate_sample(
        self,
        sample_id: str,
        *,
        backend: AngleEvaluationBackend,
        chromosome: Chromosome | None,
        stop_requested: StopRequested = lambda: False,
        commit_callback: CommitCallback = lambda: None,
    ) -> str:
        sample = self.database.get_sample(sample_id)
        self.current["current_sample_id"] = sample_id
        maximum_attempts = int(self.config.data["recovery"]["max_attempts_per_angle"])
        folder = self.sample_directory(sample_id)
        for row in self.database.angle_rows(sample_id):
            angle = float(row["angle_deg"])
            if row["status"] == AngleStatus.COMPLETED:
                continue
            if stop_requested():
                self.write_state()
                return "paused"
            self.current["current_angle_deg"] = angle
            while True:
                attempt_dir = folder / "femm" / f"angle_{angle:g}_attempt_{int(row['attempt']) + 1}"
                attempt_dir.mkdir(parents=True, exist_ok=True)
                attempt = self.database.mark_angle_running(
                    run_id=self.run_id,
                    sample_id=sample_id,
                    angle_deg=angle,
                    work_directory=str(attempt_dir),
                )
                self.write_state()
                try:
                    result = backend.evaluate_angle(
                        sample_id=sample_id,
                        sample_kind=str(sample["sample_kind"]),
                        chromosome=chromosome,
                        angle_deg=angle,
                        work_directory=attempt_dir,
                    )
                    self.database.complete_angle(
                        run_id=self.run_id,
                        sample_id=sample_id,
                        angle_deg=angle,
                        torque=result.torque_nm,
                        metrics=result.metrics(),
                    )
                    self._write_torque_snapshot(sample_id)
                    commit_callback()
                    self.write_state()
                    break
                except Exception as exc:
                    final = attempt >= maximum_attempts
                    status = AngleStatus.FAILED if final else AngleStatus.INTERRUPTED
                    self.database.fail_angle(
                        run_id=self.run_id,
                        sample_id=sample_id,
                        angle_deg=angle,
                        status=status,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                    commit_callback()
                    self.write_state()
                    if final:
                        return "failed"
                    row = self.database.angle_rows(sample_id)[
                        [float(item["angle_deg"]) for item in self.database.angle_rows(sample_id)].index(angle)
                    ]
        self.current["current_angle_deg"] = None
        self._finalize_completed_sample(sample_id, chromosome)
        commit_callback()
        self.write_state()
        return "completed"

    def evaluate_samples_parallel(
        self,
        requests: Sequence[tuple[str, Chromosome]],
        *,
        backend: AngleEvaluationBackend,
        max_workers: int,
        stop_requested: StopRequested = lambda: False,
        commit_callback: CommitCallback = lambda: None,
    ) -> dict[str, str]:
        """Evaluate different candidate chromosomes concurrently; SQLite stays main-thread only."""

        worker_count = int(max_workers)
        if worker_count <= 1 or len(requests) <= 1:
            return {
                sample_id: self.evaluate_sample(
                    sample_id,
                    backend=backend,
                    chromosome=chromosome,
                    stop_requested=stop_requested,
                    commit_callback=commit_callback,
                )
                for sample_id, chromosome in requests
            }
        if worker_count > 6:
            raise ValueError("parallel FEMM worker count cannot exceed 6")
        sample_ids = [sample_id for sample_id, _ in requests]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("parallel sample requests must be unique")
        chromosomes = {sample_id: chromosome for sample_id, chromosome in requests}
        samples = {sample_id: self.database.get_sample(sample_id) for sample_id in sample_ids}
        if any(row["sample_kind"] != "candidate" for row in samples.values()):
            raise ValueError("parallel evaluation currently accepts candidate samples only")

        status = {sample_id: "active" for sample_id in sample_ids}
        order = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        in_flight: set[str] = set()
        futures: dict[Future[Any], tuple[str, float, int]] = {}
        maximum_attempts = int(self.config.data["recovery"]["max_attempts_per_angle"])
        self.current["active_sample_ids"] = list(sample_ids)
        self.current["parallel_femm_workers"] = worker_count

        def schedule_next(executor: ThreadPoolExecutor, sample_id: str) -> bool:
            if status[sample_id] != "active" or sample_id in in_flight:
                return False
            rows = self.database.angle_rows(sample_id)
            pending = next((row for row in rows if row["status"] != AngleStatus.COMPLETED), None)
            if pending is None:
                self._finalize_completed_sample(sample_id, chromosomes[sample_id])
                status[sample_id] = "completed"
                commit_callback()
                self.write_state()
                return False
            if stop_requested():
                status[sample_id] = "paused"
                return False
            angle = float(pending["angle_deg"])
            folder = self.sample_directory(sample_id)
            attempt_dir = (
                folder / "femm" / f"angle_{angle:g}_attempt_{int(pending['attempt']) + 1}"
            )
            attempt_dir.mkdir(parents=True, exist_ok=True)
            attempt = self.database.mark_angle_running(
                run_id=self.run_id,
                sample_id=sample_id,
                angle_deg=angle,
                work_directory=str(attempt_dir),
            )
            self.current["current_sample_id"] = sample_id
            self.current["current_angle_deg"] = angle
            self.write_state()
            future = executor.submit(
                backend.evaluate_angle,
                sample_id=sample_id,
                sample_kind="candidate",
                chromosome=chromosomes[sample_id],
                angle_deg=angle,
                work_directory=attempt_dir,
            )
            futures[future] = (sample_id, angle, attempt)
            in_flight.add(sample_id)
            return True

        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="femm-candidate") as executor:
            for sample_id in sample_ids[:worker_count]:
                schedule_next(executor, sample_id)
            while futures:
                completed, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for future in sorted(
                    completed,
                    key=lambda item: (
                        order[futures[item][0]],
                        futures[item][1],
                    ),
                ):
                    sample_id, angle, attempt = futures.pop(future)
                    in_flight.remove(sample_id)
                    try:
                        result = future.result()
                        self.database.complete_angle(
                            run_id=self.run_id,
                            sample_id=sample_id,
                            angle_deg=angle,
                            torque=result.torque_nm,
                            metrics=result.metrics(),
                        )
                        self._write_torque_snapshot(sample_id)
                    except Exception as exc:
                        final = attempt >= maximum_attempts
                        failure_status = AngleStatus.FAILED if final else AngleStatus.INTERRUPTED
                        self.database.fail_angle(
                            run_id=self.run_id,
                            sample_id=sample_id,
                            angle_deg=angle,
                            status=failure_status,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                        if final:
                            status[sample_id] = "failed"
                    commit_callback()
                    self.write_state()
                    if status[sample_id] == "active":
                        schedule_next(executor, sample_id)
                for sample_id in sample_ids:
                    if len(futures) >= worker_count:
                        break
                    schedule_next(executor, sample_id)

        for sample_id in sample_ids:
            if status[sample_id] == "active":
                rows = self.database.angle_rows(sample_id)
                if all(row["status"] == AngleStatus.COMPLETED for row in rows):
                    self._finalize_completed_sample(sample_id, chromosomes[sample_id])
                    status[sample_id] = "completed"
                    commit_callback()
                else:
                    status[sample_id] = "paused" if stop_requested() else "failed"
        self.current["active_sample_ids"] = []
        self.current["parallel_femm_workers"] = None
        self.current["current_sample_id"] = None
        self.current["current_angle_deg"] = None
        self.write_state()
        return status

    def evaluate_candidate(
        self,
        chromosome: Chromosome,
        *,
        backend: AngleEvaluationBackend,
        generation: int,
        candidate_order: int,
        source: str,
        stop_requested: StopRequested = lambda: False,
        commit_callback: CommitCallback = lambda: None,
        **lineage: Any,
    ) -> CandidateEvaluation:
        self.current["generation"] = generation
        self.current["candidate_order"] = candidate_order
        candidate_id, sample_id, reasons, duplicate = self.register_candidate(
            chromosome,
            generation=generation,
            candidate_order=candidate_order,
            source=source,
            **lineage,
        )
        if sample_id is None:
            rejected_row = self.database.query_all(
                "SELECT historical_j FROM candidates WHERE candidate_id=?", (candidate_id,)
            )[0]
            return CandidateEvaluation(
                candidate_id=candidate_id,
                sample_id=None,
                status="rejected",
                torque_ratio=None,
                historical_j=float(rejected_row["historical_j"]),
                fitness_band=None,
                rejection_reasons=reasons,
            )
        sample = self.database.get_sample(sample_id)
        if sample["status"] != SampleStatus.CLASSIFIED:
            status = self.evaluate_sample(
                sample_id,
                backend=backend,
                chromosome=chromosome,
                stop_requested=stop_requested,
                commit_callback=commit_callback,
            )
            if status != "completed":
                self.database.update_candidate_status(candidate_id, status)
                return CandidateEvaluation(
                    candidate_id=candidate_id,
                    sample_id=sample_id,
                    status=status,
                    torque_ratio=None,
                    historical_j=None,
                    fitness_band=None,
                    duplicate=duplicate,
                )
            sample = self.database.get_sample(sample_id)
        self.database.update_candidate_result(
            candidate_id,
            status=CandidateStatus.COMPLETED,
            torque_ratio=float(sample["torque_ratio"]),
            historical_j=float(sample["historical_j"]),
            fitness_band=str(sample["fitness_band"]),
        )
        return CandidateEvaluation(
            candidate_id=candidate_id,
            sample_id=sample_id,
            status="completed",
            torque_ratio=float(sample["torque_ratio"]),
            historical_j=float(sample["historical_j"]),
            fitness_band=str(sample["fitness_band"]),
            duplicate=duplicate,
        )
