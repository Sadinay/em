from __future__ import annotations

from contextlib import contextmanager
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Iterator, Sequence
from datetime import datetime, timezone

from .bands import classify_torque_ratio
from .state import AngleStatus, RunStatus, SampleStatus, validate_run_transition


SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    code_version TEXT NOT NULL,
    compatibility_mode TEXT NOT NULL,
    selection_metric TEXT NOT NULL,
    selection_direction TEXT NOT NULL,
    geometry_mode TEXT NOT NULL,
    reference_model_path TEXT NOT NULL,
    reference_model_hash TEXT NOT NULL,
    reference_sample_id TEXT,
    t_avg_ref REAL,
    targets_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    paused_at TEXT,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS candidates (
    candidate_pk INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT UNIQUE,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE RESTRICT,
    generation INTEGER NOT NULL,
    candidate_order INTEGER NOT NULL,
    chromosome_json TEXT NOT NULL,
    chromosome_hash TEXT NOT NULL,
    physical_key_hash TEXT NOT NULL,
    source TEXT NOT NULL,
    parent_candidate_id TEXT REFERENCES candidates(candidate_id),
    parent_rank INTEGER,
    clone_index INTEGER,
    mutation_operator TEXT,
    mutation_strength REAL,
    mutation_indices_json TEXT,
    hamming_distance_to_parent INTEGER,
    lineage_id TEXT NOT NULL,
    status TEXT NOT NULL,
    validity_status TEXT,
    rejection_reasons_json TEXT NOT NULL DEFAULT '[]',
    duplicate_of_sample_id TEXT,
    torque_ratio REAL,
    historical_j REAL,
    fitness_band TEXT,
    created_at TEXT NOT NULL,
    validated_at TEXT,
    completed_at TEXT,
    UNIQUE(run_id, generation, candidate_order)
);

CREATE INDEX IF NOT EXISTS idx_candidates_run_hash
ON candidates(run_id, chromosome_hash);
CREATE INDEX IF NOT EXISTS idx_candidates_run_status
ON candidates(run_id, status);

CREATE TABLE IF NOT EXISTS samples (
    sample_pk INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id TEXT UNIQUE,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE RESTRICT,
    first_candidate_id TEXT REFERENCES candidates(candidate_id),
    sample_kind TEXT NOT NULL DEFAULT 'candidate',
    physical_key_hash TEXT NOT NULL,
    chromosome_json TEXT,
    material_matrix_json TEXT,
    chromosome_hash TEXT,
    config_hash TEXT NOT NULL,
    geometry_mode TEXT NOT NULL,
    angles_json TEXT NOT NULL,
    status TEXT NOT NULL,
    completed_angle_count INTEGER NOT NULL DEFAULT 0,
    t_avg REAL,
    t_min REAL,
    t_max REAL,
    t_ripple REAL,
    t_avg_ref REAL,
    torque_ratio REAL,
    historical_j REAL,
    fitness_band TEXT,
    selection_fitness REAL,
    build_time REAL,
    mesh_time REAL,
    solve_time REAL,
    postprocess_time REAL,
    total_time REAL,
    mesh_nodes INTEGER,
    mesh_elements INTEGER,
    geometry_points INTEGER,
    geometry_segments INTEGER,
    material_labels INTEGER,
    femm_version TEXT,
    solver_status TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    error_type TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    UNIQUE(run_id, physical_key_hash)
);

CREATE INDEX IF NOT EXISTS idx_samples_run_status ON samples(run_id, status);
CREATE INDEX IF NOT EXISTS idx_samples_run_band ON samples(run_id, fitness_band);

CREATE TABLE IF NOT EXISTS angle_evaluations (
    sample_id TEXT NOT NULL REFERENCES samples(sample_id) ON DELETE RESTRICT,
    angle_deg REAL NOT NULL,
    status TEXT NOT NULL,
    torque REAL,
    attempt INTEGER NOT NULL DEFAULT 0,
    build_time REAL,
    mesh_time REAL,
    solve_time REAL,
    postprocess_time REAL,
    total_time REAL,
    mesh_nodes INTEGER,
    mesh_elements INTEGER,
    worker_pid INTEGER,
    femm_pids_json TEXT NOT NULL DEFAULT '[]',
    work_directory TEXT,
    error_type TEXT,
    error_message TEXT,
    started_at TEXT,
    completed_at TEXT,
    PRIMARY KEY(sample_id, angle_deg)
);

CREATE INDEX IF NOT EXISTS idx_angles_status ON angle_evaluations(status);

CREATE TABLE IF NOT EXISTS improved_proposals (
    proposal_pk INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE RESTRICT,
    proposal_order INTEGER NOT NULL,
    generation INTEGER NOT NULL,
    campaign_id TEXT NOT NULL,
    parent_candidate_id TEXT,
    operator TEXT NOT NULL,
    scale TEXT NOT NULL,
    raw_chromosome_json TEXT NOT NULL,
    raw_hash TEXT NOT NULL,
    repaired_chromosome_json TEXT NOT NULL,
    repaired_hash TEXT NOT NULL,
    repair_status TEXT NOT NULL,
    repair_hamming INTEGER NOT NULL,
    parent_hamming INTEGER NOT NULL,
    nearest_archive_hamming INTEGER,
    admission_status TEXT NOT NULL,
    rejection_reasons_json TEXT NOT NULL DEFAULT '[]',
    candidate_id TEXT REFERENCES candidates(candidate_id),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE(run_id, proposal_order)
);

CREATE INDEX IF NOT EXISTS idx_improved_proposals_run_status
ON improved_proposals(run_id, admission_status);

CREATE TABLE IF NOT EXISTS supplement_membership (
    sample_id TEXT PRIMARY KEY REFERENCES samples(sample_id) ON DELETE RESTRICT,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE RESTRICT,
    target_band TEXT NOT NULL,
    actual_band TEXT,
    physics_archive_member INTEGER NOT NULL,
    cnn_core_dataset_member INTEGER NOT NULL,
    reason TEXT NOT NULL,
    classified_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_supplement_membership_run_band
ON supplement_membership(run_id, actual_band, cnn_core_dataset_member);

CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_pk INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE RESTRICT,
    generation INTEGER NOT NULL,
    candidate_order INTEGER,
    checkpoint_path TEXT NOT NULL,
    checkpoint_hash TEXT NOT NULL,
    rng_state_json TEXT NOT NULL,
    population_hash TEXT NOT NULL,
    completed_generation INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_pk INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE RESTRICT,
    event_type TEXT NOT NULL,
    sample_id TEXT,
    candidate_id TEXT,
    generation INTEGER,
    angle_deg REAL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
"""


class DatasetDatabase:
    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        self.path = Path(path).resolve()
        self.read_only = bool(read_only)

    def _connect(self) -> sqlite3.Connection:
        if self.read_only:
            connection = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        if not self.read_only:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
        return connection

    @contextmanager
    def transaction(self, *, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        if self.read_only:
            raise RuntimeError("cannot start a write transaction on a read-only database")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        if self.read_only:
            raise RuntimeError("cannot initialize a read-only database")
        with self._connect() as connection:
            connection.executescript(SCHEMA_SQL)
            connection.execute(
                "INSERT OR REPLACE INTO schema_info(key,value) VALUES('schema_version',?)",
                (str(SCHEMA_VERSION),),
            )

    def upsert_supplement_membership(
        self,
        *,
        sample_id: str,
        run_id: str,
        target_band: str,
        actual_band: str | None,
        physics_archive_member: bool,
        cnn_core_dataset_member: bool,
        reason: str,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO supplement_membership(
                       sample_id,run_id,target_band,actual_band,physics_archive_member,
                       cnn_core_dataset_member,reason,classified_at
                   ) VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(sample_id) DO UPDATE SET
                       target_band=excluded.target_band,
                       actual_band=excluded.actual_band,
                       physics_archive_member=excluded.physics_archive_member,
                       cnn_core_dataset_member=excluded.cnn_core_dataset_member,
                       reason=excluded.reason,
                       classified_at=excluded.classified_at""",
                (
                    sample_id,
                    run_id,
                    target_band,
                    actual_band,
                    int(physics_archive_member),
                    int(cnn_core_dataset_member),
                    reason,
                    utc_now(),
                ),
            )

    @staticmethod
    def _event(
        connection: sqlite3.Connection,
        *,
        run_id: str,
        event_type: str,
        sample_id: str | None = None,
        candidate_id: str | None = None,
        generation: int | None = None,
        angle_deg: float | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        connection.execute(
            """INSERT INTO events(
                run_id,event_type,sample_id,candidate_id,generation,angle_deg,payload_json,created_at
            ) VALUES(?,?,?,?,?,?,?,?)""",
            (
                run_id,
                event_type,
                sample_id,
                candidate_id,
                generation,
                angle_deg,
                json.dumps(payload or {}, ensure_ascii=False, sort_keys=True),
                utc_now(),
            ),
        )

    def create_run(
        self,
        *,
        run_id: str,
        config: dict[str, Any],
        config_hash: str,
        source_hash: str,
        code_version: str,
    ) -> None:
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO runs(
                    run_id,status,config_json,config_hash,source_hash,code_version,
                    compatibility_mode,selection_metric,selection_direction,geometry_mode,
                    reference_model_path,reference_model_hash,targets_json,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    RunStatus.AWAITING_REFERENCE,
                    json.dumps(config, ensure_ascii=False, sort_keys=True),
                    config_hash,
                    source_hash,
                    code_version,
                    config["compatibility_mode"],
                    config["selection"]["metric"],
                    config["selection"]["direction"],
                    config["candidate_model"]["geometry_mode"],
                    config["reference_model"]["fem_path"],
                    config["reference_model"]["fem_sha256"],
                    json.dumps(config["fitness_bands"]["targets"], sort_keys=True),
                    now,
                ),
            )
            self._event(connection, run_id=run_id, event_type="run_created")

    def get_run(self, run_id: str) -> sqlite3.Row:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown run {run_id}")
        return row

    def set_run_status(self, run_id: str, target: str) -> None:
        with self.transaction() as connection:
            row = connection.execute("SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            validate_run_transition(str(row["status"]), str(target))
            timestamp_column = {
                RunStatus.RUNNING: "started_at",
                RunStatus.PAUSED: "paused_at",
                RunStatus.COMPLETED: "completed_at",
            }.get(target)
            if timestamp_column:
                connection.execute(
                    f"UPDATE runs SET status=?, {timestamp_column}=? WHERE run_id=?",
                    (target, utc_now(), run_id),
                )
            else:
                connection.execute("UPDATE runs SET status=? WHERE run_id=?", (target, run_id))
            self._event(
                connection,
                run_id=run_id,
                event_type=f"run_{target}",
                payload={"previous_status": row["status"]},
            )

    def set_reference(self, run_id: str, *, sample_id: str, t_avg_ref: float) -> None:
        value = float(t_avg_ref)
        if not math.isfinite(value) or value == 0:
            raise ValueError("reference average torque must be finite and non-zero")
        with self.transaction() as connection:
            current = connection.execute(
                "SELECT reference_sample_id,t_avg_ref FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if current is None:
                raise KeyError(run_id)
            if current["t_avg_ref"] is not None:
                if current["reference_sample_id"] != sample_id or not math.isclose(
                    float(current["t_avg_ref"]), value, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError("reference torque is immutable once committed")
                return
            connection.execute(
                "UPDATE runs SET reference_sample_id=?,t_avg_ref=? WHERE run_id=?",
                (sample_id, value, run_id),
            )
            self._event(
                connection,
                run_id=run_id,
                sample_id=sample_id,
                event_type="reference_committed",
                payload={"t_avg_ref": value},
            )

    @staticmethod
    def _allocate_id(connection: sqlite3.Connection, table: str, prefix: str) -> str:
        cursor = connection.execute(f"INSERT INTO {table} DEFAULT VALUES")
        return f"{prefix}{int(cursor.lastrowid):06d}"

    def insert_candidate(self, record: dict[str, Any]) -> str:
        with self.transaction() as connection:
            cursor = connection.execute(
                """INSERT INTO candidates(
                    run_id,generation,candidate_order,chromosome_json,chromosome_hash,
                    physical_key_hash,source,parent_candidate_id,parent_rank,clone_index,
                    mutation_operator,mutation_strength,mutation_indices_json,
                    hamming_distance_to_parent,lineage_id,status,validity_status,
                    rejection_reasons_json,duplicate_of_sample_id,created_at,validated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record["run_id"],
                    int(record["generation"]),
                    int(record["candidate_order"]),
                    json.dumps(record["chromosome"], separators=(",", ":")),
                    record["chromosome_hash"],
                    record["physical_key_hash"],
                    record["source"],
                    record.get("parent_candidate_id"),
                    record.get("parent_rank"),
                    record.get("clone_index"),
                    record.get("mutation_operator"),
                    record.get("mutation_strength"),
                    json.dumps(record.get("mutation_indices", []), separators=(",", ":")),
                    record.get("hamming_distance_to_parent"),
                    record["lineage_id"],
                    record["status"],
                    record.get("validity_status"),
                    json.dumps(record.get("rejection_reasons", []), separators=(",", ":")),
                    record.get("duplicate_of_sample_id"),
                    utc_now(),
                    utc_now() if record.get("validity_status") is not None else None,
                ),
            )
            candidate_id = f"C{int(cursor.lastrowid):06d}"
            connection.execute(
                "UPDATE candidates SET candidate_id=? WHERE candidate_pk=?",
                (candidate_id, cursor.lastrowid),
            )
            self._event(
                connection,
                run_id=record["run_id"],
                event_type="candidate_created",
                candidate_id=candidate_id,
                generation=int(record["generation"]),
                payload={"source": record["source"], "status": record["status"]},
            )
            return candidate_id

    def find_sample(self, run_id: str, physical_key_hash: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM samples WHERE run_id=? AND physical_key_hash=?",
                (run_id, physical_key_hash),
            ).fetchone()

    def get_sample(self, sample_id: str) -> sqlite3.Row:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM samples WHERE sample_id=?", (sample_id,)
            ).fetchone()
        if row is None:
            raise KeyError(sample_id)
        return row

    def update_candidate_status(self, candidate_id: str, status: str) -> None:
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT run_id,generation FROM candidates WHERE candidate_id=?", (candidate_id,)
            ).fetchone()
            if row is None:
                raise KeyError(candidate_id)
            connection.execute(
                "UPDATE candidates SET status=?,completed_at=? WHERE candidate_id=?",
                (status, utc_now(), candidate_id),
            )
            self._event(
                connection,
                run_id=row["run_id"],
                event_type=f"candidate_{status}",
                candidate_id=candidate_id,
                generation=int(row["generation"]),
            )

    def update_candidate_result(
        self,
        candidate_id: str,
        *,
        status: str,
        torque_ratio: float | None,
        historical_j: float | None,
        fitness_band: str | None,
    ) -> None:
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT run_id,generation FROM candidates WHERE candidate_id=?", (candidate_id,)
            ).fetchone()
            if row is None:
                raise KeyError(candidate_id)
            connection.execute(
                """UPDATE candidates SET status=?,torque_ratio=?,historical_j=?,fitness_band=?,
                   completed_at=? WHERE candidate_id=?""",
                (
                    status,
                    torque_ratio,
                    historical_j,
                    fitness_band,
                    utc_now(),
                    candidate_id,
                ),
            )
            self._event(
                connection,
                run_id=row["run_id"],
                event_type="candidate_result_committed",
                candidate_id=candidate_id,
                generation=int(row["generation"]),
                payload={
                    "status": status,
                    "torque_ratio": torque_ratio,
                    "historical_j": historical_j,
                    "fitness_band": fitness_band,
                },
            )

    def create_sample(
        self,
        *,
        run_id: str,
        candidate_id: str | None,
        sample_kind: str,
        physical_key_hash: str,
        chromosome: Sequence[int] | None,
        material_matrix: Sequence[Sequence[int]] | None,
        chromosome_hash: str | None,
        config_hash: str,
        geometry_mode: str,
        angles_deg: Sequence[float],
        femm_version: str,
    ) -> str:
        with self.transaction() as connection:
            cursor = connection.execute(
                """INSERT INTO samples(
                    run_id,first_candidate_id,sample_kind,physical_key_hash,chromosome_json,
                    material_matrix_json,chromosome_hash,config_hash,geometry_mode,angles_json,
                    status,femm_version,solver_status,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    candidate_id,
                    sample_kind,
                    physical_key_hash,
                    json.dumps(list(chromosome), separators=(",", ":")) if chromosome is not None else None,
                    json.dumps(material_matrix, separators=(",", ":")) if material_matrix is not None else None,
                    chromosome_hash,
                    config_hash,
                    geometry_mode,
                    json.dumps([float(x) for x in angles_deg], separators=(",", ":")),
                    SampleStatus.QUEUED,
                    femm_version,
                    "pending",
                    utc_now(),
                ),
            )
            sample_id = f"S{int(cursor.lastrowid):06d}"
            connection.execute(
                "UPDATE samples SET sample_id=? WHERE sample_pk=?",
                (sample_id, cursor.lastrowid),
            )
            connection.executemany(
                "INSERT INTO angle_evaluations(sample_id,angle_deg,status) VALUES(?,?,?)",
                [(sample_id, float(angle), AngleStatus.PENDING) for angle in angles_deg],
            )
            self._event(
                connection,
                run_id=run_id,
                event_type="sample_created",
                sample_id=sample_id,
                candidate_id=candidate_id,
                payload={"sample_kind": sample_kind, "geometry_mode": geometry_mode},
            )
            return sample_id

    def mark_candidate_duplicate(self, candidate_id: str, sample_id: str) -> None:
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT run_id,generation FROM candidates WHERE candidate_id=?", (candidate_id,)
            ).fetchone()
            if row is None:
                raise KeyError(candidate_id)
            connection.execute(
                """UPDATE candidates SET status='duplicate',validity_status='duplicate',
                   duplicate_of_sample_id=?,validated_at=? WHERE candidate_id=?""",
                (sample_id, utc_now(), candidate_id),
            )
            self._event(
                connection,
                run_id=row["run_id"],
                event_type="candidate_duplicate",
                sample_id=sample_id,
                candidate_id=candidate_id,
                generation=int(row["generation"]),
            )

    def angle_rows(self, sample_id: str) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return list(
                connection.execute(
                    "SELECT * FROM angle_evaluations WHERE sample_id=? ORDER BY angle_deg",
                    (sample_id,),
                ).fetchall()
            )

    def mark_angle_running(
        self,
        *,
        run_id: str,
        sample_id: str,
        angle_deg: float,
        work_directory: str | None = None,
        worker_pid: int | None = None,
    ) -> int:
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT status,attempt FROM angle_evaluations WHERE sample_id=? AND angle_deg=?",
                (sample_id, float(angle_deg)),
            ).fetchone()
            if row is None:
                raise KeyError((sample_id, angle_deg))
            if row["status"] == AngleStatus.COMPLETED:
                raise ValueError("a completed angle cannot be started again")
            attempt = int(row["attempt"]) + 1
            connection.execute(
                """UPDATE angle_evaluations SET status=?,attempt=?,torque=NULL,
                   error_type=NULL,error_message=NULL,started_at=?,completed_at=NULL,
                   work_directory=?,worker_pid=? WHERE sample_id=? AND angle_deg=?""",
                (
                    AngleStatus.RUNNING,
                    attempt,
                    utc_now(),
                    work_directory,
                    worker_pid,
                    sample_id,
                    float(angle_deg),
                ),
            )
            connection.execute(
                "UPDATE samples SET status=?,started_at=COALESCE(started_at,?),solver_status='running' WHERE sample_id=?",
                (SampleStatus.FEMM_RUNNING, utc_now(), sample_id),
            )
            self._event(
                connection,
                run_id=run_id,
                event_type="angle_started",
                sample_id=sample_id,
                angle_deg=float(angle_deg),
                payload={"attempt": attempt},
            )
            return attempt

    def complete_angle(
        self,
        *,
        run_id: str,
        sample_id: str,
        angle_deg: float,
        torque: float,
        metrics: dict[str, Any],
    ) -> None:
        torque_value = float(torque)
        if not math.isfinite(torque_value):
            raise ValueError("cannot commit a non-finite torque")
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT status FROM angle_evaluations WHERE sample_id=? AND angle_deg=?",
                (sample_id, float(angle_deg)),
            ).fetchone()
            if row is None:
                raise KeyError((sample_id, angle_deg))
            if row["status"] == AngleStatus.COMPLETED:
                existing = connection.execute(
                    "SELECT torque FROM angle_evaluations WHERE sample_id=? AND angle_deg=?",
                    (sample_id, float(angle_deg)),
                ).fetchone()
                if not math.isclose(float(existing["torque"]), torque_value, rel_tol=0, abs_tol=1e-12):
                    raise ValueError("completed angle already has a different torque")
                return
            connection.execute(
                """UPDATE angle_evaluations SET status=?,torque=?,build_time=?,mesh_time=?,
                   solve_time=?,postprocess_time=?,total_time=?,mesh_nodes=?,mesh_elements=?,
                   worker_pid=?,femm_pids_json=?,error_type=NULL,error_message=NULL,completed_at=?
                   WHERE sample_id=? AND angle_deg=?""",
                (
                    AngleStatus.COMPLETED,
                    torque_value,
                    metrics.get("build_time"),
                    metrics.get("mesh_time"),
                    metrics.get("solve_time"),
                    metrics.get("postprocess_time"),
                    metrics.get("total_time"),
                    metrics.get("mesh_nodes"),
                    metrics.get("mesh_elements"),
                    metrics.get("worker_pid"),
                    json.dumps(metrics.get("femm_pids", [])),
                    utc_now(),
                    sample_id,
                    float(angle_deg),
                ),
            )
            completed = int(
                connection.execute(
                    "SELECT COUNT(*) FROM angle_evaluations WHERE sample_id=? AND status=?",
                    (sample_id, AngleStatus.COMPLETED),
                ).fetchone()[0]
            )
            total = int(
                connection.execute(
                    "SELECT COUNT(*) FROM angle_evaluations WHERE sample_id=?", (sample_id,)
                ).fetchone()[0]
            )
            connection.execute(
                "UPDATE samples SET completed_angle_count=?,status=?,solver_status='partial' WHERE sample_id=?",
                (
                    completed,
                    SampleStatus.COMPLETED if completed == total else SampleStatus.PARTIALLY_COMPLETED,
                    sample_id,
                ),
            )
            if any(
                metrics.get(key) is not None
                for key in ("geometry_points", "geometry_segments", "material_labels")
            ):
                connection.execute(
                    """UPDATE samples SET
                       geometry_points=COALESCE(?,geometry_points),
                       geometry_segments=COALESCE(?,geometry_segments),
                       material_labels=COALESCE(?,material_labels)
                       WHERE sample_id=?""",
                    (
                        metrics.get("geometry_points"),
                        metrics.get("geometry_segments"),
                        metrics.get("material_labels"),
                        sample_id,
                    ),
                )
            self._event(
                connection,
                run_id=run_id,
                event_type="angle_completed",
                sample_id=sample_id,
                angle_deg=float(angle_deg),
                payload={"torque": torque_value, "completed_angles": completed, "total_angles": total},
            )

    def fail_angle(
        self,
        *,
        run_id: str,
        sample_id: str,
        angle_deg: float,
        status: str,
        error_type: str,
        error_message: str,
    ) -> None:
        if status not in {AngleStatus.FAILED, AngleStatus.INTERRUPTED, AngleStatus.CANCELLED}:
            raise ValueError("invalid failed angle status")
        with self.transaction() as connection:
            connection.execute(
                """UPDATE angle_evaluations SET status=?,torque=NULL,error_type=?,error_message=?,
                   completed_at=? WHERE sample_id=? AND angle_deg=?""",
                (status, error_type, error_message, utc_now(), sample_id, float(angle_deg)),
            )
            sample_status = (
                SampleStatus.INTERRUPTED if status == AngleStatus.INTERRUPTED else SampleStatus.FEMM_FAILED
            )
            connection.execute(
                "UPDATE samples SET status=?,solver_status=?,error_type=?,error_message=? WHERE sample_id=?",
                (sample_status, status, error_type, error_message, sample_id),
            )
            self._event(
                connection,
                run_id=run_id,
                event_type=f"angle_{status}",
                sample_id=sample_id,
                angle_deg=float(angle_deg),
                payload={"error_type": error_type, "error_message": error_message},
            )

    def interrupt_running_angles(self, run_id: str) -> int:
        with self.transaction() as connection:
            rows = connection.execute(
                """SELECT a.sample_id,a.angle_deg FROM angle_evaluations a
                   JOIN samples s ON s.sample_id=a.sample_id
                   WHERE s.run_id=? AND a.status=?""",
                (run_id, AngleStatus.RUNNING),
            ).fetchall()
            for row in rows:
                connection.execute(
                    """UPDATE angle_evaluations SET status=?,torque=NULL,error_type='ProcessInterrupted',
                       error_message='running angle found during resume',completed_at=?
                       WHERE sample_id=? AND angle_deg=?""",
                    (AngleStatus.INTERRUPTED, utc_now(), row["sample_id"], row["angle_deg"]),
                )
                connection.execute(
                    "UPDATE samples SET status=?,solver_status='interrupted' WHERE sample_id=?",
                    (SampleStatus.INTERRUPTED, row["sample_id"]),
                )
                self._event(
                    connection,
                    run_id=run_id,
                    event_type="angle_interrupted_on_resume",
                    sample_id=row["sample_id"],
                    angle_deg=float(row["angle_deg"]),
                )
            return len(rows)

    def finalize_sample(
        self,
        *,
        run_id: str,
        sample_id: str,
        t_avg_ref: float | None,
        historical_j: float | None,
    ) -> dict[str, Any]:
        with self.transaction() as connection:
            sample = connection.execute(
                "SELECT * FROM samples WHERE sample_id=? AND run_id=?", (sample_id, run_id)
            ).fetchone()
            if sample is None:
                raise KeyError(sample_id)
            expected_angles = [float(x) for x in json.loads(sample["angles_json"])]
            rows = connection.execute(
                "SELECT angle_deg,status,torque FROM angle_evaluations WHERE sample_id=? ORDER BY angle_deg",
                (sample_id,),
            ).fetchall()
            if len(rows) != len(expected_angles) or any(
                row["status"] != AngleStatus.COMPLETED for row in rows
            ):
                raise ValueError("sample cannot be finalized without every configured angle")
            actual_angles = [float(row["angle_deg"]) for row in rows]
            if actual_angles != sorted(expected_angles):
                raise ValueError("completed angle set does not match configured angles")
            torques = [float(row["torque"]) for row in rows]
            if not all(math.isfinite(value) for value in torques):
                raise ValueError("sample contains a non-finite torque")
            t_avg = sum(torques) / len(torques)
            t_min = min(torques)
            t_max = max(torques)
            t_ripple = (t_max - t_min) / max(abs(t_avg), 1e-6)
            ratio: float | None = None
            band: str | None = None
            selection: float | None = None
            if t_avg_ref is not None:
                reference = float(t_avg_ref)
                if not math.isfinite(reference) or reference == 0:
                    raise ValueError("invalid T_avg_ref")
                ratio = t_avg / reference
                band = classify_torque_ratio(ratio)
                selection = ratio
            totals = connection.execute(
                """SELECT SUM(build_time),SUM(mesh_time),SUM(solve_time),SUM(postprocess_time),
                          SUM(total_time),MAX(mesh_nodes),MAX(mesh_elements),SUM(attempt)
                   FROM angle_evaluations WHERE sample_id=?""",
                (sample_id,),
            ).fetchone()
            connection.execute(
                """UPDATE samples SET status=?,completed_angle_count=?,t_avg=?,t_min=?,t_max=?,
                   t_ripple=?,t_avg_ref=?,torque_ratio=?,historical_j=?,fitness_band=?,
                   selection_fitness=?,build_time=?,mesh_time=?,solve_time=?,postprocess_time=?,
                   total_time=?,mesh_nodes=?,mesh_elements=?,attempt_count=?,solver_status='completed',
                   error_type=NULL,error_message=NULL,completed_at=? WHERE sample_id=?""",
                (
                    SampleStatus.CLASSIFIED,
                    len(rows),
                    t_avg,
                    t_min,
                    t_max,
                    t_ripple,
                    t_avg_ref,
                    ratio,
                    historical_j,
                    band,
                    selection,
                    totals[0],
                    totals[1],
                    totals[2],
                    totals[3],
                    totals[4],
                    totals[5],
                    totals[6],
                    totals[7],
                    utc_now(),
                    sample_id,
                ),
            )
            self._event(
                connection,
                run_id=run_id,
                event_type="sample_classified",
                sample_id=sample_id,
                payload={
                    "t_avg": t_avg,
                    "t_ripple": t_ripple,
                    "torque_ratio": ratio,
                    "historical_j": historical_j,
                    "fitness_band": band,
                },
            )
            return {
                "angles_deg": actual_angles,
                "torque_values_nm": torques,
                "T_avg": t_avg,
                "T_min": t_min,
                "T_max": t_max,
                "T_ripple": t_ripple,
                "T_avg_ref": t_avg_ref,
                "torque_ratio": ratio,
                "historical_J": historical_j,
                "fitness_band": band,
            }

    def progress_summary(self, run_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            run = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if run is None:
                raise KeyError(run_id)
            candidate_counts = {
                row["status"]: int(row["n"])
                for row in connection.execute(
                    "SELECT status,COUNT(*) n FROM candidates WHERE run_id=? GROUP BY status", (run_id,)
                )
            }
            candidate_totals = connection.execute(
                """SELECT COUNT(*) total,COUNT(DISTINCT chromosome_hash) unique_chromosomes,
                          SUM(CASE WHEN validity_status='rejected' THEN 1 ELSE 0 END) rejected,
                          SUM(CASE WHEN validity_status='duplicate' THEN 1 ELSE 0 END) duplicates,
                          SUM(CASE WHEN validity_status='valid' THEN 1 ELSE 0 END) valid_first
                   FROM candidates WHERE run_id=?""",
                (run_id,),
            ).fetchone()
            rejection_reasons: dict[str, int] = {}
            for row in connection.execute(
                "SELECT rejection_reasons_json FROM candidates WHERE run_id=? AND validity_status='rejected'",
                (run_id,),
            ):
                for reason in json.loads(row[0]):
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            sample_counts = {
                row["status"]: int(row["n"])
                for row in connection.execute(
                    "SELECT status,COUNT(*) n FROM samples WHERE run_id=? GROUP BY status", (run_id,)
                )
            }
            bands = {f"B{i}": 0 for i in range(1, 8)}
            negative = 0
            for row in connection.execute(
                "SELECT fitness_band,COUNT(*) n FROM samples WHERE run_id=? AND status=? GROUP BY fitness_band",
                (run_id, SampleStatus.CLASSIFIED),
            ):
                if row["fitness_band"] == "negative_torque":
                    negative = int(row["n"])
                elif row["fitness_band"] in bands:
                    bands[row["fitness_band"]] = int(row["n"])
            angle_counts = {
                row["status"]: int(row["n"])
                for row in connection.execute(
                    """SELECT a.status,COUNT(*) n FROM angle_evaluations a
                       JOIN samples s ON s.sample_id=a.sample_id WHERE s.run_id=? GROUP BY a.status""",
                    (run_id,),
                )
            }
            recent = [
                float(row[0])
                for row in connection.execute(
                    """SELECT a.total_time FROM angle_evaluations a JOIN samples s ON s.sample_id=a.sample_id
                       WHERE s.run_id=? AND s.sample_kind='candidate'
                         AND a.status='completed' AND a.total_time IS NOT NULL
                       ORDER BY a.completed_at DESC LIMIT 50""",
                    (run_id,),
                )
            ]
        return {
            "run_id": run_id,
            "status": run["status"],
            "geometry_mode": run["geometry_mode"],
            "config_hash": run["config_hash"],
            "T_avg_ref": run["t_avg_ref"],
            "candidate_counts": candidate_counts,
            "candidate_total": int(candidate_totals["total"] or 0),
            "unique_chromosomes": int(candidate_totals["unique_chromosomes"] or 0),
            "rejected_candidates": int(candidate_totals["rejected"] or 0),
            "duplicate_candidates": int(candidate_totals["duplicates"] or 0),
            "new_valid_candidates": int(candidate_totals["valid_first"] or 0),
            "rejection_reasons": rejection_reasons,
            "sample_counts": sample_counts,
            "angle_counts": angle_counts,
            "fitness_bands": bands,
            "negative_torque": negative,
            "recent_mean_seconds_per_angle": sum(recent) / len(recent) if recent else None,
            "valid_unique_samples": sum(bands.values()) + negative,
        }

    def record_checkpoint(
        self,
        *,
        run_id: str,
        generation: int,
        candidate_order: int | None,
        checkpoint_path: str,
        checkpoint_hash: str,
        rng_state: dict[str, Any] | None,
        population_hash: str,
        completed_generation: bool,
    ) -> None:
        """Append an auditable checkpoint record in the same authoritative DB."""

        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO checkpoints(
                    run_id,generation,candidate_order,checkpoint_path,checkpoint_hash,
                    rng_state_json,population_hash,completed_generation,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    int(generation),
                    candidate_order,
                    checkpoint_path,
                    checkpoint_hash,
                    json.dumps(rng_state or {}, ensure_ascii=False, sort_keys=True),
                    population_hash,
                    1 if completed_generation else 0,
                    utc_now(),
                ),
            )
            self._event(
                connection,
                run_id=run_id,
                event_type="checkpoint_committed",
                generation=int(generation),
                payload={
                    "checkpoint_path": checkpoint_path,
                    "checkpoint_hash": checkpoint_hash,
                    "completed_generation": bool(completed_generation),
                },
            )

    def integrity_check(self) -> str:
        with self._connect() as connection:
            return str(connection.execute("PRAGMA integrity_check").fetchone()[0])

    def insert_improved_proposal(self, record: dict[str, Any]) -> int:
        with self.transaction() as connection:
            cursor = connection.execute(
                """INSERT INTO improved_proposals(
                       run_id,proposal_order,generation,campaign_id,parent_candidate_id,
                       operator,scale,raw_chromosome_json,raw_hash,
                       repaired_chromosome_json,repaired_hash,repair_status,
                       repair_hamming,parent_hamming,nearest_archive_hamming,
                       admission_status,rejection_reasons_json,metadata_json,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record["run_id"], int(record["proposal_order"]),
                    int(record["generation"]), record["campaign_id"],
                    record.get("parent_candidate_id"), record["operator"], record["scale"],
                    json.dumps(record["raw_chromosome"], separators=(",", ":")),
                    record["raw_hash"],
                    json.dumps(record["repaired_chromosome"], separators=(",", ":")),
                    record["repaired_hash"], record["repair_status"],
                    int(record["repair_hamming"]), int(record["parent_hamming"]),
                    record.get("nearest_archive_hamming"), record["admission_status"],
                    json.dumps(record.get("rejection_reasons", []), ensure_ascii=False),
                    json.dumps(record.get("metadata", {}), ensure_ascii=False, sort_keys=True),
                    utc_now(),
                ),
            )
            return int(cursor.lastrowid)

    def link_improved_proposal(self, proposal_order: int, candidate_id: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                """UPDATE improved_proposals SET candidate_id=?
                   WHERE run_id=(SELECT run_id FROM candidates WHERE candidate_id=?)
                     AND proposal_order=?""",
                (candidate_id, candidate_id, int(proposal_order)),
            )

    def query_all(self, sql: str, parameters: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return list(connection.execute(sql, parameters).fetchall())
