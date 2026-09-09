from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator, Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    physics_hash TEXT NOT NULL,
    sampling_hash TEXT NOT NULL,
    repair_config_hash TEXT NOT NULL,
    random_seed INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS proposals(
    proposal_pk INTEGER PRIMARY KEY AUTOINCREMENT,
    proposal_id TEXT UNIQUE,
    run_id TEXT NOT NULL REFERENCES sessions(run_id),
    parent_id TEXT NOT NULL,
    lineage_id TEXT NOT NULL,
    operator TEXT NOT NULL,
    scale TEXT NOT NULL,
    requested_cells INTEGER NOT NULL,
    mutation_indices_json TEXT NOT NULL,
    raw_chromosome_json TEXT NOT NULL,
    raw_hash TEXT NOT NULL,
    raw_parent_hamming INTEGER NOT NULL,
    repaired_chromosome_json TEXT NOT NULL,
    repaired_hash TEXT NOT NULL,
    repair_status TEXT NOT NULL,
    repair_iterations INTEGER NOT NULL,
    repair_hamming INTEGER NOT NULL,
    repair_actions_json TEXT NOT NULL,
    rejection_reasons_json TEXT NOT NULL,
    legal INTEGER NOT NULL,
    unique_repaired INTEGER NOT NULL,
    duplicate_of_proposal_id TEXT,
    pre_femm_admitted INTEGER NOT NULL,
    nearest_archive_hamming INTEGER,
    nearest_all_hamming INTEGER,
    topology_family_candidate INTEGER NOT NULL,
    features_json TEXT,
    selected_as_parent INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_proposals_repaired_hash ON proposals(run_id,repaired_hash);
CREATE INDEX IF NOT EXISTS idx_proposals_legal ON proposals(run_id,legal,unique_repaired);
CREATE TABLE IF NOT EXISTS parents(
    parent_pk INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id TEXT UNIQUE,
    run_id TEXT NOT NULL REFERENCES sessions(run_id),
    source_proposal_id TEXT,
    parent_parent_id TEXT,
    lineage_id TEXT NOT NULL,
    chromosome_json TEXT NOT NULL,
    chromosome_hash TEXT NOT NULL,
    features_json TEXT NOT NULL,
    minimum_archive_hamming INTEGER,
    topology_cluster INTEGER,
    archive_order INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id,chromosome_hash),
    UNIQUE(run_id,archive_order)
);
CREATE TABLE IF NOT EXISTS campaigns(
    campaign_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES sessions(run_id),
    campaign_order INTEGER NOT NULL,
    status TEXT NOT NULL,
    random_seed INTEGER NOT NULL,
    UNIQUE(run_id,campaign_order)
);
CREATE TABLE IF NOT EXISTS campaign_members(
    campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
    parent_id TEXT NOT NULL REFERENCES parents(parent_id),
    member_order INTEGER NOT NULL,
    topology_cluster INTEGER NOT NULL,
    PRIMARY KEY(campaign_id,parent_id),
    UNIQUE(campaign_id,member_order),
    UNIQUE(parent_id)
);
CREATE TABLE IF NOT EXISTS checkpoints(
    checkpoint_pk INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES sessions(run_id),
    checkpoint_hash TEXT NOT NULL,
    proposal_count INTEGER NOT NULL,
    parent_count INTEGER NOT NULL,
    rng_state_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events(
    event_pk INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES sessions(run_id),
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class ImprovedSamplingDatabase:
    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        self.path = Path(path).resolve()
        self.read_only = read_only

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
    def transaction(self) -> Iterator[sqlite3.Connection]:
        if self.read_only:
            raise RuntimeError("read-only database")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    @staticmethod
    def _event(connection: sqlite3.Connection, run_id: str, event_type: str, payload: dict[str, Any]) -> None:
        connection.execute(
            "INSERT INTO events(run_id,event_type,payload_json,created_at) VALUES(?,?,?,?)",
            (run_id, event_type, json.dumps(payload, sort_keys=True), utc_now()),
        )

    def create_session(self, run_id: str, config: Any) -> None:
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO sessions VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    "running",
                    json.dumps(config.data, ensure_ascii=False, sort_keys=True),
                    config.config_hash,
                    config.physics_hash,
                    config.sampling_hash,
                    config.repair_config_hash,
                    config.random_seed,
                    now,
                    now,
                ),
            )
            self._event(connection, run_id, "session_created", {})

    def session(self) -> sqlite3.Row:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM sessions").fetchall()
        if len(rows) != 1:
            raise ValueError("sampling database must contain exactly one session")
        return rows[0]

    def set_status(self, status: str) -> None:
        session = self.session()
        with self.transaction() as connection:
            connection.execute(
                "UPDATE sessions SET status=?,updated_at=? WHERE run_id=?",
                (status, utc_now(), session["run_id"]),
            )
            self._event(connection, session["run_id"], f"session_{status}", {})

    def insert_proposal(self, record: dict[str, Any]) -> str:
        with self.transaction() as connection:
            cursor = connection.execute(
                """INSERT INTO proposals(
                    run_id,parent_id,lineage_id,operator,scale,requested_cells,
                    mutation_indices_json,raw_chromosome_json,raw_hash,raw_parent_hamming,
                    repaired_chromosome_json,repaired_hash,repair_status,repair_iterations,
                    repair_hamming,repair_actions_json,rejection_reasons_json,legal,
                    unique_repaired,duplicate_of_proposal_id,pre_femm_admitted,
                    nearest_archive_hamming,nearest_all_hamming,topology_family_candidate,
                    features_json,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    record["run_id"], record["parent_id"], record["lineage_id"],
                    record["operator"], record["scale"], record["requested_cells"],
                    json.dumps(record["mutation_indices"]), json.dumps(record["raw_chromosome"]),
                    record["raw_hash"], record["raw_parent_hamming"],
                    json.dumps(record["repaired_chromosome"]), record["repaired_hash"],
                    record["repair_status"], record["repair_iterations"], record["repair_hamming"],
                    json.dumps(record["repair_actions"], sort_keys=True),
                    json.dumps(record["rejection_reasons"]), int(record["legal"]),
                    int(record["unique_repaired"]), record.get("duplicate_of_proposal_id"),
                    int(record["pre_femm_admitted"]), record.get("nearest_archive_hamming"),
                    record.get("nearest_all_hamming"), int(record["topology_family_candidate"]),
                    json.dumps(record.get("features"), sort_keys=True), utc_now(),
                ),
            )
            proposal_id = f"Q{int(cursor.lastrowid):07d}"
            connection.execute(
                "UPDATE proposals SET proposal_id=? WHERE proposal_pk=?", (proposal_id, cursor.lastrowid)
            )
            self._event(connection, record["run_id"], "proposal_recorded", {"proposal_id": proposal_id, "legal": record["legal"]})
            return proposal_id

    def add_parent(self, record: dict[str, Any]) -> str:
        with self.transaction() as connection:
            cursor = connection.execute(
                """INSERT INTO parents(
                    run_id,source_proposal_id,parent_parent_id,lineage_id,chromosome_json,
                    chromosome_hash,features_json,minimum_archive_hamming,archive_order,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    record["run_id"], record.get("source_proposal_id"), record.get("parent_parent_id"),
                    record["lineage_id"], json.dumps(record["chromosome"]), record["chromosome_hash"],
                    json.dumps(record["features"], sort_keys=True), record.get("minimum_archive_hamming"),
                    record["archive_order"], utc_now(),
                ),
            )
            parent_id = f"P{int(cursor.lastrowid):04d}"
            connection.execute("UPDATE parents SET parent_id=? WHERE parent_pk=?", (parent_id, cursor.lastrowid))
            if record.get("source_proposal_id"):
                connection.execute(
                    "UPDATE proposals SET selected_as_parent=1 WHERE proposal_id=?",
                    (record["source_proposal_id"],),
                )
            self._event(connection, record["run_id"], "parent_selected", {"parent_id": parent_id, "archive_order": record["archive_order"]})
            return parent_id

    def record_checkpoint(self, run_id: str, checkpoint_hash: str, state: dict[str, Any]) -> None:
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO checkpoints(run_id,checkpoint_hash,proposal_count,parent_count,
                   rng_state_json,created_at) VALUES(?,?,?,?,?,?)""",
                (
                    run_id, checkpoint_hash, state["proposal_count"], len(state["parent_ids"]),
                    json.dumps(state["rng_state"], sort_keys=True), utc_now(),
                ),
            )

    def parents(self) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM parents ORDER BY archive_order")

    def proposal(self, proposal_id: str) -> sqlite3.Row:
        rows = self.query("SELECT * FROM proposals WHERE proposal_id=?", (proposal_id,))
        if not rows:
            raise KeyError(proposal_id)
        return rows[0]

    def legal_unique_proposals(self) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM proposals WHERE legal=1 AND unique_repaired=1 ORDER BY proposal_pk"
        )

    def find_repaired(self, repaired_hash: str) -> sqlite3.Row | None:
        rows = self.query(
            "SELECT * FROM proposals WHERE repaired_hash=? AND legal=1 ORDER BY proposal_pk LIMIT 1",
            (repaired_hash,),
        )
        return rows[0] if rows else None

    def summary(self) -> dict[str, Any]:
        session = self.session()
        counts = self.query(
            """SELECT COUNT(*) total,SUM(legal) legal,
               SUM(CASE WHEN legal=1 AND unique_repaired=1 THEN 1 ELSE 0 END) legal_unique,
               SUM(CASE WHEN legal=1 AND unique_repaired=0 THEN 1 ELSE 0 END) duplicates,
               SUM(CASE WHEN repair_status='repaired' THEN 1 ELSE 0 END) repaired
               FROM proposals"""
        )[0]
        operators = [dict(row) for row in self.query(
            """SELECT operator,scale,COUNT(*) total,SUM(legal) legal,
               ROUND(AVG(raw_parent_hamming),3) avg_parent_hamming,
               ROUND(AVG(repair_hamming),3) avg_repair_hamming
               FROM proposals GROUP BY operator,scale ORDER BY operator,scale"""
        )]
        return {
            "run_id": session["run_id"], "status": session["status"],
            "config_hash": session["config_hash"], "physics_hash": session["physics_hash"],
            "sampling_hash": session["sampling_hash"], "proposal_count": int(counts["total"] or 0),
            "legal_proposals": int(counts["legal"] or 0),
            "legal_unique_proposals": int(counts["legal_unique"] or 0),
            "duplicate_proposals": int(counts["duplicates"] or 0),
            "repaired_proposals": int(counts["repaired"] or 0),
            "parent_count": len(self.parents()), "operator_statistics": operators,
        }

    def integrity_check(self) -> str:
        with self._connect() as connection:
            return str(connection.execute("PRAGMA integrity_check").fetchone()[0])

    def query(self, sql: str, parameters: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return list(connection.execute(sql, parameters).fetchall())
