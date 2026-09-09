from __future__ import annotations

from dataclasses import dataclass
import csv
import io
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from constraints.connectivity import historical_precheck
from data_io.matlab import load_seed
from dataset_generation.checkpoint import read_checkpoint, write_checkpoint
from dataset_generation.hashing import json_sha256
from dataset_generation.state import atomic_write_bytes, atomic_write_json
from encoding.chromosome import Chromosome

from .config import (
    ImprovedSamplingConfig,
    improved_sampling_config_from_data,
    load_improved_sampling_config,
)
from .database import ImprovedSamplingDatabase, utc_now
from .features import hamming, topology_features
from .operators import mutate_2d
from .repair import repair_topology


@dataclass(frozen=True, slots=True)
class ImprovedRunLayout:
    root: Path

    @property
    def config(self) -> Path:
        return self.root / "run_config.json"

    @property
    def database(self) -> Path:
        return self.root / "sampling.sqlite"

    @property
    def checkpoint(self) -> Path:
        return self.root / "checkpoints" / "latest.json"

    @property
    def state(self) -> Path:
        return self.root / "run_state.json"

    @property
    def exports(self) -> Path:
        return self.root / "exports"

    def create(self) -> None:
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "checkpoints").mkdir()
        self.exports.mkdir()
        (self.root / "logs").mkdir()


class ParentArchiveGenerator:
    STATE_SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        layout: ImprovedRunLayout,
        config: ImprovedSamplingConfig,
        run_id: str,
        database: ImprovedSamplingDatabase,
        state: dict[str, Any],
        rng: np.random.Generator,
    ) -> None:
        self.layout = layout
        self.config = config
        self.run_id = run_id
        self.database = database
        self.state = state
        self.rng = rng
        self._reload_memory()

    @classmethod
    def create(cls, *, project_root: Path, run_name: str, config_path: Path) -> "ParentArchiveGenerator":
        if not run_name or any(char in run_name for char in '\\/:*?"<>|'):
            raise ValueError("invalid run name")
        layout = ImprovedRunLayout(Path(project_root).resolve() / "improved_runs" / run_name)
        layout.create()
        config = load_improved_sampling_config(config_path)
        atomic_write_json(
            layout.config,
            {"config_hash": config.config_hash, "config": config.data},
        )
        database = ImprovedSamplingDatabase(layout.database)
        database.initialize()
        database.create_session(run_name, config)
        rng = np.random.default_rng(config.random_seed)
        seed_info = config.data["seed"]
        seed = load_seed(Path(seed_info["mat_path"]), variable=seed_info["mat_variable"])
        if seed.sha256() != seed_info["chromosome_sha256"]:
            raise ValueError("decoded seed hash mismatch")
        precheck = historical_precheck(
            seed,
            minimum_copper_island_cells=int(config.data["constraints"]["minimum_copper_island_cells"]),
            theta_periodic=bool(config.data["constraints"]["theta_periodic"]),
        )
        if precheck.rejected or not precheck.has_any_copper:
            raise ValueError("configured seed is not legal")
        seed_parent = database.add_parent(
            {
                "run_id": run_name,
                "source_proposal_id": None,
                "parent_parent_id": None,
                "lineage_id": "L-SEED",
                "chromosome": list(seed.genes),
                "chromosome_hash": seed.sha256(),
                "features": topology_features(seed).as_dict(),
                "minimum_archive_hamming": None,
                "archive_order": 1,
            }
        )
        state = {
            "schema_version": cls.STATE_SCHEMA_VERSION,
            "proposal_count": 0,
            "parent_ids": [seed_parent],
            "window_proposal_ids": [],
            "completed": False,
            "campaigns_assigned": False,
            "stop_reason": None,
            "rng_state": rng.bit_generator.state,
        }
        generator = cls(
            layout=layout, config=config, run_id=run_name, database=database, state=state, rng=rng
        )
        generator.save_checkpoint()
        generator.write_exports()
        return generator

    @classmethod
    def resume(cls, run_directory: Path) -> "ParentArchiveGenerator":
        layout = ImprovedRunLayout(Path(run_directory).resolve())
        envelope = json.loads(layout.config.read_text(encoding="utf-8"))
        if json_sha256(envelope["config"]) != envelope["config_hash"]:
            raise ValueError("immutable run config hash mismatch")
        data = envelope["config"]
        config = improved_sampling_config_from_data(data, source_path=layout.config)
        database = ImprovedSamplingDatabase(layout.database)
        session = database.session()
        if session["config_hash"] != config.config_hash:
            raise ValueError("database and run config hashes differ")
        payload = read_checkpoint(layout.checkpoint)
        if payload["run_id"] != session["run_id"] or payload["config_hash"] != config.config_hash:
            raise ValueError("checkpoint identity mismatch")
        state = payload["state"]
        database_counts = database.summary()
        if int(state["proposal_count"]) != int(database_counts["proposal_count"]):
            raise ValueError("database/checkpoint proposal count mismatch; validation required")
        if len(state["parent_ids"]) != int(database_counts["parent_count"]):
            raise ValueError("database/checkpoint parent count mismatch")
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng_state"]
        return cls(
            layout=layout,
            config=config,
            run_id=str(session["run_id"]),
            database=database,
            state=state,
            rng=rng,
        )

    def _reload_memory(self) -> None:
        self.parent_rows = self.database.parents()
        self.parent_chromosomes = [
            Chromosome.from_iterable(json.loads(row["chromosome_json"])) for row in self.parent_rows
        ]
        legal_rows = self.database.legal_unique_proposals()
        self.legal_rows = legal_rows
        self.legal_chromosomes = [
            Chromosome.from_iterable(json.loads(row["repaired_chromosome_json"])) for row in legal_rows
        ]

    def save_checkpoint(self) -> None:
        self.state["rng_state"] = self.rng.bit_generator.state
        payload = {
            "run_id": self.run_id,
            "config_hash": self.config.config_hash,
            "physics_hash": self.config.physics_hash,
            "sampling_hash": self.config.sampling_hash,
            "state": self.state,
            "created_at": utc_now(),
        }
        envelope = write_checkpoint(self.layout.checkpoint, payload)
        self.database.record_checkpoint(
            self.run_id, str(envelope["payload_hash"]), self.state
        )
        self.write_state()

    def write_state(self) -> None:
        summary = self.database.summary()
        state = {
            **summary,
            "parent_archive_target": self.config.parent_target,
            "window_size": len(self.state["window_proposal_ids"]),
            "selection_window": int(self.config.data["proposal"]["selection_window"]),
            "completed": bool(self.state["completed"]),
            "campaigns_assigned": bool(self.state["campaigns_assigned"]),
            "stop_reason": self.state.get("stop_reason"),
            "updated_at": utc_now(),
        }
        atomic_write_json(self.layout.state, state, keep_backup=True)

    def _nearest(self, chromosome: Chromosome, archive: list[Chromosome]) -> int | None:
        return min((hamming(chromosome, other) for other in archive), default=None)

    def _select_parent_row(self) -> Any:
        index = int(self.rng.integers(0, len(self.parent_rows)))
        return self.parent_rows[index]

    def generate_one(self) -> str:
        if self.state["completed"]:
            return "completed"
        parent_row = self._select_parent_row()
        parent = Chromosome.from_iterable(json.loads(parent_row["chromosome_json"]))
        mutation = mutate_2d(parent, self.config.data, self.rng)
        repair = repair_topology(mutation.chromosome, self.config.data)
        repaired = repair.repaired
        precheck = historical_precheck(
            repaired,
            minimum_copper_island_cells=int(self.config.data["constraints"]["minimum_copper_island_cells"]),
            theta_periodic=bool(self.config.data["constraints"]["theta_periodic"]),
        )
        legal = not precheck.rejected and precheck.has_any_copper and repair.status in {
            "repaired", "unchanged_legal"
        }
        existing = self.database.find_repaired(repaired.sha256()) if legal else None
        unique = legal and existing is None
        nearest_archive = self._nearest(repaired, self.parent_chromosomes) if legal else None
        nearest_all = self._nearest(repaired, self.legal_chromosomes) if legal else None
        scale_spec = self.config.data["proposal"]["scales"][mutation.scale]
        admitted = bool(unique)
        reasons = list(repair.rejection_reasons)
        if legal and self.config.data["proposal"]["hamming_admission_mode"] == "enforce":
            if nearest_all is not None and nearest_all < int(scale_spec["minimum_archive_distance"]):
                admitted = False
                reasons.append("MINIMUM_ARCHIVE_HAMMING")
            if hamming(parent, repaired) > int(scale_spec["maximum_parent_distance"]):
                admitted = False
                reasons.append("MAXIMUM_PARENT_HAMMING")
        if legal and not unique:
            reasons.append("EXACT_DUPLICATE")
        if not legal and not reasons:
            reasons.extend(precheck.reasons)
            if not precheck.has_any_copper:
                reasons.append("NO_COPPER")
        features = topology_features(repaired).as_dict() if legal else None
        family_threshold = int(self.config.data["lineage"]["topology_family_candidate_distance"])
        proposal_id = self.database.insert_proposal(
            {
                "run_id": self.run_id,
                "parent_id": parent_row["parent_id"],
                "lineage_id": parent_row["lineage_id"],
                "operator": mutation.operator,
                "scale": mutation.scale,
                "requested_cells": mutation.requested_cells,
                "mutation_indices": list(mutation.changed_indices),
                "raw_chromosome": list(mutation.chromosome.genes),
                "raw_hash": mutation.chromosome.sha256(),
                "raw_parent_hamming": hamming(parent, mutation.chromosome),
                "repaired_chromosome": list(repaired.genes),
                "repaired_hash": repaired.sha256(),
                "repair_status": repair.status,
                "repair_iterations": repair.iterations,
                "repair_hamming": repair.repair_hamming,
                "repair_actions": list(repair.actions),
                "rejection_reasons": reasons,
                "legal": legal,
                "unique_repaired": unique,
                "duplicate_of_proposal_id": str(existing["proposal_id"]) if existing else None,
                "pre_femm_admitted": admitted,
                "nearest_archive_hamming": nearest_archive,
                "nearest_all_hamming": nearest_all,
                "topology_family_candidate": bool(
                    nearest_archive is not None and nearest_archive >= family_threshold
                ),
                "features": features,
            }
        )
        self.state["proposal_count"] = int(self.state["proposal_count"]) + 1
        if legal and unique:
            self.legal_chromosomes.append(repaired)
            self.legal_rows.append(self.database.proposal(proposal_id))
        if admitted:
            self.state["window_proposal_ids"].append(proposal_id)
        if len(self.state["window_proposal_ids"]) >= int(
            self.config.data["proposal"]["selection_window"]
        ):
            self._select_window_parent()
        if int(self.state["proposal_count"]) >= int(
            self.config.data["proposal"]["maximum_proposals"]
        ) and len(self.state["parent_ids"]) < self.config.parent_target:
            self.state["stop_reason"] = "maximum_proposals"
        self.save_checkpoint()
        return "generated"

    def _select_window_parent(self) -> None:
        candidates = [self.database.proposal(value) for value in self.state["window_proposal_ids"]]
        # Maximum-minimum selection; proposal order is the deterministic tie breaker.
        selected = max(
            candidates,
            key=lambda row: (
                int(row["nearest_archive_hamming"] or 0),
                int(row["nearest_all_hamming"] or 0),
                -int(row["proposal_pk"]),
            ),
        )
        chromosome = Chromosome.from_iterable(json.loads(selected["repaired_chromosome_json"]))
        parent_id = self.database.add_parent(
            {
                "run_id": self.run_id,
                "source_proposal_id": selected["proposal_id"],
                "parent_parent_id": selected["parent_id"],
                "lineage_id": selected["lineage_id"],
                "chromosome": list(chromosome.genes),
                "chromosome_hash": chromosome.sha256(),
                "features": json.loads(selected["features_json"]),
                "minimum_archive_hamming": selected["nearest_archive_hamming"],
                "archive_order": len(self.state["parent_ids"]) + 1,
            }
        )
        self.state["parent_ids"].append(parent_id)
        self.state["window_proposal_ids"] = []
        self._reload_memory()
        if len(self.state["parent_ids"]) >= self.config.parent_target:
            self.assign_campaigns()
            self.state["completed"] = True
            self.state["stop_reason"] = "parent_archive_complete"

    def assign_campaigns(self) -> None:
        parents = self.database.parents()
        target = self.config.parent_target
        if len(parents) != target:
            raise ValueError(f"campaign assignment requires exactly {target} parents")
        chromosomes = [
            Chromosome.from_iterable(json.loads(row["chromosome_json"])) for row in parents
        ]
        cluster_count = int(self.config.data["topology_cluster_target"])
        capacity = int(self.config.data["parents_per_cluster"])
        medoids = [0]
        while len(medoids) < cluster_count:
            choices = [index for index in range(target) if index not in medoids]
            medoids.append(
                max(choices, key=lambda index: (min(hamming(chromosomes[index], chromosomes[m]) for m in medoids), -index))
            )
        clusters: list[list[int]] = [[medoid] for medoid in medoids]
        remaining = [index for index in range(target) if index not in medoids]
        # Assign hardest-to-place members first, with deterministic cluster ties.
        remaining.sort(
            key=lambda index: (
                -min(hamming(chromosomes[index], chromosomes[m]) for m in medoids), index
            )
        )
        for index in remaining:
            available = [cluster for cluster in range(cluster_count) if len(clusters[cluster]) < capacity]
            chosen = min(
                available,
                key=lambda cluster: (hamming(chromosomes[index], chromosomes[medoids[cluster]]), cluster),
            )
            clusters[chosen].append(index)
        campaign_count = int(self.config.data["campaign_count"])
        per_campaign = int(self.config.data["parents_per_campaign"])
        interleaved: list[tuple[int, int]] = []
        for member_offset in range(capacity):
            for cluster, members in enumerate(clusters):
                interleaved.append((members[member_offset], cluster + 1))
        with self.database.transaction() as connection:
            for cluster, members in enumerate(clusters, start=1):
                for index in members:
                    connection.execute(
                        "UPDATE parents SET topology_cluster=? WHERE parent_id=?",
                        (cluster, parents[index]["parent_id"]),
                    )
            for campaign in range(1, campaign_count + 1):
                campaign_id = f"CAMPAIGN-{campaign:02d}"
                connection.execute(
                    "INSERT INTO campaigns VALUES(?,?,?,?,?)",
                    (
                        campaign_id, self.run_id, campaign, "ready",
                        self.config.random_seed + campaign,
                    ),
                )
            member_counts = [0] * campaign_count
            for order, (parent_index, cluster) in enumerate(interleaved):
                campaign = order % campaign_count
                member_counts[campaign] += 1
                connection.execute(
                    "INSERT INTO campaign_members VALUES(?,?,?,?)",
                    (
                        f"CAMPAIGN-{campaign + 1:02d}", parents[parent_index]["parent_id"],
                        member_counts[campaign], cluster,
                    ),
                )
            if any(value != per_campaign for value in member_counts):
                raise AssertionError(f"unbalanced campaign assignment {member_counts}")
        self.state["campaigns_assigned"] = True

    def write_exports(self) -> None:
        parents = self.database.parents()
        records = [
            {
                "parent_id": row["parent_id"],
                "archive_order": row["archive_order"],
                "source_proposal_id": row["source_proposal_id"],
                "parent_parent_id": row["parent_parent_id"],
                "lineage_id": row["lineage_id"],
                "chromosome_hash": row["chromosome_hash"],
                "chromosome_180": json.loads(row["chromosome_json"]),
                "minimum_archive_hamming": row["minimum_archive_hamming"],
                "topology_cluster": row["topology_cluster"],
                "features": json.loads(row["features_json"]),
                "physics_hash": self.config.physics_hash,
                "sampling_hash": self.config.sampling_hash,
            }
            for row in parents
        ]
        atomic_write_bytes(
            self.layout.exports / "parent_archive.jsonl",
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in records).encode("utf-8"),
        )
        stream = io.StringIO(newline="")
        fields = [
            "parent_id", "archive_order", "source_proposal_id", "parent_parent_id",
            "lineage_id", "chromosome_hash", "minimum_archive_hamming", "topology_cluster",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in records)
        atomic_write_bytes(self.layout.exports / "parent_archive.csv", stream.getvalue().encode("utf-8"))
        campaigns = [dict(row) for row in self.database.query(
            """SELECT m.campaign_id,m.member_order,m.parent_id,m.topology_cluster
               FROM campaign_members m ORDER BY m.campaign_id,m.member_order"""
        )]
        atomic_write_json(self.layout.exports / "campaign_assignments.json", campaigns)

    def run(
        self,
        *,
        stop_after_parents: int | None = None,
        stop_requested: Callable[[], bool] = lambda: False,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        target = self.config.parent_target
        requested_stop = min(target, int(stop_after_parents)) if stop_after_parents else target
        self.database.set_status("running")
        while len(self.state["parent_ids"]) < requested_stop:
            if stop_requested():
                self.state["stop_reason"] = "user_pause"
                self.database.set_status("paused")
                self.save_checkpoint()
                self.write_exports()
                return "paused"
            if self.state.get("stop_reason") == "maximum_proposals":
                self.database.set_status("paused")
                self.write_exports()
                return "maximum_proposals"
            self.generate_one()
            if progress and self.state["proposal_count"] % 25 == 0:
                progress(self.database.summary())
        status = "completed" if len(self.state["parent_ids"]) >= target else "pilot_stop"
        self.database.set_status(status)
        self.state["stop_reason"] = (
            "parent_archive_complete" if status == "completed" else f"requested_parent_count_{requested_stop}"
        )
        self.save_checkpoint()
        self.write_exports()
        return status
