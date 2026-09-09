from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from constraints.connectivity import historical_precheck
from dataset_generation.checkpoint import read_checkpoint
from dataset_generation.hashing import json_sha256
from encoding.chromosome import Chromosome

from .config import improved_sampling_config_from_data
from .database import ImprovedSamplingDatabase
from .features import hamming, topology_features
from .generator import ImprovedRunLayout
from .repair import repair_topology


@dataclass(frozen=True, slots=True)
class ImprovedValidation:
    ok: bool
    checks: dict[str, Any]
    errors: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": self.checks, "errors": list(self.errors)}


def validate_improved_run(run_directory: Path, *, verify_repairs: bool = True) -> ImprovedValidation:
    layout = ImprovedRunLayout(Path(run_directory).resolve())
    errors: list[str] = []
    checks: dict[str, Any] = {}
    database = ImprovedSamplingDatabase(layout.database, read_only=True)
    checks["sqlite_integrity"] = database.integrity_check()
    if checks["sqlite_integrity"] != "ok":
        errors.append("SQLite integrity check failed")
    envelope = json.loads(layout.config.read_text(encoding="utf-8"))
    if json_sha256(envelope["config"]) != envelope.get("config_hash"):
        errors.append("immutable config hash mismatch")
    config = improved_sampling_config_from_data(envelope["config"], source_path=layout.config)
    session = database.session()
    for name in ("config_hash", "physics_hash", "sampling_hash", "repair_config_hash"):
        if session[name] != getattr(config, name):
            errors.append(f"database {name} mismatch")
    try:
        checkpoint = read_checkpoint(layout.checkpoint)
        state = checkpoint["state"]
        checks["checkpoint_valid"] = True
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"checkpoint invalid: {exc}")
        state = {}
        checks["checkpoint_valid"] = False
    summary = database.summary()
    if state:
        if int(state["proposal_count"]) != summary["proposal_count"]:
            errors.append("checkpoint proposal count differs from database")
        if len(state["parent_ids"]) != summary["parent_count"]:
            errors.append("checkpoint parent count differs from database")
    parents = database.parents()
    parent_hashes: set[str] = set()
    parent_chromosomes: list[Chromosome] = []
    for row in parents:
        chromosome = Chromosome.from_iterable(json.loads(row["chromosome_json"]))
        if chromosome.sha256() != row["chromosome_hash"]:
            errors.append(f"{row['parent_id']}: chromosome hash mismatch")
        if row["chromosome_hash"] in parent_hashes:
            errors.append(f"{row['parent_id']}: duplicate parent chromosome")
        parent_hashes.add(row["chromosome_hash"])
        precheck = historical_precheck(
            chromosome,
            minimum_copper_island_cells=int(config.data["constraints"]["minimum_copper_island_cells"]),
            theta_periodic=bool(config.data["constraints"]["theta_periodic"]),
        )
        if precheck.rejected or not precheck.has_any_copper:
            errors.append(f"{row['parent_id']}: illegal archived parent")
        if topology_features(chromosome).as_dict() != json.loads(row["features_json"]):
            errors.append(f"{row['parent_id']}: feature record mismatch")
        if parent_chromosomes and int(row["minimum_archive_hamming"]) != min(
            hamming(chromosome, previous) for previous in parent_chromosomes
        ):
            errors.append(f"{row['parent_id']}: archive Hamming record mismatch")
        parent_chromosomes.append(chromosome)
    proposals = database.query("SELECT * FROM proposals ORDER BY proposal_pk")
    if verify_repairs:
        for row in proposals:
            raw = Chromosome.from_iterable(json.loads(row["raw_chromosome_json"]))
            replay = repair_topology(raw, config.data)
            if replay.repaired.sha256() != row["repaired_hash"]:
                errors.append(f"{row['proposal_id']}: deterministic repair mismatch")
            if replay.status != row["repair_status"] or replay.repair_hamming != row["repair_hamming"]:
                errors.append(f"{row['proposal_id']}: repair metadata mismatch")
    campaign_rows = database.query("SELECT * FROM campaign_members")
    if summary["parent_count"] == config.parent_target:
        expected = int(config.data["campaign_count"]) * int(config.data["parents_per_campaign"])
        if len(campaign_rows) != expected:
            errors.append("complete parent archive has incomplete campaign assignment")
    elif campaign_rows:
        errors.append("campaigns were assigned before the parent archive was complete")
    checks.update(
        {
            "proposal_count": len(proposals),
            "parent_count": len(parents),
            "campaign_member_count": len(campaign_rows),
            "deterministic_repairs_checked": len(proposals) if verify_repairs else 0,
        }
    )
    return ImprovedValidation(not errors, checks, tuple(errors))
