from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from objectives.torque import average_torque, torque_ripple_ratio

from .bands import classify_torque_ratio
from .checkpoint import read_checkpoint
from .database import DatasetDatabase
from .hashing import json_sha256
from .hashing import file_sha256
from .orchestrator import RunLayout
from .state import read_json_with_backup


@dataclass(frozen=True, slots=True)
class ValidationReport:
    ok: bool
    checks: dict[str, Any]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": self.checks,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def validate_run(run_directory: Path) -> ValidationReport:
    """Perform read-only consistency checks over a dataset run."""

    layout = RunLayout(Path(run_directory).resolve())
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}
    database = DatasetDatabase(layout.database, read_only=True)

    integrity = database.integrity_check()
    checks["sqlite_integrity"] = integrity
    if integrity != "ok":
        errors.append(f"SQLite integrity_check returned {integrity!r}")

    envelope = json.loads(layout.config.read_text(encoding="utf-8"))
    computed_config_hash = json_sha256(envelope["config"])
    checks["config_hash"] = computed_config_hash
    if computed_config_hash != envelope.get("config_hash"):
        errors.append("run_config.json content hash mismatch")
    runs = database.query_all("SELECT * FROM runs")
    if len(runs) != 1:
        errors.append(f"database contains {len(runs)} run rows instead of one")
        return ValidationReport(False, checks, tuple(errors), tuple(warnings))
    run = runs[0]
    if run["config_hash"] != computed_config_hash:
        errors.append("database config_hash differs from immutable run config")

    checkpoint: dict[str, Any] = {}
    try:
        checkpoint = read_checkpoint(layout.latest_checkpoint)
        checks["checkpoint_hash_valid"] = True
        if checkpoint.get("run_id") != run["run_id"]:
            errors.append("latest checkpoint belongs to another run")
        if checkpoint.get("config_hash") != computed_config_hash:
            errors.append("latest checkpoint config_hash differs from run")
    except (OSError, ValueError) as exc:
        checks["checkpoint_hash_valid"] = False
        errors.append(f"latest checkpoint invalid: {exc}")

    campaign_manifest_path = layout.root / "improved_campaign_manifest.json"
    if campaign_manifest_path.is_file():
        try:
            campaign_manifest = json.loads(campaign_manifest_path.read_text(encoding="utf-8"))
            campaign_config_hash = json_sha256(campaign_manifest["campaign_config"])
            checks["campaign_config_hash"] = campaign_config_hash
            if campaign_config_hash != campaign_manifest.get("campaign_config_hash"):
                errors.append("improved campaign config hash mismatch")
            if campaign_manifest.get("dataset_config_hash") != computed_config_hash:
                errors.append("improved campaign manifest refers to another physics config")
            parent_root = Path(campaign_manifest["parent_run"])
            parent_archive = parent_root / "exports" / "parent_archive.jsonl"
            assignments = parent_root / "exports" / "campaign_assignments.json"
            if file_sha256(parent_archive) != campaign_manifest.get("parent_archive_sha256"):
                errors.append("improved parent archive changed or is missing")
            if file_sha256(assignments) != campaign_manifest.get("campaign_assignments_sha256"):
                errors.append("improved campaign assignments changed or are missing")
            proposal_duplicates = database.query_all(
                """SELECT proposal_order,COUNT(*) n FROM improved_proposals WHERE run_id=?
                   GROUP BY proposal_order HAVING COUNT(*)>1""",
                (run["run_id"],),
            )
            checks["improved_proposals"] = int(
                database.query_all(
                    "SELECT COUNT(*) n FROM improved_proposals WHERE run_id=?", (run["run_id"],)
                )[0]["n"]
            )
            if proposal_duplicates:
                errors.append("duplicate improved proposal order exists")
            if checkpoint.get("algorithm_state", {}).get("runner_kind") != "improved_dataset_campaigns":
                errors.append("latest checkpoint is not an improved campaign checkpoint")
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"improved campaign manifest validation failed: {exc}")

    expected_angles = sorted(float(x) for x in envelope["config"]["physics"]["torque_angles_deg"])
    samples = database.query_all("SELECT * FROM samples ORDER BY sample_pk")
    classified = 0
    for sample in samples:
        angles = database.query_all(
            "SELECT * FROM angle_evaluations WHERE sample_id=? ORDER BY angle_deg",
            (sample["sample_id"],),
        )
        actual_angles = [float(row["angle_deg"]) for row in angles]
        if actual_angles != expected_angles:
            errors.append(f"{sample['sample_id']}: configured angle rows differ")
        completed = [row for row in angles if row["status"] == "completed"]
        if int(sample["completed_angle_count"]) != len(completed):
            errors.append(f"{sample['sample_id']}: completed angle counter differs")
        if sample["status"] != "classified":
            continue
        classified += 1
        if len(completed) != len(expected_angles):
            errors.append(f"{sample['sample_id']}: classified without six completed angles")
            continue
        torques = [float(row["torque"]) for row in completed]
        if not all(math.isfinite(value) for value in torques):
            errors.append(f"{sample['sample_id']}: non-finite committed torque")
            continue
        t_avg = average_torque(torques)
        t_ripple = torque_ripple_ratio(torques)
        if not math.isclose(t_avg, float(sample["t_avg"]), rel_tol=0, abs_tol=1e-12):
            errors.append(f"{sample['sample_id']}: T_avg cannot be reproduced")
        if not math.isclose(t_ripple, float(sample["t_ripple"]), rel_tol=0, abs_tol=1e-12):
            errors.append(f"{sample['sample_id']}: torque ripple cannot be reproduced")
        if sample["sample_kind"] == "candidate":
            if sample["historical_j"] is None or not math.isfinite(float(sample["historical_j"])):
                errors.append(f"{sample['sample_id']}: historical J missing/non-finite")
            reference = float(run["t_avg_ref"])
            ratio = t_avg / reference
            if not math.isclose(ratio, float(sample["torque_ratio"]), rel_tol=0, abs_tol=1e-12):
                errors.append(f"{sample['sample_id']}: torque_ratio cannot be reproduced")
            if classify_torque_ratio(ratio) != sample["fitness_band"]:
                errors.append(f"{sample['sample_id']}: fitness band is incorrect")

    duplicate_keys = database.query_all(
        """SELECT physical_key_hash,COUNT(*) n FROM samples
           GROUP BY physical_key_hash HAVING COUNT(*)>1"""
    )
    if duplicate_keys:
        errors.append("duplicate physical FEMM evaluations exist in samples")
    checks["samples"] = len(samples)
    checks["classified_samples"] = classified
    checks["duplicate_physical_evaluations"] = len(duplicate_keys)

    try:
        state = read_json_with_backup(layout.state)
        summary = database.progress_summary(str(run["run_id"]))
        if state.get("config_hash") != summary["config_hash"]:
            errors.append("run_state config_hash differs from database")
        if state.get("sample_counts") != summary["sample_counts"]:
            warnings.append("run_state sample counts are stale; DB remains authoritative")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"run_state is unreadable: {exc}")

    return ValidationReport(not errors, checks, tuple(errors), tuple(warnings))
