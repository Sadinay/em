from __future__ import annotations

from pathlib import Path
from typing import Any

from .database import DatasetDatabase
from .orchestrator import RunLayout
from .checkpoint import read_checkpoint


def read_status(run_directory: Path) -> dict[str, Any]:
    layout = RunLayout(Path(run_directory).resolve())
    database = DatasetDatabase(layout.database, read_only=True)
    rows = database.query_all("SELECT run_id,targets_json FROM runs")
    if len(rows) != 1:
        raise ValueError("run database must contain exactly one run")
    summary = database.progress_summary(str(rows[0]["run_id"]))
    mean = summary["recent_mean_seconds_per_angle"]
    import json

    targets = json.loads(rows[0]["targets_json"])
    config = json.loads(
        database.query_all("SELECT config_json FROM runs WHERE run_id=?", (rows[0]["run_id"],))[0][0]
    )
    remaining_total = max(
        0,
        int(config["limits"]["target_valid_unique_samples"])
        - int(summary["valid_unique_samples"]),
    )
    summary["solver_only_seconds_for_total_target"] = (
        float(mean) * 6 * remaining_total if mean is not None else None
    )
    summary["eta_is_estimate"] = mean is not None
    summary["eta_note"] = (
        "solver-only lower bound; band completion and feasible-candidate discovery are not predictable"
        if mean is not None
        else "insufficient completed candidate angles"
    )
    summary["band_targets_met"] = all(
        summary["fitness_bands"][band] >= targets[band] for band in targets
    )
    summary["band_completion_eta"] = None
    try:
        algorithm = read_checkpoint(layout.latest_checkpoint).get("algorithm_state", {})
        if algorithm.get("runner_kind") == "improved_dataset_campaigns":
            best_by_campaign: dict[str, float | None] = {}
            for campaign in algorithm.get("campaigns", []):
                values = [
                    member.get("result", {}).get("historical_j")
                    for member in campaign.get("population", [])
                    if member.get("result") and member["result"].get("historical_j") is not None
                ]
                best_by_campaign[str(campaign.get("campaign_id"))] = (
                    min(float(value) for value in values) if values else None
                )
            summary["improved_campaign"] = {
                "generation": algorithm.get("generation"),
                "phase": algorithm.get("phase"),
                "campaign_index": algorithm.get("campaign_index"),
                "proposal_count": algorithm.get("proposal_count"),
                "completed_campaign_generations": len(algorithm.get("history", [])),
                "best_historical_j_by_campaign": best_by_campaign,
                "stop_reason": algorithm.get("stop_reason"),
                "completed": algorithm.get("completed", False),
            }
    except (OSError, ValueError):
        pass
    return summary
