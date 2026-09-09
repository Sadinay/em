from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .database import DatasetDatabase
from .orchestrator import RunLayout
from .state import atomic_write_bytes


def export_dataset(run_directory: Path, output_directory: Path) -> dict[str, Any]:
    """Export only valid unique candidate samples; rejected/failed rows stay in SQLite."""

    layout = RunLayout(Path(run_directory).resolve())
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    database = DatasetDatabase(layout.database, read_only=True)
    runs = database.query_all("SELECT * FROM runs")
    if len(runs) != 1:
        raise ValueError("run database must contain exactly one run")
    run = runs[0]
    samples = database.query_all(
        """SELECT s.*,c.candidate_id,c.generation,c.source,c.parent_candidate_id,
                  c.lineage_id,c.hamming_distance_to_parent
           FROM samples s LEFT JOIN candidates c ON c.candidate_id=s.first_candidate_id
           WHERE s.run_id=? AND s.sample_kind='candidate' AND s.status='classified'
           ORDER BY s.sample_pk""",
        (run["run_id"],),
    )
    records: list[dict[str, Any]] = []
    chromosomes: list[list[int]] = []
    grids_18x10: list[list[list[int]]] = []
    torques: list[list[float]] = []
    labels: list[float] = []
    historical_j: list[float] = []
    for sample in samples:
        chromosome = json.loads(sample["chromosome_json"])
        grid = json.loads(sample["material_matrix_json"])
        angle_rows = database.query_all(
            """SELECT angle_deg,torque FROM angle_evaluations
               WHERE sample_id=? ORDER BY angle_deg""",
            (sample["sample_id"],),
        )
        torque_values = [float(row["torque"]) for row in angle_rows]
        record = {
            "sample_id": sample["sample_id"],
            "candidate_id": sample["candidate_id"],
            "generation": sample["generation"],
            "source": sample["source"],
            "parent_candidate_id": sample["parent_candidate_id"],
            "lineage_id": sample["lineage_id"],
            "hamming_distance_to_parent": sample["hamming_distance_to_parent"],
            "chromosome_180": chromosome,
            "material_matrix_18x10": grid,
            "material_matrix_10x18": np.asarray(grid, dtype=np.uint8).T.tolist(),
            "angles_deg": [float(row["angle_deg"]) for row in angle_rows],
            "torque_values_nm": torque_values,
            "T_avg": float(sample["t_avg"]),
            "T_min": float(sample["t_min"]),
            "T_max": float(sample["t_max"]),
            "T_ripple": float(sample["t_ripple"]),
            "T_avg_ref": float(sample["t_avg_ref"]),
            "torque_ratio": float(sample["torque_ratio"]),
            "historical_J": float(sample["historical_j"]),
            "fitness_band": sample["fitness_band"],
            "chromosome_hash": sample["chromosome_hash"],
            "physical_key_hash": sample["physical_key_hash"],
            "config_hash": sample["config_hash"],
            "geometry_mode": sample["geometry_mode"],
        }
        records.append(record)
        chromosomes.append(chromosome)
        grids_18x10.append(grid)
        torques.append(torque_values)
        labels.append(float(sample["torque_ratio"]))
        historical_j.append(float(sample["historical_j"]))

    jsonl = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records)
    atomic_write_bytes(output / "samples.jsonl", jsonl.encode("utf-8"))
    csv_path = output / "samples.csv"
    csv_rows = [
        {
            "sample_id": row["sample_id"],
            "candidate_id": row["candidate_id"],
            "generation": row["generation"],
            "lineage_id": row["lineage_id"],
            "T_avg": row["T_avg"],
            "T_ripple": row["T_ripple"],
            "torque_ratio": row["torque_ratio"],
            "historical_J": row["historical_J"],
            "fitness_band": row["fitness_band"],
            "chromosome_hash": row["chromosome_hash"],
            "config_hash": row["config_hash"],
        }
        for row in records
    ]
    lines: list[str] = []
    if csv_rows:
        import io

        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
        lines.append(stream.getvalue())
    atomic_write_bytes(csv_path, "".join(lines).encode("utf-8"))
    npz_path = output / "dataset.npz"
    temporary = output / ".dataset.npz.tmp"
    np.savez_compressed(
        temporary,
        chromosome_180=np.asarray(chromosomes, dtype=np.uint8).reshape((-1, 180)),
        material_matrix_18x10=np.asarray(grids_18x10, dtype=np.uint8).reshape((-1, 18, 10)),
        torque_values_nm=np.asarray(torques, dtype=float).reshape((-1, 6)),
        torque_ratio=np.asarray(labels, dtype=float),
        historical_J=np.asarray(historical_j, dtype=float),
        sample_id=np.asarray([row["sample_id"] for row in records], dtype=str),
    )
    generated = temporary.with_suffix(temporary.suffix + ".npz")
    generated.replace(npz_path)
    manifest = {
        "run_id": run["run_id"],
        "config_hash": run["config_hash"],
        "geometry_mode": run["geometry_mode"],
        "sample_count": len(records),
        "continuous_label": "torque_ratio",
        "selection_metric": "historical_j",
        "files": ["samples.jsonl", "samples.csv", "dataset.npz"],
    }
    atomic_write_bytes(
        output / "manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return manifest
