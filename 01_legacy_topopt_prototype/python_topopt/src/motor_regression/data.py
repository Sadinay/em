from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class SampleRecord:
    key: str
    run_id: str
    sample_id: str
    candidate_id: str
    matrix_18x10: np.ndarray
    chromosome_hash: str
    fitness_band: str
    targets: np.ndarray
    generation: int
    direct_parent: str
    root_parent: str
    lineage_id: str
    topology_family: str = ""


@dataclass(frozen=True, slots=True)
class TargetScaler:
    names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, names: Sequence[str]) -> "TargetScaler":
        array = np.asarray(values, dtype=np.float64)
        mean = array.mean(axis=0)
        std = array.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        return cls(tuple(names), mean, std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.std + self.mean

    def as_dict(self) -> dict[str, Any]:
        return {"target_names": list(self.names), "mean": self.mean.tolist(), "std": self.std.tolist()}


def _candidate_graph(connection: sqlite3.Connection) -> dict[str, str | None]:
    return {str(row[0]): (str(row[1]) if row[1] is not None else None)
            for row in connection.execute("SELECT candidate_id,parent_candidate_id FROM candidates")}


def _root(candidate_id: str, graph: dict[str, str | None]) -> str:
    current = candidate_id
    seen: set[str] = set()
    while graph.get(current) is not None:
        if current in seen:
            raise ValueError(f"candidate parent cycle at {current}")
        seen.add(current)
        current = str(graph[current])
    return current


def _proposal_external_parents(connection: sqlite3.Connection) -> dict[str, str]:
    result: dict[str, str] = {}
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='improved_proposals'"
    ).fetchone()
    if not exists:
        return result
    for candidate_id, metadata_json in connection.execute(
        "SELECT candidate_id,metadata_json FROM improved_proposals WHERE candidate_id IS NOT NULL"
    ):
        metadata = json.loads(metadata_json or "{}")
        external = metadata.get("external_parent_candidate_id")
        if external:
            result[str(candidate_id)] = str(external)
    return result


def load_records(
    run_directories: Sequence[Path],
    *,
    target_specs: Sequence[dict[str, str]],
) -> list[SampleRecord]:
    columns = [str(spec["column"]) for spec in target_specs]
    source_graph: dict[str, str | None] | None = None
    source_run_id: str | None = None
    raw_runs: list[tuple[str, list[sqlite3.Row], dict[str, str | None], dict[str, str]]] = []
    for run_directory in run_directories:
        database = Path(run_directory) / "dataset.sqlite"
        if not database.is_file():
            raise FileNotFoundError(database)
        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            run_row = connection.execute("SELECT run_id,status FROM runs").fetchone()
            if run_row is None or run_row["status"] != "completed":
                raise ValueError(f"dataset run is not completed: {run_directory}")
            graph = _candidate_graph(connection)
            external = _proposal_external_parents(connection)
            query_columns = ",".join(f"s.{column}" for column in columns)
            rows = list(
                connection.execute(
                    f"""SELECT s.sample_id,s.first_candidate_id,s.chromosome_json,
                               s.material_matrix_json,s.chromosome_hash,s.fitness_band,{query_columns},
                               c.generation,c.parent_candidate_id,c.lineage_id
                        FROM samples s JOIN candidates c
                          ON c.candidate_id=s.first_candidate_id
                        WHERE s.sample_kind='candidate' AND s.status='classified'
                        ORDER BY s.sample_pk"""
                )
            )
        if source_graph is None and any(str(row["lineage_id"]) == "L-SEED" for row in rows):
            source_graph = graph
            source_run_id = str(run_row["run_id"])
        raw_runs.append((str(run_row["run_id"]), rows, graph, external))
    if source_graph is None:
        source_graph = raw_runs[0][2]
        source_run_id = raw_runs[0][0]

    records: list[SampleRecord] = []
    for run_id, rows, graph, external in raw_runs:
        for row in rows:
            candidate_id = str(row["first_candidate_id"])
            stored = np.asarray(json.loads(row["material_matrix_json"]), dtype=np.int64)
            if stored.shape != (18, 10):
                raise ValueError(f"{run_id}:{row['sample_id']} stored matrix shape {stored.shape} != (18,10)")
            matrix = stored.copy()
            target = np.asarray([row[column] for column in columns], dtype=np.float64)
            parent = row["parent_candidate_id"]
            if parent is not None:
                direct_parent = (
                    f"source:{parent}" if run_id == source_run_id else f"{run_id}:{parent}"
                )
            elif candidate_id in external:
                direct_parent = f"source:{external[candidate_id]}"
            else:
                direct_parent = (
                    f"source:{candidate_id}"
                    if run_id == source_run_id
                    else f"{run_id}:{candidate_id}"
                )
            lineage = str(row["lineage_id"])
            if lineage.startswith("BASE-C"):
                external_candidate = lineage.removeprefix("BASE-")
                root_id = _root(external_candidate, source_graph)
                root_parent = f"source:{root_id}"
            elif run_id == source_run_id:
                root_parent = f"source:{_root(candidate_id, graph)}"
            else:
                root_parent = f"{run_id}:{_root(candidate_id, graph)}"
            records.append(
                SampleRecord(
                    key=f"{run_id}:{row['sample_id']}",
                    run_id=run_id,
                    sample_id=str(row["sample_id"]),
                    candidate_id=candidate_id,
                    matrix_18x10=matrix,
                    chromosome_hash=str(row["chromosome_hash"]),
                    fitness_band=str(row["fitness_band"]),
                    targets=target,
                    generation=int(row["generation"]),
                    direct_parent=direct_parent,
                    root_parent=root_parent,
                    lineage_id=lineage,
                )
            )
    return records


def assign_topology_families(records: Sequence[SampleRecord], threshold: int) -> list[SampleRecord]:
    if threshold < 0 or threshold > 180:
        raise ValueError("topology family Hamming threshold must be within [0,180]")
    representatives: list[np.ndarray] = []
    result: list[SampleRecord] = []
    for record in records:
        flat = record.matrix_18x10.reshape(-1)
        family = None
        best_distance = 181
        for index, representative in enumerate(representatives):
            distance = int(np.count_nonzero(flat != representative))
            if distance <= threshold and distance < best_distance:
                family = index
                best_distance = distance
        if family is None:
            family = len(representatives)
            representatives.append(flat.copy())
        result.append(
            SampleRecord(
                **{field: getattr(record, field) for field in (
                    "key", "run_id", "sample_id", "candidate_id", "matrix_18x10",
                    "chromosome_hash", "fitness_band", "targets", "generation", "direct_parent",
                    "root_parent", "lineage_id"
                )},
                topology_family=f"TOPO-{family:04d}",
            )
        )
    return result


def group_value(record: SampleRecord, mode: str) -> str:
    if mode == "generation":
        return f"{record.run_id}:GEN-{record.generation:04d}"
    if mode == "parent":
        return record.direct_parent
    if mode == "root_parent":
        return record.root_parent
    if mode == "topology_family":
        return record.topology_family
    raise ValueError(f"unsupported group mode {mode}")


def grouped_split(
    records: Sequence[SampleRecord],
    *,
    group_mode: str,
    fractions: tuple[float, float, float],
    seed: int,
    stratification_bins: int = 10,
    band_balance_weight: float = 4.0,
    target_balance_weight: float = 1.0,
    search_restarts: int = 128,
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(group_value(record, group_mode), []).append(index)
    if len(groups) < 3:
        raise ValueError(f"group split needs at least 3 groups; found {len(groups)}")
    names = ("train", "val", "test")
    target_counts = np.asarray(fractions, dtype=float) * len(records)
    values = np.asarray([record.targets[0] for record in records], dtype=float)
    quantiles = np.unique(
        np.quantile(values, np.linspace(0.0, 1.0, max(2, int(stratification_bins)) + 1))[1:-1]
    )
    value_bins = np.searchsorted(quantiles, values, side="right")
    bin_count = len(quantiles) + 1
    global_histogram = np.bincount(value_bins, minlength=bin_count).astype(float)
    target_histograms = np.asarray(fractions, dtype=float)[:, None] * global_histogram[None, :]
    band_names = tuple(f"B{index}" for index in range(1, 8))
    band_lookup = {name: index for index, name in enumerate(band_names)}
    try:
        band_bins = np.asarray([band_lookup[record.fitness_band] for record in records], dtype=int)
    except KeyError as exc:
        raise ValueError(f"unsupported fitness band {exc.args[0]}") from exc
    global_bands = np.bincount(band_bins, minlength=len(band_names)).astype(float)
    target_bands = np.asarray(fractions, dtype=float)[:, None] * global_bands[None, :]

    def score_state(
        counts: np.ndarray, histograms: np.ndarray, bands: np.ndarray
    ) -> float:
        count_error = np.mean(((counts - target_counts) / target_counts) ** 2)
        target_error = np.mean(
            ((histograms - target_histograms) / (target_histograms + 1.0)) ** 2
        )
        band_error = np.mean(((bands - target_bands) / (target_bands + 1.0)) ** 2)
        return float(
            count_error
            + float(target_balance_weight) * target_error
            + float(band_balance_weight) * band_error
        )

    best: tuple[float, dict[str, list[int]]] | None = None
    base_items = list(groups.items())
    for restart in range(max(1, int(search_restarts))):
        rng = np.random.default_rng(int(seed) + restart)
        jitter = {name: float(rng.random()) for name, _ in base_items}
        items = sorted(base_items, key=lambda item: (-len(item[1]), jitter[item[0]], item[0]))
        counts = np.zeros(3, dtype=float)
        histograms = np.zeros((3, bin_count), dtype=float)
        bands = np.zeros((3, len(band_names)), dtype=float)
        assigned: dict[str, list[int]] = {name: [] for name in names}
        for _, indices in items:
            group_histogram = np.bincount(value_bins[indices], minlength=bin_count).astype(float)
            group_bands = np.bincount(band_bins[indices], minlength=len(band_names)).astype(float)
            scores = []
            for split_index in range(3):
                projected_counts = counts.copy()
                projected_histograms = histograms.copy()
                projected_bands = bands.copy()
                projected_counts[split_index] += len(indices)
                projected_histograms[split_index] += group_histogram
                projected_bands[split_index] += group_bands
                scores.append(score_state(projected_counts, projected_histograms, projected_bands))
            choice = int(np.argmin(scores))
            assigned[names[choice]].extend(indices)
            counts[choice] += len(indices)
            histograms[choice] += group_histogram
            bands[choice] += group_bands
        final_score = score_state(counts, histograms, bands)
        if best is None or final_score < best[0]:
            best = (final_score, assigned)
    assert best is not None
    assigned = best[1]
    for name in names:
        assigned[name].sort()
        if not assigned[name]:
            raise ValueError(f"group split produced empty {name} partition")
    return assigned


class MotorTopologyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        records: Sequence[SampleRecord],
        indices: Sequence[int],
        *,
        categories: Sequence[int],
        scaler: TargetScaler,
    ) -> None:
        self.records = records
        self.indices = list(indices)
        self.categories = tuple(int(value) for value in categories)
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[self.indices[item]]
        matrix = record.matrix_18x10
        channels = np.stack([(matrix == value) for value in self.categories], axis=0).astype(np.float32)
        if not np.all(channels.sum(axis=0) == 1):
            raise ValueError(f"unknown material value in {record.key}")
        target = self.scaler.transform(record.targets.reshape(1, -1))[0].astype(np.float32)
        return torch.from_numpy(channels), torch.from_numpy(target)
