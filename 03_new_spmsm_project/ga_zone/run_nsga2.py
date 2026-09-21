"""Checkpointed binary NSGA-II search with a frozen four-CNN committee.

Default execution pauses at each generation's FEMM queue.  Use --solve-pending
to run those arbitrary 120-bit genes through the project's six-angle FEMM
physics, then --resume.  --femm-mode live explicitly enables the full loop.
No CNN weights are changed by this program.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, fields
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
GA_DATA = Path(__file__).resolve().parent / "data"
MODEL_ROOT = PROJECT / "experiments/cnn_comprehensive_v3/models/training_runs"
MODEL_NAMES = ("polar90_resnet18", "polar90_vgg16", "gene_6x20_resnet20", "gene_gap90_6x97_resnet20")
BITS = 120
CANDIDATE_COLUMNS = ["gene_id", "bits", "first_generation", "magnet_cells", "pred_tavg_nm",
                     "pred_delta_t_nm", "committee_mean_tavg_nm", "committee_mean_delta_t_nm",
                     "disagreement_tavg_nm", "disagreement_delta_t_nm", "disagreement_scaled",
                     "femm_tavg_nm", "femm_delta_t_nm", "truth_source", "femm_generation"]
QUEUE_COLUMNS = ["generation", "gene_id", "bits", "selection_reason", "pred_tavg_nm",
                 "pred_delta_t_nm", "disagreement_scaled", "status"]


@dataclass(frozen=True)
class SearchConfig:
    # Keep all tunable search settings in this one file.
    population: int = 96
    generations: int = 80
    seed: int = 20260921
    initial_magnet_min: int = 12
    initial_magnet_max: int = 108
    search_magnet_min: int = 12
    search_magnet_max: int = 108
    crossover_probability: float = 0.9
    mutation_probability: float = 1.0 / BITS
    inference_batch_size: int = 16
    femm_fraction: float = 0.10
    femm_max_per_generation: int = 10
    verified_elite_fraction: float = 0.10
    cnn_update_interval: int = 100  # RESERVED ONLY; hook below is intentionally inert.

    def check(self) -> None:
        if self.population < 4 or self.generations < 1:
            raise ValueError("population >= 4 and generations >= 1 are required")
        if not (0 <= self.initial_magnet_min <= self.initial_magnet_max <= BITS):
            raise ValueError("invalid initial magnet count range")
        if not (0 <= self.search_magnet_min <= self.search_magnet_max <= BITS):
            raise ValueError("invalid search magnet count range")
        if not (0 <= self.crossover_probability <= 1 and 0 <= self.mutation_probability <= 1):
            raise ValueError("invalid crossover or mutation probability")
        if self.inference_batch_size < 1 or not (0 < self.femm_fraction <= 1):
            raise ValueError("invalid inference batch size or FEMM fraction")
        if self.femm_max_per_generation < 1 or not (0 < self.verified_elite_fraction <= 1):
            raise ValueError("invalid FEMM cap or verified-elite fraction")
        if self.cnn_update_interval < 1:
            raise ValueError("invalid future CNN update interval")


def validate_bits(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.uint8)
    if array.ndim != 2 or array.shape[1] != BITS or np.any((array != 0) & (array != 1)):
        raise ValueError("expected a binary [N,120] matrix")
    return array


def bit_string(bits: np.ndarray) -> str:
    return "".join(str(int(v)) for v in bits)


def parse_bits(value: str) -> np.ndarray:
    if len(value) != BITS or set(value) - {"0", "1"}:
        raise ValueError("invalid 120-bit gene string")
    return np.fromiter((int(v) for v in value), dtype=np.uint8, count=BITS)


def gene_id(bits: np.ndarray) -> str:
    # Matches femm_zone/scripts/spmsm_mapping.py genotype_sha256.
    return hashlib.sha256(np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()).hexdigest()


def isolated_mask(grid: np.ndarray) -> np.ndarray:
    same = np.zeros_like(grid, dtype=bool)
    for dr in (-1, 0, 1):
        for da in (-1, 0, 1):
            if dr == da == 0:
                continue
            target = (slice(max(0, dr), min(6, 6 + dr)), slice(max(0, da), min(20, 20 + da)))
            neighbour = (slice(max(0, -dr), min(6, 6 - dr)), slice(max(0, -da), min(20, 20 - da)))
            same[target] |= grid[target] == grid[neighbour]
    return ~same


def repair_isolated(bits: np.ndarray) -> np.ndarray:
    grid = np.asarray(bits, dtype=np.uint8).reshape(20, 6).T.copy()
    seen: set[bytes] = set()
    while True:
        mask = isolated_mask(grid)
        if not mask.any():
            return grid.T.reshape(BITS)
        key = grid.tobytes()
        if key in seen:
            raise ValueError("isolated-cell repair oscillated")
        seen.add(key)
        grid[mask] ^= 1


def file_hash(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def write_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class FrozenProxy:
    def __init__(self, model_name: str, device_name: str, batch_size: int) -> None:
        import torch

        cnn_root = PROJECT / "cnn_zone"
        for path in (PROJECT, cnn_root):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from cnn_zone.src.training import TrainingSpec, build_model, build_renderer, make_inputs

        self.torch = torch
        self.make_inputs = make_inputs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else
                                   "cpu" if device_name == "auto" else device_name)
        model_dir = MODEL_ROOT / model_name / "seed_20260917"
        config_path, checkpoint_path = model_dir / "config.json", model_dir / "best_checkpoint.pt"
        configuration = read_json(config_path)
        if configuration["experiment_id"] != model_name:
            raise ValueError("unexpected proxy model configuration")
        keys = {item.name for item in fields(TrainingSpec)}
        self.spec = TrainingSpec(**{key: configuration[key] for key in keys})
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["experiment_id"] != model_name:
            raise ValueError("model checkpoint identity mismatch")
        self.model = build_model(self.spec).to(self.device).eval()
        self.model.load_state_dict(checkpoint["model_state"])
        self.renderer = build_renderer(self.spec, PROJECT, self.device)
        self.mean = checkpoint["target_mean"].to(self.device).float()
        self.std = checkpoint["target_std"].to(self.device).float()
        if not np.allclose(self.mean.cpu().numpy(), configuration["target_mean"]) or not np.allclose(
            self.std.cpu().numpy(), configuration["target_std"]
        ):
            raise ValueError("checkpoint and configuration target scaling differ")
        self.identity = {"experiment_id": model_name, "checkpoint_sha256": file_hash(checkpoint_path),
                         "config_sha256": file_hash(config_path), "device": str(self.device)}
        for key in ("lookup", "gap_mapping"):
            if getattr(self.spec, key):
                self.identity[key + "_sha256"] = file_hash(PROJECT / getattr(self.spec, key))

    def predict(self, bits: np.ndarray) -> np.ndarray:
        bits = validate_bits(bits)
        if not len(bits):
            return np.empty((0, 2), dtype=np.float64)
        outputs = []
        with self.torch.inference_mode():
            for start in range(0, len(bits), self.batch_size):
                chunk = self.torch.as_tensor(bits[start:start + self.batch_size].copy(), device=self.device)
                values = self.model(self.make_inputs(chunk, self.spec, self.renderer)).float() * self.std + self.mean
                outputs.append(values.cpu().numpy())
        result = np.concatenate(outputs, axis=0).astype(np.float64)
        if result.shape != (len(bits), 2) or not np.isfinite(result).all():
            raise ValueError("proxy returned invalid torque predictions")
        return result


class FrozenCommittee:
    def __init__(self, device_name: str, batch_size: int) -> None:
        self.models = [FrozenProxy(name, device_name, batch_size) for name in MODEL_NAMES]
        self.identity = [model.identity for model in self.models]
        self.scale = self.models[0].std.cpu().numpy().astype(float)
        if (self.scale <= 0).any():
            raise ValueError("target scales must be positive")

    def predict(self, bits: np.ndarray) -> np.ndarray:
        return np.stack([model.predict(bits) for model in self.models], axis=1)  # [genes,4,2]


def candidate_record(bits: np.ndarray, generation: int, predictions: np.ndarray, scale: np.ndarray) -> dict:
    predictions = np.asarray(predictions, dtype=float)
    if predictions.shape != (4, 2) or not np.isfinite(predictions).all():
        raise ValueError("four models must each predict two finite targets")
    mean = predictions.mean(axis=0)
    spread = predictions.std(axis=0, ddof=0)  # the four selected models are the complete committee
    score = float(np.sqrt(np.mean((spread / scale) ** 2)))
    return {"gene_id": gene_id(bits), "bits": bit_string(bits), "first_generation": generation,
            "magnet_cells": int(bits.sum()), "pred_tavg_nm": float(predictions[0, 0]),
            "pred_delta_t_nm": float(predictions[0, 1]), "committee_mean_tavg_nm": float(mean[0]),
            "committee_mean_delta_t_nm": float(mean[1]), "disagreement_tavg_nm": float(spread[0]),
            "disagreement_delta_t_nm": float(spread[1]), "disagreement_scaled": score,
            "femm_tavg_nm": "", "femm_delta_t_nm": "", "truth_source": "", "femm_generation": ""}


def objectives(row: dict) -> tuple[float, float]:
    # A complete FEMM label ALWAYS supersedes the CNN estimate for this gene.
    verified = row["femm_tavg_nm"] != "" and row["femm_delta_t_nm"] != ""
    tavg = float(row["femm_tavg_nm"] if verified else row["pred_tavg_nm"])
    delta = float(row["femm_delta_t_nm"] if verified else row["pred_delta_t_nm"])
    return -tavg, delta


def pareto_fronts(ids: list[str], records: dict[str, dict]) -> list[list[str]]:
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    if not ids:
        return []
    values = np.asarray([objectives(records[key]) for key in ids], dtype=float)
    return [[ids[int(i)] for i in front] for front in NonDominatedSorting().do(values)]


def snapshot_generations(total: int) -> tuple[int, ...]:
    """Sparse progress figures: requested milestones plus the actual final generation."""
    if total < 1:
        raise ValueError("generation count must be positive")
    return tuple(sorted({generation for generation in (1, 20, 40, 60, 80, total) if generation <= total}))


def save_pareto_snapshot(run_dir: Path, generation: int, records: dict[str, dict]) -> Path:
    """Plot two separate, discrete nondominated sets at the current checkpoint."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    rows = list(records.values())
    if not rows:
        raise ValueError("cannot plot Pareto snapshot without candidates")
    predicted = np.asarray([[float(row["pred_tavg_nm"]), float(row["pred_delta_t_nm"])] for row in rows])
    unverified = np.asarray([[float(row["pred_tavg_nm"]), float(row["pred_delta_t_nm"])]
                             for row in rows if row["femm_tavg_nm"] == ""], dtype=float).reshape(-1, 2)
    pred_front = (unverified[NonDominatedSorting().do(np.column_stack((-unverified[:, 0], unverified[:, 1])),
                                                       only_non_dominated_front=True)]
                  if len(unverified) else np.empty((0, 2)))
    verified_rows = [row for row in rows if row["femm_tavg_nm"] != ""]
    verified = np.asarray([[float(row["femm_tavg_nm"]), float(row["femm_delta_t_nm"])]
                           for row in verified_rows], dtype=float).reshape(-1, 2)
    true_front = (verified[NonDominatedSorting().do(np.column_stack((-verified[:, 0], verified[:, 1])),
                                                      only_non_dominated_front=True)]
                  if len(verified) else np.empty((0, 2)))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(predicted[:, 0], predicted[:, 1], s=5, c="#9b9b9b", alpha=.17,
               label="All proxy predictions")
    if len(pred_front):
        ax.scatter(pred_front[:, 0], pred_front[:, 1], s=15, c="#d63c3c", alpha=.9,
                   label="Proxy-predicted Pareto (unverified)")
    if len(true_front):
        ax.scatter(true_front[:, 0], true_front[:, 1], s=28, facecolors="none", edgecolors="#126caa",
                   linewidths=1.2, label="FEMM-verified Pareto")
    ax.set(xlabel="Average torque Tavg (N m)", ylabel="Torque ripple DeltaT (N m)",
           title=f"Generation {generation}: cumulative candidate trade-offs")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.text(.1, .015, "Discrete candidate points only. CNN predictions are not FEMM evidence.", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    directory = run_dir / "plots"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"pareto_generation_{generation:03d}.png"
    temporary = output.with_name(output.stem + ".tmp.png")
    try:
        fig.savefig(temporary, dpi=160)
        temporary.replace(output)
    finally:
        plt.close(fig)
    return output


def crowding(front: list[str], records: dict[str, dict]) -> dict[str, float]:
    if len(front) <= 2:
        return {key: math.inf for key in front}
    values = np.asarray([objectives(records[key]) for key in front], dtype=float)
    distance = np.zeros(len(front), dtype=float)
    for target in range(2):
        order = np.argsort(values[:, target], kind="stable")
        distance[order[[0, -1]]] = math.inf
        span = values[order[-1], target] - values[order[0], target]
        if span > 0:
            for position in range(1, len(order) - 1):
                distance[order[position]] += (values[order[position + 1], target] -
                                              values[order[position - 1], target]) / span
    return dict(zip(front, distance))


def nsga_take(ids: list[str], limit: int, records: dict[str, dict]) -> list[str]:
    chosen: list[str] = []
    for front in pareto_fronts(ids, records):
        if len(chosen) + len(front) <= limit:
            chosen.extend(front)
        else:
            distance = crowding(front, records)
            chosen.extend(sorted(front, key=lambda key: (-distance[key], key))[:limit - len(chosen)])
            break
    return chosen


def choose_population(parent_ids: list[str], offspring_ids: list[str], records: dict[str, dict],
                      config: SearchConfig) -> list[str]:
    # Keep a separate, truth-only Pareto archive.  Earlier FEMM-verified designs
    # can re-enter the mating pool even if an optimistic CNN generation displaced
    # them temporarily.  The archive itself is not bounded by the parent count.
    verified = [key for key, row in records.items() if row["femm_tavg_nm"] != ""]
    archive = pareto_fronts(verified, records)[0] if verified else []
    union = list(dict.fromkeys(parent_ids + offspring_ids + archive))
    reserve = min(len(verified), max(1, math.ceil(config.population * config.verified_elite_fraction)))
    protected = nsga_take(archive, reserve, records)
    selected = protected + nsga_take([key for key in union if key not in protected],
                                      config.population - len(protected), records)
    if len(selected) != min(config.population, len(union)):
        raise RuntimeError("population selection did not fill all available slots")
    return selected


def select_femm_queries(new_ids: list[str], records: dict[str, dict], config: SearchConfig,
                        rng: np.random.Generator) -> list[tuple[str, str]]:
    """Pareto coverage, near-front disagreement, then structural/random audit.

    No scalar weighting of torque and ripple is used for Pareto ranking.
    Disagreement is only a heuristic query priority, never a confidence bound.
    """
    eligible = [key for key in dict.fromkeys(new_ids) if records[key]["femm_tavg_nm"] == ""]
    budget = min(len(eligible), config.femm_max_per_generation,
                 max(1, math.ceil(len(eligible) * config.femm_fraction))) if eligible else 0
    if not budget:
        return []
    quotas = [math.ceil(budget * .5), math.floor(budget * .3)]
    selected: list[tuple[str, str]] = []
    fronts = pareto_fronts(eligible, records)
    # Cover both ends and interior of the predicted front. Spread breaks ties
    # within interior strata; it does not collapse the two objectives.
    for front in fronts:
        free = quotas[0] - len(selected)
        if free <= 0:
            break
        ordered = sorted(front, key=lambda key: (float(records[key]["pred_tavg_nm"]), key))
        if len(ordered) <= free:
            chosen = ordered
        else:
            chosen = [ordered[0], ordered[-1]] if free >= 2 else [ordered[len(ordered) // 2]]
            while len(chosen) < free:
                remaining = [key for key in ordered if key not in chosen]
                # Maximise position-space coverage; disagreement breaks ties.
                positions = {key: ordered.index(key) / max(1, len(ordered)-1) for key in ordered}
                choice = max(remaining, key=lambda key: (min(abs(positions[key]-positions[prior]) for prior in chosen),
                                                          float(records[key]["disagreement_scaled"]), key))
                chosen.append(choice)
        selected.extend((key, "predicted_pareto") for key in chosen)
    already = {key for key, _ in selected}
    near = [key for front in fronts[:min(3, len(fronts))] for key in front if key not in already]
    near.sort(key=lambda key: (-float(records[key]["disagreement_scaled"]), key))
    for key in near[:quotas[1]]:
        selected.append((key, "competitive_disagreement"))
    already = {key for key, _ in selected}
    remaining = [key for key in eligible if key not in already]
    # One structurally distant candidate, then random audits. Both cover the
    # correlated-error risk when all four models make a similar mistake.
    if remaining and len(selected) < budget:
        anchors = [parse_bits(records[key]["bits"]) for key, _ in selected] or [parse_bits(records[eligible[0]]["bits"])]
        distant = max(remaining, key=lambda key: (min(int(np.count_nonzero(parse_bits(records[key]["bits"]) != a))
                                                      for a in anchors), key))
        selected.append((distant, "structural_audit"))
    remaining = [key for key in eligible if key not in {key for key, _ in selected}]
    if len(selected) < budget:
        for index in rng.choice(len(remaining), budget - len(selected), replace=False):
            selected.append((remaining[int(index)], "random_audit"))
    return selected


def sample_initial(config: SearchConfig, rng: np.random.Generator) -> list[np.ndarray]:
    rows: list[np.ndarray] = []
    seen: set[str] = set()
    for _ in range(config.population * 200):
        count = int(rng.integers(config.initial_magnet_min, config.initial_magnet_max + 1))
        row = np.zeros(BITS, dtype=np.uint8)
        row[rng.choice(BITS, count, replace=False)] = 1
        row = repair_isolated(row)
        key = gene_id(row)
        if config.search_magnet_min <= row.sum() <= config.search_magnet_max and key not in seen:
            rows.append(row)
            seen.add(key)
            if len(rows) == config.population:
                return rows
    raise RuntimeError("could not generate a full unique feasible initial population")


def make_offspring(parent_ids: list[str], records: dict[str, dict], config: SearchConfig,
                   rng: np.random.Generator) -> list[np.ndarray]:
    fronts = pareto_fronts(parent_ids, records)
    rank = {key: index for index, front in enumerate(fronts) for key in front}
    crowd = {key: score for front in fronts for key, score in crowding(front, records).items()}

    def tournament() -> str:
        a, b = rng.choice(parent_ids, 2, replace=False)
        return min((a, b), key=lambda key: (rank[key], -crowd[key], key))

    children: list[np.ndarray] = []
    seen: set[str] = set()
    for _ in range(config.population * 300):
        a, b = parse_bits(records[tournament()]["bits"]), parse_bits(records[tournament()]["bits"])
        child = a.copy()
        if rng.random() < config.crossover_probability:
            lo, hi = sorted(rng.choice(np.arange(1, BITS), size=2, replace=False))
            child[lo:hi] = b[lo:hi]
        child[rng.random(BITS) < config.mutation_probability] ^= 1
        child = repair_isolated(child)
        key = gene_id(child)
        if config.search_magnet_min <= child.sum() <= config.search_magnet_max and key not in seen:
            children.append(child)
            seen.add(key)
            if len(children) == config.population:
                return children
    raise RuntimeError("could not generate a full unique feasible offspring generation")


class FemmAdapter:
    """Fresh GA-local adapter; never mutates the frozen pilot/expansion queue."""

    def __init__(self, run_dir: Path) -> None:
        if str(PROJECT) not in sys.path:
            sys.path.insert(0, str(PROJECT))
        from experiments.input_distribution_pilot_v1 import pilot as core
        self.core = core
        self.run_dir = run_dir
        self.physical = core.physical_config()
        self.physics_identity = {"physical": self.physical,
                                 "template_sha256": core.sha(core.physical.TEMPLATE_FILE),
                                 "mat_sha256": core.sha(core.physical.MAT_FILE),
                                 "mapping_sha256": core.sha(core.mapping.__file__),
                                 "physical_source_sha256": core.sha(core.physical.__file__)}
        self.physics_fingerprint = core.digest(self.physics_identity)
        (run_dir / "femm_results").mkdir(exist_ok=True)

    source = "six_angle_femm_verified"

    def _contract(self) -> dict:
        # Capturing the solver binary identity is deliberately deferred until
        # an explicit live FEMM request; a paused search needs no COM process.
        return {**self.physics_identity, "solver": self.core.solver_identity(),
                "ga_adapter_sha256": file_hash(Path(__file__))}

    def solve(self, row: dict) -> tuple[float, float, str]:
        import scipy.io

        core = self.core
        bits = parse_bits(row["bits"])
        if gene_id(bits) != row["gene_id"]:
            raise ValueError("FEMM queue gene hash mismatch")
        contract = self._contract()
        fingerprint = core.digest(contract)
        root = self.run_dir / "femm_results" / ("c_" + fingerprint[:12])
        gene_folder = root / ("g_" + row["gene_id"][:16])
        root.mkdir(parents=True, exist_ok=True)
        contract_path = root / "contract.json"
        if contract_path.exists():
            if read_json(contract_path) != contract:
                raise ValueError("FEMM contract changed within result directory")
        else:
            write_json(contract_path, contract)
        template = core.physical.TEMPLATE_FILE.read_text(encoding="utf-8")
        positions = scipy.io.loadmat(core.physical.MAT_FILE, variable_names=["MaterialPosition"],
                                     squeeze_me=True)["MaterialPosition"]
        mapped = core.mapping.match_material_positions(template, positions)
        base = core.mapping.replace_cell_materials(template, bits)
        core.mapping.validate_generated_model(base, mapped, bits)
        raw: list[float] = []
        for angle in self.physical["inner_angles_deg"]:
            folder = gene_folder / f"angle_{angle:g}"
            model, state_path = folder / "model.fem", folder / "state.json"
            content = core.physical.configure_fem(base, self.physical, angle, 0).encode("utf-8")
            identity = {"gene_id": row["gene_id"], "bits": row["bits"], "inner_angle_deg": angle,
                        "rotor_travel_deg": angle - self.physical["airgap"]["initial_inner_angle_deg"],
                        "currents_a": core.physical.phase_currents(self.physical, angle, 0),
                        "condition_fingerprint": fingerprint,
                        "prepared_sha256": hashlib.sha256(content).hexdigest()}
            if state_path.exists():
                state = read_json(state_path)
                if any(state.get(key) != value for key, value in identity.items()):
                    raise ValueError("existing FEMM angle belongs to a different gene or condition")
                if state["status"] == "succeeded":
                    result = core.validate_success(folder, identity)
                else:
                    if not model.exists() or file_hash(model) not in (identity["prepared_sha256"],
                                                                       state.get("last_fem_sha256")):
                        raise ValueError("pending FEMM model was changed")
                    result = core.solve_angle(folder, contract)
            else:
                if model.exists():
                    raise ValueError("unregistered FEMM model already exists")
                folder.mkdir(parents=True, exist_ok=True)
                model.write_bytes(content)
                write_json(state_path, {**identity, "status": "pending", "attempts": []})
                result = core.solve_angle(folder, contract)
            if result is None:
                raise RuntimeError("FEMM angle failed twice; search remains paused")
            raw.append(float(core.validate_success(folder, identity)["raw_torque_nm"]))
        if len(raw) != len(self.physical["inner_angles_deg"]) or len(raw) != 6:
            raise RuntimeError("all six FEMM angles must finish before using a truth label")
        metric = core.physical.torque_metrics(raw, self.physical)
        label = {"gene_id": row["gene_id"], "bits": row["bits"],
                 "inner_angles_deg": self.physical["inner_angles_deg"], "raw_torques_nm": raw,
                 "tavg_nm": metric["tavg_nm"], "delta_t_nm": metric["peak_to_peak_nm"],
                 "physics_fingerprint": self.physics_fingerprint, "contract_fingerprint": fingerprint,
                 "source": "six_angle_femm_verified"}
        write_json(gene_folder / "label.json", label)
        return float(metric["tavg_nm"]), float(metric["peak_to_peak_nm"]), str(gene_folder / "label.json")

    def load_label(self, row: dict) -> tuple[float, float]:
        """Accept a paused query only after all six cached angle files validate."""
        core = self.core
        matches = list((self.run_dir / "femm_results").glob(
            f"c_*/g_{row['gene_id'][:16]}/label.json"))
        if len(matches) != 1:
            raise ValueError("expected exactly one complete FEMM label for queued gene")
        path = matches[0]
        label = read_json(path)
        if (label.get("gene_id") != row["gene_id"] or label.get("bits") != row["bits"] or
                label.get("source") != self.source or
                label.get("physics_fingerprint") != self.physics_fingerprint or
                label.get("inner_angles_deg") != self.physical["inner_angles_deg"]):
            raise ValueError("FEMM label identity or physics mismatch")
        root = path.parents[1]
        contract = read_json(root / "contract.json")
        fingerprint = core.digest(contract)
        if fingerprint != label.get("contract_fingerprint") or contract.get("physical") != self.physical:
            raise ValueError("FEMM contract differs from the queued physics")
        raw = []
        for angle in self.physical["inner_angles_deg"]:
            folder = path.parent / f"angle_{angle:g}"
            state = read_json(folder / "state.json")
            identity = {key: state[key] for key in ("gene_id", "bits", "inner_angle_deg", "rotor_travel_deg",
                                                    "currents_a", "condition_fingerprint", "prepared_sha256")}
            if identity["gene_id"] != row["gene_id"] or identity["bits"] != row["bits"]:
                raise ValueError("FEMM angle belongs to a different gene")
            raw.append(float(core.validate_success(folder, identity)["raw_torque_nm"]))
        if len(raw) != 6 or not np.allclose(raw, label["raw_torques_nm"], atol=0, rtol=0):
            raise ValueError("FEMM six-angle waveform differs from label")
        metrics = core.physical.torque_metrics(raw, self.physical)
        if (not math.isclose(metrics["tavg_nm"], label["tavg_nm"], rel_tol=1e-12, abs_tol=1e-12) or
                not math.isclose(metrics["peak_to_peak_nm"], label["delta_t_nm"], rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("FEMM label metrics differ from verified waveform")
        return float(metrics["tavg_nm"]), float(metrics["peak_to_peak_nm"])


def reserved_cnn_update_hook(generation: int, config: SearchConfig) -> bool:
    """Reserved checkpoint only: returning False prevents unapproved retraining."""
    return False  # future: generation % config.cnn_update_interval == 0 after dataset/retrain validation


def _state_path(run_dir: Path) -> Path:
    return run_dir / "state.json"


def _manifest(run_dir: Path, state: dict, config: SearchConfig, committee, adapter: FemmAdapter) -> None:
    write_json(run_dir / "run.json", {
        "status": state["stage"], "algorithm": "binary NSGA-II with verified-elite reservation",
        "config": asdict(config), "committee": committee.identity,
        "primary_performance_model": MODEL_NAMES[0],
        "four_model_disagreement_ddof": 0,
        "disagreement_is_calibrated_confidence_bound": False,
        "femm_physics_fingerprint": adapter.physics_fingerprint,
        "generation_completed": state["generation_completed"],
        "pending_generation": state.get("pending_generation"),
        "cnn_weights_updated": False,
    })


def _persist(run_dir: Path, state: dict, records: dict[str, dict], queue: list[dict],
             config: SearchConfig, committee, adapter: FemmAdapter) -> None:
    verified = [key for key, row in records.items() if row["femm_tavg_nm"] != ""]
    state["verified_archive"] = pareto_fronts(verified, records)[0] if verified else []
    write_csv(run_dir / "candidates.csv", CANDIDATE_COLUMNS,
              sorted(records.values(), key=lambda row: row["gene_id"]))
    write_csv(run_dir / "femm_queue.csv", QUEUE_COLUMNS, queue)
    write_json(_state_path(run_dir), state)
    _manifest(run_dir, state, config, committee, adapter)


def run_search(config: SearchConfig, name: str, device_name: str = "auto", *, resume: bool = False,
               femm_mode: str = "pause", committee_override=None, femm_override=None) -> Path:
    config.check()
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", name):
        raise ValueError("run name must contain only letters, digits, dash or underscore")
    if femm_mode not in ("pause", "live"):
        raise ValueError("FEMM mode must be pause or live")
    run_dir = GA_DATA / name
    if resume:
        if not _state_path(run_dir).exists():
            raise FileNotFoundError("no checkpoint to resume")
        state = read_json(_state_path(run_dir))
        original = SearchConfig(**read_json(run_dir / "run.json")["config"])
        if original != config:
            raise ValueError("resume configuration differs from saved run")
        records = {row["gene_id"]: row for row in read_csv(run_dir / "candidates.csv")}
        queue = read_csv(run_dir / "femm_queue.csv")
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng_state"]
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        state = {"stage": "new", "generation_completed": 0, "population": [], "pending_generation": None,
                 "pending_offspring": [], "pending_queries": [], "rng_state": None}
        records, queue = {}, []
        rng = np.random.default_rng(config.seed)
    committee = committee_override or FrozenCommittee(device_name, config.inference_batch_size)
    adapter = femm_override or FemmAdapter(run_dir)
    if resume:
        previous = read_json(run_dir / "run.json")
        if previous["committee"] != committee.identity or previous["femm_physics_fingerprint"] != adapter.physics_fingerprint:
            raise ValueError("CNN checkpoint or FEMM physics differs from saved run")
    else:
        state["rng_state"] = rng.bit_generator.state
        _persist(run_dir, state, records, queue, config, committee, adapter)

    while state["generation_completed"] < config.generations:
        if state["stage"] != "awaiting_femm":
            generation = state["generation_completed"] + 1
            offspring = (sample_initial(config, rng) if generation == 1 else
                         make_offspring(state["population"], records, config, rng))
            ids = [gene_id(bits) for bits in offspring]
            newly_seen = [bits for bits, key in zip(offspring, ids) if key not in records]
            if newly_seen:
                predictions = committee.predict(np.stack(newly_seen))
                if predictions.shape != (len(newly_seen), 4, 2):
                    raise ValueError("committee prediction shape differs from [N,4,2]")
                for bits, values in zip(newly_seen, predictions):
                    record = candidate_record(bits, generation, values, np.asarray(committee.scale, dtype=float))
                    records[record["gene_id"]] = record
            fresh_ids = [key for key in ids if int(records[key]["first_generation"]) == generation]
            choices = select_femm_queries(fresh_ids, records, config, rng)
            for key, reason in choices:
                row = records[key]
                if not any(int(q["generation"]) == generation and q["gene_id"] == key for q in queue):
                    queue.append({"generation": generation, "gene_id": key, "bits": row["bits"],
                                  "selection_reason": reason, "pred_tavg_nm": row["pred_tavg_nm"],
                                  "pred_delta_t_nm": row["pred_delta_t_nm"],
                                  "disagreement_scaled": row["disagreement_scaled"], "status": "pending"})
            state.update(stage="awaiting_femm", pending_generation=generation,
                         pending_offspring=ids, pending_queries=[key for key, _ in choices],
                         rng_state=rng.bit_generator.state)
            if generation == 1:
                write_csv(run_dir / "seeds.csv", ["gene_id", "bits", "magnet_cells"],
                          [{"gene_id": key, "bits": records[key]["bits"],
                            "magnet_cells": records[key]["magnet_cells"]} for key in ids])
            _persist(run_dir, state, records, queue, config, committee, adapter)

        if state["pending_queries"] and femm_mode == "pause":
            # An explicit solve-pending action creates and validates labels.
            # No unverified number is ever silently promoted to FEMM truth.
            labels = {row["gene_id"]: row for row in read_csv(run_dir / "femm_labels.csv")}
            if not all(key in labels for key in state["pending_queries"]):
                print(f"Paused before generation {state['pending_generation']} selection: "
                      f"{len(state['pending_queries'])} FEMM genes await complete six-angle labels", flush=True)
                return run_dir
            for key in state["pending_queries"]:
                label = labels[key]
                if (label["bits"] != records[key]["bits"] or
                        label["physics_fingerprint"] != adapter.physics_fingerprint or
                        label["source"] != adapter.source):
                    raise ValueError("pending FEMM label gene or physics mismatch")
                tavg, delta = adapter.load_label(records[key])
                if (not math.isclose(tavg, float(label["tavg_nm"]), rel_tol=1e-12, abs_tol=1e-12) or
                        not math.isclose(delta, float(label["delta_t_nm"]), rel_tol=1e-12, abs_tol=1e-12)):
                    raise ValueError("FEMM label table differs from verified six-angle files")
                records[key].update(femm_tavg_nm=tavg,
                                    femm_delta_t_nm=delta,
                                    truth_source=label["source"], femm_generation=state["pending_generation"])
                for q in queue:
                    if q["gene_id"] == key:
                        q["status"] = "succeeded"
        elif state["pending_queries"]:
            for key in state["pending_queries"]:
                row = records[key]
                tavg, delta, label_path = adapter.solve(row)
                if not (math.isfinite(tavg) and math.isfinite(delta) and delta >= 0):
                    raise ValueError("FEMM returned invalid complete-gene metrics")
                row.update(femm_tavg_nm=tavg, femm_delta_t_nm=delta,
                           truth_source=adapter.source, femm_generation=state["pending_generation"])
                for q in queue:
                    if q["gene_id"] == key:
                        q["status"] = "succeeded"
                _save_labels(run_dir, records, adapter)
                _persist(run_dir, state, records, queue, config, committee, adapter)

        state["population"] = choose_population(state["population"], state["pending_offspring"], records, config)
        state["generation_completed"] = state["pending_generation"]
        state.update(stage="complete" if state["generation_completed"] == config.generations else "ready",
                     pending_generation=None, pending_offspring=[], pending_queries=[],
                     rng_state=rng.bit_generator.state)
        with (run_dir / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"generation": state["generation_completed"],
                                     "unique_candidates": len(records),
                                     "femm_verified": sum(row["femm_tavg_nm"] != "" for row in records.values()),
                                     "population": len(state["population"]),
                                     "cnn_update_due_but_inert": state["generation_completed"] % config.cnn_update_interval == 0},
                                    ensure_ascii=False) + "\n")
        _persist(run_dir, state, records, queue, config, committee, adapter)
        if state["generation_completed"] in snapshot_generations(config.generations):
            save_pareto_snapshot(run_dir, state["generation_completed"], records)
        print(f"generation {state['generation_completed']}/{config.generations}: "
              f"{len(records)} unique, {sum(row['femm_tavg_nm'] != '' for row in records.values())} FEMM verified", flush=True)
        reserved_cnn_update_hook(state["generation_completed"], config)
    return run_dir


def _save_labels(run_dir: Path, records: dict[str, dict], adapter: FemmAdapter) -> None:
    rows = []
    for row in records.values():
        if row["femm_tavg_nm"] != "":
            rows.append({"gene_id": row["gene_id"], "bits": row["bits"],
                         "tavg_nm": row["femm_tavg_nm"], "delta_t_nm": row["femm_delta_t_nm"],
                         "source": row["truth_source"], "physics_fingerprint": adapter.physics_fingerprint})
    write_csv(run_dir / "femm_labels.csv", ["gene_id", "bits", "tavg_nm", "delta_t_nm", "source", "physics_fingerprint"], rows)


def solve_pending(name: str) -> Path:
    """Explicit expensive operation, separate from paused GA search."""
    run_dir = GA_DATA / name
    state = read_json(_state_path(run_dir))
    if state["stage"] != "awaiting_femm":
        raise ValueError("run has no pending FEMM generation")
    manifest = read_json(run_dir / "run.json")
    adapter = FemmAdapter(run_dir)
    if manifest["femm_physics_fingerprint"] != adapter.physics_fingerprint:
        raise ValueError("FEMM physics changed since queue creation")
    records = {row["gene_id"]: row for row in read_csv(run_dir / "candidates.csv")}
    labels = {row["gene_id"]: row for row in read_csv(run_dir / "femm_labels.csv")}
    for key in state["pending_queries"]:
        if key in labels:
            if labels[key]["bits"] != records[key]["bits"] or labels[key]["physics_fingerprint"] != adapter.physics_fingerprint:
                raise ValueError("cached FEMM label identity mismatch")
            adapter.load_label(records[key])
            continue
        row = records[key]
        tavg, delta, _ = adapter.solve(row)
        if not (math.isfinite(tavg) and math.isfinite(delta) and delta >= 0):
            raise ValueError("FEMM returned invalid metrics")
        labels[key] = {"gene_id": key, "bits": row["bits"], "tavg_nm": tavg, "delta_t_nm": delta,
                       "source": "six_angle_femm_verified", "physics_fingerprint": adapter.physics_fingerprint}
        write_csv(run_dir / "femm_labels.csv",
                  ["gene_id", "bits", "tavg_nm", "delta_t_nm", "source", "physics_fingerprint"],
                  list(labels.values()))
    return run_dir


def self_test() -> None:
    from tempfile import TemporaryDirectory

    global GA_DATA
    initial_data = GA_DATA
    test = np.zeros((2, BITS), dtype=np.uint8)
    test[0, 0] = 1
    assert gene_id(test[0]) != gene_id(test[1])
    assert not isolated_mask(repair_isolated(test[0]).reshape(20, 6).T).any()
    values = np.array([[2.0, .2], [4.0, .4], [6.0, .6], [8.0, .8]])
    record = candidate_record(test[0], 1, values, np.ones(2))
    assert np.isclose(record["disagreement_tavg_nm"], np.std(values[:, 0], ddof=0))
    record["femm_tavg_nm"], record["femm_delta_t_nm"] = 1.0, .1
    assert objectives(record) == (-1.0, .1)

    class FakeCommittee:
        identity = [{"experiment_id": name, "sha": "test", "device": "cpu"} for name in MODEL_NAMES]
        scale = np.ones(2)

        def predict(self, bits: np.ndarray) -> np.ndarray:
            m = bits.sum(axis=1).astype(float)
            main = np.column_stack((m / 30, m / 60))
            return np.stack([main + np.array([i * .03, i * .02]) for i in range(4)], axis=1)

    class FakeAdapter:
        physics_fingerprint = "synthetic_test_only"
        source = "synthetic_test_only"

        def __init__(self, run_dir: Path) -> None:
            self.run_dir = run_dir

        def solve(self, row: dict) -> tuple[float, float, str]:
            m = int(row["magnet_cells"])
            value = {"gene_id": row["gene_id"], "bits": row["bits"],
                     "tavg_nm": m / 35, "delta_t_nm": m / 65}
            write_json(self.run_dir / f"mock_{row['gene_id']}.json", value)
            return value["tavg_nm"], value["delta_t_nm"], "synthetic_test_only"

        def load_label(self, row: dict) -> tuple[float, float]:
            proof = read_json(self.run_dir / f"mock_{row['gene_id']}.json")
            if proof["gene_id"] != row["gene_id"] or proof["bits"] != row["bits"]:
                raise ValueError("synthetic proof identity mismatch")
            return float(proof["tavg_nm"]), float(proof["delta_t_nm"])

    with TemporaryDirectory() as temp:
        GA_DATA = Path(temp)
        try:
            assert snapshot_generations(80) == (1, 20, 40, 60, 80)
            assert snapshot_generations(23) == (1, 20, 23)
            assert save_pareto_snapshot(Path(temp), 1, {record["gene_id"]: record}).exists()
            try:
                save_pareto_snapshot(Path(temp), 1, {})
            except ValueError:
                pass
            else:
                raise AssertionError("empty candidate snapshot was accepted")
            cfg = SearchConfig(population=8, generations=2, femm_max_per_generation=2)
            proxy, adapter = FakeCommittee(), FakeAdapter(Path(temp))
            run = run_search(cfg, "synthetic", "cpu", femm_mode="live", committee_override=proxy,
                             femm_override=adapter)
            saved = read_json(run / "state.json")
            assert saved["stage"] == "complete" and saved["generation_completed"] == 2
            candidates = read_csv(run / "candidates.csv")
            assert sum(row["femm_tavg_nm"] != "" for row in candidates) >= 2
            assert all(row["truth_source"] == "synthetic_test_only" for row in candidates if row["femm_tavg_nm"] != "")
            assert len(read_csv(run / "femm_queue.csv")) <= 4
            assert sorted(path.name for path in (run / "plots").glob("*.png")) == [
                "pareto_generation_001.png", "pareto_generation_002.png"]

            cfg = SearchConfig(population=4, generations=1, femm_max_per_generation=1)
            paused_adapter = FakeAdapter(Path(temp) / "paused")
            paused = run_search(cfg, "paused", "cpu", committee_override=proxy, femm_override=paused_adapter)
            pending = read_json(paused / "state.json")
            assert pending["stage"] == "awaiting_femm" and len(pending["pending_queries"]) == 1
            key = pending["pending_queries"][0]
            candidate = {row["gene_id"]: row for row in read_csv(paused / "candidates.csv")}[key]
            tavg, delta, _ = paused_adapter.solve(candidate)
            label = {"gene_id": key, "bits": candidate["bits"], "tavg_nm": tavg, "delta_t_nm": delta,
                     "source": paused_adapter.source, "physics_fingerprint": "stale_condition"}
            columns = ["gene_id", "bits", "tavg_nm", "delta_t_nm", "source", "physics_fingerprint"]
            write_csv(paused / "femm_labels.csv", columns, [label])
            try:
                run_search(cfg, "paused", "cpu", resume=True, committee_override=proxy, femm_override=paused_adapter)
            except ValueError as exc:
                assert "physics mismatch" in str(exc)
            else:
                raise AssertionError("stale FEMM label was accepted")
            label["physics_fingerprint"] = paused_adapter.physics_fingerprint
            write_csv(paused / "femm_labels.csv", columns, [label])
            completed = run_search(cfg, "paused", "cpu", resume=True, committee_override=proxy,
                                   femm_override=paused_adapter)
            assert read_json(completed / "state.json")["stage"] == "complete"
            saved_cfg, saved_device = resolve_resume_options(read_json(completed / "run.json"))
            assert saved_cfg == cfg and saved_device == "cpu"
            try:
                resolve_resume_options(read_json(completed / "run.json"), population=96)
            except ValueError as exc:
                assert "population" in str(exc)
            else:
                raise AssertionError("conflicting resume override was accepted")
        finally:
            GA_DATA = initial_data
    print("NSGA-II selection, fixed-four disagreement and synthetic two-generation loop passed")


def resolve_resume_options(saved: dict, *, population=None, generations=None, seed=None,
                           femm_fraction=None, device=None) -> tuple[SearchConfig, str]:
    config = SearchConfig(**saved["config"])
    for flag, saved_value, value in (("population", config.population, population),
                                     ("generations", config.generations, generations),
                                     ("seed", config.seed, seed),
                                     ("femm-fraction", config.femm_fraction, femm_fraction)):
        if value is not None and value != saved_value:
            raise ValueError(f"--{flag} conflicts with the saved run configuration")
    saved_device = saved["committee"][0]["device"]
    if device not in (None, "auto", saved_device):
        raise ValueError("--device conflicts with the saved run device")
    return config, saved_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", help="run directory name; new runs default to the local time")
    parser.add_argument("--population", type=int)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--femm-fraction", type=float)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--femm-mode", choices=("pause", "live"), default="pause")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--solve-pending", action="store_true", help="explicitly solve only the queued six-angle FEMM genes")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    name = args.name or datetime.now().strftime("nsga2_%Y%m%d_%H%M%S")
    if args.solve_pending:
        if not args.name:
            parser.error("--solve-pending requires --name")
        print(f"Queued FEMM labels saved under {solve_pending(name)}; resume the GA separately")
        return
    if args.resume:
        if not args.name:
            parser.error("--resume requires --name")
        saved = read_json(GA_DATA / name / "run.json")
        try:
            config, device = resolve_resume_options(saved, population=args.population,
                                                    generations=args.generations, seed=args.seed,
                                                    femm_fraction=args.femm_fraction, device=args.device)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        config = SearchConfig(population=args.population if args.population is not None else SearchConfig.population,
                              generations=args.generations if args.generations is not None else SearchConfig.generations,
                              seed=args.seed if args.seed is not None else SearchConfig.seed,
                              femm_fraction=args.femm_fraction if args.femm_fraction is not None else SearchConfig.femm_fraction)
        device = args.device or "auto"
    path = run_search(config, name, device, resume=args.resume, femm_mode=args.femm_mode)
    print(f"Run state: {read_json(path / 'run.json')['status']}; data: {path}")


if __name__ == "__main__":
    main()
