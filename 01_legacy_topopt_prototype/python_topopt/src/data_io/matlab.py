from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.io import loadmat

from encoding.chromosome import CHROMOSOME_LENGTH, Chromosome


@dataclass(frozen=True, slots=True)
class BestResult:
    chromosome: Chromosome
    objective: float
    generation_index_matlab: int | None
    objective_history: np.ndarray | None
    available_variables: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TraceData:
    populations: np.ndarray  # [generation, population, gene]
    objectives: np.ndarray  # [generation, population]
    sorted_indices: np.ndarray  # zero-based [generation, population]
    floating_iron_cells: np.ndarray
    small_copper_cells: np.ndarray
    copper_component_counts: np.ndarray
    best_objective_history: np.ndarray
    population_size: int
    generations: int
    beta: float
    random_immigrant_fraction: float


def load_seed(path: Path, variable: str = "seed_bits") -> Chromosome:
    data = loadmat(Path(path), simplify_cells=True)
    if variable not in data:
        raise KeyError(f"{variable!r} is absent from {path}")
    return Chromosome.from_iterable(np.asarray(data[variable]).reshape(-1))


def load_best_result(path: Path) -> BestResult:
    data: dict[str, Any] = loadmat(Path(path), simplify_cells=True)
    if "best_bits" not in data or "bestOverall" not in data:
        raise KeyError("best result requires best_bits and bestOverall")
    history = data.get("J_hist")
    return BestResult(
        chromosome=Chromosome.from_iterable(np.asarray(data["best_bits"]).reshape(-1)),
        objective=float(np.asarray(data["bestOverall"]).reshape(-1)[0]),
        generation_index_matlab=(
            int(np.asarray(data["genIdx"]).reshape(-1)[0])
            if "genIdx" in data
            else None
        ),
        objective_history=(
            np.asarray(history, dtype=float).reshape(-1) if history is not None else None
        ),
        available_variables=tuple(sorted(key for key in data if not key.startswith("__"))),
    )


def _scalar(dataset: h5py.Dataset) -> float:
    return float(np.asarray(dataset).reshape(-1)[0])


def load_trace_all(path: Path) -> TraceData:
    with h5py.File(Path(path), "r") as handle:
        populations_disk = np.asarray(handle["Ab_hist"], dtype=np.uint8)
        if populations_disk.ndim != 3 or populations_disk.shape[0] != CHROMOSOME_LENGTH:
            raise ValueError(f"unexpected Ab_hist disk shape {populations_disk.shape}")
        populations = populations_disk.transpose(2, 1, 0).copy()
        objectives = np.asarray(handle["J_hist_all"], dtype=float).T.copy()
        sorted_indices = np.asarray(handle["ind_hist"], dtype=int).T.copy() - 1
        floating_iron = np.asarray(handle["nFloatFe_hist"], dtype=int).T.copy()
        floating_copper = np.asarray(handle["nFloatCu_hist"], dtype=int).T.copy()
        component_counts = np.asarray(handle["nCompCu_hist"], dtype=int).T.copy()
        best_history = np.asarray(handle["J_hist"], dtype=float).reshape(-1)
        result = TraceData(
            populations=populations,
            objectives=objectives,
            sorted_indices=sorted_indices,
            floating_iron_cells=floating_iron,
            small_copper_cells=floating_copper,
            copper_component_counts=component_counts,
            best_objective_history=best_history,
            population_size=int(_scalar(handle["N"])),
            generations=int(_scalar(handle["gen"])),
            beta=_scalar(handle["beta"]),
            random_immigrant_fraction=_scalar(handle["d"]),
        )
    expected = (result.generations, result.population_size, CHROMOSOME_LENGTH)
    if result.populations.shape != expected:
        raise ValueError(f"logical Ab_hist shape {result.populations.shape} != {expected}")
    return result

