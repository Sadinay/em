from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .chromosome import Chromosome, ensure_chromosome


DEFAULT_CIRCUIT_NAMES = ("A+", "A-", "B+", "B-", "C+", "C-")
DEFAULT_SECTOR_CIRCUIT_IDS = (4, 3, 6, 5, 2, 1)
DEFAULT_TOTAL_TURNS = (100.0,) * 6


@dataclass(frozen=True, slots=True)
class TurnsAllocation:
    copper_cells_per_circuit: tuple[int, ...]
    turns_per_copper_cell: tuple[float, ...]
    circuit_names: tuple[str, ...]

    @property
    def has_any_copper(self) -> bool:
        return any(count > 0 for count in self.copper_cells_per_circuit)


def compute_turns_per_circuit(
    chromosome: Chromosome | Sequence[int],
    *,
    sector_circuit_ids: Sequence[int] = DEFAULT_SECTOR_CIRCUIT_IDS,
    total_turns_per_circuit: Sequence[float] = DEFAULT_TOTAL_TURNS,
    circuit_names: Sequence[str] = DEFAULT_CIRCUIT_NAMES,
) -> TurnsAllocation:
    """Port of ``compute_turns_per_cell.m`` for constant circuit ID per sector."""

    genes = ensure_chromosome(chromosome).to_numpy()
    n_circuits = len(total_turns_per_circuit)
    if len(circuit_names) != n_circuits:
        raise ValueError("circuit names and total-turn arrays must have equal length")
    counts = np.zeros(n_circuits, dtype=int)
    copper_count = int(np.count_nonzero(genes == 2))
    for phase_id in sector_circuit_ids:
        if phase_id < 0 or phase_id > n_circuits:
            raise ValueError(f"circuit ID {phase_id} is outside 0..{n_circuits}")
        if phase_id > 0:
            counts[phase_id - 1] += copper_count
    turns = tuple(
        float(total_turns_per_circuit[index]) / int(count)
        if count > 0
        else 0.0
        for index, count in enumerate(counts)
    )
    return TurnsAllocation(
        copper_cells_per_circuit=tuple(int(value) for value in counts),
        turns_per_copper_cell=turns,
        circuit_names=tuple(str(value) for value in circuit_names),
    )

