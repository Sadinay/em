from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from encoding.chromosome import Chromosome
from encoding.layout import ANGULAR_CELLS, RADIAL_CELLS, matlab_vector_to_grid


@dataclass(frozen=True, slots=True)
class IronConnectivityResult:
    n_iron: int
    n_floating: int
    ratio: float
    anchored_mask: np.ndarray
    floating_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class CopperIslandResult:
    n_copper: int
    n_floating: int
    ratio: float
    component_sizes: tuple[int, ...]
    small_component_sizes: tuple[int, ...]
    floating_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class PrecheckResult:
    iron: IronConnectivityResult
    copper: CopperIslandResult
    has_any_copper: bool
    reasons: tuple[str, ...]
    rejected: bool
    objective_if_rejected: float | None


def _neighbors(
    row: int,
    column: int,
    *,
    theta_periodic: bool,
) -> Iterator[tuple[int, int]]:
    for next_row, next_column in (
        (row - 1, column),
        (row + 1, column),
        (row, column - 1),
        (row, column + 1),
    ):
        if theta_periodic:
            next_column %= ANGULAR_CELLS
        if 0 <= next_row < RADIAL_CELLS and 0 <= next_column < ANGULAR_CELLS:
            yield next_row, next_column


def detect_floating_iron(
    chromosome: Chromosome | Sequence[int],
    *,
    theta_periodic: bool = False,
) -> IronConnectivityResult:
    grid = matlab_vector_to_grid(chromosome)
    is_iron = grid == 1
    n_iron = int(np.count_nonzero(is_iron))
    anchored = np.zeros_like(is_iron, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for column in range(ANGULAR_CELLS):
        if is_iron[RADIAL_CELLS - 1, column]:
            anchored[RADIAL_CELLS - 1, column] = True
            queue.append((RADIAL_CELLS - 1, column))

    while queue:
        row, column = queue.popleft()
        for next_row, next_column in _neighbors(
            row, column, theta_periodic=theta_periodic
        ):
            if is_iron[next_row, next_column] and not anchored[next_row, next_column]:
                anchored[next_row, next_column] = True
                queue.append((next_row, next_column))

    floating = is_iron & ~anchored
    n_floating = int(np.count_nonzero(floating))
    return IronConnectivityResult(
        n_iron=n_iron,
        n_floating=n_floating,
        ratio=(n_floating / n_iron) if n_iron else 0.0,
        anchored_mask=anchored,
        floating_mask=floating,
    )


def detect_small_copper_islands(
    chromosome: Chromosome | Sequence[int],
    *,
    minimum_cells: int = 4,
    theta_periodic: bool = False,
) -> CopperIslandResult:
    if minimum_cells < 1:
        raise ValueError("minimum_cells must be positive")
    grid = matlab_vector_to_grid(chromosome)
    is_copper = grid == 2
    n_copper = int(np.count_nonzero(is_copper))
    visited = np.zeros_like(is_copper, dtype=bool)
    floating = np.zeros_like(is_copper, dtype=bool)
    component_sizes: list[int] = []
    small_sizes: list[int] = []

    for start_row in range(RADIAL_CELLS):
        for start_column in range(ANGULAR_CELLS):
            if not is_copper[start_row, start_column] or visited[start_row, start_column]:
                continue
            queue: deque[tuple[int, int]] = deque([(start_row, start_column)])
            visited[start_row, start_column] = True
            component: list[tuple[int, int]] = []
            while queue:
                row, column = queue.popleft()
                component.append((row, column))
                for next_row, next_column in _neighbors(
                    row, column, theta_periodic=theta_periodic
                ):
                    if (
                        is_copper[next_row, next_column]
                        and not visited[next_row, next_column]
                    ):
                        visited[next_row, next_column] = True
                        queue.append((next_row, next_column))
            size = len(component)
            component_sizes.append(size)
            if size < minimum_cells:
                small_sizes.append(size)
                for row, column in component:
                    floating[row, column] = True

    n_floating = int(np.count_nonzero(floating))
    return CopperIslandResult(
        n_copper=n_copper,
        n_floating=n_floating,
        ratio=(n_floating / n_copper) if n_copper else 0.0,
        component_sizes=tuple(component_sizes),
        small_component_sizes=tuple(small_sizes),
        floating_mask=floating,
    )


def historical_precheck(
    chromosome: Chromosome | Sequence[int],
    *,
    minimum_copper_island_cells: int = 4,
    theta_periodic: bool = False,
    invalid_base_objective: float = 1_000_000.0,
    per_floating_iron_cell: float = 50.0,
    per_small_copper_cell: float = 50.0,
) -> PrecheckResult:
    iron = detect_floating_iron(chromosome, theta_periodic=theta_periodic)
    copper = detect_small_copper_islands(
        chromosome,
        minimum_cells=minimum_copper_island_cells,
        theta_periodic=theta_periodic,
    )
    reasons: list[str] = []
    if iron.n_floating > 0:
        reasons.append("FLOATING_IRON")
    if copper.n_floating > 0:
        reasons.append("SMALL_COPPER_ISLAND")
    has_any_copper = copper.n_copper > 0
    rejected = bool(reasons)
    objective = (
        invalid_base_objective
        + per_floating_iron_cell * iron.n_floating
        + per_small_copper_cell * copper.n_floating
        if rejected
        else None
    )
    return PrecheckResult(
        iron=iron,
        copper=copper,
        has_any_copper=has_any_copper,
        reasons=tuple(reasons),
        rejected=rejected,
        objective_if_rejected=float(objective) if objective is not None else None,
    )

