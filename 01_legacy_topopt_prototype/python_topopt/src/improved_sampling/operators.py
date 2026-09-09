from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from encoding.chromosome import Chromosome
from encoding.layout import matlab_vector_to_grid, grid_to_matlab_vector


@dataclass(frozen=True, slots=True)
class MutationRecord:
    chromosome: Chromosome
    operator: str
    scale: str
    requested_cells: int
    changed_indices: tuple[int, ...]
    metadata: dict[str, Any]


def _index(row: int, column: int) -> int:
    return column + 10 * row


def _neighbors(row: int, column: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (rr, cc)
        for rr, cc in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1))
        if 0 <= rr < 18 and 0 <= cc < 10
    )


def _weighted_choice(mapping: dict[str, float], rng: np.random.Generator) -> str:
    # Canonical ordering is required because immutable JSON is written with
    # sorted keys; resumed and uninterrupted runs must consume RNG identically.
    names = sorted(mapping)
    probabilities = np.asarray([float(mapping[name]) for name in names], dtype=float)
    probabilities /= probabilities.sum()
    return names[int(rng.choice(len(names), p=probabilities))]


def _different_material(value: int, rng: np.random.Generator) -> int:
    choices = [material for material in (0, 1, 2) if material != int(value)]
    return int(choices[int(rng.integers(0, len(choices)))])


def _grow_same_material(
    grid: np.ndarray,
    start: tuple[int, int],
    count: int,
    fixed: set[int],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    original = int(grid[start])
    selected: list[tuple[int, int]] = []
    selected_set: set[tuple[int, int]] = set()
    frontier: list[tuple[int, int]] = [start]
    while frontier and len(selected) < count:
        pick = int(rng.integers(0, len(frontier)))
        cell = frontier.pop(pick)
        if cell in selected_set or _index(*cell) in fixed or int(grid[cell]) != original:
            continue
        selected.append(cell)
        selected_set.add(cell)
        for neighbor in _neighbors(*cell):
            if neighbor not in selected_set and neighbor not in frontier:
                frontier.append(neighbor)
    return selected


def mutate_2d(
    parent: Chromosome,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> MutationRecord:
    proposal = config["proposal"]
    operator = _weighted_choice(proposal["operator_weights"], rng)
    scale = _weighted_choice(
        {name: float(spec["probability"]) for name, spec in proposal["scales"].items()}, rng
    )
    scale_spec = proposal["scales"][scale]
    requested = int(
        rng.integers(int(scale_spec["minimum_cells"]), int(scale_spec["maximum_cells"]) + 1)
    )
    fixed = {int(key) for key in config["constraints"].get("fixed_cells", {})}
    grid = matlab_vector_to_grid(parent)
    mutable = [(row, column) for row in range(18) for column in range(10) if _index(row, column) not in fixed]
    if not mutable:
        raise ValueError("no mutable design cells")

    cells: list[tuple[int, int]]
    if operator == "connected_region_repaint":
        start = mutable[int(rng.integers(0, len(mutable)))]
        target = _different_material(int(grid[start]), rng)
        cells = _grow_same_material(grid, start, requested, fixed, rng)
    elif operator == "boundary_growth":
        boundaries: list[tuple[tuple[int, int], int]] = []
        for cell in mutable:
            for neighbor in _neighbors(*cell):
                if int(grid[cell]) != int(grid[neighbor]):
                    boundaries.append((cell, int(grid[neighbor])))
        if not boundaries:
            start = mutable[int(rng.integers(0, len(mutable)))]
            target = _different_material(int(grid[start]), rng)
        else:
            start, target = boundaries[int(rng.integers(0, len(boundaries)))]
        cells = _grow_same_material(grid, start, requested, fixed, rng)
    elif operator == "rectangular_patch":
        height = max(1, min(18, int(round(requested ** 0.5))))
        width = max(1, min(10, int(np.ceil(requested / height))))
        row0 = int(rng.integers(0, 18 - height + 1))
        col0 = int(rng.integers(0, 10 - width + 1))
        center = (row0 + height // 2, col0 + width // 2)
        target = _different_material(int(grid[center]), rng)
        cells = [
            (row, column)
            for row in range(row0, row0 + height)
            for column in range(col0, col0 + width)
            if _index(row, column) not in fixed
        ][:requested]
    else:
        raise ValueError(f"unsupported operator {operator}")

    mutated = grid.copy()
    for cell in cells:
        mutated[cell] = target
    chromosome = grid_to_matlab_vector(mutated)
    changed = tuple(
        index for index, (before, after) in enumerate(zip(parent.genes, chromosome.genes)) if before != after
    )
    return MutationRecord(
        chromosome=chromosome,
        operator=operator,
        scale=scale,
        requested_cells=requested,
        changed_indices=changed,
        metadata={"target_material": target, "selected_grid_cells": [list(cell) for cell in cells]},
    )
