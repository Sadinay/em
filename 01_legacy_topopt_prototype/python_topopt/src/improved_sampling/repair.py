from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from constraints.connectivity import historical_precheck
from encoding.chromosome import Chromosome
from encoding.layout import matlab_vector_to_grid, grid_to_matlab_vector

from .features import hamming


@dataclass(frozen=True, slots=True)
class RepairResult:
    raw: Chromosome
    repaired: Chromosome
    status: str
    iterations: int
    repair_hamming: int
    actions: tuple[dict[str, Any], ...]
    rejection_reasons: tuple[str, ...]


def repair_topology(raw: Chromosome, config: dict[str, Any]) -> RepairResult:
    repair = config["repair"]
    constraints = config["constraints"]
    fixed = {int(key): int(value) for key, value in constraints.get("fixed_cells", {}).items()}
    for index, value in fixed.items():
        if raw.genes[index] != value:
            return RepairResult(raw, raw, "fixed_cell_violation", 0, 0, (), ("FIXED_CELL",))
    grid = matlab_vector_to_grid(raw)
    actions: list[dict[str, Any]] = []
    iterations = 0
    for iterations in range(1, int(repair["maximum_iterations"]) + 1):
        current = grid_to_matlab_vector(grid)
        precheck = historical_precheck(
            current,
            minimum_copper_island_cells=int(constraints["minimum_copper_island_cells"]),
            theta_periodic=bool(constraints["theta_periodic"]),
        )
        if not precheck.rejected and precheck.has_any_copper:
            distance = hamming(raw, current)
            status = "repaired" if distance else "unchanged_legal"
            return RepairResult(raw, current, status, iterations - 1, distance, tuple(actions), ())
        changed = False
        if precheck.iron.n_floating:
            if repair["floating_iron_strategy"] != "remove":
                raise ValueError("only deterministic floating-iron remove is implemented")
            positions = sorted(tuple(int(x) for x in row) for row in np.argwhere(precheck.iron.floating_mask))
            blocked = [cell for cell in positions if cell[1] + 10 * cell[0] in fixed]
            if blocked:
                return RepairResult(raw, current, "repair_failed", iterations, hamming(raw, current), tuple(actions), ("FIXED_FLOATING_IRON",))
            for cell in positions:
                grid[cell] = 0
            actions.append({"iteration": iterations, "action": "remove_floating_iron", "cells": [list(x) for x in positions]})
            changed = changed or bool(positions)
        if precheck.copper.n_floating:
            if repair["small_copper_island_strategy"] != "remove":
                raise ValueError("only deterministic small-copper remove is implemented")
            positions = sorted(tuple(int(x) for x in row) for row in np.argwhere(precheck.copper.floating_mask))
            blocked = [cell for cell in positions if cell[1] + 10 * cell[0] in fixed]
            if blocked:
                return RepairResult(raw, current, "repair_failed", iterations, hamming(raw, current), tuple(actions), ("FIXED_SMALL_COPPER",))
            for cell in positions:
                grid[cell] = 0
            actions.append({"iteration": iterations, "action": "remove_small_copper", "cells": [list(x) for x in positions]})
            changed = changed or bool(positions)
        repaired = grid_to_matlab_vector(grid)
        distance = hamming(raw, repaired)
        if distance > int(repair["maximum_repair_hamming"]):
            return RepairResult(raw, repaired, "repair_hamming_exceeded", iterations, distance, tuple(actions), ("MAXIMUM_REPAIR_HAMMING",))
        if not changed:
            reasons = list(precheck.reasons)
            if not precheck.has_any_copper:
                reasons.append("NO_COPPER")
            return RepairResult(raw, repaired, "repair_failed", iterations, distance, tuple(actions), tuple(reasons))
    repaired = grid_to_matlab_vector(grid)
    final = historical_precheck(
        repaired,
        minimum_copper_island_cells=int(constraints["minimum_copper_island_cells"]),
        theta_periodic=bool(constraints["theta_periodic"]),
    )
    reasons = list(final.reasons)
    if not final.has_any_copper:
        reasons.append("NO_COPPER")
    return RepairResult(raw, repaired, "maximum_iterations", iterations, hamming(raw, repaired), tuple(actions), tuple(reasons))
