from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from encoding.chromosome import Chromosome
from encoding.layout import matlab_vector_to_grid


def hamming(first: Chromosome, second: Chromosome) -> int:
    return int(np.count_nonzero(first.to_numpy() != second.to_numpy()))


def _components(mask: np.ndarray) -> list[int]:
    seen = np.zeros_like(mask, dtype=bool)
    sizes: list[int] = []
    rows, columns = mask.shape
    for row in range(rows):
        for column in range(columns):
            if not mask[row, column] or seen[row, column]:
                continue
            queue: deque[tuple[int, int]] = deque([(row, column)])
            seen[row, column] = True
            size = 0
            while queue:
                rr, cc = queue.popleft()
                size += 1
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < columns and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            sizes.append(size)
    return sorted(sizes, reverse=True)


@dataclass(frozen=True, slots=True)
class TopologyFeatures:
    air_cells: int
    iron_cells: int
    copper_cells: int
    iron_components: int
    copper_components: int
    largest_iron_component: int
    largest_copper_component: int
    material_transition_edges: int
    copper_centroid_radial: float | None
    copper_centroid_angular: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def topology_features(chromosome: Chromosome) -> TopologyFeatures:
    grid = matlab_vector_to_grid(chromosome)
    counts = [int(np.count_nonzero(grid == value)) for value in (0, 1, 2)]
    iron = _components(grid == 1)
    copper = _components(grid == 2)
    transitions = int(
        np.count_nonzero(grid[1:, :] != grid[:-1, :])
        + np.count_nonzero(grid[:, 1:] != grid[:, :-1])
    )
    copper_positions = np.argwhere(grid == 2)
    return TopologyFeatures(
        air_cells=counts[0],
        iron_cells=counts[1],
        copper_cells=counts[2],
        iron_components=len(iron),
        copper_components=len(copper),
        largest_iron_component=iron[0] if iron else 0,
        largest_copper_component=copper[0] if copper else 0,
        material_transition_edges=transitions,
        copper_centroid_radial=(float(np.mean(copper_positions[:, 0])) if copper_positions.size else None),
        copper_centroid_angular=(float(np.mean(copper_positions[:, 1])) if copper_positions.size else None),
    )
