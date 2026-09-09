from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from encoding.chromosome import Chromosome, ensure_chromosome
from encoding.layout import ANGULAR_CELLS, RADIAL_CELLS, matlab_vector_to_grid
from encoding.turns import (
    DEFAULT_CIRCUIT_NAMES,
    DEFAULT_SECTOR_CIRCUIT_IDS,
    TurnsAllocation,
    compute_turns_per_circuit,
)


MATERIAL_NAMES = {0: "Air", 1: "Pure Iron", 2: "Copper"}


@dataclass(frozen=True, slots=True)
class HistoricalInsetCell:
    gene_index: int
    radial_index: int
    angular_index: int
    sector_index: int
    mirrored: bool
    material_code: int
    material_name: str
    background_material_name: str
    circuit_name: str | None
    turns: float
    radial_bounds_mm: tuple[float, float]
    angular_bounds_deg: tuple[float, float]
    copper_inset_radial_bounds_mm: tuple[float, float] | None
    copper_inset_angular_bounds_deg: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class HistoricalInsetTopology:
    chromosome: Chromosome
    base_material_grid: np.ndarray
    large_region_material_grid: np.ndarray
    merge_key_grid: np.ndarray
    turns_allocation: TurnsAllocation
    cells: tuple[HistoricalInsetCell, ...]
    geometry_mode: str = "historical_inset"

    def cells_for_sector(self, sector_index: int) -> tuple[HistoricalInsetCell, ...]:
        return tuple(cell for cell in self.cells if cell.sector_index == sector_index)


def decode_historical_inset(
    chromosome: Chromosome | Sequence[int],
    *,
    radial_inner_mm: float = 31.5,
    radial_outer_mm: float = 52.0,
    sector_span_deg: float = 15.0,
    sector_count: int = 6,
    inset_radial_ratio: float = 0.2,
    inset_angular_ratio: float = 0.2,
    sector_circuit_ids: Sequence[int] = DEFAULT_SECTOR_CIRCUIT_IDS,
    circuit_names: Sequence[str] = DEFAULT_CIRCUIT_NAMES,
) -> HistoricalInsetTopology:
    """Decode the old per-cell inset topology without issuing FEMM commands."""

    gene = ensure_chromosome(chromosome)
    if not 0 <= inset_radial_ratio < 0.5 or not 0 <= inset_angular_ratio < 0.5:
        raise ValueError("inset ratios must be in [0, 0.5)")
    if sector_count != len(sector_circuit_ids):
        raise ValueError("sector_count must match sector_circuit_ids")

    grid = matlab_vector_to_grid(gene)
    large_grid = np.where(grid == 1, 1, 0).astype(np.uint8)
    merge_keys = np.where(grid == 1, 200, 100).astype(np.uint16)
    turns = compute_turns_per_circuit(
        gene,
        sector_circuit_ids=sector_circuit_ids,
        circuit_names=circuit_names,
    )

    r_edges = np.linspace(radial_inner_mm, radial_outer_mm, RADIAL_CELLS + 1)
    theta_edges = np.linspace(0.0, sector_span_deg, ANGULAR_CELLS + 1)
    cells: list[HistoricalInsetCell] = []

    for sector in range(sector_count):
        mirrored = (sector + 1) % 2 == 0  # MATLAB sectors are one-based.
        phase_id = int(sector_circuit_ids[sector])
        circuit_name = circuit_names[phase_id - 1] if phase_id > 0 else None
        turns_value = turns.turns_per_copper_cell[phase_id - 1] if phase_id > 0 else 0.0
        offset = sector * sector_span_deg

        for radial_index in range(RADIAL_CELLS):
            r1, r2 = float(r_edges[radial_index]), float(r_edges[radial_index + 1])
            for angular_index in range(ANGULAR_CELLS):
                base_t1 = float(theta_edges[angular_index])
                base_t2 = float(theta_edges[angular_index + 1])
                if mirrored:
                    t1 = offset + sector_span_deg - base_t2
                    t2 = offset + sector_span_deg - base_t1
                else:
                    t1 = offset + base_t1
                    t2 = offset + base_t2

                code = int(grid[radial_index, angular_index])
                gene_index = radial_index * ANGULAR_CELLS + angular_index
                if code == 2:
                    dr = inset_radial_ratio * (r2 - r1)
                    dt = inset_angular_ratio * (base_t2 - base_t1)
                    ir1, ir2 = r1 + dr, r2 - dr
                    if mirrored:
                        it1 = offset + sector_span_deg - (base_t2 - dt)
                        it2 = offset + sector_span_deg - (base_t1 + dt)
                    else:
                        it1 = offset + base_t1 + dt
                        it2 = offset + base_t2 - dt
                    inset_r = (float(min(ir1, ir2)), float(max(ir1, ir2)))
                    inset_t = (float(min(it1, it2)), float(max(it1, it2)))
                    background = "Air"
                    cell_circuit = circuit_name
                    cell_turns = float(turns_value)
                else:
                    inset_r = None
                    inset_t = None
                    background = MATERIAL_NAMES[code]
                    cell_circuit = None
                    cell_turns = 0.0

                cells.append(
                    HistoricalInsetCell(
                        gene_index=gene_index,
                        radial_index=radial_index,
                        angular_index=angular_index,
                        sector_index=sector,
                        mirrored=mirrored,
                        material_code=code,
                        material_name=MATERIAL_NAMES[code],
                        background_material_name=background,
                        circuit_name=cell_circuit,
                        turns=cell_turns,
                        radial_bounds_mm=(r1, r2),
                        angular_bounds_deg=(float(t1), float(t2)),
                        copper_inset_radial_bounds_mm=inset_r,
                        copper_inset_angular_bounds_deg=inset_t,
                    )
                )

    return HistoricalInsetTopology(
        chromosome=gene,
        base_material_grid=grid,
        large_region_material_grid=large_grid,
        merge_key_grid=merge_keys,
        turns_allocation=turns,
        cells=tuple(cells),
    )

