from __future__ import annotations

import numpy as np
import pytest

from encoding.layout import matlab_vector_to_grid
from encoding.turns import compute_turns_per_circuit
from topology.historical_inset import decode_historical_inset


def test_turns_are_distributed_to_every_phase(valid_copper_chromosome) -> None:
    result = compute_turns_per_circuit(valid_copper_chromosome)
    assert result.copper_cells_per_circuit == (4, 4, 4, 4, 4, 4)
    assert result.turns_per_copper_cell == (25.0,) * 6


def test_no_copper_produces_zero_turns() -> None:
    from encoding.chromosome import Chromosome

    result = compute_turns_per_circuit(Chromosome.from_iterable([1] * 180))
    assert result.copper_cells_per_circuit == (0,) * 6
    assert result.turns_per_copper_cell == (0.0,) * 6


def test_historical_inset_materials_mirroring_and_copper_bounds(
    valid_copper_chromosome,
) -> None:
    topology = decode_historical_inset(valid_copper_chromosome)
    assert topology.geometry_mode == "historical_inset"
    assert topology.base_material_grid.shape == (18, 10)
    assert np.array_equal(
        topology.base_material_grid, matlab_vector_to_grid(valid_copper_chromosome)
    )
    assert len(topology.cells) == 6 * 180

    copper = next(cell for cell in topology.cells_for_sector(0) if cell.material_code == 2)
    assert copper.background_material_name == "Air"
    assert copper.circuit_name == "B-"
    assert copper.turns == 25.0
    assert copper.copper_inset_radial_bounds_mm is not None
    assert copper.copper_inset_angular_bounds_deg is not None
    r1, r2 = copper.radial_bounds_mm
    ir1, ir2 = copper.copper_inset_radial_bounds_mm
    assert ir1 == pytest.approx(r1 + 0.2 * (r2 - r1))
    assert ir2 == pytest.approx(r2 - 0.2 * (r2 - r1))

    sector0 = next(
        cell
        for cell in topology.cells_for_sector(0)
        if cell.radial_index == 0 and cell.angular_index == 0
    )
    sector1 = next(
        cell
        for cell in topology.cells_for_sector(1)
        if cell.radial_index == 0 and cell.angular_index == 0
    )
    assert sector0.angular_bounds_deg == pytest.approx((0.0, 1.5))
    assert sector1.angular_bounds_deg == pytest.approx((28.5, 30.0))
    assert sector1.mirrored

