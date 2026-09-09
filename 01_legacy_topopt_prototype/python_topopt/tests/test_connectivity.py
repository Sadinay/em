from __future__ import annotations

import numpy as np

from constraints.connectivity import (
    detect_floating_iron,
    detect_small_copper_islands,
    historical_precheck,
)
from encoding.layout import grid_to_matlab_vector


def test_floating_iron_uses_outer_row_as_anchor() -> None:
    grid = np.zeros((18, 10), dtype=np.uint8)
    grid[17, 0] = 1
    grid[16, 0] = 1
    grid[0, 5] = 1
    result = detect_floating_iron(grid_to_matlab_vector(grid))
    assert result.n_iron == 3
    assert result.n_floating == 1
    assert result.anchored_mask[16, 0]
    assert result.floating_mask[0, 5]


def test_small_copper_island_threshold_is_strict() -> None:
    three = np.ones((18, 10), dtype=np.uint8)
    three[5, 4:7] = 2
    result_three = detect_small_copper_islands(
        grid_to_matlab_vector(three), minimum_cells=4
    )
    assert result_three.component_sizes == (3,)
    assert result_three.n_floating == 3

    four = np.ones((18, 10), dtype=np.uint8)
    four[5:7, 4:6] = 2
    result_four = detect_small_copper_islands(
        grid_to_matlab_vector(four), minimum_cells=4
    )
    assert result_four.component_sizes == (4,)
    assert result_four.n_floating == 0


def test_precheck_reports_overlapping_reasons() -> None:
    grid = np.zeros((18, 10), dtype=np.uint8)
    grid[0, 0] = 1
    grid[4, 4] = 2
    result = historical_precheck(grid_to_matlab_vector(grid))
    assert result.reasons == ("FLOATING_IRON", "SMALL_COPPER_ISLAND")
    assert result.objective_if_rejected == 1_000_100.0

