from __future__ import annotations

from pathlib import Path

import numpy as np

from encoding.layout import grid_to_matlab_vector
from evaluator.femm import IsolatedFemmEvaluator
from femm_runner.config import FemmRunConfig
from femm_runner.historical_inset_commands import apply_historical_inset
from femm_runner.merged_copper_v5_commands import apply_merged_copper_v5
from femm_runner.merged_copper_v5_commands import (
    _air_band_component_points,
    _component_information,
    _copper_neighbors,
    _has_air_band,
)


class RecordingFemmApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def __getattr__(self, name: str):
        def record(*args):
            self.calls.append((name, args))

        return record


def test_historical_inset_command_bridge_assigns_copper_circuits_and_turns(
    valid_copper_chromosome,
) -> None:
    api = RecordingFemmApi()
    apply_historical_inset(api, valid_copper_chromosome)
    copper = [
        args
        for name, args in api.calls
        if name == "mi_setblockprop" and args[0] == "Copper"
    ]
    assert len(copper) == 6 * 4
    assert {args[3] for args in copper} == {"A+", "A-", "B+", "B-", "C+", "C-"}
    assert all(args[6] == 25.0 for args in copper)
    assert all(args[5] == 30 for args in copper)


def test_merged_copper_v5_uses_one_label_per_copper_component(
    valid_copper_chromosome,
) -> None:
    historical_api = RecordingFemmApi()
    merged_api = RecordingFemmApi()
    apply_historical_inset(historical_api, valid_copper_chromosome)
    apply_merged_copper_v5(merged_api, valid_copper_chromosome)
    historical_copper = [
        args
        for name, args in historical_api.calls
        if name == "mi_setblockprop" and args[0] == "Copper"
    ]
    merged_copper = [
        args
        for name, args in merged_api.calls
        if name == "mi_setblockprop" and args[0] == "Copper"
    ]
    assert len(historical_copper) == 24
    assert len(merged_copper) == 6
    assert {args[3] for args in merged_copper} == {
        "A+",
        "A-",
        "B+",
        "B-",
        "C+",
        "C-",
    }
    assert all(args[6] == 100.0 for args in merged_copper)
    assert any(name == "mi_addarc" for name, _ in merged_api.calls)


def test_merged_copper_v5_labels_actual_air_regions_not_cell_components() -> None:
    # Regression topology P0018: MATLAB V5 sees two cell-level air-band
    # components, while the inset polygons create three actual closed regions.
    rows = [
        "2222222200", "0222222220", "0222221110", "0222011110",
        "0222011111", "0222211110", "2222222111", "2020211111",
        "2022110111", "1111110011", "2211100011", "2222201111",
        "2022221111", "2222211111", "2211111111", "1111111111",
        "1111111111", "1111111111",
    ]
    grid = np.asarray([[int(value) for value in row] for row in rows], dtype=np.uint8)
    copper = grid == 2
    air_cells = np.zeros(grid.shape, dtype=bool)
    for radial, angular in np.argwhere(copper):
        air_cells[radial, angular] = _has_air_band(
            _copper_neighbors(copper, int(radial), int(angular))
        )
    cell_representatives, _ = _component_information(
        np.where(air_cells, 350.0, np.nan)
    )
    points = _air_band_component_points(
        copper,
        air_cells,
        np.linspace(31.5, 52.0, 19),
        np.linspace(0.0, 15.0, 11),
        0.2,
    )
    assert len(cell_representatives) == 2
    assert len(points) == 3


def test_femm_evaluator_hard_reject_does_not_start_femm(tmp_path: Path) -> None:
    base_model = tmp_path / "base.fem"
    base_model.write_text("placeholder", encoding="ascii")
    grid = np.zeros((18, 10), dtype=np.uint8)
    grid[0, 0] = 1
    grid[4, 4] = 2
    evaluator = IsolatedFemmEvaluator(
        FemmRunConfig(base_model=base_model, work_root=tmp_path / "cases")
    )
    result = evaluator.evaluate(grid_to_matlab_vector(grid))
    assert result.status == "PRECHECK_REJECTED"
    assert result.metadata["femm_started"] is False
    assert not (tmp_path / "cases").exists()
