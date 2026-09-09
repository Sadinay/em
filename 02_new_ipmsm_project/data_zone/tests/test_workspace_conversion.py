from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


converter = load_script("convert_workspace", ROOT / "data_zone" / "scripts" / "convert_workspace.py")
analyzer = load_script("analyze_ipmsm", ROOT / "femm_zone" / "scripts" / "analyze_ipmsm_structure.py")


def test_adjacent_bit_pair_decode() -> None:
    pairs = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8), (25, 1))
    decoded = converter.decode_bit_pairs(pairs.reshape(1, 200))
    assert decoded.shape == (1, 100)
    assert decoded[0, :4].tolist() == [0, 1, 2, 3]


def test_workspace_decode_and_grid_order() -> None:
    data = loadmat(ROOT / "data_zone" / "raw" / "workspace_600.mat", squeeze_me=True, struct_as_record=False)
    decoded = converter.decode_bit_pairs(np.asarray(data["population"], dtype=np.uint8))
    assert np.array_equal(decoded, data["Material_change"])
    grid = converter.vectors_to_grids(decoded[:1])
    assert grid.shape == (1, 10, 10)
    # vector cell = angular*10 + radial; grid axes are [radial, angular]
    assert grid[0, 7, 3] == decoded[0, 3 * 10 + 7]


def test_exported_tensor_shapes_and_best_alignment() -> None:
    current = np.load(ROOT / "data_zone" / "processed" / "current_generation.npz")
    best = np.load(ROOT / "data_zone" / "processed" / "generation_best.npz")
    assert current["genome_bits_corrected"].shape == (504, 200)
    assert current["material_grid_corrected"].shape == (504, 10, 10)
    assert current["t_avg_nm"].shape == (504,)
    assert best["material_grid_corrected"].shape == (600, 10, 10)
    assert best["fitness"].shape == (600,)


def test_all_material_positions_match_four_fem_labels() -> None:
    data = loadmat(ROOT / "data_zone" / "raw" / "workspace_600.mat", squeeze_me=True, struct_as_record=False)
    properties, labels, _ = analyzer.parse_fem(ROOT / "femm_zone" / "models" / "IPMSM.fem")
    cells = analyzer.match_design_cells(data["MaterialPosition"], properties, labels)
    assert len(cells) == 100
    assert sum(len(cell["copies"]) for cell in cells) == 400
    assert all(len({copy["physical_class"] for copy in cell["copies"]}) == 1 for cell in cells)
