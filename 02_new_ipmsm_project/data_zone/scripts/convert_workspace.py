"""Convert the MATLAB GA workspace into traceable NumPy/JSON/CSV datasets.

The original MAT file is never modified.  Numeric tensors are stored in NPZ
files because JSON is inefficient for the 302,904 historical records; JSON is
used for metadata and a single human-readable example.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAT = ROOT / "data_zone" / "raw" / "workspace_600.mat"
DEFAULT_PROCESSED = ROOT / "data_zone" / "processed"
DEFAULT_EXPORTS = ROOT / "data_zone" / "exports"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_bit_pairs(bits: np.ndarray) -> np.ndarray:
    """Decode adjacent MATLAB bits: 00->0, 01->1, 10->2, 11->3."""
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.shape[-1] != 200:
        raise ValueError(f"Expected a final dimension of 200 bits, got {bits.shape}")
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("Chromosome contains values outside {0, 1}")
    return 2 * bits[..., 0::2] + bits[..., 1::2]


def vectors_to_grids(material_vectors: np.ndarray) -> np.ndarray:
    """Map cell vectors to [sample, radial_index, angular_index] 10x10 grids.

    MATLAB vector ordering is angular-major: ten consecutive entries run from
    the inner to outer radius at one angular position.  The transpose makes the
    exported image axes spatially intuitive: rows=radius, columns=angle.
    """
    values = np.asarray(material_vectors, dtype=np.uint8)
    if values.shape[-1] != 100:
        raise ValueError(f"Expected 100 material cells, got {values.shape}")
    leading = values.shape[:-1]
    return values.reshape(*leading, 10, 10).swapaxes(-2, -1)


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def _write_csv(path: Path, columns: dict[str, np.ndarray]) -> None:
    names = list(columns)
    rows = zip(*(np.asarray(columns[name]).reshape(-1).tolist() for name in names))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        writer.writerows(rows)


def _current_dataset(data: dict[str, Any]) -> dict[str, np.ndarray]:
    # Despite the ambiguous variable names, cross-checking the final state
    # proves that population/population_all are the structure-corrected genes
    # evaluated by FEMM.  population_noChange_all stores the pre-repair genes.
    corrected_bits = np.asarray(data["population"], dtype=np.uint8)
    material_corrected = decode_bit_pairs(corrected_bits)
    stored_corrected = np.asarray(data["Material_change"], dtype=np.uint8)
    stored_raw = np.asarray(data["Material"], dtype=np.uint8)
    if not np.array_equal(material_corrected, stored_corrected):
        raise AssertionError("Adjacent-bit decoding does not match corrected Material_change")

    raw_bits = np.asarray(data["population_noChange_all"], dtype=np.uint8)[:, :, -1]
    if not np.array_equal(decode_bit_pairs(raw_bits), stored_raw):
        raise AssertionError("Final pre-repair chromosome does not match Material")

    return {
        "genome_bits_raw": raw_bits,
        "genome_bits_corrected": corrected_bits,
        "material_vector_raw": stored_raw,
        "material_vector_corrected": stored_corrected,
        "material_grid_raw": vectors_to_grids(stored_raw),
        "material_grid_corrected": vectors_to_grids(stored_corrected),
        "t_avg_nm": np.asarray(data["Tavg"], dtype=np.float64),
        "delta_t_nm": np.asarray(data["DeltaT"], dtype=np.float64),
        "fitness": np.asarray(data["Fitvalue"], dtype=np.float64),
        "volume_pm_cells": np.asarray(data["VolumePM"], dtype=np.uint8),
        "material_connectivity": np.asarray(data["MaterialConectivity"], dtype=np.float64),
        "rectangularity_pm": np.asarray(data["Rectangularity_PM"], dtype=np.float64),
        "sample_index_0based": np.arange(raw_bits.shape[0], dtype=np.int32),
    }


def _best_dataset(data: dict[str, Any]) -> dict[str, np.ndarray]:
    fitness_history = np.asarray(data["Fitvalue_all"], dtype=np.float64)
    # Column 0 is the initial state; columns 1..600 correspond to generations.
    generation_count = int(data["Generation"])
    best_rows = np.argmax(fitness_history[:, 1 : generation_count + 1], axis=0)
    generation_cols = np.arange(1, generation_count + 1)
    raw_history = np.asarray(data["population_noChange_all"], dtype=np.uint8)
    corrected_history = np.asarray(data["population_all"], dtype=np.uint8)
    best_raw_bits = raw_history[best_rows, :, generation_cols]
    best_corrected_bits = corrected_history[best_rows, :, generation_cols]
    best_raw_vectors = decode_bit_pairs(best_raw_bits)
    best_corrected_vectors = decode_bit_pairs(best_corrected_bits)
    stored_best = np.asarray(data["Bestpopulation"], dtype=np.uint8)
    if not np.array_equal(best_raw_vectors, stored_best):
        raise AssertionError("Bestpopulation is not aligned with pre-repair historical chromosomes")
    if not np.array_equal(
        fitness_history[best_rows, generation_cols], np.asarray(data["Bestfitness"], dtype=np.float64)
    ):
        raise AssertionError("Bestfitness does not equal the per-generation maximum")

    def select(name: str, dtype: Any) -> np.ndarray:
        return np.asarray(data[name], dtype=dtype)[best_rows, generation_cols]

    return {
        "generation_1based": generation_cols.astype(np.int32),
        "population_row_0based": best_rows.astype(np.int32),
        "genome_bits_raw": best_raw_bits,
        "genome_bits_corrected": best_corrected_bits,
        "material_vector_raw": best_raw_vectors,
        "material_vector_corrected": best_corrected_vectors,
        "material_grid_raw": vectors_to_grids(best_raw_vectors),
        "material_grid_corrected": vectors_to_grids(best_corrected_vectors),
        "t_avg_nm": select("Tavg_all", np.float64),
        "delta_t_nm": select("DeltaT_all", np.float64),
        "fitness": select("Fitvalue_all", np.float64),
        "volume_pm_cells": select("VolumePM_all", np.uint8),
        "material_connectivity": select("MaterialConectivity_all", np.float64),
        "rectangularity_pm": select("Rectangularity_PM_all", np.float64),
    }


def _full_history_dataset(data: dict[str, Any]) -> dict[str, np.ndarray]:
    # Convert from MATLAB [population, bit, state] to Python
    # [state, population, ...].  State 0 is initialization, 1..600 are GA gens.
    raw_bits = np.asarray(data["population_noChange_all"], dtype=np.uint8).transpose(2, 0, 1)
    corrected_bits = np.asarray(data["population_all"], dtype=np.uint8).transpose(2, 0, 1)
    corrected_vectors = decode_bit_pairs(corrected_bits)
    raw_vectors = decode_bit_pairs(raw_bits)
    return {
        "genome_bits_raw": raw_bits,
        "genome_bits_corrected": corrected_bits,
        "material_vector_raw": raw_vectors,
        "material_vector_corrected": corrected_vectors,
        "material_grid_raw": vectors_to_grids(raw_vectors),
        "material_grid_corrected": vectors_to_grids(corrected_vectors),
        "t_avg_nm": np.asarray(data["Tavg_all"], dtype=np.float64).T,
        "delta_t_nm": np.asarray(data["DeltaT_all"], dtype=np.float64).T,
        "fitness": np.asarray(data["Fitvalue_all"], dtype=np.float64).T,
        "volume_pm_cells": np.asarray(data["VolumePM_all"], dtype=np.uint8).T,
        "material_connectivity": np.asarray(data["MaterialConectivity_all"], dtype=np.float64).T,
        "rectangularity_pm": np.asarray(data["Rectangularity_PM_all"], dtype=np.float64).T,
        "state_index_0based": np.arange(corrected_bits.shape[0], dtype=np.int32),
        "population_row_0based": np.arange(corrected_bits.shape[1], dtype=np.int32),
    }


def _material_position_rows(position: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell, values in enumerate(np.asarray(position, dtype=np.float64)):
        angular = cell // 10
        radial = cell % 10
        row: dict[str, Any] = {
            "cell_index_1based": cell + 1,
            "angular_index_0based": angular,
            "radial_index_0based": radial,
        }
        for copy in range(4):
            x = float(values[2 * copy])
            y = float(values[2 * copy + 1])
            row[f"copy_{copy + 1}_x_mm"] = x
            row[f"copy_{copy + 1}_y_mm"] = y
        rows.append(row)
    return rows


def convert(mat_path: Path, processed_dir: Path, exports_dir: Path, include_full_history: bool) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    current = _current_dataset(data)
    best = _best_dataset(data)
    np.savez_compressed(processed_dir / "current_generation.npz", **current)
    np.savez_compressed(processed_dir / "generation_best.npz", **best)

    if include_full_history:
        history = _full_history_dataset(data)
        np.savez_compressed(processed_dir / "full_history.npz", **history)

    _write_csv(
        exports_dir / "current_generation_targets.csv",
        {key: current[key] for key in (
            "sample_index_0based", "t_avg_nm", "delta_t_nm", "fitness",
            "volume_pm_cells", "material_connectivity", "rectangularity_pm",
        )},
    )
    _write_csv(
        exports_dir / "generation_best_targets.csv",
        {key: best[key] for key in (
            "generation_1based", "population_row_0based", "t_avg_nm", "delta_t_nm",
            "fitness", "volume_pm_cells", "material_connectivity", "rectangularity_pm",
        )},
    )

    position_rows = _material_position_rows(np.asarray(data["MaterialPosition"]))
    with (exports_dir / "material_positions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(position_rows[0]))
        writer.writeheader()
        writer.writerows(position_rows)

    example_index = int(np.argmax(current["fitness"]))
    example = {
        "description": "当前第600代中适应度最高的一个可直接阅读样本",
        "sample_index_0based": example_index,
        "genome_bits_raw": current["genome_bits_raw"][example_index].tolist(),
        "genome_bit_pairs": current["genome_bits_raw"][example_index].reshape(100, 2).tolist(),
        "material_vector_raw": current["material_vector_raw"][example_index].tolist(),
        "material_vector_corrected": current["material_vector_corrected"][example_index].tolist(),
        "material_grid_corrected_rows_radius_columns_angle": current["material_grid_corrected"][example_index].tolist(),
        "targets": {
            "t_avg_nm": float(current["t_avg_nm"][example_index]),
            "delta_t_nm": float(current["delta_t_nm"][example_index]),
            "fitness": float(current["fitness"][example_index]),
            "volume_pm_cells": int(current["volume_pm_cells"][example_index]),
        },
    }
    with (exports_dir / "readable_example_best_current.json").open("w", encoding="utf-8") as stream:
        json.dump(example, stream, ensure_ascii=False, indent=2)

    packed = np.packbits(
        np.asarray(data["population_all"], dtype=np.uint8).transpose(2, 0, 1).reshape(-1, 200),
        axis=1,
    )
    unique_history_count = int(np.unique(packed, axis=0).shape[0])
    metadata = {
        "source": {
            "path": str(mat_path.resolve()),
            "sha256": sha256(mat_path),
            "matlab_header": str(data.get("__header__", b"").decode("latin-1", errors="replace") if isinstance(data.get("__header__"), bytes) else data.get("__header__", "")),
        },
        "dimensions": {
            "chromosome_bits": 200,
            "material_cells": 100,
            "grid_radial_by_angular": [10, 10],
            "population_size": int(data["popsize"]),
            "completed_generations": int(data["Generation"]),
            "saved_states_including_initial": int(np.asarray(data["Fitvalue_all"]).shape[1]),
            "historical_records_with_repetition": int(np.asarray(data["Fitvalue_all"]).size),
            "unique_corrected_historical_chromosomes": unique_history_count,
        },
        "encoding": {
            "bit_pair_order": "adjacent, most-significant bit first",
            "formula": "material_code = 2 * bit[2*i] + bit[2*i+1]",
            "mapping": {"00": 0, "01": 1, "10": 2, "11": 3},
            "vector_order": "angular-major; every ten entries traverse radius inner-to-outer",
            "cnn_grid_axes": "[radial_index, angular_index]",
            "cnn_one_hot_example": "np.eye(4, dtype=np.float32)[grid].transpose(2, 0, 1)",
        },
        "material_identity_status": {
            "code_0": "Air; supported by the reference FEM mapping and corrected FEMM replay",
            "code_1": "N38 permanent magnet; independently verified by VolumePM",
            "codes_2_and_3": "Pure Iron in FEMM; both map to the same physical material",
            "recommended_cnn_encoding": "collapse codes 2/3 and use three physical one-hot channels; retain four-state data for audit/ablation",
        },
        "stored_files": {
            "current_generation.npz": "504 final-state samples; raw and corrected input plus scalar targets",
            "generation_best.npz": "one corrected best individual for each of generations 1..600",
            "full_history.npz": "all 601x504 records" if include_full_history else "not generated; rerun with --include-full-history",
        },
        "ga_parameters_observed": {
            "pcrossover": float(data["pcrossover"]),
            "pmutation": float(data["pmutation"]),
            "nmax": int(data["nmax"]),
            "generationnmax": int(data["Generationnmax"]),
        },
        "motor_parameters_observed": {
            "steps": int(data["inp"].steps),
            "stps": int(data["inp"].stps),
            "poles_or_parameter_P": int(data["inp"].P),
            "rotor_outer_diameter_mm": float(data["inp"].RotorOD),
            "stator_inner_diameter_mm": float(data["inp"].StatorID),
            "stack_depth_mm": float(data["inp"].Lfe),
        },
    }
    with (exports_dir / "dataset_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)

    print(f"Converted: {mat_path}")
    print(f"Current generation: {current['material_grid_corrected'].shape}")
    print(f"Generation best: {best['material_grid_corrected'].shape}")
    print(f"Historical records: {np.asarray(data['Fitvalue_all']).size:,}; unique corrected: {unique_history_count:,}")
    print(f"Outputs: {processed_dir} and {exports_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, default=DEFAULT_MAT)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--exports-dir", type=Path, default=DEFAULT_EXPORTS)
    parser.add_argument("--include-full-history", action="store_true")
    args = parser.parse_args()
    convert(args.mat, args.processed_dir, args.exports_dir, args.include_full_history)


if __name__ == "__main__":
    main()
