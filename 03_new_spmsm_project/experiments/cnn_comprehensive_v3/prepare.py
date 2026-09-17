"""Prepare the v3 CNN comparison inputs and unified candidate pool.

Commands are intentionally concentrated in this file.  This stage prepares data
and reports only; it never trains a CNN or chooses a train/validation/test split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
MAT_FILE = PROJECT / "data_zone/raw/workspace_200.mat"
FEM_FILE = PROJECT / "femm_zone/models/SPMSM_discrete.fem"
MAPPING_SOURCE = PROJECT / "femm_zone/scripts/spmsm_mapping.py"
GAP_DIR = HERE / "data/gap90"
DATA_DIR = HERE / "data"
POOL_DIR = DATA_DIR / "candidate_pool"
SOURCE_DIR = DATA_DIR / "sources"
MODEL_DIR = HERE / "models"
REPORT_DIR = HERE / "reports"

OLD_SELECTED = SOURCE_DIR / "historical40000_lcmd/selected_20000.csv"
OLD_LCMD_DIR = SOURCE_DIR / "historical40000_lcmd"
OLD_FEATURE_ROOT = PROJECT / "experiments/input_distribution_expansion_f1_v1"
PRIOR_3299_ROOT = PROJECT / "femm_zone/results/FEMM_results_20260913"
LATEST_10000_ROOT = PROJECT / "femm_zone/results/FEMM_10000_results_20260917"
LATEST_TASK_ZIP = LATEST_10000_ROOT / "original_task/FEMM_10000_f1_v1.zip"
F1_CHECKPOINT_SOURCE = PROJECT / "experiments/cnn_replay_update_v2/models/vgg16/runs/F-S/best_unconstrained.pt"
F1_EXPECTED_SHA256 = "9612f9892e76bc354ca6c6e885a57b29f865da8b84d0f3b67b195befc6980346"
LABEL_TOLERANCE = 1e-6


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_fem_scalar(name: str) -> float:
    import re

    text = FEM_FILE.read_text(encoding="utf-8")
    values = re.findall(rf"^\[{re.escape(name)}\]\s*=\s*([^\r\n]+)", text, re.MULTILINE)
    require(len(values) == 1, f"Expected one FEM field [{name}], found {len(values)}")
    return float(values[0].strip().strip('"'))


def derive_gap90() -> dict:
    """Derive the 6xN representation from the frozen MAT/FEM geometry."""
    raw = loadmat(MAT_FILE, variable_names=["inp", "MaterialPosition"], squeeze_me=True,
                  struct_as_record=False)
    inp = raw["inp"]
    positions = np.asarray(raw["MaterialPosition"], dtype=np.float64)
    radial_cells, angular_cells, copies = int(inp.y), int(inp.x), 4
    require((radial_cells, angular_cells, copies) == (6, 20, 4), "Unexpected 03 gene geometry")
    require(positions.shape == (radial_cells * angular_cells, 2 * copies),
            f"Unexpected MaterialPosition shape: {positions.shape}")

    theta = float(inp.theta)
    pm_angle = float(inp.PM_angle)
    pole_pitch = float(inp.PM_fan_angle)
    sector_angle = 90.0
    require(math.isclose(theta, pm_angle / (2 * angular_cells), rel_tol=0, abs_tol=1e-12),
            "inp.theta no longer equals PM_angle/(2*x)")
    require(math.isclose(pole_pitch, 45.0, rel_tol=0, abs_tol=1e-12), "Expected 45 degree pole pitch")
    require(read_fem_scalar("MinAngle") == 15.0, "Unexpected FEM MinAngle")

    equivalent_columns = sector_angle / theta
    output_columns = math.ceil(equivalent_columns - 1e-12)
    require(output_columns == 97, f"Geometry-derived N changed to {output_columns}")

    # Every pair of MaterialPosition columns contains one physical copy's x/y coordinates.
    angles = np.empty((radial_cells * angular_cells, copies), dtype=np.float64)
    for copy in range(copies):
        x = positions[:, 2 * copy]
        y = positions[:, 2 * copy + 1]
        angles[:, copy] = np.degrees(np.arctan2(y, x))
    require(np.all((angles > 0) & (angles < sector_angle)), "Design center outside the 90 degree sector")

    rows: list[dict] = []
    column_map = np.empty((angular_cells, copies), dtype=np.int64)
    for angular in range(angular_cells):
        source = angular * radial_cells  # all six radii share the same polar angle
        for copy in range(copies):
            sample = angles[source:source + radial_cells, copy]
            require(float(np.ptp(sample)) < 1e-10, "Radial cells do not share an angular center")
            angle = float(sample.mean())
            target = int(math.floor((sector_angle - angle) / theta + 1e-10))
            require(0 <= target < output_columns, "Derived slot outside gap-aware matrix")
            column_map[angular, copy] = target
            rows.append({
                "original_angular_index_0based": angular,
                "copy_index_0based": copy,
                "physical_center_angle_deg": f"{angle:.12f}",
                "gap90_column_0based": target,
            })

    design_columns = sorted(set(column_map.ravel().tolist()))
    require(len(design_columns) == angular_cells * copies == 80, "Physical design columns collided")
    zero_columns = sorted(set(range(output_columns)) - set(design_columns))
    require(zero_columns == [*range(0, 4), *range(44, 53), *range(93, 97)],
            f"Unexpected gap columns: {zero_columns}")

    # Exact source-to-copy pattern is useful for both reviewers and future data loaders.
    expected = np.stack([
        4 + np.arange(20),
        43 - np.arange(20),
        53 + np.arange(20),
        92 - np.arange(20),
    ], axis=1)
    require(np.array_equal(column_map, expected), "MaterialPosition-derived copy order changed")

    gap_total = pole_pitch - pm_angle
    result = {
        "status": "derived_and_verified",
        "input_shape": [radial_cells, angular_cells],
        "output_shape": [radial_cells, output_columns],
        "independent_design_variables": radial_cells * angular_cells,
        "physical_copies_in_90deg": copies,
        "encoded_design_cells": radial_cells * angular_cells * copies,
        "fixed_zero_cells": radial_cells * len(zero_columns),
        "theta_design_step_deg": theta,
        "design_band_per_pole_deg": pm_angle,
        "pole_pitch_deg": pole_pitch,
        "gap_per_pole_deg": gap_total,
        "gap_per_pole_equivalent_columns": gap_total / theta,
        "sector_deg": sector_angle,
        "sector_equivalent_columns": equivalent_columns,
        "integer_quantization": "ceil(sector_deg/theta); centers use floor((90-angle)/theta)",
        "quantized_sector_coverage_deg": output_columns * theta,
        "quantization_excess_deg": output_columns * theta - sector_angle,
        "design_columns_0based": design_columns,
        "fixed_zero_columns_0based": zero_columns,
        "fixed_zero_ranges_0based_inclusive": [[0, 3], [44, 52], [93, 96]],
        "source_to_gap90_columns": column_map.tolist(),
        "material_encoding": {"0": "Air/fixed pole-gap padding", "1": "N38 permanent magnet"},
        "sources": {
            str(MAT_FILE.relative_to(PROJECT)): sha256(MAT_FILE),
            str(FEM_FILE.relative_to(PROJECT)): sha256(FEM_FILE),
            str(MAPPING_SOURCE.relative_to(PROJECT)): sha256(MAPPING_SOURCE),
        },
    }
    GAP_DIR.mkdir(parents=True, exist_ok=True)
    save_json(GAP_DIR / "geometry.json", result)
    write_csv(GAP_DIR / "angular_mapping.csv", rows,
              ["original_angular_index_0based", "copy_index_0based",
               "physical_center_angle_deg", "gap90_column_0based"])
    np.save(GAP_DIR / "source_to_gap90_columns.npy", column_map)
    return result


def encode_gap90(bits: np.ndarray, geometry: dict | None = None) -> np.ndarray:
    """Convert (...,120) or (...,6,20) genes to (...,6,97); no new variable is created."""
    geometry = derive_gap90() if geometry is None else geometry
    values = np.asarray(bits, dtype=np.uint8)
    require(np.isin(values, (0, 1)).all(), "Gene contains a non-binary value")
    if values.shape[-2:] == (6, 20):
        grid = values
    else:
        require(values.shape[-1] == 120, f"Expected (...,120) or (...,6,20), got {values.shape}")
        grid = values.reshape(*values.shape[:-1], 20, 6).swapaxes(-1, -2)
    output = np.zeros((*grid.shape[:-1], geometry["output_shape"][1]), dtype=np.uint8)
    mapping = np.asarray(geometry["source_to_gap90_columns"], dtype=np.int64)
    for angular in range(20):
        output[..., mapping[angular].tolist()] = grid[..., angular][..., None]
    return output


def gap90_examples() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geometry = derive_gap90()
    examples = {
        "全部为永磁体": np.ones((6, 20), dtype=np.uint8),
        "仅原始角向列0": np.pad(np.ones((6, 1), dtype=np.uint8), ((0, 0), (0, 19))),
        "角向交替": np.tile((np.arange(20) % 2).astype(np.uint8), (6, 1)),
    }
    figure, axes = plt.subplots(len(examples), 2, figsize=(15, 7), constrained_layout=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    for row, (name, original) in enumerate(examples.items()):
        transformed = encode_gap90(original, geometry)
        axes[row, 0].imshow(original, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap="viridis")
        axes[row, 0].set_title(f"{name}：原始 6×20")
        axes[row, 1].imshow(transformed, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap="viridis")
        axes[row, 1].set_title(f"{name}：真实极间距 6×97")
        for start, end in geometry["fixed_zero_ranges_0based_inclusive"]:
            axes[row, 1].axvspan(start - .5, end + .5, color="white", alpha=.28, hatch="//")
        for axis in axes[row]:
            axis.set_ylabel("径向索引")
            axis.set_xlabel("角向列索引")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    figure.suptitle("03 SPMSM：6×20 到 gap-aware 90° 6×97（斜线为固定补零区）")
    figure.savefig(REPORT_DIR / "01_gap90映射样例.png", dpi=180)
    plt.close(figure)


def read_rows(path: Path) -> list[dict[str, str]]:
    require(path.exists(), f"Missing input: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def copy_file(source: Path, destination: Path) -> None:
    require(source.exists(), f"Missing source file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def prepare_model_registry() -> dict:
    """Freeze the feature-model identity and the eight future comparison slots."""
    require(sha256(F1_CHECKPOINT_SOURCE) == F1_EXPECTED_SHA256,
            "The selected f1 feature checkpoint identity changed")
    local_checkpoint = MODEL_DIR / "f1_polar90_vgg16_F-S_step4000.pt"
    if not local_checkpoint.exists() or sha256(local_checkpoint) != F1_EXPECTED_SHA256:
        copy_file(F1_CHECKPOINT_SOURCE, local_checkpoint)

    identity_source = OLD_FEATURE_ROOT / "model_identity.json"
    identity = json.loads(identity_source.read_text(encoding="utf-8"))
    require(identity["generation"] == "f1" and identity["step"] == 4000,
            "Unexpected f1 model generation or step")
    require(identity["architecture"].endswith("Semantic224VGG16V2"),
            "Unexpected f1 feature architecture")
    require(identity["sha256"] == F1_EXPECTED_SHA256, "Model identity hash mismatch")
    identity["source_checkpoint"] = identity["checkpoint"]
    identity["checkpoint"] = str(local_checkpoint.relative_to(HERE)).replace("\\", "/")
    identity["purpose_in_v3"] = "frozen 512D F feature extractor for historical LCMD only"
    save_json(MODEL_DIR / "feature_model_identity.json", identity)

    configurations = [
        ("gene_6x20_smallcnn", "Logical6x20SmallCNNV2", "gene_6x20"),
        ("gene_6x20_resnet20", "Logical6x20ResNet20V2", "gene_6x20"),
        ("gene_gap90_6x97_smallcnn", "Logical6x20SmallCNNV2", "gap90_6x97"),
        ("gene_gap90_6x97_resnet20", "Logical6x20ResNet20V2", "gap90_6x97"),
        ("polar90_vgg16", "Semantic224VGG16V2", "Polar90_224"),
        ("polar90_resnet18", "Semantic224ResNet18V2", "Polar90_224"),
        ("polar360_vgg16", "Semantic224VGG16V2", "Polar360_224"),
        ("polar360_resnet18", "Semantic224ResNet18V2", "Polar360_224"),
    ]
    registry = {
        "status": "registered_not_trained",
        "comparison_principle": "all configurations train from scratch on one future frozen split",
        "split_status": "frozen 70/15/15; historical20000 and expanded13299 split separately, then merged",
        "configurations": [
            {"id": key, "architecture": architecture, "input": input_name,
             "training_status": "not_started", "initialization": "from_scratch"}
            for key, architecture, input_name in configurations
        ],
        "excluded": ["Mini-Inception", "all XY224 inputs", "all other architectures"],
    }
    save_json(MODEL_DIR / "configurations.json", registry)
    return identity


def extract_original_task_zip() -> dict:
    """Safely extract and verify the user-named task ZIP, then remove that ZIP only."""
    destination = SOURCE_DIR / "latest10000/original_task"
    receipt_path = destination / "extraction_receipt.json"
    if not LATEST_TASK_ZIP.exists():
        require(receipt_path.exists(), "Original task ZIP is absent and no verified extraction receipt exists")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        require(receipt.get("verified") is True, "Existing extraction receipt is not verified")
        return receipt

    archive_hash = sha256(LATEST_TASK_ZIP)
    root = destination.resolve()
    require(HERE.resolve() in root.parents, "Refusing to replace extraction outside v3")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    path_map = []
    with zipfile.ZipFile(LATEST_TASK_ZIP, "r") as archive:
        require(archive.testzip() is None, "Original task ZIP failed CRC verification")
        members = [item for item in archive.infolist() if not item.is_dir()]
        for item in members:
            member = Path(item.filename.replace("\\", "/"))
            require(not member.is_absolute() and ".." not in member.parts,
                    f"Unsafe ZIP member: {item.filename}")
            parts = list(member.parts)
            if parts and parts[0] == "03_new_spmsm_project":
                parts = parts[1:]
            # Windows path limits make the two 64-character run identities impractical
            # inside the already-deep v3 folder.  Shorten only directory names and keep
            # an explicit reversible path table with the original archive name.
            mapped = [f"hash_{part[:16]}" if len(part) == 64 and all(c in "0123456789abcdef" for c in part)
                      else part for part in parts]
            target = (destination.joinpath(*mapped)).resolve()
            require(target == root or root in target.parents, f"ZIP member escapes destination: {item.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item, "r") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            require(target.exists() and target.stat().st_size == item.file_size,
                    f"Extracted member mismatch: {item.filename}")
            path_map.append({"archive_path": item.filename,
                             "extracted_path": str(target.relative_to(destination)).replace("\\", "/"),
                             "bytes": item.file_size, "crc32": f"{item.CRC:08x}"})
    write_csv(destination / "path_map.csv", path_map,
              ["archive_path", "extracted_path", "bytes", "crc32"])
    receipt = {
        "verified": True,
        "archive_name": LATEST_TASK_ZIP.name,
        "archive_sha256": archive_hash,
        "file_entries": len(members),
        "extracted_bytes": int(sum(item.file_size for item in members)),
        "destination": str(destination.relative_to(HERE)).replace("\\", "/"),
        "source_zip_removed_after_verification": True,
        "note": "This archive is the original FEMM task input; completed labels are copied separately.",
    }
    save_json(receipt_path, receipt)
    LATEST_TASK_ZIP.unlink()
    return receipt


def copy_frozen_sources() -> dict:
    """Keep the v3 experiment self-contained without copying per-angle run-record ZIPs."""
    files = {
        OLD_FEATURE_ROOT / "cache/features_reference.json": DATA_DIR / "features/f1_reference_40000_512d.json",
        OLD_FEATURE_ROOT / "feature_scaler.json": DATA_DIR / "features/f1_feature_scaler.json",
        OLD_FEATURE_ROOT / "reference_pool.csv": DATA_DIR / "features/reference_pool_40000.csv",
        PRIOR_3299_ROOT / "tables/dataset_all.csv": SOURCE_DIR / "prior3299/dataset_all.csv",
        PRIOR_3299_ROOT / "tables/labels.csv": SOURCE_DIR / "prior3299/labels.csv",
        PRIOR_3299_ROOT / "tables/waveforms.csv": SOURCE_DIR / "prior3299/waveforms.csv",
        PRIOR_3299_ROOT / "manifests/memberships.csv": SOURCE_DIR / "prior3299/memberships.csv",
        LATEST_10000_ROOT / "train_dev/dataset_labeled.csv": SOURCE_DIR / "latest10000/train_dev_dataset_labeled.csv",
        LATEST_10000_ROOT / "train_dev/labels.csv": SOURCE_DIR / "latest10000/train_dev_labels.csv",
        LATEST_10000_ROOT / "train_dev/waveforms.csv": SOURCE_DIR / "latest10000/train_dev_waveforms.csv",
        LATEST_10000_ROOT / "sealed_test/dataset_labeled.csv": SOURCE_DIR / "latest10000/sealed_test_dataset_labeled.csv",
        LATEST_10000_ROOT / "sealed_test/labels.csv": SOURCE_DIR / "latest10000/sealed_test_labels.csv",
        LATEST_10000_ROOT / "sealed_test/waveforms.csv": SOURCE_DIR / "latest10000/sealed_test_waveforms.csv",
        LATEST_10000_ROOT / "package_summary.json": SOURCE_DIR / "latest10000/package_summary.json",
        LATEST_10000_ROOT / "checksums.json": SOURCE_DIR / "latest10000/checksums.json",
    }
    records = []
    for local in (OLD_SELECTED, OLD_LCMD_DIR / "audit.json", OLD_LCMD_DIR / "anchors_trace.json",
                  OLD_LCMD_DIR / "selection_trace.json"):
        require(local.exists(), f"Missing frozen historical LCMD artifact: {local}")
        records.append({"path": str(local.relative_to(HERE)).replace("\\", "/"),
                        "bytes": local.stat().st_size, "sha256": sha256(local)})
    for source, destination in files.items():
        if not destination.exists() or sha256(destination) != sha256(source):
            copy_file(source, destination)
        records.append({
            "path": str(destination.relative_to(HERE)).replace("\\", "/"),
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        })
    feature_source = OLD_FEATURE_ROOT / "cache/features_reference.npy"
    feature_destination = DATA_DIR / "features/f1_reference_old40000_512d.npy"
    features = np.load(feature_source, mmap_mode="r")
    require(features.ndim == 2 and features.shape[0] >= 40000 and features.shape[1] == 512,
            f"Unexpected reference feature cache: {features.shape}")
    if not feature_destination.exists() or np.load(feature_destination, mmap_mode="r").shape != (40000, 512):
        feature_destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = feature_destination.with_suffix(".tmp.npy")
        np.save(temporary, np.asarray(features[:40000], dtype=np.float32))
        temporary.replace(feature_destination)
    obsolete_full_cache = DATA_DIR / "features/f1_reference_40000_512d.npy"
    if obsolete_full_cache.exists():
        obsolete_full_cache.unlink()
    records.append({"path": str(feature_destination.relative_to(HERE)).replace("\\", "/"),
                    "bytes": feature_destination.stat().st_size,
                    "sha256": sha256(feature_destination)})
    result = {"status": "copied_and_verified", "files": records,
              "excluded_large_archives": ["train_dev/run_records.zip", "sealed_test/run_records.zip"]}
    save_json(DATA_DIR / "source_files.json", result)
    return result


def normalized_occurrence(row: dict[str, str], batch: str, order: int) -> dict:
    bits = row["bits"].strip()
    require(len(bits) == 120 and set(bits) <= {"0", "1"}, f"Invalid 120-bit gene in {batch}")
    gene_id = row["gene_id"].strip()
    require(len(gene_id) == 64, f"Invalid gene_id in {batch}")
    tavg = float(row["tavg_nm"])
    delta = float(row["delta_t_nm"])
    require(math.isfinite(tavg) and math.isfinite(delta) and delta >= 0, f"Invalid label in {batch}")
    return {
        "gene_id": gene_id,
        "bits": bits,
        "batch": batch,
        "batch_order": order,
        "generation_source": row.get("source", ""),
        "original_role": row.get("split_role", "old_train" if batch == "historical40000_lcmd" else ""),
        "family_id": row.get("family_id", ""),
        "selection_method": row.get("selection_method", ""),
        "selection_rank": row.get("selection_rank", ""),
        "selection_distance_squared": row.get("selection_distance_squared", ""),
        "condition_fingerprint": row.get("condition_fingerprint", ""),
        "tavg_nm": tavg,
        "delta_t_nm": delta,
        "magnet_cells": int(row.get("magnet_cells") or bits.count("1")),
        "raw_gene_id": row.get("raw_gene_id", ""),
        "repair_changed_cells": row.get("repair_changed_cells", ""),
    }


def describe(values: np.ndarray) -> dict:
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def plot_candidate_distributions(groups: dict[str, list[dict]], final_rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"历史40000→20000": "#377eb8", "上一轮3299": "#ff7f00", "最新10000": "#4daf4a"}
    display = {
        "historical40000_lcmd": "历史40000→20000",
        "prior3299": "上一轮3299",
        "latest10000": "最新10000",
    }
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for key, rows in groups.items():
        name = display[key]
        tavg = np.asarray([row["tavg_nm"] for row in rows])
        delta = np.asarray([row["delta_t_nm"] for row in rows])
        magnets = np.asarray([row["magnet_cells"] for row in rows])
        axes[0, 0].hist(tavg, bins=45, density=True, histtype="step", linewidth=1.6,
                        label=f"{name} ({len(rows)})", color=colors[name])
        axes[0, 1].hist(delta, bins=45, density=True, histtype="step", linewidth=1.6,
                        label=name, color=colors[name])
        axes[1, 0].hist(magnets, bins=np.arange(-.5, 121.5, 4), density=True,
                        histtype="step", linewidth=1.6, label=name, color=colors[name])
        axes[1, 1].scatter(tavg, delta, s=4, alpha=.22, label=name, color=colors[name], rasterized=True)
    axes[0, 0].set(title="平均转矩分布", xlabel="Tavg (N·m)", ylabel="密度")
    axes[0, 1].set(title="转矩波动分布", xlabel="DeltaT (N·m)", ylabel="密度")
    axes[1, 0].set(title="永磁体格数分布", xlabel="120位基因中的永磁体格数", ylabel="密度")
    axes[1, 1].set(title="两项标签联合分布", xlabel="Tavg (N·m)", ylabel="DeltaT (N·m)")
    for axis in axes.ravel():
        axis.grid(alpha=.18)
        axis.legend(fontsize=8)
    figure.suptitle(f"统一候选池：{len(final_rows)} 个唯一基因（未划分 Train/Val/Test）")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(REPORT_DIR / "02_候选池分布.png", dpi=180)
    plt.close(figure)


def plot_real_gene_mapping(final_rows: list[dict]) -> None:
    """Show one actual candidate before and after the verified physical mapping."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    row = final_rows[0]
    values = np.fromiter((int(bit) for bit in row["bits"]), dtype=np.uint8, count=120)
    logical = values.reshape(20, 6).T
    geometry = json.loads((GAP_DIR / "geometry.json").read_text(encoding="utf-8"))
    gap90 = encode_gap90(logical, geometry)
    colors = ListedColormap(["#f5f5f5", "#d95f02"])
    figure, axes = plt.subplots(2, 1, figsize=(16, 5.8), constrained_layout=True,
                                gridspec_kw={"height_ratios": [1, 1.25]})
    axes[0].imshow(logical, aspect="auto", interpolation="nearest", cmap=colors, vmin=0, vmax=1)
    axes[0].set_title("同一个真实基因：原始逻辑表示 6×20")
    axes[0].set_xticks(np.arange(20))
    axes[0].set_ylabel("径向行")
    axes[0].set_xlabel("原始角向列 j（0–19）")
    axes[1].imshow(gap90, aspect="auto", interpolation="nearest", cmap=colors, vmin=0, vmax=1)
    axes[1].set_title("映射后：真实极间距 90° 表示 6×97")
    axes[1].set_xticks([0, 3, 4, 23, 24, 43, 44, 52, 53, 72, 73, 92, 93, 96])
    axes[1].set_ylabel("径向行")
    axes[1].set_xlabel("90°角向列（0–96）")
    for start, end in geometry["fixed_zero_ranges_0based_inclusive"]:
        axes[1].axvspan(start - .5, end + .5, facecolor="#808080", alpha=.33,
                        edgecolor="#444444", hatch="////", linewidth=.8)
    for boundary in (3.5, 23.5, 43.5, 52.5, 72.5, 92.5):
        axes[1].axvline(boundary, color="#333333", linewidth=.7, alpha=.75)
    for axis in axes:
        axis.set_yticks(np.arange(6))
        axis.grid(False)
    figure.suptitle("橙色=永磁体(1)，浅色=基因空气(0)，灰色斜线=新增的固定极间空气")
    figure.savefig(REPORT_DIR / "03_真实基因_6x20与6x97核对.png", dpi=190)
    plt.close(figure)


def build_candidate_pool() -> dict:
    """Merge the three requested sources by gene identity and quarantine label conflicts."""
    sources = {
        "historical40000_lcmd": read_rows(OLD_SELECTED),
        "prior3299": read_rows(PRIOR_3299_ROOT / "tables/dataset_all.csv"),
        "latest10000": read_rows(LATEST_10000_ROOT / "train_dev/dataset_labeled.csv")
        + read_rows(LATEST_10000_ROOT / "sealed_test/dataset_labeled.csv"),
    }
    expected = {"historical40000_lcmd": 20000, "prior3299": 3299, "latest10000": 10000}
    require({key: len(value) for key, value in sources.items()} == expected, "Source count changed")
    normalized = {
        key: [normalized_occurrence(row, key, index) for index, row in enumerate(rows)]
        for key, rows in sources.items()
    }
    for key, rows in normalized.items():
        require(len({row["gene_id"] for row in rows}) == len(rows), f"Duplicate gene_id inside {key}")
        require(len({row["bits"] for row in rows}) == len(rows), f"Duplicate bits inside {key}")

    by_gene: dict[str, list[dict]] = defaultdict(list)
    for rows in normalized.values():
        for row in rows:
            by_gene[row["gene_id"]].append(row)
    conflicts, final_rows, provenance = [], [], []
    batch_priority = {"historical40000_lcmd": 0, "prior3299": 1, "latest10000": 2}
    ordered_occurrences = sorted(
        (row for rows in normalized.values() for row in rows),
        key=lambda row: (batch_priority[row["batch"]], row["batch_order"]),
    )
    first_order = {row["gene_id"]: index for index, row in enumerate(ordered_occurrences)}
    for gene_id in sorted(by_gene, key=lambda item: first_order[item]):
        occurrences = sorted(by_gene[gene_id], key=lambda row: batch_priority[row["batch"]])
        require(len({row["bits"] for row in occurrences}) == 1,
                f"Same gene_id maps to different bits: {gene_id}")
        tavg = np.asarray([row["tavg_nm"] for row in occurrences], dtype=np.float64)
        delta = np.asarray([row["delta_t_nm"] for row in occurrences], dtype=np.float64)
        if np.ptp(tavg) > LABEL_TOLERANCE or np.ptp(delta) > LABEL_TOLERANCE:
            conflicts.extend(occurrences)
            continue
        primary = occurrences[0]
        memberships = [row["batch"] for row in occurrences]
        final_rows.append({
            "candidate_index": len(final_rows),
            "gene_id": gene_id,
            "bits": primary["bits"],
            "source": ";".join(memberships),
            "source_primary": primary["batch"],
            "source_count": len(occurrences),
            "generation_source": primary["generation_source"],
            "original_role": primary["original_role"],
            "family_id": primary["family_id"],
            "magnet_cells": primary["magnet_cells"],
            "tavg_nm": float(np.median(tavg)),
            "delta_t_nm": float(np.median(delta)),
            "selection_method": primary["selection_method"],
            "selection_rank": primary["selection_rank"],
            "selection_distance_squared": primary["selection_distance_squared"],
            "condition_fingerprint": primary["condition_fingerprint"],
        })
        for row in occurrences:
            provenance.append({
                "candidate_index": len(final_rows) - 1,
                "gene_id": gene_id,
                "source": row["batch"],
                "source_row": row["batch_order"],
                "generation_source": row["generation_source"],
                "original_role": row["original_role"],
                "family_id": row["family_id"],
                "tavg_nm": row["tavg_nm"],
                "delta_t_nm": row["delta_t_nm"],
            })

    POOL_DIR.mkdir(parents=True, exist_ok=True)
    manifest_fields = list(final_rows[0])
    write_csv(POOL_DIR / "manifest.csv", final_rows, manifest_fields)
    write_csv(POOL_DIR / "provenance.csv", provenance, list(provenance[0]))
    conflict_fields = ["gene_id", "bits", "batch", "batch_order", "tavg_nm", "delta_t_nm"]
    write_csv(POOL_DIR / "label_conflicts.csv",
              [{key: row[key] for key in conflict_fields} for row in conflicts], conflict_fields)

    bits = np.fromiter((int(bit) for row in final_rows for bit in row["bits"]),
                       dtype=np.uint8, count=len(final_rows) * 120).reshape(len(final_rows), 120)
    logical = bits.reshape(len(final_rows), 20, 6).swapaxes(1, 2)
    targets = np.asarray([[row["tavg_nm"], row["delta_t_nm"]] for row in final_rows], dtype=np.float32)
    geometry = json.loads((GAP_DIR / "geometry.json").read_text(encoding="utf-8"))
    gap90 = encode_gap90(logical, geometry)
    np.save(POOL_DIR / "topology_bits_120.npy", bits)
    np.save(POOL_DIR / "topology_6x20.npy", logical)
    np.save(POOL_DIR / "topology_gap90_6x97.npy", gap90)
    np.save(POOL_DIR / "targets_tavg_delta_nm.npy", targets)
    require(np.all(gap90[:, :, geometry["fixed_zero_columns_0based"]] == 0),
            "Gap-aware fixed columns are not zero")

    keys = list(normalized)
    sets = {key: {row["gene_id"] for row in rows} for key, rows in normalized.items()}
    intersections = {}
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1:]:
            intersections[f"{left}__{right}"] = len(sets[left] & sets[right])
    intersections["all_three"] = len(set.intersection(*(sets[key] for key in keys)))
    label_stats = {}
    for key, rows in normalized.items():
        label_stats[key] = {
            "tavg_nm": describe(np.asarray([row["tavg_nm"] for row in rows])),
            "delta_t_nm": describe(np.asarray([row["delta_t_nm"] for row in rows])),
            "magnet_cells": describe(np.asarray([row["magnet_cells"] for row in rows])),
            "generation_source_counts": dict(sorted(Counter(row["generation_source"] for row in rows).items())),
            "original_role_counts": dict(sorted(Counter(row["original_role"] for row in rows).items())),
        }
    audit = {
        "status": "passed" if not conflicts else "passed_with_quarantined_conflicts",
        "source_counts": expected,
        "pairwise_and_triple_intersections": intersections,
        "duplicate_occurrences_removed": sum(expected.values()) - len(by_gene),
        "label_conflict_genes_quarantined": len({row["gene_id"] for row in conflicts}),
        "candidate_pool_unique_genes": len(final_rows),
        "expected_theoretical_total": 33299,
        "train_validation_test_split": "stored separately in data/split_audit.json and data/split_membership.csv",
        "label_tolerance_nm": LABEL_TOLERANCE,
        "array_shapes": {
            "topology_bits_120": list(bits.shape),
            "topology_6x20": list(logical.shape),
            "topology_gap90_6x97": list(gap90.shape),
            "targets_tavg_delta_nm": list(targets.shape),
        },
        "label_and_gene_statistics": label_stats,
    }
    save_json(POOL_DIR / "audit.json", audit)
    plot_candidate_distributions(normalized, final_rows)
    plot_real_gene_mapping(final_rows)
    return audit


def allocate_counts(total: int, ratios: tuple[float, float, float]) -> list[int]:
    raw = np.asarray(ratios, dtype=np.float64) * total
    counts = np.floor(raw).astype(int)
    remainder = total - int(counts.sum())
    fractions = raw - counts
    order = sorted(range(len(ratios)), key=lambda index: (-fractions[index], index))
    for index in order[:remainder]:
        counts[index] += 1
    return counts.tolist()


def allocate_strata(sizes: dict[str, int], target: int) -> dict[str, int]:
    total = sum(sizes.values())
    raw = {key: sizes[key] * target / total for key in sizes}
    counts = {key: math.floor(value) for key, value in raw.items()}
    remainder = target - sum(counts.values())
    order = sorted(sizes, key=lambda key: (-(raw[key] - counts[key]), key))
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def freeze_split(seed: int = 20260917) -> dict:
    """Split historical and expanded pools independently at 70/15/15, then merge."""
    manifest = read_rows(POOL_DIR / "manifest.csv")
    require(len(manifest) == 33299, "Candidate pool size changed before split")
    for row in manifest:
        row["super_pool"] = ("historical20000" if row["source_primary"] == "historical40000_lcmd"
                             else "expanded13299")
        row["stratum"] = f"{row['source_primary']}|{row['generation_source']}"

    memberships: dict[str, str] = {}
    targets_by_pool = {}
    for super_pool in ("historical20000", "expanded13299"):
        pool_rows = [row for row in manifest if row["super_pool"] == super_pool]
        train_target, validation_target, test_target = allocate_counts(len(pool_rows), (.70, .15, .15))
        targets_by_pool[super_pool] = {
            "total": len(pool_rows), "train": train_target,
            "validation": validation_target, "test": test_target,
        }
        strata: dict[str, list[dict]] = defaultdict(list)
        for row in pool_rows:
            strata[row["stratum"]].append(row)
        train_by_stratum = allocate_strata({key: len(rows) for key, rows in strata.items()}, train_target)
        remaining_sizes = {key: len(rows) - train_by_stratum[key] for key, rows in strata.items()}
        validation_by_stratum = allocate_strata(remaining_sizes, validation_target)
        for key, rows in strata.items():
            ordered = sorted(rows, key=lambda row: hashlib.sha256(
                f"v3-split|{seed}|{super_pool}|{key}|{row['gene_id']}".encode("utf-8")
            ).hexdigest())
            train_end = train_by_stratum[key]
            validation_end = train_end + validation_by_stratum[key]
            for row in ordered[:train_end]:
                memberships[row["gene_id"]] = "train"
            for row in ordered[train_end:validation_end]:
                memberships[row["gene_id"]] = "validation"
            for row in ordered[validation_end:]:
                memberships[row["gene_id"]] = "test"

    require(len(memberships) == len(manifest), "Split did not assign every candidate")
    split_rows = []
    for row in manifest:
        split_rows.append({
            "candidate_index": int(row["candidate_index"]),
            "gene_id": row["gene_id"],
            "split": memberships[row["gene_id"]],
            "super_pool": row["super_pool"],
            "source_primary": row["source_primary"],
            "generation_source": row["generation_source"],
            "original_role": row["original_role"],
            "stratum": row["stratum"],
        })
    split_rows.sort(key=lambda row: row["candidate_index"])
    write_csv(DATA_DIR / "split_membership.csv", split_rows, list(split_rows[0]))
    for split in ("train", "validation", "test"):
        indices = np.asarray([row["candidate_index"] for row in split_rows if row["split"] == split],
                             dtype=np.int64)
        np.save(DATA_DIR / f"{split}_indices.npy", indices)

    total_counts = Counter(row["split"] for row in split_rows)
    pool_counts = {
        pool: dict(Counter(row["split"] for row in split_rows if row["super_pool"] == pool))
        for pool in ("historical20000", "expanded13299")
    }
    source_counts = {
        source: dict(Counter(row["split"] for row in split_rows if row["source_primary"] == source))
        for source in sorted({row["source_primary"] for row in split_rows})
    }
    require(dict(total_counts) == {"train": 23309, "validation": 4995, "test": 4995},
            f"Unexpected final split counts: {dict(total_counts)}")
    require(pool_counts["historical20000"] == {"train": 14000, "validation": 3000, "test": 3000},
            "Historical split count mismatch")
    require(pool_counts["expanded13299"] == {"train": 9309, "validation": 1995, "test": 1995},
            "Expanded split count mismatch")
    audit = {
        "status": "frozen",
        "seed": seed,
        "method": "split historical20000 and expanded13299 independently; stratify by source batch and U/L/B/P generation source; stable SHA256 ordering; then merge",
        "requested_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "total_counts": dict(total_counts),
        "super_pool_counts": pool_counts,
        "source_counts": source_counts,
        "membership_sha256": sha256(DATA_DIR / "split_membership.csv"),
        "test_policy": "test identities are frozen; training and model selection must not read test labels",
    }
    save_json(DATA_DIR / "split_audit.json", audit)
    return audit


def write_report(geometry: dict, pool: dict, split: dict, zip_receipt: dict, feature_identity: dict) -> None:
    stats = pool["label_and_gene_statistics"]
    source_names = {
        "historical40000_lcmd": "历史40000经F＋LCMD选出的20000",
        "prior3299": "上一轮3299",
        "latest10000": "最新10000",
    }
    lines = [
        "# CNN 综合对比 v3：输入表示与统一候选池", "",
        "## 当前结论", "",
        f"- 真实极间距输入由几何推导为 **6×{geometry['output_shape'][1]}**；120个独立设计变量保持不变。",
        f"- 三批数据全局去重后为 **{pool['candidate_pool_unique_genes']} 个唯一基因**。",
        f"- 标签冲突基因：**{pool['label_conflict_genes_quarantined']}**。",
        f"- 已按用户确认的70%/15%/15%冻结划分：训练 **{split['total_counts']['train']}**、验证 **{split['total_counts']['validation']}**、测试 **{split['total_counts']['test']}**。",
        "- 本阶段未训练任何CNN，也未运行FEMM。", "",
        "## 1. 6×97 gap-aware 90° 推导", "",
        f"实际设计步长 θ = `{geometry['theta_design_step_deg']:.12f}°`，单极设计带宽 = `{geometry['design_band_per_pole_deg']:.12f}°`，极距 = `{geometry['pole_pitch_deg']:.1f}°`。",
        f"因此单极物理间隙为 `{geometry['gap_per_pole_deg']:.12f}°`，等价 `{geometry['gap_per_pole_equivalent_columns']:.6f}` 个 θ 单元；90° 等价 `{geometry['sector_equivalent_columns']:.6f}` 个单元，向上取整得到 `N={geometry['output_shape'][1]}`。", "",
        "四个20列设计带按真实中心角投影到97列：", "",
        "- 第1带：原始列 j → `4+j`；",
        "- 第2带：原始列 j → `43-j`；",
        "- 第3带：原始列 j → `53+j`；",
        "- 第4带：原始列 j → `92-j`。", "",
        "固定空气补零列（0-based）为 `0–3`、`44–52`、`93–96`，共17列；四个设计带仍复制同一组120位变量，没有新增可优化变量。", "",
        "![6×20到6×97映射](01_gap90映射样例.png)", "",
        "## 2. 候选池来源与去重", "",
        "| 来源 | 输入数量 | 与其他两批交集 | Tavg均值 (N·m) | DeltaT均值 (N·m) |",
        "|---|---:|---:|---:|---:|",
    ]
    intersections = pool["pairwise_and_triple_intersections"]
    for key in ("historical40000_lcmd", "prior3299", "latest10000"):
        other_overlap = sum(value for name, value in intersections.items()
                            if name != "all_three" and key in name)
        lines.append(f"| {source_names[key]} | {pool['source_counts'][key]} | {other_overlap} | "
                     f"{stats[key]['tavg_nm']['mean']:.6f} | {stats[key]['delta_t_nm']['mean']:.6f} |")
    lines += [
        "",
        f"三批两两交集和三批共同交集均为0，所以理论总数 `20000+3299+10000` 未减少，实际仍为 **{pool['candidate_pool_unique_genes']}**。",
        "相同基因若出现多份标签，本流程按1e-6 N·m容差核对；超出容差会进入 `label_conflicts.csv` 并从候选池隔离。本次没有发生冲突。", "",
        "![候选池分布](02_候选池分布.png)", "",
        "![真实基因映射核对](03_真实基因_6x20与6x97核对.png)", "",
        "U/L/B/P只表示生成来源，不解释为互斥的物理拓扑类别。最新一万原有角色仅作为来源记录保留。", "",
        "## 3. 冻结数据划分", "",
        "历史20000与新增13299先各自按70%/15%/15%确定身份，再合并为所有架构共用的一套划分。新增部分还按批次与U/L/B/P来源分层；固定种子和基因哈希排序保证可复现。", "",
        "| 数据块 | 训练 | 验证 | 测试 |",
        "|---|---:|---:|---:|",
        f"| 历史20000 | {split['super_pool_counts']['historical20000']['train']} | {split['super_pool_counts']['historical20000']['validation']} | {split['super_pool_counts']['historical20000']['test']} |",
        f"| 新增13299 | {split['super_pool_counts']['expanded13299']['train']} | {split['super_pool_counts']['expanded13299']['validation']} | {split['super_pool_counts']['expanded13299']['test']} |",
        f"| 合计 | {split['total_counts']['train']} | {split['total_counts']['validation']} | {split['total_counts']['test']} |", "",
        "测试身份已经冻结；训练、目标归一化和检查点选择不读取测试标签。每个配置完成后只评价一次测试集。", "",
        "## 4. F＋LCMD身份", "",
        f"历史四万的筛选使用冻结的 f1 Polar90 VGG16、F-S、12.5%新数据、第4000步无约束最佳候选；SHA256为 `{feature_identity['sha256']}`。",
        "F为两个回归分支最终线性层之前的256维特征拼接成512维，并使用冻结参照池的分支RMS标准化。LCMD采用簇累计最近距离平方选簇，再选该簇最远成员；不是按单一距离直接取前20000。", "",
        "该f1检查点没有通过当时的旧分布5%接受约束，只作为冻结特征空间使用，没有被登记为已验收预测模型。", "",
        "## 5. 数据与压缩包处理", "",
        f"原始任务包已完成CRC、逐文件大小和路径安全核验，解出 `{zip_receipt['file_entries']}` 个文件后删除两份哈希相同的本地ZIP副本；已记录原ZIP哈希。完成的10000条FEMM标签另行复制到本实验数据目录。",
        "两份逐角度 `run_records.zip` 没有复制进v3；其标签、六角度波形、合同和校验摘要均已保留，原结果目录仍保留这两份归档。", "",
        "## 6. 后续八配置", "",
        "已登记且尚未训练：6×20的SmallCNN/ResNet20、6×97 gap-aware的SmallCNN/ResNet20、Polar90的VGG16/ResNet18、Polar360的VGG16/ResNet18。后续统一从头训练并共享同一套已冻结的数据身份划分。", "",
        "启动入口是 `train.py`。无参数运行只执行八配置前向/反向预检；显式使用 `--train --models all` 才会顺序启动正式训练，中断后用 `--resume` 恢复。", "",
        "## 范围说明", "",
        "这是一轮输入表示推导和统一候选池准备。它不能证明6×97一定优于6×20，也不能在正式训练前宣称预测精度提高。", "",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def create_checksums() -> None:
    records = []
    for root in (DATA_DIR, MODEL_DIR, REPORT_DIR):
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != "checksums.json":
                records.append({"path": str(path.relative_to(HERE)).replace("\\", "/"),
                                "bytes": path.stat().st_size, "sha256": sha256(path)})
    save_json(HERE / "checksums.json", {"files": records})


def prepare_all() -> dict:
    geometry = derive_gap90()
    gap90_examples()
    feature_identity = prepare_model_registry()
    source_files = copy_frozen_sources()
    zip_receipt = extract_original_task_zip()
    pool = build_candidate_pool()
    split = freeze_split()
    write_report(geometry, pool, split, zip_receipt, feature_identity)
    create_checksums()
    result = {
        "status": "complete",
        "gap90_N": geometry["output_shape"][1],
        "candidate_pool_unique_genes": pool["candidate_pool_unique_genes"],
        "label_conflicts": pool["label_conflict_genes_quarantined"],
        "source_files": len(source_files["files"]),
        "split_counts": split["total_counts"],
        "training_started": False,
        "split_decided": True,
    }
    save_json(HERE / "status.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("gap90", "pool", "split", "all", "status"),
                        nargs="?", default="status")
    args = parser.parse_args()
    if args.command == "gap90":
        result = derive_gap90()
        gap90_examples()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "pool":
        print(json.dumps(build_candidate_pool(), ensure_ascii=False, indent=2))
    elif args.command == "split":
        print(json.dumps(freeze_split(), ensure_ascii=False, indent=2))
    elif args.command == "all":
        print(json.dumps(prepare_all(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps({
            "gap90_ready": (GAP_DIR / "geometry.json").exists(),
            "candidate_pool_ready": (HERE / "data/candidate_pool/manifest.csv").exists(),
            "split_ready": (DATA_DIR / "split_membership.csv").exists(),
            "cnn_training_started": False,
            "split_decided": (DATA_DIR / "split_membership.csv").exists(),
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
