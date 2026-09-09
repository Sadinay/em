from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import yaml
from scipy.io import loadmat

from fem_raster import CHANNEL_NAMES, rasterize_fem


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def result_from_mat(path: Path) -> tuple[float, float, dict[str, Any]]:
    data = loadmat(path, simplify_cells=True)
    required = ("bestOverall", "ctx")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError("Missing: " + ", ".join(missing))
    ctx = data["ctx"]
    return float(ctx["T_min"]), float(data["bestOverall"]), {
        "penalty_coef": float(ctx["penaltyCoef"]),
        "base_fem_file": str(ctx["baseFemFile"]),
    }


def assign_split(rows: list[dict[str, Any]], validation_samples: int) -> None:
    # Rank-interleaved split keeps the tiny train/validation target ranges representative.
    ranked = sorted(range(len(rows)), key=lambda index: (rows[index]["target_j"], rows[index]["sample_id"]))
    validation_indices = set(ranked[1::2][:validation_samples])
    if len(validation_indices) != validation_samples:
        raise ValueError("Not enough independent samples for requested validation size")
    for index, row in enumerate(rows):
        row["split"] = "validation" if index in validation_indices else "train"


def save_preview(grid: np.ndarray, destination: Path, title: str) -> None:
    material_index = np.argmax(grid, axis=0)
    cmap = ListedColormap(["#ffffff", "#dbeafe", "#4b5563", "#d97706", "#dc2626"])
    fig, ax = plt.subplots(figsize=(4.2, 6.2))
    ax.imshow(material_index, cmap=cmap, vmin=0, vmax=4, origin="lower", interpolation="nearest", aspect="equal")
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(destination, dpi=120)
    plt.close(fig)


def build(raw_root: Path, audit_output: Path, output: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    pairs = read_csv(audit_output / "fem_mat_pairs.csv")
    accepted = {"confirmed", "probable"}
    candidates: list[dict[str, Any]] = []
    images: list[np.ndarray] = []
    errors: list[dict[str, str]] = []
    for pair in pairs:
        if pair["match_status"] not in accepted or not pair["mat_path"]:
            continue
        try:
            image, fem_metadata = rasterize_fem(
                raw_root / Path(pair["fem_path"]),
                height=int(config["image"]["height"]),
                width=int(config["image"]["width"]),
                bbox=tuple(float(value) for value in config["image"]["bbox"]),
            )
            if int(fem_metadata["ambiguous_face_count"]) > int(config["raster_quality"]["max_ambiguous_faces"]):
                raise ValueError(
                    "FEM raster rejected: ambiguous faces "
                    f"{fem_metadata['ambiguous_face_count']} > "
                    f"{config['raster_quality']['max_ambiguous_faces']}"
                )
            t_min, target_j, metadata = result_from_mat(raw_root / Path(pair["mat_path"]))
        except Exception as exc:
            errors.append({"sample_id": pair["sample_id"], "mat_path": pair["mat_path"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        topology_id = "FEMRASTER_" + __import__("hashlib").sha256(image.tobytes()).hexdigest()[:12]
        candidates.append({
            "sample_id": pair["sample_id"], "fem_path": pair["fem_path"], "mat_path": pair["mat_path"],
            "match_status": pair["match_status"], "match_score": float(pair["match_score"]),
            "t_min": t_min, "target_j": target_j, "topology_id": topology_id,
            **metadata, **fem_metadata,
        })
        images.append(image)

    grouped: dict[tuple[str, float], list[int]] = defaultdict(list)
    for index, row in enumerate(candidates):
        grouped[(row["topology_id"], row["t_min"])].append(index)
    rows: list[dict[str, Any]] = []
    unique_images: list[np.ndarray] = []
    for (topology_id, t_min), indices in sorted(grouped.items()):
        targets = [candidates[index]["target_j"] for index in indices]
        if max(targets) - min(targets) > 1e-9:
            raise ValueError(f"Same topology/T_min has inconsistent J: {topology_id}, {targets}")
        representative = min(indices, key=lambda index: candidates[index]["sample_id"])
        row = dict(candidates[representative])
        row["duplicate_run_count"] = len(indices)
        row["duplicate_sample_ids"] = "|".join(sorted(candidates[index]["sample_id"] for index in indices))
        rows.append(row)
        unique_images.append(images[representative])
    if len(rows) < 20:
        raise RuntimeError(f"Unexpectedly few independent FEM raster/T_min samples: {len(rows)}")
    assign_split(rows, int(config["validation_samples"]))
    output.mkdir(parents=True, exist_ok=True)
    image_array = np.stack(unique_images).astype(np.float32)
    t_min_array = np.asarray([row["t_min"] for row in rows], dtype=np.float32).reshape(-1, 1)
    target_array = np.asarray([row["target_j"] for row in rows], dtype=np.float32).reshape(-1, 1)
    np.savez_compressed(
        output / "structure_dataset.npz",
        images=image_array, t_min=t_min_array, target_j=target_array,
        sample_ids=np.asarray([row["sample_id"] for row in rows]),
        splits=np.asarray([row["split"] for row in rows]),
        topology_ids=np.asarray([row["topology_id"] for row in rows]),
    )
    fields = [
        "sample_id", "topology_id", "split", "fem_path", "mat_path", "match_status", "match_score",
        "t_min", "target_j", "penalty_coef", "base_fem_file",
        "encoding", "node_count", "face_count", "assigned_face_count", "ambiguous_face_count", "bbox", "channel_names",
        "duplicate_run_count", "duplicate_sample_ids",
    ]
    write_csv(output / "structure_manifest.csv", rows, fields)
    write_csv(output / "dataset_build_errors.csv", errors, ["sample_id", "mat_path", "error"])
    preview_dir = output / "structure_previews"
    preview_dir.mkdir(exist_ok=True)
    for row, image in zip(rows, unique_images):
        save_preview(image, preview_dir / f"{row['sample_id']}.png", f"{row['sample_id']} | J={row['target_j']:.6f}")
    train_ids = {row["topology_id"] for row in rows if row["split"] == "train"}
    validation_ids = {row["topology_id"] for row in rows if row["split"] == "validation"}
    split_info = {
        "source_run_records": len(candidates),
        "independent_topology_tmin_samples": len(rows),
        "train_samples": sum(row["split"] == "train" for row in rows),
        "validation_samples": sum(row["split"] == "validation" for row in rows),
        "train_target_range": [
            min(row["target_j"] for row in rows if row["split"] == "train"),
            max(row["target_j"] for row in rows if row["split"] == "train"),
        ],
        "validation_target_range": [
            min(row["target_j"] for row in rows if row["split"] == "validation"),
            max(row["target_j"] for row in rows if row["split"] == "validation"),
        ],
        "topology_overlap": sorted(train_ids & validation_ids),
        "leakage_check": not bool(train_ids & validation_ids),
        "input_shape": list(image_array.shape),
        "channel_order": CHANNEL_NAMES,
        "split_strategy": "target-rank alternating, independent topology/T_min only",
    }
    (output / "split_info.json").write_text(json.dumps(split_info, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deduplicated material-topology image dataset.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = build(args.raw_root.resolve(), args.audit_output.resolve(), args.output.resolve(), config)
    print(f"Built {len(rows)} independent topology samples: train={sum(r['split']=='train' for r in rows)}, validation={sum(r['split']=='validation' for r in rows)}")


if __name__ == "__main__":
    main()
