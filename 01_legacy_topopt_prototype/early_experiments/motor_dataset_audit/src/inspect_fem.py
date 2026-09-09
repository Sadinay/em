from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .common import compact_json, safe_rel, write_csv


SUMMARY_FIELDS = [
    "relative_path", "absolute_path", "encoding", "read_status", "format", "frequency",
    "precision", "min_angle", "depth", "length_unit", "problem_type", "coordinates",
    "point_count", "boundary_count", "material_count", "circuit_count", "node_count",
    "segment_count", "arc_count", "block_label_count", "material_names",
    "assigned_material_names", "circuit_names", "current_settings", "magnetization_directions",
    "geometry_bbox", "model_center", "geometry_signature", "error",
]


def decode_text(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "cp1252", "latin-1"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace"), "utf-8-replace"


def inspect_fem_files(root: Path, output: Path, inventory: list[dict[str, Any]], config: dict[str, Any], logger) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    material_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    fem_rows = [r for r in inventory if str(r["extension"]).casefold() == ".fem"]
    parsed_by_path: dict[str, dict[str, Any]] = {}
    for index, file_row in enumerate(fem_rows, 1):
        path = Path(str(file_row["absolute_path"]))
        rel = str(file_row["relative_path"])
        try:
            text, encoding = decode_text(path)
            parsed = parse_fem(text)
            parsed["relative_path"] = rel
            parsed["absolute_path"] = str(path)
            parsed["encoding"] = encoding
            parsed["read_status"] = "ok"
            parsed["error"] = ""
            parsed_by_path[rel] = parsed
            summaries.append({field: parsed.get(field, "") for field in SUMMARY_FIELDS})
            assigned_counts = Counter(parsed.get("_assigned_material_names", []))
            for material in parsed.get("_material_names", []):
                material_rows.append({
                    "relative_path": rel,
                    "material_name": material,
                    "definition_count": 1,
                    "assignment_count": assigned_counts.get(material, 0),
                })
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            summaries.append({"relative_path": rel, "absolute_path": str(path), "read_status": "error", "error": error})
            errors.append({"relative_path": rel, "error": error})
        if index % 50 == 0:
            logger.info("Inspected %d/%d FEM files", index, len(fem_rows))
    write_csv(output / "fem_file_summary.csv", summaries, SUMMARY_FIELDS)
    write_csv(output / "fem_material_inventory.csv", material_rows, ["relative_path", "material_name", "definition_count", "assignment_count"])
    write_csv(output / "fem_read_errors.csv", errors, ["relative_path", "error"])
    make_previews(output / "fem_previews", parsed_by_path, config, logger)
    logger.info("FEM: %d files, %d readable, %d material records", len(summaries), sum(s.get("read_status") == "ok" for s in summaries), len(material_rows))
    return summaries


def setting(text: str, key: str) -> str:
    match = re.search(rf"(?im)^\s*\[{re.escape(key)}\]\s*=\s*(.+?)\s*$", text)
    return unquote(match.group(1).strip()) if match else ""


def unquote(value: str) -> str:
    value = value.strip()
    return value[1:-1] if len(value) >= 2 and value[0] == value[-1] == '"' else value


def names_in_blocks(text: str, begin: str, end: str, key: str) -> list[str]:
    results = []
    for body in re.findall(rf"(?is)<{begin}>(.*?)<{end}>", text):
        match = re.search(rf"(?im)^\s*<{re.escape(key)}>\s*=\s*(.+?)\s*$", body)
        if match:
            results.append(unquote(match.group(1)))
    return results


def section_rows(text: str, count_key: str, minimum_columns: int = 2) -> list[list[str]]:
    match = re.search(rf"(?im)^\s*\[{re.escape(count_key)}\]\s*=\s*(\d+)\s*$", text)
    if not match:
        return []
    count = int(match.group(1))
    rows = []
    for line in text[match.end():].splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") or stripped.startswith("<"):
            break
        parts = re.findall(r'"[^"]*"|\S+', stripped)
        if len(parts) >= minimum_columns:
            rows.append(parts)
        if len(rows) >= count:
            break
    return rows


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_fem(text: str) -> dict[str, Any]:
    materials = names_in_blocks(text, "BeginBlock", "EndBlock", "BlockName")
    boundaries = names_in_blocks(text, "BeginBdry", "EndBdry", "BdryName")
    circuits = names_in_blocks(text, "BeginCircuit", "EndCircuit", "CircuitName")
    current_settings = []
    for body in re.findall(r"(?is)<BeginCircuit>(.*?)<EndCircuit>", text):
        name_match = re.search(r'(?im)^\s*<CircuitName>\s*=\s*(.+?)\s*$', body)
        amps_match = re.search(r'(?im)^\s*<TotalAmps_re>\s*=\s*(.+?)\s*$', body)
        if name_match:
            current_settings.append({"name": unquote(name_match.group(1)), "current_re": unquote(amps_match.group(1)) if amps_match else ""})
    nodes_raw = section_rows(text, "NumPoints", 2)
    segments_raw = section_rows(text, "NumSegments", 2)
    arcs_raw = section_rows(text, "NumArcSegments", 2)
    labels_raw = section_rows(text, "NumBlockLabels", 3)
    nodes = []
    for row in nodes_raw:
        x, y = parse_float(row[0]), parse_float(row[1])
        if x is not None and y is not None:
            nodes.append((x, y))
    segments = []
    for row in segments_raw:
        try:
            segments.append((int(float(row[0])), int(float(row[1]))))
        except (ValueError, IndexError):
            continue
    arcs = []
    for row in arcs_raw:
        try:
            arcs.append((int(float(row[0])), int(float(row[1])), float(row[2]) if len(row) > 2 else 0.0))
        except (ValueError, IndexError):
            continue
    labels = []
    assigned_materials = []
    magnetization = []
    for row in labels_raw:
        x, y = parse_float(row[0]), parse_float(row[1])
        material_index = None
        try:
            material_index = int(float(row[2]))
        except (ValueError, IndexError):
            pass
        material = ""
        # FEMM stores block type as 1-based index; zero means no material.
        if material_index and 1 <= material_index <= len(materials):
            material = materials[material_index - 1]
            assigned_materials.append(material)
        mag = parse_float(row[6]) if len(row) > 6 else None
        if mag is not None:
            magnetization.append(mag)
        if x is not None and y is not None:
            labels.append((x, y, material))
    bbox = ""
    center = ""
    if nodes:
        xs, ys = zip(*nodes)
        bounds = [min(xs), min(ys), max(xs), max(ys)]
        bbox = compact_json(bounds)
        center = compact_json([(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2])
    signature_payload = compact_json({
        "nodes": [[round(x, 9), round(y, 9)] for x, y in nodes],
        "segments": segments,
        "arcs": [[a, b, round(angle, 9)] for a, b, angle in arcs],
        "labels": [[round(x, 9), round(y, 9), material] for x, y, material in labels],
        "depth": setting(text, "Depth"),
        "problem": setting(text, "ProblemType"),
    })
    geometry_signature = hashlib.sha256(signature_payload.encode("utf-8")).hexdigest()
    result = {
        "format": setting(text, "Format"), "frequency": setting(text, "Frequency"),
        "precision": setting(text, "Precision"), "min_angle": setting(text, "MinAngle"),
        "depth": setting(text, "Depth"), "length_unit": setting(text, "LengthUnits"),
        "problem_type": setting(text, "ProblemType"), "coordinates": setting(text, "Coordinates"),
        "point_count": setting(text, "PointProps"), "boundary_count": setting(text, "BdryProps"),
        "material_count": setting(text, "BlockProps"), "circuit_count": setting(text, "CircuitProps"),
        "node_count": len(nodes_raw), "segment_count": len(segments_raw), "arc_count": len(arcs_raw),
        "block_label_count": len(labels_raw), "material_names": "|".join(materials),
        "assigned_material_names": "|".join(sorted(set(assigned_materials))),
        "circuit_names": "|".join(circuits), "current_settings": compact_json(current_settings),
        "magnetization_directions": compact_json(sorted(set(magnetization))),
        "geometry_bbox": bbox, "model_center": center, "geometry_signature": geometry_signature,
        "_material_names": materials, "_assigned_material_names": assigned_materials,
        "_nodes": nodes, "_segments": segments, "_arcs": arcs, "_labels": labels,
    }
    return result


def make_previews(destination: Path, parsed: dict[str, dict[str, Any]], config: dict[str, Any], logger) -> None:
    if not parsed:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = int(config.get("fem", {}).get("preview_count", 20))
    dpi = int(config.get("fem", {}).get("preview_dpi", 110))
    records = list(parsed.items())
    # Deterministic representative selection: diverse directories and sizes, then evenly spread.
    selected: list[tuple[str, dict[str, Any]]] = []
    seen_dirs: set[str] = set()
    for item in records:
        directory = str(Path(item[0]).parent)
        if directory not in seen_dirs:
            selected.append(item)
            seen_dirs.add(directory)
        if len(selected) >= min(5, count):
            break
    for item in sorted(records, key=lambda x: len(x[1].get("_nodes", []))):
        if item not in selected:
            selected.append(item)
        if len(selected) >= min(10, count):
            break
    stride = max(1, len(records) // max(1, count))
    for item in records[::stride]:
        if item not in selected:
            selected.append(item)
        if len(selected) >= count:
            break
    destination.mkdir(parents=True, exist_ok=True)
    for rel, data in selected[:count]:
        nodes = data.get("_nodes", [])
        if not nodes:
            continue
        fig, ax = plt.subplots(figsize=(6, 6))
        for a, b in data.get("_segments", []):
            if 0 <= a < len(nodes) and 0 <= b < len(nodes):
                ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]], color="#3b4652", linewidth=0.45)
        for a, b, _angle in data.get("_arcs", []):
            if 0 <= a < len(nodes) and 0 <= b < len(nodes):
                ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]], color="#6a7480", linewidth=0.35, linestyle=":")
        material_names = sorted({m for _x, _y, m in data.get("_labels", []) if m})
        cmap = plt.get_cmap("tab20")
        for i, material in enumerate(material_names):
            pts = [(x, y) for x, y, m in data.get("_labels", []) if m == material]
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, s=14, color=cmap(i % 20), label=material, zorder=3)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(Path(rel).name, fontsize=9)
        ax.axis("off")
        if material_names:
            ax.legend(fontsize=6, loc="best", framealpha=0.75)
        name = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:10] + "_" + Path(rel).stem[:50] + ".png"
        fig.savefig(destination / name, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    logger.info("Generated %d low-resolution FEM previews", len(list(destination.glob("*.png"))))
