from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import compact_json, keyword_matches, write_csv, write_json


MANIFEST_FIELDS = [
    "sample_id", "geometry_id", "operating_point_id", "simulation_id", "fem_path",
    "mat_path", "match_status", "match_score", "problem_type", "length_unit",
    "axial_length", "material_names", "geometry_bbox", "operating_current",
    "current_angle", "speed_rpm", "rotor_position", "candidate_target_names",
    "validation_status", "notes",
]


def first_number(patterns: list[str], text: str) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def build_manifest(output: Path, inventory: list[dict[str, Any]], fem_summaries: list[dict[str, Any]], mat_variables: list[dict[str, Any]], pairs: list[dict[str, Any]], logger) -> list[dict[str, Any]]:
    fem_lookup = {str(row["relative_path"]): row for row in fem_summaries}
    inventory_lookup = {str(row["relative_path"]): row for row in inventory}
    candidate_lookup: dict[str, list[str]] = defaultdict(list)
    keywords = ("torque", "tavg", "tmean", "t_avg", "t_ripple", "cogging", "ripple", "flux", "bemf", "emf", "loss", "efficiency", "power", "objective", "fitness", "best", "j_hist", "j_ripple")
    for row in mat_variables:
        name = str(row["variable_path"])
        if keyword_matches(name, keywords):
            candidate_lookup[str(row["relative_path"])].append(name)
    geometry_ids: dict[str, str] = {}
    geometry_counter = 0
    manifest: list[dict[str, Any]] = []
    for pair in pairs:
        fem_path = str(pair["fem_path"])
        mat_path = str(pair["mat_path"])
        fem = fem_lookup.get(fem_path, {})
        signature = str(fem.get("geometry_signature", ""))
        if signature and signature not in geometry_ids:
            geometry_counter += 1
            geometry_ids[signature] = f"GEOM_{geometry_counter:04d}"
        geometry_id = geometry_ids.get(signature, "")
        searchable = " ".join((Path(fem_path).stem, Path(mat_path).stem if mat_path else ""))
        rotor = first_number([r"(?:rotor|angle|pos(?:ition)?)[_-]?(-?\d+(?:\.\d+)?)", r"(?:theta)[_-]?(-?\d+(?:\.\d+)?)"], searchable)
        current = first_number([r"(?:current|amp|i)[_-]?(-?\d+(?:\.\d+)?)a?\b"], searchable)
        current_source = "filename" if current else ""
        if not current:
            try:
                settings = json.loads(str(fem.get("current_settings", "[]")))
                values = [float(item["current_re"]) for item in settings if str(item.get("current_re", "")).strip()]
                if values:
                    current = f"{max(abs(value) for value in values):.12g}"
                    current_source = "FEM circuit max_abs(current_re)"
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
        current_angle = first_number([r"(?:current_angle|phase_angle|beta)[_-]?(-?\d+(?:\.\d+)?)"], searchable)
        speed = first_number([r"(?:rpm|speed)[_-]?(\d+(?:\.\d+)?)", r"(\d+(?:\.\d+)?)[_-]?rpm"], searchable)
        op_payload = compact_json({"rotor": rotor, "current": current, "current_angle": current_angle, "speed": speed})
        operating_point_id = "OP_" + hashlib.sha1(op_payload.encode("utf-8")).hexdigest()[:10] if any((rotor, current, current_angle, speed)) else ""
        inv = inventory_lookup.get(fem_path, {})
        simulation_id = "SIM_" + str(inv.get("quick_hash_sha256", ""))[:12] if inv else ""
        notes = []
        if pair["match_status"] in ("uncertain", "conflict"):
            notes.append("需要人工复核配对")
        if not operating_point_id:
            notes.append("未从文件名可靠提取工况")
        elif current_source:
            notes.append("operating_current 来源: " + current_source)
        manifest.append({
            "sample_id": pair["sample_id"], "geometry_id": geometry_id,
            "operating_point_id": operating_point_id, "simulation_id": simulation_id,
            "fem_path": fem_path, "mat_path": mat_path,
            "match_status": pair["match_status"], "match_score": pair["match_score"],
            "problem_type": fem.get("problem_type", ""), "length_unit": fem.get("length_unit", ""),
            "axial_length": fem.get("depth", ""), "material_names": fem.get("material_names", ""),
            "geometry_bbox": fem.get("geometry_bbox", ""), "operating_current": current,
            "current_angle": current_angle, "speed_rpm": speed, "rotor_position": rotor,
            "candidate_target_names": "|".join(sorted(set(candidate_lookup.get(mat_path, [])))),
            "validation_status": "pending_review" if pair["match_status"] != "confirmed" else "auto_high_confidence",
            "notes": "；".join(notes),
            "_nested_metadata": {
                "pairing_evidence": pair["matching_evidence"],
                "conflicting_evidence": pair["conflicting_evidence"],
                "fem": fem,
                "candidate_targets": sorted(set(candidate_lookup.get(mat_path, []))),
            },
        })
    write_csv(output / "dataset_manifest.csv", manifest, MANIFEST_FIELDS)
    write_json(output / "dataset_manifest.json", manifest)
    write_design_summary(output / "design_group_summary.csv", manifest)
    logger.info("Manifest: %d samples, %d exact geometry signatures", len(manifest), len(geometry_ids))
    return manifest


def write_design_summary(path: Path, manifest: list[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        groups[str(row["geometry_id"]) or "<unknown>"].append(row)
    rows = []
    for geometry_id, members in sorted(groups.items()):
        op_ids = {str(x["operating_point_id"]) for x in members if x["operating_point_id"]}
        rows.append({
            "geometry_id": geometry_id,
            "simulation_count": len(members),
            "operating_point_count": len(op_ids),
            "confirmed_pair_count": sum(x["match_status"] == "confirmed" for x in members),
            "fem_paths": "|".join(str(x["fem_path"]) for x in members),
        })
    write_csv(path, rows, ["geometry_id", "simulation_count", "operating_point_count", "confirmed_pair_count", "fem_paths"])
