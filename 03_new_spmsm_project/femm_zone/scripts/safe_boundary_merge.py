"""Conservatively merge physically equivalent adjacent design cells in a FEM file.

Only the 120-cell, four-copy design-domain boundaries described by
``match_material_positions`` are touched.  Sector borders, air-gap boundaries,
and boundaries between permanent magnets with different magnetisation angles
are never selected.
"""

from __future__ import annotations

from collections import defaultdict, deque
import math
import re
from typing import Any

import numpy as np

from femm_zone.scripts import spmsm_mapping as mapping


_HEADER = re.compile(r"\[(NumPoints|NumSegments|NumArcSegments|NumBlockLabels)\]\s*=\s*(\d+)\s*\r?\n")


def _section(text: str, name: str) -> tuple[re.Match[str], list[str], int]:
    match = re.search(rf"\[{name}\]\s*=\s*(\d+)\s*\r?\n", text)
    if not match:
        raise ValueError(f"[{name}] was not found")
    count = int(match.group(1))
    lines = text[match.end():].splitlines(keepends=True)[:count]
    if len(lines) != count:
        raise ValueError(f"[{name}] is truncated")
    end = match.end() + sum(len(line) for line in lines)
    return match, lines, end


def _replace_section(text: str, name: str, lines: list[str]) -> str:
    match, _, end = _section(text, name)
    newline = "\r\n" if "\r\n" in match.group(0) else "\n"
    header = f"[{name}] = {len(lines)}{newline}"
    body = "".join(line if line.endswith(("\n", "\r")) else line + newline for line in lines)
    return text[:match.start()] + header + body + text[end:]


def _angle_delta(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def _between_angle(value: float, a: float, b: float, tolerance: float = 1e-7) -> bool:
    span = _angle_delta(b, a)
    offset = _angle_delta(value, a)
    if span < 0:
        span, offset = -span, -offset
    return -tolerance <= offset <= span + tolerance


def _label_signature(label: dict[str, Any], bit: int) -> tuple[Any, ...]:
    # Magnetisation direction is physics for PMs; it is irrelevant in Air.
    direction = round(float(label["magnetization_deg"]), 12) if bit else None
    return (int(bit), direction, round(float(label["max_area"]), 12),
            int(label["circuit_1based"]), int(label["group"]),
            round(float(label["turns"]), 12), int(label["external"]))


def merge_equivalent_design_cells(fem_text: str, bits: np.ndarray,
                                  mapping_rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Delete only shared edges whose two design cells have equal physics.

    Returns the rewritten FEM text and an audit dictionary.  The caller should
    validate the unmerged material assignment before invoking this function.
    """
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bits.shape != (mapping.GENE_COUNT,) or np.any((bits != 0) & (bits != 1)):
        raise ValueError("Expected exactly 120 binary gene values")
    if len(mapping_rows) != mapping.GENE_COUNT * mapping.COPIES_PER_GENE:
        raise ValueError("Expected the complete 120-cell, four-copy mapping")

    _, node_lines, _ = _section(fem_text, "NumPoints")
    _, segment_lines, _ = _section(fem_text, "NumSegments")
    _, arc_lines, _ = _section(fem_text, "NumArcSegments")
    _, label_lines, _ = _section(fem_text, "NumBlockLabels")
    nodes = [tuple(map(float, line.split()[:2])) for line in node_lines]
    segments = [line.split() for line in segment_lines]
    arcs = [line.split() for line in arc_lines]
    labels = mapping.parse_labels(fem_text)

    cells: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in mapping_rows:
        key = (int(row["copy_index_1based"]), int(row["angular_index_0based"]),
               int(row["radial_index_0based"]))
        cells[key] = row
    if len(cells) != mapping.GENE_COUNT * mapping.COPIES_PER_GENE:
        raise AssertionError("Design mapping contains duplicate grid cells")

    def point(index: str) -> tuple[float, float]:
        return nodes[int(index)]

    def find_arc(radius: float, theta: float) -> int:
        candidates = []
        for index, fields in enumerate(arcs):
            p0, p1 = point(fields[0]), point(fields[1])
            r0, r1 = math.hypot(*p0), math.hypot(*p1)
            if abs(r0 - radius) > 1e-7 or abs(r1 - radius) > 1e-7:
                continue
            a0, a1 = math.degrees(math.atan2(p0[1], p0[0])), math.degrees(math.atan2(p1[1], p1[0]))
            if _between_angle(theta, a0, a1) or _between_angle(theta, a1, a0):
                candidates.append(index)
        if len(candidates) != 1:
            raise AssertionError(f"Expected one radial-cell arc at r={radius}, theta={theta}; got {candidates}")
        return candidates[0]

    def find_segment(radius: float, theta: float) -> int:
        q = np.array([radius * math.cos(math.radians(theta)), radius * math.sin(math.radians(theta))])
        candidates = []
        for index, fields in enumerate(segments):
            a, b = np.array(point(fields[0])), np.array(point(fields[1]))
            ab = b - a
            if float(ab @ ab) == 0:
                continue
            fraction = float((q - a) @ ab / (ab @ ab))
            distance = float(np.linalg.norm(q - (a + np.clip(fraction, 0, 1) * ab)))
            if -1e-7 <= fraction <= 1 + 1e-7 and distance <= 1e-7:
                candidates.append(index)
        if len(candidates) != 1:
            raise AssertionError(f"Expected one angular-cell segment at r={radius}, theta={theta}; got {candidates}")
        return candidates[0]

    removed_segments: set[int] = set()
    removed_arcs: set[int] = set()
    joined: dict[int, list[tuple[tuple[int, int], tuple[int, int]]]] = defaultdict(list)
    skipped_different_direction = 0
    for copy in range(1, mapping.COPIES_PER_GENE + 1):
        for angular in range(mapping.ANGULAR_CELLS):
            for radial in range(mapping.RADIAL_CELLS):
                left = cells[(copy, angular, radial)]
                left_gene = int(left["gene_index_1based"]) - 1
                left_label = labels[int(left["label_index_1based"]) - 1]
                for da, dr, edge_kind in ((0, 1, "arc"), (1, 0, "segment")):
                    other_key = (copy, angular + da, radial + dr)
                    if other_key not in cells:
                        continue
                    right = cells[other_key]
                    right_gene = int(right["gene_index_1based"]) - 1
                    right_label = labels[int(right["label_index_1based"]) - 1]
                    same_bit = int(bits[left_gene]) == int(bits[right_gene])
                    equivalent = (_label_signature(left_label, int(bits[left_gene])) ==
                                  _label_signature(right_label, int(bits[right_gene])))
                    if same_bit and not equivalent and bits[left_gene] and (
                            left_label["magnetization_deg"] != right_label["magnetization_deg"]):
                        skipped_different_direction += 1
                    if not equivalent:
                        continue
                    if edge_kind == "arc":
                        radius = (float(left["radius_mm"]) + float(right["radius_mm"])) / 2
                        removed_arcs.add(find_arc(radius, float(left["polar_angle_deg"])))
                    else:
                        radius = float(left["radius_mm"])
                        theta = (float(left["polar_angle_deg"]) + float(right["polar_angle_deg"])) / 2
                        removed_segments.add(find_segment(radius, theta))
                    joined[copy].append(((angular, radial), (angular + da, radial + dr)))

    # One block label per newly connected component.  No component can cross a
    # sector boundary because those edges are never candidates above.
    removed_labels: set[int] = set()
    for copy in range(1, mapping.COPIES_PER_GENE + 1):
        graph: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
        for a, b in joined[copy]:
            graph[a].add(b); graph[b].add(a)
        visited: set[tuple[int, int]] = set()
        for start in graph:
            if start in visited:
                continue
            queue, component = deque([start]), []
            visited.add(start)
            while queue:
                item = queue.popleft(); component.append(item)
                for neighbor in graph[item]:
                    if neighbor not in visited:
                        visited.add(neighbor); queue.append(neighbor)
            indices = sorted(int(cells[(copy, a, r)]["label_index_1based"]) - 1 for a, r in component)
            removed_labels.update(indices[1:])

    referenced_before = {int(fields[i]) for fields in segments + arcs for i in (0, 1)}
    kept_segments = [fields for index, fields in enumerate(segments) if index not in removed_segments]
    kept_arcs = [fields for index, fields in enumerate(arcs) if index not in removed_arcs]
    referenced_after = {int(fields[i]) for fields in kept_segments + kept_arcs for i in (0, 1)}
    candidate_nodes = {int(fields[i]) for index, fields in enumerate(segments) if index in removed_segments for i in (0, 1)}
    candidate_nodes |= {int(fields[i]) for index, fields in enumerate(arcs) if index in removed_arcs for i in (0, 1)}
    removed_nodes = (candidate_nodes & referenced_before) - referenced_after
    node_map = {}
    kept_nodes = []
    for old_index, line in enumerate(node_lines):
        if old_index not in removed_nodes:
            node_map[old_index] = len(kept_nodes); kept_nodes.append(line)
    for fields in kept_segments + kept_arcs:
        fields[0], fields[1] = str(node_map[int(fields[0])]), str(node_map[int(fields[1])])

    def lines(fields_list: list[list[str]]) -> list[str]:
        return ["\t".join(fields) + "\n" for fields in fields_list]

    result = fem_text
    result = _replace_section(result, "NumPoints", kept_nodes)
    result = _replace_section(result, "NumSegments", lines(kept_segments))
    result = _replace_section(result, "NumArcSegments", lines(kept_arcs))
    result = _replace_section(result, "NumBlockLabels",
                              [line for index, line in enumerate(label_lines) if index not in removed_labels])
    audit = {"removed_segments": len(removed_segments), "removed_arc_segments": len(removed_arcs),
             "removed_block_labels": len(removed_labels), "removed_orphan_nodes": len(removed_nodes),
             "remaining_nodes": len(kept_nodes), "remaining_segments": len(kept_segments),
             "remaining_arc_segments": len(kept_arcs),
             "remaining_block_labels": len(label_lines) - len(removed_labels),
             "skipped_pm_boundaries_with_different_direction": skipped_different_direction}
    return result, audit
