from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from shapely import contains_xy
from shapely.geometry import LineString, Point
from shapely.ops import polygonize, unary_union


AUDIT_PROJECT = Path(__file__).resolve().parents[1] / "motor_dataset_audit"
if str(AUDIT_PROJECT) not in sys.path:
    sys.path.insert(0, str(AUDIT_PROJECT))

from src.inspect_fem import decode_text, parse_fem  # noqa: E402


CHANNEL_NAMES = ["Background", "Air", "Iron", "Copper", "Magnet"]


def arc_points(
    start: tuple[float, float],
    end: tuple[float, float],
    angle_degrees: float,
    maximum_step_degrees: float = 1.0,
) -> list[tuple[float, float]]:
    theta = math.radians(angle_degrees)
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    chord = math.hypot(dx, dy)
    if chord == 0 or abs(theta) < 1e-12:
        return [start, end]
    midpoint_x, midpoint_y = (x0 + x1) / 2, (y0 + y1) / 2
    offset = chord / (2 * math.tan(theta / 2))
    normal_x, normal_y = -dy / chord, dx / chord
    center_x = midpoint_x + normal_x * offset
    center_y = midpoint_y + normal_y * offset
    radius = math.hypot(x0 - center_x, y0 - center_y)
    start_angle = math.atan2(y0 - center_y, x0 - center_x)
    segments = max(2, int(math.ceil(abs(angle_degrees) / maximum_step_degrees)))
    return [
        (
            center_x + radius * math.cos(start_angle + theta * index / segments),
            center_y + radius * math.sin(start_angle + theta * index / segments),
        )
        for index in range(segments + 1)
    ]


def material_channel(name: str) -> int:
    normalized = name.casefold()
    if "air" in normalized:
        return 1
    if any(token in normalized for token in ("iron", "steel", "m-19", "m19")):
        return 2
    if "copper" in normalized or normalized == "cu":
        return 3
    if any(token in normalized for token in ("n40", "ndfeb", "magnet")):
        return 4
    return 0


def rasterize_fem(
    fem_path: Path,
    height: int = 96,
    width: int = 96,
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 74.0, 74.0),
) -> tuple[np.ndarray, dict[str, Any]]:
    text, encoding = decode_text(fem_path)
    parsed = parse_fem(text)
    nodes = parsed["_nodes"]
    linework: list[LineString] = []
    for start, end in parsed["_segments"]:
        if 0 <= start < len(nodes) and 0 <= end < len(nodes):
            linework.append(LineString([nodes[start], nodes[end]]))
    for start, end, angle in parsed["_arcs"]:
        if 0 <= start < len(nodes) and 0 <= end < len(nodes):
            linework.append(LineString(arc_points(nodes[start], nodes[end], angle)))
    if not linework:
        raise ValueError("FEM contains no usable segments/arcs")
    faces = list(polygonize(unary_union(linework)))
    labels = [
        (Point(x, y), material)
        for x, y, material in parsed["_labels"]
        if material and material_channel(material) > 0
    ]
    assigned: list[tuple[Any, int]] = []
    ambiguous_faces = 0
    for face in faces:
        hits = [(point, material) for point, material in labels if face.covers(point)]
        if not hits:
            continue
        unique_channels = {material_channel(material) for _point, material in hits}
        if len(unique_channels) > 1:
            ambiguous_faces += 1
            representative = face.representative_point()
            point, material = min(hits, key=lambda item: representative.distance(item[0]))
        else:
            material = hits[0][1]
        assigned.append((face, material_channel(material)))
    if not assigned:
        raise ValueError("No FEM material regions could be reconstructed")
    xmin, ymin, xmax, ymax = bbox
    x_coordinates = np.linspace(xmin, xmax, width, endpoint=False) + (xmax - xmin) / (2 * width)
    y_coordinates = np.linspace(ymin, ymax, height, endpoint=False) + (ymax - ymin) / (2 * height)
    xx, yy = np.meshgrid(x_coordinates, y_coordinates)
    material_map = np.zeros((height, width), dtype=np.uint8)
    for face, channel in assigned:
        material_map[contains_xy(face, xx, yy)] = channel
    one_hot = np.stack([(material_map == channel).astype(np.float32) for channel in range(5)], axis=0)
    return one_hot, {
        "encoding": encoding,
        "node_count": len(nodes),
        "face_count": len(faces),
        "assigned_face_count": len(assigned),
        "ambiguous_face_count": ambiguous_faces,
        "bbox": list(bbox),
        "channel_names": CHANNEL_NAMES,
    }
