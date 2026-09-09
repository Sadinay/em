from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.ops import unary_union

from encoding.chromosome import Chromosome
from encoding.layout import ANGULAR_CELLS, RADIAL_CELLS, matlab_vector_to_grid
from encoding.turns import (
    DEFAULT_CIRCUIT_NAMES,
    DEFAULT_SECTOR_CIRCUIT_IDS,
    compute_turns_per_circuit,
)
from femm_runner.historical_inset_commands import (
    _delete_internal_boundaries,
    _map_theta,
    _remove_isolated_nodes,
    _xy,
)


def _component_information(
    keys: np.ndarray,
) -> tuple[set[tuple[int, int]], np.ndarray]:
    visited = np.zeros(keys.shape, dtype=bool)
    representatives: set[tuple[int, int]] = set()
    sizes = np.ones(keys.shape, dtype=int)
    for radial in range(RADIAL_CELLS):
        for angular in range(ANGULAR_CELLS):
            if visited[radial, angular] or np.isnan(keys[radial, angular]):
                continue
            representatives.add((radial, angular))
            target = float(keys[radial, angular])
            component: list[tuple[int, int]] = []
            queue: deque[tuple[int, int]] = deque([(radial, angular)])
            visited[radial, angular] = True
            while queue:
                row, column = queue.popleft()
                component.append((row, column))
                for next_row, next_column in (
                    (row - 1, column),
                    (row + 1, column),
                    (row, column - 1),
                    (row, column + 1),
                ):
                    if not (
                        0 <= next_row < RADIAL_CELLS
                        and 0 <= next_column < ANGULAR_CELLS
                    ):
                        continue
                    if visited[next_row, next_column]:
                        continue
                    if keys[next_row, next_column] == target:
                        visited[next_row, next_column] = True
                        queue.append((next_row, next_column))
            for row, column in component:
                sizes[row, column] = len(component)
    return representatives, sizes


def _copper_neighbors(grid: np.ndarray, radial: int, angular: int) -> dict[str, bool]:
    def copper(row: int, column: int) -> bool:
        return (
            0 <= row < RADIAL_CELLS
            and 0 <= column < ANGULAR_CELLS
            and bool(grid[row, column])
        )

    return {
        "in": copper(radial - 1, angular),
        "out": copper(radial + 1, angular),
        "lft": copper(radial, angular - 1),
        "rgt": copper(radial, angular + 1),
        "ul": copper(radial - 1, angular - 1),
        "ur": copper(radial - 1, angular + 1),
        "dl": copper(radial + 1, angular - 1),
        "dr": copper(radial + 1, angular + 1),
    }


def _air_band_flags(neighbors: dict[str, bool]) -> tuple[bool, ...]:
    has_in = neighbors["in"]
    has_out = neighbors["out"]
    has_left = neighbors["lft"]
    has_right = neighbors["rgt"]
    return (
        has_in,
        has_out,
        has_left,
        has_right,
        has_in and has_left and not neighbors["ul"],
        has_in and has_right and not neighbors["ur"],
        has_out and has_left and not neighbors["dl"],
        has_out and has_right and not neighbors["dr"],
    )


def _has_air_band(neighbors: dict[str, bool]) -> bool:
    flags = _air_band_flags(neighbors)
    return not all(flags[:4]) or any(flags[4:])


def _air_band_label_points(
    radial: int,
    angular: int,
    neighbors: dict[str, bool],
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    center: float,
    sector_center: float,
    mirrored: bool,
    alpha: float,
) -> tuple[tuple[float, float], ...]:
    r1, r2 = float(r_edges[radial]), float(r_edges[radial + 1])
    t1, t2 = float(theta_edges[angular]), float(theta_edges[angular + 1])
    dr, dt = alpha * (r2 - r1), alpha * (t2 - t1)
    has_in, has_out, has_left, has_right, need_ul, need_ur, need_dl, need_dr = (
        _air_band_flags(neighbors)
    )
    ri1 = r1 + (0.0 if has_in else dr)
    ri2 = r2 - (0.0 if has_out else dr)
    ti1 = t1 + (0.0 if has_left else dt)
    ti2 = t2 - (0.0 if has_right else dt)
    theta_mid = (ti1 + ti2) / 2
    radius_mid = (ri1 + ri2) / 2
    polar: list[tuple[float, float]] = []
    if not (has_in and has_out and has_left and has_right):
        if not has_in and not has_out and has_left and has_right:
            polar.extend(
                (((r1 + ri1) / 2, theta_mid), ((ri2 + r2) / 2, theta_mid))
            )
        elif not has_left and not has_right and has_in and has_out:
            polar.extend(
                ((radius_mid, (t1 + ti1) / 2), (radius_mid, (ti2 + t2) / 2))
            )
        elif not has_out:
            polar.append(((ri2 + r2) / 2, theta_mid))
        elif not has_in:
            polar.append(((r1 + ri1) / 2, theta_mid))
        elif not has_right:
            polar.append((radius_mid, (ti2 + t2) / 2))
        else:
            polar.append((radius_mid, (t1 + ti1) / 2))
    fraction = 0.25
    if need_ul:
        polar.append((r1 + fraction * dr, t1 + fraction * dt))
    if need_ur:
        polar.append((r1 + fraction * dr, t2 - fraction * dt))
    if need_dl:
        polar.append((r2 - fraction * dr, t1 + fraction * dt))
    if need_dr:
        polar.append((r2 - fraction * dr, t2 - fraction * dt))
    return tuple(
        _xy(radius, _map_theta(theta, center, sector_center, mirrored))
        for radius, theta in polar
    )


def _add_polar_edge(
    api: Any,
    first: tuple[float, float],
    second: tuple[float, float],
    *,
    center: float,
    sector_center: float,
    mirrored: bool,
    group: int,
) -> None:
    r1, t1 = first
    r2, t2 = second
    x1, y1 = _xy(r1, _map_theta(t1, center, sector_center, mirrored))
    x2, y2 = _xy(r2, _map_theta(t2, center, sector_center, mirrored))
    if abs(t1 - t2) < 1.0e-12:
        api.mi_addsegment(x1, y1, x2, y2)
        api.mi_selectsegment((x1 + x2) / 2, (y1 + y2) / 2)
        api.mi_setsegmentprop("", 0, 1, 0, group)
    elif abs(r1 - r2) < 1.0e-12:
        sweep = t2 - t1
        if mirrored:
            sweep = -sweep
        if sweep >= 0:
            api.mi_addarc(x1, y1, x2, y2, abs(sweep), 1)
        else:
            api.mi_addarc(x2, y2, x1, y1, abs(sweep), 1)
        theta_mid = _map_theta((t1 + t2) / 2, center, sector_center, mirrored)
        xm, ym = _xy(r1, theta_mid)
        api.mi_selectarcsegment(xm, ym)
        api.mi_setarcsegmentprop(1, "", 0, group)
    else:
        api.mi_addsegment(x1, y1, x2, y2)
        api.mi_selectsegment((x1 + x2) / 2, (y1 + y2) / 2)
        api.mi_setsegmentprop("", 0, 1, 0, group)
    api.mi_clearselected()


def _add_copper_boundary(
    api: Any,
    radial: int,
    angular: int,
    copper_grid: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    center: float,
    sector_center: float,
    mirrored: bool,
    alpha: float,
    group: int,
) -> None:
    deduplicated = _copper_polar_vertices(
        radial, angular, copper_grid, r_edges, theta_edges, alpha
    )
    for radius, theta in deduplicated:
        x, y = _xy(radius, _map_theta(theta, center, sector_center, mirrored))
        api.mi_addnode(x, y)
        api.mi_selectnode(x, y)
        api.mi_setnodeprop("", group)
        api.mi_clearselected()
    for index, point in enumerate(deduplicated):
        _add_polar_edge(
            api,
            point,
            deduplicated[(index + 1) % len(deduplicated)],
            center=center,
            sector_center=sector_center,
            mirrored=mirrored,
            group=group,
        )


def _copper_polar_vertices(
    radial: int,
    angular: int,
    copper_grid: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    alpha: float,
) -> list[tuple[float, float]]:
    """Return the exact V5 copper polygon in (radius, theta) coordinates."""

    r1, r2 = float(r_edges[radial]), float(r_edges[radial + 1])
    t1, t2 = float(theta_edges[angular]), float(theta_edges[angular + 1])
    dr, dt = alpha * (r2 - r1), alpha * (t2 - t1)
    neighbors = _copper_neighbors(copper_grid, radial, angular)
    has_in, has_out, has_left, has_right, need_ul, need_ur, need_dl, need_dr = (
        _air_band_flags(neighbors)
    )
    ri1 = r1 + (0.0 if has_in else dr)
    ri2 = r2 - (0.0 if has_out else dr)
    ti1 = t1 + (0.0 if has_left else dt)
    ti2 = t2 - (0.0 if has_right else dt)
    if ri2 <= ri1:
        midpoint = (r1 + r2) / 2
        ri1, ri2 = midpoint - 0.1 * (r2 - r1), midpoint + 0.1 * (r2 - r1)
    if ti2 <= ti1:
        midpoint = (t1 + t2) / 2
        ti1, ti2 = midpoint - 0.1 * (t2 - t1), midpoint + 0.1 * (t2 - t1)

    if any((need_ul, need_ur, need_dl, need_dr)):
        polar: list[tuple[float, float]] = []
        polar.append((ri1 + dr, ti1) if need_ul else (ri1, ti1))
        if need_dl:
            polar.extend(
                ((ri2 - dr, ti1), (ri2 - dr, ti1 + dt), (ri2, ti1 + dt))
            )
        else:
            polar.append((ri2, ti1))
        if need_dr:
            polar.extend(
                ((ri2, ti2 - dt), (ri2 - dr, ti2 - dt), (ri2 - dr, ti2))
            )
        else:
            polar.append((ri2, ti2))
        if need_ur:
            polar.extend(
                ((ri1 + dr, ti2), (ri1 + dr, ti2 - dt), (ri1, ti2 - dt))
            )
        else:
            polar.append((ri1, ti2))
        if need_ul:
            polar.extend(((ri1, ti1 + dt), (ri1 + dr, ti1 + dt)))
    else:
        polar = [(ri1, ti1), (ri2, ti1), (ri2, ti2), (ri1, ti2)]

    deduplicated: list[tuple[float, float]] = []
    for point in polar:
        if not deduplicated or not (
            abs(point[0] - deduplicated[-1][0]) < 1.0e-12
            and abs(point[1] - deduplicated[-1][1]) < 1.0e-12
        ):
            deduplicated.append(point)
    if len(deduplicated) > 1 and (
        abs(deduplicated[0][0] - deduplicated[-1][0]) < 1.0e-12
        and abs(deduplicated[0][1] - deduplicated[-1][1]) < 1.0e-12
    ):
        deduplicated.pop()

    return deduplicated


def _air_band_component_points(
    copper_grid: np.ndarray,
    air_cells: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    alpha: float,
) -> tuple[tuple[float, float], ...]:
    """One deterministic interior point per actual V5 air-band region.

    MATLAB V5 grouped air bands by cell adjacency.  That is not equivalent to
    connectivity of the inset polygons and can both miss regions and merge
    disconnected pockets.  Polygon difference models the geometry that FEMM
    actually sees after the selected grid boundaries are removed.
    """

    band_boxes = []
    copper_polygons = []
    for radial, angular in np.argwhere(copper_grid):
        radial_i, angular_i = int(radial), int(angular)
        vertices = _copper_polar_vertices(
            radial_i, angular_i, copper_grid, r_edges, theta_edges, alpha
        )
        copper_polygons.append(Polygon([(theta, radius) for radius, theta in vertices]))
        if air_cells[radial_i, angular_i]:
            band_boxes.append(
                box(
                    float(theta_edges[angular_i]),
                    float(r_edges[radial_i]),
                    float(theta_edges[angular_i + 1]),
                    float(r_edges[radial_i + 1]),
                )
            )
    if not band_boxes:
        return ()
    air_geometry = unary_union(band_boxes).difference(unary_union(copper_polygons))
    if isinstance(air_geometry, Polygon):
        polygons = [air_geometry]
    elif isinstance(air_geometry, MultiPolygon):
        polygons = list(air_geometry.geoms)
    elif isinstance(air_geometry, GeometryCollection):
        polygons = [item for item in air_geometry.geoms if isinstance(item, Polygon)]
    else:
        polygons = []
    points: list[tuple[float, float]] = []
    for polygon in polygons:
        if polygon.area <= 1.0e-10:
            continue
        point = polygon.representative_point()
        points.append((float(point.y), float(point.x)))
    return tuple(sorted(points, key=lambda value: (value[0], value[1])))


def apply_merged_copper_v5(
    api: Any,
    chromosome: Chromosome,
    *,
    core_group: int = 30,
    ring_group: int = 31,
    copper_geometry_group: int = 31,
    radial_inner_mm: float = 31.5,
    radial_outer_mm: float = 52.0,
    sector_span_deg: float = 15.0,
    inset_ratio: float = 0.2,
) -> None:
    """Port the actively called MATLAB V5 merged-copper geometry mode."""

    alpha = max(0.0, min(float(inset_ratio), 0.45))
    grid = matlab_vector_to_grid(chromosome)
    copper_grid = grid == 2
    background_keys = np.full(grid.shape, np.nan, dtype=float)
    background_keys[grid == 0] = 100.0
    background_keys[grid == 1] = 200.0
    copper_keys = np.where(copper_grid, 300.0, np.nan)
    background_reps, _ = _component_information(background_keys)
    copper_reps, copper_sizes = _component_information(copper_keys)
    air_cells = np.zeros(grid.shape, dtype=bool)
    for radial, angular in np.argwhere(copper_grid):
        air_cells[radial, angular] = _has_air_band(
            _copper_neighbors(copper_grid, int(radial), int(angular))
        )
    air_keys = np.where(air_cells, 350.0, np.nan)
    turns = compute_turns_per_circuit(chromosome)
    r_edges = np.linspace(radial_inner_mm, radial_outer_mm, RADIAL_CELLS + 1)
    theta_edges = np.linspace(0.0, sector_span_deg, ANGULAR_CELLS + 1)
    air_label_points = _air_band_component_points(
        copper_grid, air_cells, r_edges, theta_edges, alpha
    )
    center = float(np.mean(theta_edges[:-1] + np.diff(theta_edges) / 2))

    for group in sorted({core_group, ring_group, copper_geometry_group}):
        api.mi_clearselected()
        api.mi_selectgroup(group)
        api.mi_deleteselected()
    api.mi_clearselected()

    for sector in range(6):
        mirrored = (sector + 1) % 2 == 0
        sector_center = center + sector * sector_span_deg
        _delete_internal_boundaries(
            api,
            background_keys,
            r_edges,
            theta_edges,
            center,
            sector_center,
            mirrored,
        )
        _remove_isolated_nodes(
            api,
            background_keys,
            r_edges,
            theta_edges,
            center,
            sector_center,
            mirrored,
        )
        _delete_internal_boundaries(
            api, air_keys, r_edges, theta_edges, center, sector_center, mirrored
        )
        _remove_isolated_nodes(
            api, air_keys, r_edges, theta_edges, center, sector_center, mirrored
        )
        for radius, theta_base in air_label_points:
            x, y = _xy(
                radius, _map_theta(theta_base, center, sector_center, mirrored)
            )
            api.mi_addblocklabel(x, y)
            api.mi_selectlabel(x, y)
            api.mi_setblockprop("Air", 1, 0, "", 0, ring_group, 0)
            api.mi_clearselected()
        phase_id = DEFAULT_SECTOR_CIRCUIT_IDS[sector]
        circuit_name = DEFAULT_CIRCUIT_NAMES[phase_id - 1]
        turns_per_cell = turns.turns_per_copper_cell[phase_id - 1]

        for radial in range(RADIAL_CELLS):
            r1, r2 = float(r_edges[radial]), float(r_edges[radial + 1])
            ring_radius = (r2 - alpha * (r2 - r1) + r2) / 2
            for angular in range(ANGULAR_CELLS):
                theta_base = float(
                    (theta_edges[angular] + theta_edges[angular + 1]) / 2
                )
                theta = _map_theta(theta_base, center, sector_center, mirrored)
                code = int(grid[radial, angular])
                if code != 2 and (radial, angular) in background_reps:
                    x, y = _xy(ring_radius, theta)
                    api.mi_addblocklabel(x, y)
                    api.mi_selectlabel(x, y)
                    material = "Pure Iron" if code == 1 else "Air"
                    api.mi_setblockprop(material, 1, 0, "", 0, ring_group, 0)
                    api.mi_clearselected()
                if code == 2:
                    _add_copper_boundary(
                        api,
                        radial,
                        angular,
                        copper_grid,
                        r_edges,
                        theta_edges,
                        center,
                        sector_center,
                        mirrored,
                        alpha,
                        copper_geometry_group,
                    )

        _delete_internal_boundaries(
            api,
            copper_keys,
            r_edges,
            theta_edges,
            center,
            sector_center,
            mirrored,
        )
        _remove_isolated_nodes(
            api,
            copper_keys,
            r_edges,
            theta_edges,
            center,
            sector_center,
            mirrored,
        )
        for radial, angular in copper_reps:
            theta_base = float(
                (theta_edges[angular] + theta_edges[angular + 1]) / 2
            )
            theta = _map_theta(theta_base, center, sector_center, mirrored)
            radius = float((r_edges[radial] + r_edges[radial + 1]) / 2)
            x, y = _xy(radius, theta)
            component_turns = float(copper_sizes[radial, angular] * turns_per_cell)
            api.mi_addblocklabel(x, y)
            api.mi_selectlabel(x, y)
            api.mi_setblockprop(
                "Copper", 1, 0, circuit_name, 0, core_group, component_turns
            )
            api.mi_clearselected()
