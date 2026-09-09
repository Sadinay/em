from __future__ import annotations

from collections import deque
import math
from typing import Any, Iterator

import numpy as np

from encoding.chromosome import Chromosome
from encoding.layout import ANGULAR_CELLS, RADIAL_CELLS, matlab_vector_to_grid
from encoding.turns import DEFAULT_CIRCUIT_NAMES, DEFAULT_SECTOR_CIRCUIT_IDS
from topology.historical_inset import decode_historical_inset


def _xy(radius: float, theta_deg: float) -> tuple[float, float]:
    theta = math.radians(theta_deg)
    return radius * math.cos(theta), radius * math.sin(theta)


def _map_theta(theta: float, center: float, sector_center: float, mirrored: bool) -> float:
    delta = theta - center
    return sector_center - delta if mirrored else sector_center + delta


def _components_representatives(keys: np.ndarray) -> set[tuple[int, int]]:
    visited = np.zeros_like(keys, dtype=bool)
    representatives: set[tuple[int, int]] = set()
    for radial in range(RADIAL_CELLS):
        for angular in range(ANGULAR_CELLS):
            if visited[radial, angular]:
                continue
            representatives.add((radial, angular))
            target = int(keys[radial, angular])
            queue: deque[tuple[int, int]] = deque([(radial, angular)])
            visited[radial, angular] = True
            while queue:
                row, column = queue.popleft()
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
                    if int(keys[next_row, next_column]) == target:
                        visited[next_row, next_column] = True
                        queue.append((next_row, next_column))
    return representatives


def _delete_selected(api: Any, selector: str, x: float, y: float) -> None:
    api.mi_clearselected()
    getattr(api, selector)(x, y)
    api.mi_deleteselected()
    api.mi_clearselected()


def _delete_internal_boundaries(
    api: Any,
    keys: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    center: float,
    sector_center: float,
    mirrored: bool,
) -> None:
    for angular in range(ANGULAR_CELLS):
        theta_mid = _map_theta(
            float((theta_edges[angular] + theta_edges[angular + 1]) / 2),
            center,
            sector_center,
            mirrored,
        )
        for radial in range(RADIAL_CELLS - 1):
            if keys[radial, angular] != keys[radial + 1, angular]:
                continue
            x, y = _xy(float(r_edges[radial + 1]), theta_mid)
            try:
                _delete_selected(api, "mi_selectarcsegment", x, y)
            except Exception:
                _delete_selected(api, "mi_selectsegment", x, y)

    for angular in range(ANGULAR_CELLS - 1):
        theta = _map_theta(
            float(theta_edges[angular + 1]), center, sector_center, mirrored
        )
        for radial in range(RADIAL_CELLS):
            if keys[radial, angular] != keys[radial, angular + 1]:
                continue
            radius = float((r_edges[radial] + r_edges[radial + 1]) / 2)
            x, y = _xy(radius, theta)
            _delete_selected(api, "mi_selectsegment", x, y)


def _remove_isolated_nodes(
    api: Any,
    keys: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    center: float,
    sector_center: float,
    mirrored: bool,
) -> None:
    for radial_edge in range(1, RADIAL_CELLS):
        for angular_edge in range(1, ANGULAR_CELLS):
            values = (
                keys[radial_edge - 1, angular_edge - 1],
                keys[radial_edge, angular_edge - 1],
                keys[radial_edge - 1, angular_edge],
                keys[radial_edge, angular_edge],
            )
            if not all(value == values[0] for value in values[1:]):
                continue
            theta = _map_theta(
                float(theta_edges[angular_edge]), center, sector_center, mirrored
            )
            x, y = _xy(float(r_edges[radial_edge]), theta)
            _delete_selected(api, "mi_selectnode", x, y)


def _inner_corners(
    radial: int,
    angular: int,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    center: float,
    sector_center: float,
    mirrored: bool,
    radial_ratio: float,
    angular_ratio: float,
) -> Iterator[tuple[float, float]]:
    r1, r2 = float(r_edges[radial]), float(r_edges[radial + 1])
    t1, t2 = float(theta_edges[angular]), float(theta_edges[angular + 1])
    dr = radial_ratio * (r2 - r1)
    dt = angular_ratio * (t2 - t1)
    ri1, ri2 = r1 + dr, r2 - dr
    ti1, ti2 = t1 + dt, t2 - dt
    mapped1 = _map_theta(ti1, center, sector_center, mirrored)
    mapped2 = _map_theta(ti2, center, sector_center, mirrored)
    for radius, theta in (
        (ri1, mapped1),
        (ri2, mapped1),
        (ri2, mapped2),
        (ri1, mapped2),
    ):
        yield _xy(radius, theta)


def apply_historical_inset(
    api: Any,
    chromosome: Chromosome,
    *,
    core_group: int = 30,
    ring_group: int = 31,
    copper_geometry_group: int = 31,
    radial_inner_mm: float = 31.5,
    radial_outer_mm: float = 52.0,
    sector_span_deg: float = 15.0,
    inset_radial_ratio: float = 0.2,
    inset_angular_ratio: float = 0.2,
) -> None:
    """Port the command-producing part of historical MATLAB inset geometry."""

    topology = decode_historical_inset(
        chromosome,
        radial_inner_mm=radial_inner_mm,
        radial_outer_mm=radial_outer_mm,
        sector_span_deg=sector_span_deg,
        inset_radial_ratio=inset_radial_ratio,
        inset_angular_ratio=inset_angular_ratio,
    )
    grid = matlab_vector_to_grid(chromosome)
    keys = np.where(grid == 1, 200, 100).astype(np.uint16)
    representatives = _components_representatives(keys)
    r_edges = np.linspace(radial_inner_mm, radial_outer_mm, RADIAL_CELLS + 1)
    theta_edges = np.linspace(0.0, sector_span_deg, ANGULAR_CELLS + 1)
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
            api, keys, r_edges, theta_edges, center, sector_center, mirrored
        )
        _remove_isolated_nodes(
            api, keys, r_edges, theta_edges, center, sector_center, mirrored
        )
        phase_id = DEFAULT_SECTOR_CIRCUIT_IDS[sector]
        circuit_name = DEFAULT_CIRCUIT_NAMES[phase_id - 1]
        turns = topology.turns_allocation.turns_per_copper_cell[phase_id - 1]

        for radial in range(RADIAL_CELLS):
            r1, r2 = float(r_edges[radial]), float(r_edges[radial + 1])
            dr = inset_radial_ratio * (r2 - r1)
            ring_radius = (r2 - dr + r2) / 2
            for angular in range(ANGULAR_CELLS):
                code = int(grid[radial, angular])
                base_theta = float(
                    (theta_edges[angular] + theta_edges[angular + 1]) / 2
                )
                theta = _map_theta(
                    base_theta, center, sector_center, mirrored
                )
                if (radial, angular) in representatives:
                    x, y = _xy(ring_radius, theta)
                    api.mi_addblocklabel(x, y)
                    api.mi_selectlabel(x, y)
                    material = "Pure Iron" if code == 1 else "Air"
                    api.mi_setblockprop(material, 1, 0, "", 0, ring_group, 0)
                    api.mi_clearselected()

                if code != 2:
                    continue
                corners = tuple(
                    _inner_corners(
                        radial,
                        angular,
                        r_edges,
                        theta_edges,
                        center,
                        sector_center,
                        mirrored,
                        inset_radial_ratio,
                        inset_angular_ratio,
                    )
                )
                for x, y in corners:
                    api.mi_addnode(x, y)
                    api.mi_selectnode(x, y)
                    api.mi_setnodeprop("", copper_geometry_group)
                    api.mi_clearselected()
                for index, (x1, y1) in enumerate(corners):
                    x2, y2 = corners[(index + 1) % len(corners)]
                    api.mi_addsegment(x1, y1, x2, y2)
                    api.mi_selectsegment((x1 + x2) / 2, (y1 + y2) / 2)
                    api.mi_setsegmentprop("", 0, 1, 0, copper_geometry_group)
                    api.mi_clearselected()
                core_radius = (r1 + r2) / 2
                x, y = _xy(core_radius, theta)
                api.mi_addblocklabel(x, y)
                api.mi_selectlabel(x, y)
                api.mi_setblockprop(
                    "Copper", 1, 0, circuit_name, 0, core_group, float(turns)
                )
                api.mi_clearselected()

