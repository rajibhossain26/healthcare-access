"""Build travel-cost matrices for the accessibility models."""

from __future__ import annotations

from typing import Hashable, Mapping, Optional

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def haversine_matrix(
    demand_lat, demand_lon, supply_lat, supply_lon, demand_ids=None, supply_ids=None
) -> pd.DataFrame:
    """Great-circle distance in km between every demand and supply point."""
    lat1 = np.radians(np.asarray(demand_lat, dtype=float))[:, None]
    lon1 = np.radians(np.asarray(demand_lon, dtype=float))[:, None]
    lat2 = np.radians(np.asarray(supply_lat, dtype=float))[None, :]
    lon2 = np.radians(np.asarray(supply_lon, dtype=float))[None, :]
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    d = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return pd.DataFrame(d, index=demand_ids, columns=supply_ids)


def euclidean_matrix(demand_xy, supply_xy, demand_ids=None, supply_ids=None) -> pd.DataFrame:
    """Straight-line distance between projected coordinates (same units)."""
    a = np.asarray(demand_xy, dtype=float)
    b = np.asarray(supply_xy, dtype=float)
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    return pd.DataFrame(d, index=demand_ids, columns=supply_ids)


def from_long(
    table: pd.DataFrame,
    origin: str = "origin",
    destination: str = "destination",
    cost: str = "cost",
) -> pd.DataFrame:
    """Pivot an origin-destination table (one row per pair) into a matrix.

    Duplicate pairs keep the smallest cost. Pairs not in the table become NaN,
    which the models treat as unreachable.
    """
    missing = {origin, destination, cost} - set(table.columns)
    if missing:
        raise KeyError(f"columns not found: {sorted(missing)}")
    return table.pivot_table(index=origin, columns=destination, values=cost, aggfunc="min")


def network_cost_matrix(
    graph,
    demand_nodes: Mapping[Hashable, Hashable],
    supply_nodes: Mapping[Hashable, Hashable],
    weight: str = "travel_time",
    cutoff: Optional[float] = None,
) -> pd.DataFrame:
    """Shortest-path travel cost on a networkx graph.

    ``demand_nodes`` and ``supply_nodes`` map location ids to the graph node
    each location snaps to. One Dijkstra search is run *backwards* from each
    provider, so directed networks (one-way streets) are handled correctly:
    the result is the cost of travelling from the demand location to the
    provider. ``cutoff`` stops each search early; pairs beyond it are NaN.
    """
    import networkx as nx

    g = graph.reverse(copy=False) if graph.is_directed() else graph
    demand_ids = list(demand_nodes)
    columns = {}
    for sid, snode in supply_nodes.items():
        lengths = nx.single_source_dijkstra_path_length(g, snode, cutoff=cutoff, weight=weight)
        columns[sid] = [lengths.get(demand_nodes[did], np.nan) for did in demand_ids]
    return pd.DataFrame(columns, index=demand_ids, columns=list(supply_nodes))


def osm_drive_cost_matrix(place: str, demand_points: pd.DataFrame, supply_points: pd.DataFrame,
                          cutoff_minutes: Optional[float] = 60.0) -> pd.DataFrame:
    """Drive-time matrix (minutes) from OpenStreetMap using OSMnx.

    ``demand_points`` and ``supply_points`` need ``lon`` and ``lat`` columns and
    are indexed by id. Requires ``pip install osmnx`` and an internet
    connection. Speeds are imputed from OSM speed limits and road types.
    """
    import osmnx as ox

    graph = ox.graph_from_place(place, network_type="drive")
    routing = getattr(ox, "routing", ox)
    graph = routing.add_edge_speeds(graph)
    graph = routing.add_edge_travel_times(graph)  # seconds, edge attribute "travel_time"

    d_nodes = ox.distance.nearest_nodes(graph, demand_points["lon"], demand_points["lat"])
    s_nodes = ox.distance.nearest_nodes(graph, supply_points["lon"], supply_points["lat"])
    cutoff_s = None if cutoff_minutes is None else cutoff_minutes * 60.0
    seconds = network_cost_matrix(
        graph,
        dict(zip(demand_points.index, d_nodes)),
        dict(zip(supply_points.index, s_nodes)),
        weight="travel_time",
        cutoff=cutoff_s,
    )
    return seconds / 60.0
