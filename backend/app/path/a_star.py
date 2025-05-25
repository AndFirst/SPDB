import heapq
import math
from typing import Dict, List, Optional, Tuple

import networkx as nx

from .intersection import count_yield_directions
from .schema import Stats

DEFAULT_YIELD_PENALTY: float = 30.0
OPTIMIZE_DISTANCE: str = "distance"
OPTIMIZE_TIME: str = "time"
ALLOWED_OPTIMIZE_FOR: Tuple[str, ...] = (OPTIMIZE_DISTANCE, OPTIMIZE_TIME)
AVERAGE_SPEED_MS: float = 50 * 1000 / 3600  # 50 km/h in m/s
INF: float = float("inf")

Node = Tuple[int, int, dict]


def euclidean_distance(graph: nx.MultiDiGraph, node_1: int, node_2: int) -> float:
    """Calculate Euclidean distance between two nodes."""
    try:
        x1, y1 = graph.nodes[node_1]["x"], graph.nodes[node_1]["y"]
        x2, y2 = graph.nodes[node_2]["x"], graph.nodes[node_2]["y"]
        return math.hypot(x2 - x1, y2 - y1)
    except KeyError as e:
        raise KeyError(f"Node {node_1} or {node_2} missing 'x' or 'y' attribute: {e}")


def get_edge_data(
    graph: nx.MultiDiGraph, from_node: int, to_node: int, edge_key: int
) -> dict:
    """Safely retrieve edge data with error handling."""
    try:
        return graph[from_node][to_node][edge_key]
    except KeyError:
        raise KeyError(f"Edge ({from_node}, {to_node}, {edge_key}) not found in graph")


def get_in_and_out_edges(
    graph: nx.MultiDiGraph, node_idx: int
) -> Tuple[List[Node], List[Node]]:
    """Retrieve incoming and outgoing edges for a node."""
    in_edges = [(u, v, d) for u, v, d in graph.in_edges(node_idx, data=True)]
    out_edges = [(u, v, d) for u, v, d in graph.out_edges(node_idx, data=True)]
    return in_edges, out_edges


def compute_yield_directions(
    entry_edge: Optional[Node], in_edges: List[Node], out_edges: List[Node]
) -> Dict[int, int]:
    """Compute yield directions for a node, handling edge cases."""
    if not entry_edge or not in_edges:
        return {}
    in_edges_other = [edge for edge in in_edges if edge != entry_edge]
    if not in_edges_other:
        return {}
    try:
        return count_yield_directions(entry_edge, in_edges_other, out_edges)
    except Exception as e:
        print(f"Warning: Failed to compute yield directions: {e}")
        return {}


def reconstruct_path(
    came_from: Dict[int, int],
    edge_keys: Dict[int, int],
    yield_mappings: Dict[int, Dict[int, int]],
    graph: nx.MultiDiGraph,
    end_idx: int,
    yield_penalty: float,
) -> Tuple[List[int], int, float, float]:
    """Reconstruct path and compute total yield directions, distance, and time."""
    path = [end_idx]
    total_yield_directions = 0
    total_distance = 0.0
    total_time = 0.0
    current = end_idx

    while current in came_from:
        next_node = current
        current = came_from[current]
        path.append(current)
        if next_node not in edge_keys:
            continue
        edge_key = edge_keys[next_node]
        if not graph.has_edge(current, next_node, edge_key):
            continue
        edge_data = get_edge_data(graph, current, next_node, edge_key)
        total_distance += edge_data["length"]
        total_time += edge_data["travel_time"]
        osmid = edge_data.get("osmid")
        if osmid is not None:
            yield_mapping = yield_mappings.get(current, {})
            yield_directions = yield_mapping.get(osmid, 0)
            total_yield_directions += yield_directions
            total_time += yield_directions * yield_penalty

    return path[::-1], total_yield_directions, total_distance, total_time


def initialize_scores(
    graph: nx.MultiDiGraph, start_idx: int
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """Initialize g_score, time_score, and distance_score dictionaries."""
    g_score = {node: INF for node in graph.nodes}
    time_score = {node: INF for node in graph.nodes}
    distance_score = {node: INF for node in graph.nodes}
    g_score[start_idx] = 0.0
    time_score[start_idx] = 0.0
    distance_score[start_idx] = 0.0
    return g_score, time_score, distance_score


def a_star(
    graph: nx.MultiDiGraph,
    start_idx: int,
    end_idx: int,
    optimize_for: str = OPTIMIZE_DISTANCE,
    yield_penalty: float = DEFAULT_YIELD_PENALTY,
) -> Tuple[List[int], Stats]:
    """
    A* algorithm optimizing for distance or time with yield penalties.

    Args:
        graph: Directed multigraph with edge attributes 'length' and 'travel_time'.
        start_idx: Starting node index.
        end_idx: Ending node index.
        optimize_for: Optimization criterion, either 'distance' or 'time'.
        yield_penalty: Penalty per yield direction (in seconds or meters).

    Returns:
        Tuple of (path, Stats) where path is the node list and Stats contains
        distance, time, and num_yield_directions.
    """
    if optimize_for not in ALLOWED_OPTIMIZE_FOR:
        raise ValueError(f"optimize_for must be one of {ALLOWED_OPTIMIZE_FOR}")
    if start_idx not in graph.nodes or end_idx not in graph.nodes:
        return [], Stats(distance=INF, time=INF, num_yield_directions=0)

    open_set = [(0.0, start_idx)]
    heapq.heapify(open_set)

    came_from: Dict[int, int] = {}
    edge_keys: Dict[int, int] = {}
    entry_edges: Dict[int, Node] = {}
    yield_mappings: Dict[int, Dict[int, int]] = {}
    g_score, time_score, distance_score = initialize_scores(graph, start_idx)

    while open_set:
        f_score, current = heapq.heappop(open_set)

        if current == end_idx:
            path, num_yield_directions, total_distance, total_time = reconstruct_path(
                came_from, edge_keys, yield_mappings, graph, end_idx, yield_penalty
            )
            return path, Stats(
                distance=total_distance,
                time=total_time,
                num_yield_directions=num_yield_directions,
            )

        entry_edge = entry_edges.get(current)
        in_edges, out_edges = get_in_and_out_edges(graph, current)
        yield_mappings[current] = compute_yield_directions(
            entry_edge, in_edges, out_edges
        )

        for neighbor in graph.successors(current):
            for key in graph[current][neighbor]:
                edge_data = get_edge_data(graph, current, neighbor, key)
                length = edge_data["length"]
                travel_time = edge_data["travel_time"]
                osmid = edge_data.get("osmid")
                yield_directions = yield_mappings.get(current, {}).get(osmid, 0)

                cost = (
                    length
                    if optimize_for == OPTIMIZE_DISTANCE
                    else travel_time + yield_directions * yield_penalty
                )
                tentative_g = g_score[current] + cost
                tentative_time = (
                    time_score[current] + travel_time + yield_directions * yield_penalty
                )
                tentative_distance = distance_score[current] + length

                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    edge_keys[neighbor] = key
                    entry_edges[neighbor] = (current, neighbor, edge_data)
                    g_score[neighbor] = tentative_g
                    time_score[neighbor] = tentative_time
                    distance_score[neighbor] = tentative_distance

                    h_score = euclidean_distance(graph, neighbor, end_idx)
                    if optimize_for == OPTIMIZE_TIME:
                        h_score /= AVERAGE_SPEED_MS

                    f_score = tentative_g + h_score
                    heapq.heappush(open_set, (f_score, neighbor))

    return [], Stats(distance=INF, time=INF, num_yield_directions=0)
