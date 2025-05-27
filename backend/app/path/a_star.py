import heapq
import math
from typing import Dict, List, Optional, Tuple

import networkx as nx
from app.logger import logger

from .intersection import count_yield_directions
from .schema import Stats

# Constants
DEFAULT_YIELD_PENALTY: float = 30.0
OPTIMIZE_BY_DISTANCE: str = "distance"
OPTIMIZE_BY_TIME: str = "time"
ALLOWED_OPTIMIZATION_MODES: Tuple[str, ...] = (OPTIMIZE_BY_DISTANCE, OPTIMIZE_BY_TIME)
AVERAGE_SPEED_MS: float = 50 * 1000 / 3600  # 50 km/h in meters per second
INFINITY: float = float("inf")

Edge = Tuple[int, int, dict]


def calculate_euclidean_distance(
    graph: nx.MultiDiGraph, node_1: int, node_2: int
) -> float:
    """
    Calculate the Euclidean distance between two nodes in the graph.

    Args:
        graph: Directed multigraph containing node coordinates.
        node_1: First node index.
        node_2: Second node index.

    Returns:
        float: Euclidean distance between the nodes.

    Raises:
        KeyError: If nodes are missing 'x' or 'y' coordinates.
    """
    try:
        x1, y1 = graph.nodes[node_1]["x"], graph.nodes[node_1]["y"]
        x2, y2 = graph.nodes[node_2]["x"], graph.nodes[node_2]["y"]
        return math.hypot(x2 - x1, y2 - y1)
    except KeyError as e:
        raise KeyError(f"Node {node_1} or {node_2} missing 'x' or 'y' coordinates: {e}")


def get_edge_attributes(
    graph: nx.MultiDiGraph, from_node: int, to_node: int, edge_key: int
) -> dict:
    """
    Retrieve edge attributes with error handling.

    Args:
        graph: Directed multigraph containing edge data.
        from_node: Source node index.
        to_node: Target node index.
        edge_key: Edge key for multi-edges.

    Returns:
        dict: Edge attributes.

    Raises:
        KeyError: If the edge does not exist in the graph.
    """
    try:
        return graph[from_node][to_node][edge_key]
    except KeyError:
        raise KeyError(f"Edge ({from_node}, {to_node}, {edge_key}) not found in graph")


def get_node_edges(
    graph: nx.MultiDiGraph, node_idx: int
) -> Tuple[List[Edge], List[Edge]]:
    """
    Retrieve incoming and outgoing edges for a node.

    Args:
        graph: Directed multigraph containing edge data.
        node_idx: Node index to query.

    Returns:
        Tuple[List[Edge], List[Edge]]: Lists of incoming and outgoing edges.
    """
    incoming_edges = [(u, v, d) for u, v, d in graph.in_edges(node_idx, data=True)]
    outgoing_edges = [(u, v, d) for u, v, d in graph.out_edges(node_idx, data=True)]
    return incoming_edges, outgoing_edges


def calculate_yield_directions(
    entry_edge: Optional[Edge], incoming_edges: List[Edge], outgoing_edges: List[Edge]
) -> Dict[int, int]:
    """
    Calculate yield directions for a node, handling edge cases.

    Args:
        entry_edge: The edge used to enter the node, if any.
        incoming_edges: List of incoming edges to the node.
        outgoing_edges: List of outgoing edges from the node.

    Returns:
        Dict[int, int]: Mapping of edge OSM IDs to yield direction counts.
    """
    if not entry_edge or not incoming_edges:
        return {}
    other_incoming = [edge for edge in incoming_edges if edge != entry_edge]
    if not other_incoming:
        return {}
    try:
        return count_yield_directions(entry_edge, other_incoming, outgoing_edges)
    except Exception as e:
        logger.warning(f"Failed to compute yield directions: {e}")
        return {}


def build_path(
    predecessors: Dict[int, int],
    edge_keys: Dict[int, int],
    yield_mappings: Dict[int, Dict[int, int]],
    graph: nx.MultiDiGraph,
    target_node: int,
    yield_penalty: float,
) -> Tuple[List[int], int, float, float]:
    """
    Reconstruct the path and compute path statistics.

    Args:
        predecessors: Dictionary mapping nodes to their predecessors.
        edge_keys: Dictionary mapping nodes to edge keys.
        yield_mappings: Dictionary mapping nodes to yield direction counts.
        graph: Directed multigraph containing edge data.
        target_node: Target node index for path reconstruction.
        yield_penalty: Penalty per yield direction (in seconds or meters).

    Returns:
        Tuple[List[int], int, float, float]: Path nodes, number of yield directions,
        total distance, and total time.
    """
    path = [target_node]
    total_yields = 0
    total_distance = 0.0
    total_time = 0.0
    current_node = target_node

    while current_node in predecessors:
        next_node = current_node
        current_node = predecessors[current_node]
        path.append(current_node)
        if next_node not in edge_keys:
            continue
        edge_key = edge_keys[next_node]
        if not graph.has_edge(current_node, next_node, edge_key):
            continue
        edge_data = get_edge_attributes(graph, current_node, next_node, edge_key)
        total_distance += edge_data["length"]
        total_time += edge_data["travel_time"]
        osmid = edge_data.get("osmid")
        if osmid is not None:
            yield_counts = yield_mappings.get(current_node, {})
            total_yields += yield_counts.get(osmid, 0)
            total_time += yield_counts.get(osmid, 0) * yield_penalty

    return path[::-1], total_yields, total_distance, total_time


def initialize_costs(
    graph: nx.MultiDiGraph, start_node: int
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """
    Initialize cost dictionaries for A* algorithm.

    Args:
        graph: Directed multigraph containing nodes.
        start_node: Starting node index.

    Returns:
        Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]: Dictionaries for
        g_score, time_score, and distance_score.
    """
    g_score = {node: INFINITY for node in graph.nodes}
    time_score = {node: INFINITY for node in graph.nodes}
    distance_score = {node: INFINITY for node in graph.nodes}
    g_score[start_node] = 0.0
    time_score[start_node] = 0.0
    distance_score[start_node] = 0.0
    return g_score, time_score, distance_score


def a_star(
    graph: nx.MultiDiGraph,
    start_node: int,
    target_node: int,
    optimize_by: str = OPTIMIZE_BY_DISTANCE,
    yield_penalty: float = DEFAULT_YIELD_PENALTY,
) -> Tuple[List[int], Stats]:
    """
    Find the shortest path using A* algorithm, optimizing for distance or time.

    Args:
        graph: Directed multigraph with edge attributes 'length' and 'travel_time'.
        start_node: Starting node index.
        target_node: Target node index.
        optimize_by: Optimization mode, either 'distance' or 'time'.
        yield_penalty: Penalty per yield direction (in seconds or meters).

    Returns:
        Tuple[List[int], Stats]: List of node indices in the path and statistics
        including distance, time, and number of yield directions.

    Raises:
        ValueError: If optimize_by is not one of ALLOWED_OPTIMIZATION_MODES.
    """
    if optimize_by not in ALLOWED_OPTIMIZATION_MODES:
        raise ValueError(f"optimize_by must be one of {ALLOWED_OPTIMIZATION_MODES}")
    if start_node not in graph.nodes or target_node not in graph.nodes:
        return [], Stats(distance=INFINITY, time=INFINITY, num_yield_directions=0)

    priority_queue = [(0.0, start_node)]
    heapq.heapify(priority_queue)

    predecessors: Dict[int, int] = {}
    edge_keys: Dict[int, int] = {}
    entry_edges: Dict[int, Edge] = {}
    yield_mappings: Dict[int, Dict[int, int]] = {}
    g_score, time_score, distance_score = initialize_costs(graph, start_node)

    while priority_queue:
        f_score, current_node = heapq.heappop(priority_queue)

        if current_node == target_node:
            path, num_yields, total_distance, total_time = build_path(
                predecessors,
                edge_keys,
                yield_mappings,
                graph,
                target_node,
                yield_penalty,
            )
            return path, Stats(
                distance=total_distance,
                time=total_time,
                num_yield_directions=num_yields,
            )

        entry_edge = entry_edges.get(current_node)
        incoming_edges, outgoing_edges = get_node_edges(graph, current_node)
        yield_mappings[current_node] = calculate_yield_directions(
            entry_edge, incoming_edges, outgoing_edges
        )

        for neighbor in graph.successors(current_node):
            for key in graph[current_node][neighbor]:
                edge_data = get_edge_attributes(graph, current_node, neighbor, key)
                length = edge_data["length"]
                travel_time = edge_data["travel_time"]
                osmid = edge_data.get("osmid")
                yield_count = yield_mappings.get(current_node, {}).get(osmid, 0)

                cost = (
                    length
                    if optimize_by == OPTIMIZE_BY_DISTANCE
                    else travel_time + yield_count * yield_penalty
                )
                tentative_g = g_score[current_node] + cost
                tentative_time = (
                    time_score[current_node] + travel_time + yield_count * yield_penalty
                )
                tentative_distance = distance_score[current_node] + length

                if tentative_g < g_score[neighbor]:
                    predecessors[neighbor] = current_node
                    edge_keys[neighbor] = key
                    entry_edges[neighbor] = (current_node, neighbor, edge_data)
                    g_score[neighbor] = tentative_g
                    time_score[neighbor] = tentative_time
                    distance_score[neighbor] = tentative_distance

                    heuristic = calculate_euclidean_distance(
                        graph, neighbor, target_node
                    )
                    if optimize_by == OPTIMIZE_BY_TIME:
                        heuristic /= AVERAGE_SPEED_MS

                    f_score = tentative_g + heuristic
                    heapq.heappush(priority_queue, (f_score, neighbor))

    return [], Stats(distance=INFINITY, time=INFINITY, num_yield_directions=0)
