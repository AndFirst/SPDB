import math
import os
import pickle
import random
from itertools import product
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import osmnx as ox
from app.logger import logger

from .a_star import a_star
from .schema import PathRequest, PathResponse, Point, Route, Stats

IndexPair = Tuple[int, int]


def calculate_angle(x1, y1, x2, y2):
    angle = math.degrees(math.atan2(x2 - x1, y2 - y1))
    # Normalize to (-180, 180]
    if angle > 180:
        angle -= 360
    elif angle <= -180:
        angle += 360
    return angle * -1


class PathService:
    """Service for calculating optimal driving routes between landmarks in a city graph."""

    CITY: str = "Warsaw"
    CACHE_DIR: str = "./cache"
    CACHE_FILE: str = os.path.join(
        CACHE_DIR, f"{CITY.lower().replace(', ', '_')}_graph.pkl"
    )
    NUM_PROCESSES: int = cpu_count()  # Default to number of CPU cores

    def __init__(self, num_processes: int = None):
        """Initialize PathService with a cached or newly fetched city graph.

        Args:
            num_processes: Number of processes to use for parallel A* computation.
                          Defaults to NUM_PROCESSES if None.
        """
        self._city_graph_full, self._city_graph_simplified = self._load_city_graphs()
        self.num_processes = (
            num_processes if num_processes is not None else self.NUM_PROCESSES
        )

    def _load_city_graphs(self) -> Tuple[nx.MultiDiGraph, nx.MultiDiGraph]:
        """Load full and simplified city graphs from cache or fetch and cache them."""
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        full_path = os.path.join(self.CACHE_DIR, f"{self.CITY}_full.pkl")
        simple_path = os.path.join(self.CACHE_DIR, f"{self.CITY}_simple.pkl")

        def load_or_fetch(path, simplify):
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        g = pickle.load(f)
                    if isinstance(g, nx.MultiDiGraph):
                        return g
                    logger.warning(f"Invalid cached graph at {path}, refetching.")
                except Exception as e:
                    logger.error(f"Error loading {path}: {e}")
            g = ox.graph_from_place(self.CITY, network_type="drive", simplify=simplify)
            g = ox.add_edge_speeds(g)
            g = ox.add_edge_travel_times(g)
            g = ox.bearing.add_edge_bearings(g)
            g = self._preprocess_graph(g)
            with open(path, "wb") as f:
                pickle.dump(g, f)
            return g

        return load_or_fetch(full_path, simplify=False), load_or_fetch(
            simple_path, simplify=True
        )

    def _preprocess_graph(self, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        new_graph = nx.MultiDiGraph()
        new_graph.graph.update(graph.graph)
        new_graph.add_nodes_from(
            (node_id, {"x": float(data["x"]), "y": float(data["y"])})
            for node_id, data in graph.nodes(data=True)
        )
        for u, v, data in graph.edges(data=True):
            x1 = graph.nodes[u]["x"]
            y1 = graph.nodes[u]["y"]
            x2 = graph.nodes[v]["x"]
            y2 = graph.nodes[v]["y"]
            angle = calculate_angle(x1, y1, x2, y2)
            new_graph.add_edge(
                u,
                v,
                **{
                    "osmid": tuple(data["osmid"])
                    if isinstance(data.get("osmid"), list)
                    else (data["osmid"],)
                    if not isinstance(data.get("osmid"), tuple)
                    else data["osmid"],
                    "length": data.get("length", 0.0),
                    "travel_time": data.get("travel_time", 0.0),
                    "speed_kph": data.get("speed_kph", 0.0),
                    "angle": angle,
                    "highway": data.get("highway", None),
                },
            )
        return new_graph

    def _get_start_idx(self, landmarks: List[Point]) -> int:
        """Get the index of the start point from landmarks."""
        start_idx = next(
            (i for i, point in enumerate(landmarks) if point.is_start), None
        )
        if start_idx is None:
            raise ValueError("Exactly one landmark must have is_start=True")
        return start_idx

    async def calculate_path(self, request: PathRequest) -> PathResponse:
        """Calculate default and optimized routes between landmarks."""
        landmarks = request.landmarks
        start_idx = self._get_start_idx(landmarks)
        default_route = self._create_route(
            landmarks, start_idx, optimize_yield_directions=False
        )
        optimized_route = self._create_route(
            landmarks, start_idx, optimize_yield_directions=True
        )
        return PathResponse(
            landmarks=landmarks,
            default_route=default_route,
            optimized_route=optimized_route,
        )

    def _create_route(
        self,
        landmarks: List[Point],
        start_idx: int,
        optimize_yield_directions: bool = False,
    ) -> Route:
        stats, _ = self._compute_stats_and_paths(
            landmarks, optimize_yield_directions, self._city_graph_simplified
        )

        default_stats = Stats(
            distance=float("inf"), time=float("inf"), num_yield_directions=0
        )
        cost_key = "distance" if optimize_yield_directions else "time"
        cost_matrix = np.array(
            [
                [
                    stats.get((i, j), default_stats).__getattribute__(cost_key)
                    for j in range(len(landmarks))
                ]
                for i in range(len(landmarks))
            ]
        )
        order = self._greedy_order(cost_matrix, start_idx)

        paths, stats = self._compute_stats_and_paths_from_order(
            landmarks, order, optimize_yield_directions
        )
        path, route_stats = self._construct_path(paths, stats, order)
        return Route(path=path, stats=route_stats)

    def _compute_stats_and_paths(
        self,
        landmarks: List[Point],
        optimize_yield_directions: bool,
        graph: nx.MultiDiGraph,
    ) -> Tuple[Dict[Tuple[int, int], Stats], Dict[Tuple[int, int], List[int]]]:
        paths, stats = {}, {}
        pairs = get_possible_pairs(landmarks)
        tasks = [
            (
                i,
                j,
                landmarks[i].lon,
                landmarks[i].lat,
                landmarks[j].lon,
                landmarks[j].lat,
                optimize_yield_directions,
                graph,
            )
            for i, j in pairs
        ]
        import time

        start_time = time.time()
        with Pool(processes=self.num_processes) as pool:
            results = pool.starmap(PathService._compute_path_for_pair_static, tasks)
        end_time = time.time()
        logger.info(f"Pathfinding (simplified) in {end_time - start_time:.2f} seconds")
        for (i, j), (path, stat) in zip(pairs, results):
            paths[(i, j)] = path
            stats[(i, j)] = stat
        return stats, paths

    def _compute_stats_and_paths_from_order(
        self, landmarks: List[Point], order: List[int], optimize_yield_directions: bool
    ) -> Tuple[Dict[Tuple[int, int], List[int]], Dict[Tuple[int, int], Stats]]:
        paths = {}
        stats = {}
        for i, j in zip(order, order[1:]):
            path, stat = self._compute_path_for_pair_static(
                i,
                j,
                landmarks[i].lon,
                landmarks[i].lat,
                landmarks[j].lon,
                landmarks[j].lat,
                optimize_yield_directions,
                self._city_graph_full,
            )
            paths[(i, j)] = path
            stats[(i, j)] = stat
        return paths, stats

    @staticmethod
    def _compute_path_for_pair_static(
        i: int,
        j: int,
        from_lon: float,
        from_lat: float,
        to_lon: float,
        to_lat: float,
        optimize_yield_directions: bool,
        graph: nx.MultiDiGraph,
    ) -> Tuple[List[int], Stats]:
        from_node = ox.nearest_nodes(graph, from_lon, from_lat)
        to_node = ox.nearest_nodes(graph, to_lon, to_lat)
        path, stat = a_star(
            graph,
            from_node,
            to_node,
            optimize_for="time" if optimize_yield_directions else "distance",
            yield_penalty=30.0,
        )
        return path, stat

    def _greedy_order(self, costs: np.ndarray, start_idx: int) -> List[int]:
        """Determine visit order using greedy nearest-neighbor approach."""
        order = [start_idx]
        visited = {start_idx}
        while len(order) < len(costs):
            current = order[-1]
            next_idx = np.argmin(
                [
                    costs[current, i] if i not in visited else float("inf")
                    for i in range(len(costs))
                ]
            )
            if next_idx in visited:
                break
            order.append(next_idx)
            visited.add(next_idx)
        return order

    def _construct_path(
        self,
        paths: Dict[Tuple[int, int], List[int]],
        stats: Dict[Tuple[int, int], Stats],
        order: List[int],
    ) -> Tuple[List[Point], Stats]:
        """Construct full path and stats from ordered segments."""
        path: List[Point] = []
        stats_list: List[Stats] = []
        for start_idx, end_idx in zip(order, order[1:]):
            segment = paths[(start_idx, end_idx)]
            path.extend(
                Point(
                    lat=self._city_graph_full.nodes[node]["y"],
                    lon=self._city_graph_full.nodes[node]["x"],
                    is_start=False,
                )
                for node in segment
            )
            stats_list.append(stats[(start_idx, end_idx)])
        total_stats = Stats(
            distance=sum(s.distance for s in stats_list) / 1000,
            time=sum(s.time for s in stats_list) / 60,
            num_yield_directions=sum(s.num_yield_directions for s in stats_list),
        )
        logger.info(f"Route stats: {total_stats}")
        return path, total_stats


class PathError(Exception):
    """Base exception for path-related errors."""


class TooShortPathError(PathError):
    """Raised when fewer than two landmarks are provided."""


class NoStartPointError(PathError):
    """Raised when no start point is specified."""


def get_possible_pairs(points: List[Point]) -> List[Tuple[int, int]]:
    """Generate valid index pairs for path calculation."""
    if len(points) < 2:
        raise TooShortPathError("At least two points required")
    start_idx = next((i for i, point in enumerate(points) if point.is_start), None)
    if start_idx is None:
        raise NoStartPointError("No start point found")
    return [
        (i, j)
        for i, j in product(range(len(points)), repeat=2)
        if i != j and j != start_idx
    ]


path_service = PathService(num_processes=4)
