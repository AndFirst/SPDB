import os
import pickle
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
        self._city_graph = self._load_city_graph()
        self.num_processes = (
            num_processes if num_processes is not None else self.NUM_PROCESSES
        )

    def _load_city_graph(self) -> nx.MultiDiGraph:
        """Load city graph from cache or fetch and cache a new one.

        Returns:
            nx.MultiDiGraph: Directed graph of the city's road network.
        """
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        if os.path.exists(self.CACHE_FILE):
            try:
                with open(self.CACHE_FILE, "rb") as f:
                    graph = pickle.load(f)
                if isinstance(graph, nx.MultiDiGraph):
                    return graph
                logger.warning("Invalid cached graph, fetching new one.")
            except Exception as e:
                logger.error(f"Error loading cached graph: {e}")
        graph = ox.graph_from_place(self.CITY, network_type="drive", simplify=False)
        graph = ox.add_edge_speeds(graph)
        graph = ox.add_edge_travel_times(graph)
        graph = ox.bearing.add_edge_bearings(graph)
        graph = self._preprocess_graph(graph)  # Fixed method call
        with open(self.CACHE_FILE, "wb") as f:
            pickle.dump(graph, f)
        return graph

    def _preprocess_graph(self, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """
        Preprocesses a directed multigraph by copying nodes and edges with specific attributes.

        Creates a new `MultiDiGraph` containing all nodes from the input graph with their
        `x` and `y` coordinates converted to floats, and all edges with a subset of attributes
        (`osmid`, `length`, `travel_time`, `speed_kph`, `angle`, `highway`).

        Args:
            graph (nx.MultiDiGraph): The input directed multigraph to preprocess.

        Returns:
            nx.MultiDiGraph: A new directed multigraph with copied nodes and edges, including
                only the specified attributes.
        """
        new_graph = nx.MultiDiGraph()
        new_graph.add_nodes_from(
            (node_id, {"x": float(data["x"]), "y": float(data["y"])})
            for node_id, data in graph.nodes(data=True)
        )
        new_graph.add_edges_from(
            (
                from_id,
                to_id,
                {
                    "osmid": data["osmid"],
                    "length": data["length"],
                    "travel_time": data["travel_time"],
                    "speed_kph": data["speed_kph"],
                    "angle": data["angle"],
                    "highway": data["highway"],
                },
            )
            for from_id, to_id, data in graph.edges(data=True)
        )
        return new_graph

    def _get_start_idx(self, landmarks: List[Point]) -> int:
        """Get the index of the start point from landmarks.

        Args:
            landmarks: List of points to search.

        Returns:
            Index of the start point.

        Raises:
            ValueError: If no start point is found or multiple start points exist.
        """
        start_idx = next(
            (i for i, point in enumerate(landmarks) if point.is_start), None
        )
        if start_idx is None:
            raise ValueError("Exactly one landmark must have is_start=True")
        return start_idx

    async def calculate_path(self, request: PathRequest) -> PathResponse:
        """
        Calculate default and optimized routes between landmarks.

        Args:
            request: PathRequest containing list of landmarks with one start point.

        Returns:
            PathResponse with default and left-turn-optimized routes.

        Raises:
            ValueError: If fewer than two landmarks or no/invalid start point.
        """
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
        """Creates a route through landmarks using A* pathfinding and greedy ordering.

        Constructs a route starting from the specified landmark, optimizing either for
        total travel time or fewer left turns, based on the provided flag. Uses A* to
        compute paths between landmark pairs and a greedy algorithm to determine the order.

        Args:
            landmarks: List of points to visit.
            start_idx: Index of the starting landmark in the landmarks list.
            optimize_yield_directions: If True, optimizes for fewer left turns; otherwise,
                optimizes for minimal travel time.

        Returns:
            Route: A Route object containing the ordered path points and route statistics.
        """
        stats, paths = self._compute_stats_and_paths(
            landmarks, optimize_yield_directions
        )

        default_stats = Stats(
            distance=float("inf"), time=float("inf"), num_yield_directions=0
        )

        cost_key = "distance" if optimize_yield_directions else "time"
        logger.info(f"Cost key: {cost_key}")
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

        # Construct final path and stats
        path, route_stats = self._construct_path(paths, stats, order)

        return Route(path=path, stats=route_stats)

    def _compute_stats_and_paths(
        self, landmarks: List[Point], optimize_yield_directions: bool
    ) -> Tuple[Dict[Tuple[int, int], Stats], Dict[Tuple[int, int], List[int]]]:
        """Compute paths and stats for all valid landmark pairs using A* pathfinding in parallel.

        Args:
            landmarks: List of points to calculate paths between.
            optimize_yield_directions: If True, optimize A* for fewer left turns.

        Returns:
            Tuple of dictionaries mapping (i,j) pairs to stats and paths.
        """
        paths: Dict[IndexPair, List[int]] = {}
        stats: Dict[IndexPair, Stats] = {}
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
            )
            for i, j in pairs
        ]
        import time

        start_time = time.time()
        with Pool(processes=self.num_processes) as pool:
            results = pool.starmap(self._compute_path_for_pair, tasks)
        end_time = time.time()
        logger.info(f"Pathfinding completed in {end_time - start_time:.2f} seconds")
        for (i, j), (path, stat) in zip(pairs, results):
            paths[(i, j)] = path
            stats[(i, j)] = stat
            logger.info(f"Path {i}->{j}: {stat}")

        return stats, paths

    def _compute_path_for_pair(
        self,
        i: int,
        j: int,
        from_lon: float,
        from_lat: float,
        to_lon: float,
        to_lat: float,
        optimize_yield_directions: bool,
    ) -> Tuple[List[int], Stats]:
        """Compute A* path for a single landmark pair.

        Args:
            i: Index of the starting landmark.
            j: Index of the ending landmark.
            from_lon: Longitude of the starting landmark.
            from_lat: Latitude of the starting landmark.
            to_lon: Longitude of the ending landmark.
            to_lat: Latitude of the ending landmark.
            optimize_yield_directions: If True, optimize for fewer left turns.

        Returns:
            Tuple of path and stats for the pair.
        """
        from_node = ox.nearest_nodes(self._city_graph, from_lon, from_lat)
        to_node = ox.nearest_nodes(self._city_graph, to_lon, to_lat)
        path, stat = a_star(
            self._city_graph,
            from_node,
            to_node,
            optimize_for="time" if optimize_yield_directions else "distance",
            yield_penalty=1.0,
        )
        return path, stat

    def _greedy_order(self, costs: np.ndarray, start_idx: int) -> List[int]:
        """Determine visit order using greedy nearest-neighbor approach.

        Args:
            costs: Matrix of costs between landmarks.
            start_idx: Index of starting landmark.

        Returns:
            List of landmark indices in visit order.
        """
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
        """Construct full path and stats from ordered segments.

        Args:
            paths: Dictionary of paths between landmark pairs.
            stats: Dictionary of stats for each path segment.
            order: List of landmark indices in visit order.

        Returns:
            Tuple of path points and aggregated route statistics.
        """
        path: List[Point] = []
        stats_list: List[Stats] = []
        for start_idx, end_idx in zip(order, order[1:]):
            segment = paths[(start_idx, end_idx)]
            path.extend(
                Point(
                    lat=self._city_graph.nodes[node]["y"],
                    lon=self._city_graph.nodes[node]["x"],
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
    """Generate valid index pairs for path calculation.

    Args:
        points: List of landmarks.

    Returns:
        List of (i,j) tuples for valid point pairs.

    Raises:
        TooShortPathError: If fewer than two points.
        NoStartPointError: If no start point found.
    """
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


path_service = PathService()
