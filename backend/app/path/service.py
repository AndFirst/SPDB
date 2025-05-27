import math
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

NodePair = Tuple[int, int]


class PathService:
    """
    Service for computing optimal driving routes between landmarks in a city graph.

    Attributes:
        CITY_NAME: Name of the city for graph generation.
        CACHE_DIRECTORY: Directory for caching graph data.
        FULL_GRAPH_CACHE: Path to the cached full graph.
        SIMPLIFIED_GRAPH_CACHE: Path to the cached simplified graph.
        DEFAULT_NUM_PROCESSES: Number of CPU cores for parallel processing.
    """

    CITY_NAME: str = "Warsaw"
    CACHE_DIRECTORY: str = "./cache"
    AVERAGE_SPEED_MS: float = 50 * 1000 / 3600  # 50 km/h in meters per second
    DEFAULT_YIELD_PENALTY: float = 30.0
    FULL_GRAPH_CACHE: str = os.path.join(
        CACHE_DIRECTORY, f"{CITY_NAME.lower().replace(', ', '_')}_full.pkl"
    )
    SIMPLIFIED_GRAPH_CACHE: str = os.path.join(
        CACHE_DIRECTORY, f"{CITY_NAME.lower().replace(', ', '_')}_simple.pkl"
    )
    DEFAULT_NUM_PROCESSES: int = cpu_count()

    def __init__(self, num_processes: int | None = None):
        """
        Initialize the PathService with cached or newly fetched city graphs.

        Args:
            num_processes: Number of processes for parallel A* computation.
                           Defaults to DEFAULT_NUM_PROCESSES if None.
        """
        self._full_graph, self._simplified_graph = self._load_graphs()
        self.num_processes = (
            num_processes if num_processes is not None else self.DEFAULT_NUM_PROCESSES
        )

    def _load_graphs(self) -> Tuple[nx.MultiDiGraph, nx.MultiDiGraph]:
        """
        Load full and simplified city graphs from cache or fetch and cache them.

        Returns:
            Tuple[nx.MultiDiGraph, nx.MultiDiGraph]: Full and simplified city graphs.
        """
        os.makedirs(self.CACHE_DIRECTORY, exist_ok=True)

        def load_or_fetch_graph(cache_path: str, simplify: bool) -> nx.MultiDiGraph:
            logger.info(f"Loading graph from {cache_path} (simplified={simplify})")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "rb") as f:
                        graph = pickle.load(f)
                    if isinstance(graph, nx.MultiDiGraph):
                        logger.info(f"Loaded graph from {cache_path}")
                        return graph
                    logger.warning(f"Invalid cached graph at {cache_path}, refetching.")
                except Exception as e:
                    logger.error(f"Error loading graph from {cache_path}: {e}")
            logger.info(
                f"Fetching new graph for {self.CITY_NAME} (simplified={simplify})"
            )
            graph = ox.graph_from_place(
                self.CITY_NAME, network_type="drive", simplify=simplify
            )
            graph = ox.add_edge_speeds(graph)
            graph = ox.add_edge_travel_times(graph)
            graph = ox.bearing.add_edge_bearings(graph)
            logger.info(
                f"Processing graph for {self.CITY_NAME} (simplified={simplify})"
            )
            graph = self._preprocess_graph(graph)
            logger.info("Graph processed.")
            with open(cache_path, "wb") as f:
                pickle.dump(graph, f)
            logger.info(f"Graph saved to {cache_path}")
            return graph

        return load_or_fetch_graph(self.FULL_GRAPH_CACHE, False), load_or_fetch_graph(
            self.SIMPLIFIED_GRAPH_CACHE, True
        )

    def _preprocess_graph(self, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """
        Preprocess the graph by adding normalized node and edge attributes.

        Args:
            graph: Input graph to preprocess.

        Returns:
            nx.MultiDiGraph: Preprocessed graph with updated attributes.
        """
        processed_graph = nx.MultiDiGraph()
        processed_graph.graph.update(graph.graph)
        processed_graph.add_nodes_from(
            (node_id, {"x": float(data["x"]), "y": float(data["y"])})
            for node_id, data in graph.nodes(data=True)
        )
        for u, v, data in graph.edges(data=True):
            x1, y1 = graph.nodes[u]["x"], graph.nodes[u]["y"]
            x2, y2 = graph.nodes[v]["x"], graph.nodes[v]["y"]
            angle = calculate_bearing(x1, y1, x2, y2)
            osmid = (
                tuple(data["osmid"])
                if isinstance(data.get("osmid"), list)
                else (data["osmid"],)
                if not isinstance(data.get("osmid"), tuple)
                else data["osmid"]
            )
            processed_graph.add_edge(
                u,
                v,
                osmid=osmid,
                length=data.get("length", 0.0),
                travel_time=data.get("travel_time", 0.0),
                speed_kph=data.get("speed_kph", 0.0),
                angle=angle,
                highway=data.get("highway", None),
            )
        return processed_graph

    def _find_start_index(self, landmarks: List[Point]) -> int:
        """
        Find the index of the starting point in the landmarks list.

        Args:
            landmarks: List of points to search for the start point.

        Returns:
            int: Index of the start point.

        Raises:
            NoStartPointError: If no landmark has is_start=True.
        """
        start_index = next(
            (i for i, point in enumerate(landmarks) if point.is_start), None
        )
        if start_index is None:
            raise NoStartPointError("Exactly one landmark must have is_start=True")
        return start_index

    async def calculate_path(self, request: PathRequest) -> PathResponse:
        """
        Compute default and optimized routes between landmarks.

        Args:
            request: Path request containing landmarks.

        Returns:
            PathResponse: Response containing default and optimized routes.
        """
        landmarks = request.landmarks
        start_index = self._find_start_index(landmarks)
        default_route = self._build_route(landmarks, start_index, optimize_yield=False)
        optimized_route = self._build_route(landmarks, start_index, optimize_yield=True)
        return PathResponse(
            landmarks=landmarks,
            default_route=default_route,
            optimized_route=optimized_route,
        )

    def _build_route(
        self,
        landmarks: List[Point],
        start_index: int,
        optimize_yield: bool = False,
    ) -> Route:
        """
        Build a route between landmarks using A* algorithm.

        Args:
            landmarks: List of points to route through.
            start_index: Index of the starting point.
            optimize_yield: Whether to optimize for fewer yield directions.

        Returns:
            Route: Computed route with path and statistics.
        """
        stats, paths = self._compute_pairwise_stats_and_paths(
            landmarks, optimize_yield, self._simplified_graph
        )
        cost_key = "distance" if optimize_yield else "time"
        cost_matrix = np.array(
            [
                [
                    stats.get(
                        (i, j),
                        Stats(
                            distance=float("inf"),
                            time=float("inf"),
                            num_yield_directions=0,
                        ),
                    ).__getattribute__(cost_key)
                    for j in range(len(landmarks))
                ]
                for i in range(len(landmarks))
            ]
        )
        visit_order = self._compute_greedy_order(cost_matrix, start_index)
        paths, stats = self._compute_paths_from_order(
            landmarks, visit_order, optimize_yield
        )
        route_path, route_stats = self._construct_full_path(paths, stats, visit_order)
        return Route(path=route_path, stats=route_stats)

    def _compute_pairwise_stats_and_paths(
        self,
        landmarks: List[Point],
        optimize_yield: bool,
        graph: nx.MultiDiGraph,
    ) -> Tuple[Dict[NodePair, Stats], Dict[NodePair, List[int]]]:
        """
        Compute paths and stats for all valid landmark pairs in parallel.

        Args:
            landmarks: List of points to compute paths between.
            optimize_yield: Whether to optimize for fewer yield directions.
            graph: Graph to use for pathfinding.

        Returns:
            Tuple[Dict[NodePair, Stats], Dict[NodePair, List[int]]]: Stats and paths for each pair.
        """
        paths: Dict[NodePair, List[int]] = {}
        stats: Dict[NodePair, Stats] = {}
        pairs = get_valid_pairs(landmarks)
        tasks = [
            (
                i,
                j,
                landmarks[i].lon,
                landmarks[i].lat,
                landmarks[j].lon,
                landmarks[j].lat,
                optimize_yield,
                graph,
            )
            for i, j in pairs
        ]
        import time

        start_time = time.time()
        with Pool(processes=self.num_processes) as pool:
            results = pool.starmap(self._compute_path_for_pair, tasks)
        logger.info(f"Pathfinding completed in {time.time() - start_time:.2f} seconds")
        for (i, j), (path, stat) in zip(pairs, results):
            paths[(i, j)] = path
            stats[(i, j)] = stat
        return stats, paths

    def _compute_paths_from_order(
        self, landmarks: List[Point], visit_order: List[int], optimize_yield: bool
    ) -> Tuple[Dict[NodePair, List[int]], Dict[NodePair, Stats]]:
        """
        Compute paths and stats for a given visit order using the full graph.

        Args:
            landmarks: List of points to route through.
            visit_order: Order of landmarks to visit.
            optimize_yield: Whether to optimize for fewer yield directions.

        Returns:
            Tuple[Dict[NodePair, List[int]], Dict[NodePair, Stats]]: Paths and stats for ordered segments.
        """
        paths: Dict[NodePair, List[int]] = {}
        stats: Dict[NodePair, Stats] = {}
        for i, j in zip(visit_order, visit_order[1:]):
            path, stat = self._compute_path_for_pair(
                i,
                j,
                landmarks[i].lon,
                landmarks[i].lat,
                landmarks[j].lon,
                landmarks[j].lat,
                optimize_yield,
                self._full_graph,
            )
            paths[(i, j)] = path
            stats[(i, j)] = stat
        return paths, stats

    @staticmethod
    def _compute_path_for_pair(
        start_idx: int,
        end_idx: int,
        start_lon: float,
        start_lat: float,
        end_lon: float,
        end_lat: float,
        optimize_yield: bool,
        graph: nx.MultiDiGraph,
    ) -> Tuple[List[int], Stats]:
        """
        Compute the shortest path between two points using A* algorithm.

        Args:
            start_idx: Index of the start landmark.
            end_idx: Index of the end landmark.
            start_lon: Longitude of the start point.
            start_lat: Latitude of the start point.
            end_lon: Longitude of the end point.
            end_lat: Latitude of the end point.
            optimize_yield: Whether to optimize for fewer yield directions.
            graph: Graph to use for pathfinding.

        Returns:
            Tuple[List[int], Stats]: Path node indices and route statistics.
        """
        start_node = ox.nearest_nodes(graph, start_lon, start_lat)
        end_node = ox.nearest_nodes(graph, end_lon, end_lat)
        path, stats = a_star(
            graph,
            start_node,
            end_node,
            optimize_by="time" if optimize_yield else "distance",
            yield_penalty=PathService.DEFAULT_YIELD_PENALTY,
        )
        return path, stats

    def _compute_greedy_order(
        self, cost_matrix: np.ndarray, start_index: int
    ) -> List[int]:
        """
        Compute a visit order using a greedy nearest-neighbor approach.

        Args:
            cost_matrix: Matrix of costs between landmarks.
            start_index: Index of the starting landmark.

        Returns:
            List[int]: Ordered list of landmark indices.
        """
        order = [start_index]
        visited = {start_index}
        while len(order) < len(cost_matrix):
            current = order[-1]
            next_idx = np.argmin(
                [
                    cost_matrix[current, i] if i not in visited else float("inf")
                    for i in range(len(cost_matrix))
                ]
            )
            if next_idx in visited:
                break
            order.append(next_idx)
            visited.add(next_idx)
        return order

    def _construct_full_path(
        self,
        paths: Dict[NodePair, List[int]],
        stats: Dict[NodePair, Stats],
        visit_order: List[int],
    ) -> Tuple[List[Point], Stats]:
        """
        Construct the full path and aggregate statistics from ordered segments.

        Args:
            paths: Dictionary of paths between landmark pairs.
            stats: Dictionary of statistics for each path segment.
            visit_order: Order of landmarks to visit.

        Returns:
            Tuple[List[Point], Stats]: Full path as points and aggregated route statistics.
        """
        route_points: List[Point] = []
        segment_stats: List[Stats] = []
        for start_idx, end_idx in zip(visit_order, visit_order[1:]):
            segment = paths[(start_idx, end_idx)]
            route_points.extend(
                Point(
                    lat=self._full_graph.nodes[node]["y"],
                    lon=self._full_graph.nodes[node]["x"],
                    is_start=False,
                )
                for node in segment
            )
            segment_stats.append(stats[(start_idx, end_idx)])
        total_stats = Stats(
            distance=sum(s.distance for s in segment_stats)
            / 1000,  # Convert to kilometers
            time=sum(s.time for s in segment_stats) / 60,  # Convert to minutes
            num_yield_directions=sum(s.num_yield_directions for s in segment_stats),
        )
        logger.info(f"Computed route statistics: {total_stats}")
        return route_points, total_stats


class PathError(Exception):
    """Base exception for path-related errors."""


class TooFewLandmarksError(PathError):
    """Raised when fewer than two landmarks are provided."""


class NoStartPointError(PathError):
    """Raised when no start point is specified in the landmarks."""


def get_valid_pairs(landmarks: List[Point]) -> List[NodePair]:
    """
    Generate valid index pairs for path calculation between landmarks.

    Args:
        landmarks: List of points to generate pairs for.

    Returns:
        List[NodePair]: List of valid (start, end) index pairs.

    Raises:
        TooFewLandmarksError: If fewer than two landmarks are provided.
        NoStartPointError: If no landmark has is_start=True.
    """
    if len(landmarks) < 2:
        raise TooFewLandmarksError(
            "At least two landmarks are required for path calculation"
        )
    start_index = next((i for i, point in enumerate(landmarks) if point.is_start), None)
    if start_index is None:
        raise NoStartPointError("No start point found in landmarks")
    return [
        (i, j)
        for i, j in product(range(len(landmarks)), repeat=2)
        if i != j and j != start_index
    ]


def calculate_bearing(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate the bearing (angle) between two points in degrees.

    Args:
        x1: X-coordinate (longitude) of the first point.
        y1: Y-coordinate (latitude) of the first point.
        x2: X-coordinate (longitude) of the second point.
        y2: Y-coordinate (latitude) of the second point.

    Returns:
        float: Normalized bearing in degrees (-180, 180].
    """
    angle = math.degrees(math.atan2(x2 - x1, y2 - y1))
    angle = angle % 360
    if angle > 180:
        angle -= 360
    return -angle


path_service = PathService(num_processes=4)
