import heapq
import os
import pickle
import timeit
from math import atan2, degrees, sqrt
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import osmnx as ox

from .schema import PathRequest, PathResponse, Point, Route, Stats


class PathService:
    CITY: str = "Warsaw, Poland"
    CACHE_DIR: str = "./cache"
    CACHE_FILE: str = os.path.join(CACHE_DIR, "warsaw_graph.pkl")

    def __init__(self):
        self._city_graph = self._get_city_graph()

    def _get_city_graph(self) -> nx.MultiDiGraph:
        os.makedirs(self.CACHE_DIR, exist_ok=True)

        if os.path.exists(self.CACHE_FILE):
            try:
                with open(self.CACHE_FILE, "rb") as f:
                    g = pickle.load(f)
                if isinstance(g, nx.MultiDiGraph):
                    print("Graf załadowany z bufora")
                    return g
                else:
                    print(
                        "Nieprawidłowy format grafu w buforze, pobieranie nowego grafu."
                    )
            except Exception as e:
                print(f"Błąd podczas wczytywania bufora: {e}, pobieranie nowego grafu.")

        g = ox.graph_from_place(self.CITY, network_type="drive", simplify=False)
        g = ox.add_edge_speeds(g)
        g = ox.add_edge_travel_times(g)

        for u, v, key, data in g.edges(keys=True, data=True):
            if "length" not in data:
                print(f"Krawędź {u} -> {v} (key={key}) nie ma atrybutu 'length'")
                data["length"] = data.get("length", 0) / 10

        try:
            with open(self.CACHE_FILE, "wb") as f:
                pickle.dump(g, f)
            print(f"Graf zapisany do bufora: {self.CACHE_FILE}")
        except Exception as e:
            print(f"Błąd podczas zapisywania grafu do bufora: {e}")

        return g

    def _greedy_order(
        self,
        distances: np.ndarray,
        left_turns: np.ndarray,
        start_idx: int,
        optimize_left_turns: bool = False,
    ) -> Tuple[List[int], float, int]:
        n = distances.shape[0]
        if n < 2:
            return [start_idx], 0.0, 0

        order = [start_idx]
        visited = {start_idx}
        total_distance = 0.0
        total_left_turns = 0

        while len(order) < n:
            current = order[-1]
            min_score = float("inf")
            next_node = None
            next_distance = 0.0
            next_left_turns = 0

            for i in range(n):
                if i not in visited:
                    distance = distances[current][i]
                    turns = left_turns[current][i] if optimize_left_turns else 0
                    score = distance + (turns * 10000 if optimize_left_turns else 0)
                    if score < min_score:
                        min_score = score
                        next_node = i
                        next_distance = distance
                        next_left_turns = turns

            if next_node is None:
                break

            order.append(next_node)
            visited.add(next_node)
            total_distance += next_distance
            total_left_turns += next_left_turns

        return order, total_distance, total_left_turns

    def _create_route(
        self,
        order: List[int],
        landmarks: List[Point],
        distances: np.ndarray,
        paths: Dict[Tuple[int, int], List[int]],
    ) -> Route:
        route_points = [landmarks[i] for i in order]
        total_distance = sum(
            distances[order[i]][order[i + 1]] for i in range(len(order) - 1)
        )

        detailed_path = []
        for i in range(len(order) - 1):
            start_idx, end_idx = order[i], order[i + 1]
            path_key = (
                (start_idx, end_idx) if start_idx < end_idx else (end_idx, start_idx)
            )
            path = paths.get(path_key, [])
            if start_idx > end_idx:
                path = path[::-1]
            detailed_path.extend(path[:-1] if i < len(order) - 2 else path)

        # Calculate left turns and travel time
        left_turns = 0
        total_time = 0.0
        for i in range(len(detailed_path) - 2):
            n1, n2, n3 = detailed_path[i], detailed_path[i + 1], detailed_path[i + 2]
            try:
                x1, y1 = (
                    self._city_graph.nodes[n1]["x"],
                    self._city_graph.nodes[n1]["y"],
                )
                x2, y2 = (
                    self._city_graph.nodes[n2]["x"],
                    self._city_graph.nodes[n2]["y"],
                )
                x3, y3 = (
                    self._city_graph.nodes[n3]["x"],
                    self._city_graph.nodes[n3]["y"],
                )

                # Vectors: v1 = n2->n1, v2 = n2->n3
                v1 = (x1 - x2, y1 - y2)
                v2 = (x3 - x2, y3 - y2)

                # Angle between vectors using atan2
                angle = degrees(atan2(v2[1], v2[0]) - atan2(v1[1], v1[0]))
                angle = (angle + 360) % 360  # Normalize to [0, 360]
                if 30 <= angle <= 150:  # Consider as left turn
                    left_turns += 1
            except KeyError as e:
                print(f"Błąd w obliczaniu skrętu: Węzeł {e} nie ma współrzędnych")
                continue

        # Calculate total travel time
        for u, v in zip(detailed_path[:-1], detailed_path[1:]):
            edge_data = self._city_graph.get_edge_data(u, v)
            if not edge_data:
                print(f"Brak krawędzi między {u} a {v}")
                continue
            # Find the edge with minimum travel time (in case of multiple edges)
            min_time_edge = min(
                edge_data.values(), key=lambda x: x.get("travel_time", float("inf"))
            )
            travel_time = min_time_edge.get("travel_time", 0.0)
            total_time += travel_time

        points = [
            Point(
                lat=self._city_graph.nodes[node]["y"],
                lon=self._city_graph.nodes[node]["x"],
                is_start=(node == detailed_path[0]),
            )
            for node in detailed_path
        ]
        return Route(
            path=points,
            stats=Stats(
                distance=total_distance,
                time=total_time,  # Time in seconds
                left_turns=left_turns,
            ),
        )

    def _euclidean_distance(self, p1: Point, p2: Point) -> float:
        return sqrt((p1.lat - p2.lat) ** 2 + (p1.lon - p2.lon) ** 2)

    def _get_distance_matrix(
        self, landmarks: list[Point]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[int, int], List[int]]]:
        n = len(landmarks)
        distances = np.zeros((n, n))
        left_turns = np.zeros((n, n), dtype=int)
        paths: Dict[Tuple[int, int], List[int]] = {}

        for i in range(n):
            for j in range(i + 1, n):
                dist, path = self._a_star_distance(landmarks[i], landmarks[j])
                distances[i][j] = dist
                distances[j][i] = dist
                paths[(i, j)] = path
                # Calculate left turns for this path
                turns = 0
                for k in range(len(path) - 2):
                    n1, n2, n3 = path[k], path[k + 1], path[k + 2]
                    try:
                        x1, y1 = (
                            self._city_graph.nodes[n1]["x"],
                            self._city_graph.nodes[n1]["y"],
                        )
                        x2, y2 = (
                            self._city_graph.nodes[n2]["x"],
                            self._city_graph.nodes[n2]["y"],
                        )
                        x3, y3 = (
                            self._city_graph.nodes[n3]["x"],
                            self._city_graph.nodes[n3]["y"],
                        )

                        v1 = (x1 - x2, y1 - y2)
                        v2 = (x3 - x2, y3 - y2)
                        angle = degrees(atan2(v2[1], v2[0]) - atan2(v1[1], v1[0]))
                        angle = (angle + 360) % 360
                        if 30 <= angle <= 150:
                            turns += 1
                    except KeyError:
                        continue
                left_turns[i][j] = turns
                left_turns[j][i] = turns

        return distances, left_turns, paths

    def _a_star_distance(self, p1: Point, p2: Point) -> Tuple[float, List[int]]:
        start_time = timeit.default_timer()
        start_xy = (p1.lon, p1.lat)
        end_xy = (p2.lon, p2.lat)

        start_node = ox.distance.nearest_nodes(
            self._city_graph, X=start_xy[0], Y=start_xy[1]
        )
        end_node = ox.distance.nearest_nodes(self._city_graph, X=end_xy[0], Y=end_xy[1])

        print(
            f"Start node: {start_node}, End node: {end_node}, Start point: {start_xy}, End point: {end_xy}"
        )

        if start_node not in self._city_graph:
            print(f"Błąd: Węzeł startowy {start_node} nie istnieje w grafie")
            return float("inf"), []
        if end_node not in self._city_graph:
            print(f"Błąd: Węzeł końcowy {end_node} nie istnieje w grafie")
            return float("inf"), []

        def heuristic(node1: int, node2: int) -> float:
            try:
                x1, y1 = (
                    self._city_graph.nodes[node1]["x"],
                    self._city_graph.nodes[node1]["y"],
                )
                x2, y2 = (
                    self._city_graph.nodes[node2]["x"],
                    self._city_graph.nodes[node2]["y"],
                )
                distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                distance_meters = distance * 111_000
                max_speed_ms = 33.33
                return distance_meters / max_speed_ms
            except KeyError as e:
                print(f"Błąd w heurystyce: Węzeł {e} nie ma współrzędnych")
                return float("inf")

        def a_star(
            graph: nx.MultiDiGraph, start: int, goal: int
        ) -> Tuple[List[int], float]:
            open_set = [(0, start, [start], 0.0)]  # (f_score, node, path, g_score)
            closed_set = set()
            g_score: Dict[int, float] = {start: 0.0}
            f_score: Dict[int, float] = {start: heuristic(start, goal)}

            print(f"Rozpoczynam A* od {start} do {goal}")

            while open_set:
                current_f, current, current_path, current_g = heapq.heappop(open_set)

                if current == goal:
                    return current_path, current_g

                if current in closed_set:
                    continue

                closed_set.add(current)

                neighbors_found = False
                for neighbor, edges in graph[current].items():
                    neighbors_found = True
                    for edge_id, edge_data in edges.items():
                        length = edge_data.get("length", float("inf"))
                        if length == float("inf"):
                            print(
                                f"Pominięto krawędź {current} -> {neighbor}: brak atrybutu 'length'"
                            )
                            continue

                        tentative_g = current_g + length
                        if tentative_g < g_score.get(neighbor, float("inf")):
                            g_score[neighbor] = tentative_g
                            f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                            new_path = current_path + [neighbor]
                            heapq.heappush(
                                open_set,
                                (f_score[neighbor], neighbor, new_path, tentative_g),
                            )

                if not neighbors_found:
                    print(f"Węzeł {current} nie ma sąsiadów")

            print(f"Brak ścieżki z {start} do {goal}")
            return [], float("inf")

        try:
            route, total_length = a_star(self._city_graph, start_node, end_node)
            if not route:
                print(f"Nie znaleziono ścieżki między {start_node} a {end_node}")
                return float("inf"), []
        except KeyError as e:
            print(f"Błąd: Węzeł {e} nie znajduje się w grafie")
            return float("inf"), []

        distance = 0.0
        for u, v in zip(route[:-1], route[1:]):
            edge_data = self._city_graph.get_edge_data(u, v)
            if not edge_data:
                print(f"Brak krawędzi między {u} a {v}")
                return float("inf"), []
            min_length_edge = min(
                edge_data.values(), key=lambda x: x.get("length", float("inf"))
            )
            distance += min_length_edge["length"]
        print(f"{timeit.default_timer() - start_time:.2f}")
        return distance, route

    async def calculate_path(self, request: PathRequest) -> PathResponse:
        landmarks = request.landmarks
        print(f"Punkty orientacyjne: {[(p.lat, p.lon, p.is_start) for p in landmarks]}")
        start_points = [i for i, point in enumerate(landmarks) if point.is_start]
        if len(start_points) != 1:
            raise ValueError("Exactly one landmark must have is_start=True")
        start_idx = start_points[0]

        if len(landmarks) < 2:
            route = Route(
                path=[landmarks[0]],
                stats=Stats(distance=0.0, time=0.0, left_turns=0),
            )
            return PathResponse(
                landmarks=landmarks,
                default_route=route,
                optimized_route=route,
            )

        distances, left_turns, paths = self._get_distance_matrix(landmarks)
        print_distance_matrix(distances)

        # Default route: greedy based on distance only
        default_order, _, _ = self._greedy_order(
            distances, left_turns, start_idx, optimize_left_turns=False
        )
        default_route = self._create_route(default_order, landmarks, distances, paths)

        # Optimized route: greedy with left turn penalty
        optimized_order, _, _ = self._greedy_order(
            distances, left_turns, start_idx, optimize_left_turns=True
        )
        optimized_route = self._create_route(
            optimized_order, landmarks, distances, paths
        )

        return PathResponse(
            landmarks=landmarks,
            default_route=default_route,
            optimized_route=optimized_route,
        )


def print_distance_matrix(distances: np.ndarray):
    print("Macierz odległości (w formacie %6.2f):")
    n = distances.shape[0]
    for i in range(n):
        row = ""
        for j in range(n):
            row += f"{distances[i][j]:6.2f} "
        print(row)


path_service = PathService()
