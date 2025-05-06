import heapq
import os
import pickle
import timeit
from math import sqrt
from tracemalloc import start
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import osmnx as ox

from .schema import PathRequest, PathResponse, Point, Route, Stats


class PathService:
    CITY: str = "Warsaw, Poland"
    CACHE_DIR: str = "./cache"
    CACHE_FILE: str = os.path.join(CACHE_DIR, "warsaw_graph.pkl")  # Zmiana na .pkl

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

        # Sprawdzenie atrybutu time
        for u, v, key, data in g.edges(keys=True, data=True):
            if "length" not in data:
                print(f"Krawędź {u} -> {v} (key={key}) nie ma atrybutu 'length'")
                data["length"] = data.get("length", 0) / 10  # Domyślna prędkość 36 km/h

        try:
            with open(self.CACHE_FILE, "wb") as f:
                pickle.dump(g, f)
            print(f"Graf zapisany do bufora: {self.CACHE_FILE}")
        except Exception as e:
            print(f"Błąd podczas zapisywania grafu do bufora: {e}")

        return g

    def _greedy_order(
        self, distances: np.ndarray, start_idx: int
    ) -> Tuple[List[int], float]:
        n = distances.shape[0]
        if n < 2:
            return [start_idx], 0.0

        order = [start_idx]
        visited = {start_idx}
        total_distance = 0.0

        while len(order) < n:
            current = order[-1]
            min_dist = float("inf")
            next_node = None

            for i in range(n):
                if i not in visited and distances[current][i] < min_dist:
                    min_dist = distances[current][i]
                    next_node = i

            if next_node is None:
                break

            order.append(next_node)
            visited.add(next_node)
            total_distance += min_dist

        return order, total_distance

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

        # Składanie szczegółowej ścieżki
        detailed_path = []
        for i in range(len(order) - 1):
            start_idx, end_idx = order[i], order[i + 1]
            path_key = (
                (start_idx, end_idx) if start_idx < end_idx else (end_idx, start_idx)
            )
            path = paths.get(path_key, [])
            # Jeśli ścieżka jest odwrócona, odwróć ją
            if start_idx > end_idx:
                path = path[::-1]
            # Dodaj ścieżkę bez ostatniego węzła, aby uniknąć powtórek
            detailed_path.extend(path[:-1] if i < len(order) - 2 else path)

        points = [
            Point(
                lat=self._city_graph.nodes[node]["y"],
                lon=self._city_graph.nodes[node]["x"],
                is_start=(node == order[0]),
            )
            for node in detailed_path
        ]
        return Route(
            path=points,
            stats=Stats(
                distance=total_distance,
                time=0.0,
                left_turns=0,
            ),
        )

    def _euclidean_distance(self, p1: Point, p2: Point) -> float:
        return sqrt((p1.lat - p2.lat) ** 2 + (p1.lon - p2.lon) ** 2)

    def _get_distance_matrix(
        self, landmarks: list[Point]
    ) -> Tuple[np.ndarray, Dict[Tuple[int, int], List[int]]]:
        n = len(landmarks)
        distances = np.zeros((n, n))
        paths: Dict[Tuple[int, int], List[int]] = {}

        for i in range(n):
            for j in range(i + 1, n):
                dist, path = self._a_star_distance(landmarks[i], landmarks[j])
                distances[i][j] = dist
                distances[j][i] = dist
                paths[(i, j)] = path

        return distances, paths

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
                detailed_path=[],
            )
            return PathResponse(
                landmarks=landmarks,
                default_route=route,
                optimized_route=route,
            )

        distances, paths = self._get_distance_matrix(landmarks)
        print_distance_matrix(distances)

        best_order, _ = self._greedy_order(distances, start_idx)
        optimized_route = self._create_route(best_order, landmarks, distances, paths)

        return PathResponse(
            landmarks=landmarks,
            default_route=optimized_route,
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
