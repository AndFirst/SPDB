import pickle
import networkx as nx
import osmnx as ox
import heapq
import timeit
from math import atan2, degrees, sqrt
from typing import List, Tuple, Dict
from .schema import Point


_graph = None


def load_graph(path: str):
    global _graph
    if _graph is None:
        with open(path, 'rb') as f:
            _graph = pickle.load(f)


def _a_star_distance(p1: Point, p2: Point) -> Tuple[float, List[int]]:
        start_time = timeit.default_timer()
        start_xy = (p1.lon, p1.lat)
        end_xy = (p2.lon, p2.lat)

        start_node = ox.distance.nearest_nodes(
            _graph, X=start_xy[0], Y=start_xy[1]
        )
        end_node = ox.distance.nearest_nodes(_graph, X=end_xy[0], Y=end_xy[1])

        print(
            f"Start node: {start_node}, End node: {end_node}, Start point: {start_xy}, End point: {end_xy}"
        )

        if start_node not in _graph:
            print(f"Błąd: Węzeł startowy {start_node} nie istnieje w grafie")
            return float("inf"), []
        if end_node not in _graph:
            print(f"Błąd: Węzeł końcowy {end_node} nie istnieje w grafie")
            return float("inf"), []

        def heuristic(node1: int, node2: int) -> float:
            try:
                x1, y1 = (
                    _graph.nodes[node1]["x"],
                    _graph.nodes[node1]["y"],
                )
                x2, y2 = (
                    _graph.nodes[node2]["x"],
                    _graph.nodes[node2]["y"],
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
            route, total_length = a_star(_graph, start_node, end_node)
            if not route:
                print(f"Nie znaleziono ścieżki między {start_node} a {end_node}")
                return float("inf"), []
        except KeyError as e:
            print(f"Błąd: Węzeł {e} nie znajduje się w grafie")
            return float("inf"), []

        distance = 0.0
        for u, v in zip(route[:-1], route[1:]):
            edge_data = _graph.get_edge_data(u, v)
            if not edge_data:
                print(f"Brak krawędzi między {u} a {v}")
                return float("inf"), []
            min_length_edge = min(
                edge_data.values(), key=lambda x: x.get("length", float("inf"))
            )
            distance += min_length_edge["length"]
        print(f"{timeit.default_timer() - start_time:.2f}")
        return distance, route


def compute_distance_and_path(args):
    global _graph
    i, j, p1, p2 = args
    dist, path = _a_star_distance(p1, p2)

    # Oblicz lewoskręty (jak wcześniej)
    turns = 0
    for k in range(len(path) - 2):
        try:
            x1, y1 = _graph.nodes[path[k]]["x"], _graph.nodes[path[k]]["y"]
            x2, y2 = _graph.nodes[path[k+1]]["x"], _graph.nodes[path[k+1]]["y"]
            x3, y3 = _graph.nodes[path[k+2]]["x"], _graph.nodes[path[k+2]]["y"]
            angle = (degrees(atan2(y3 - y2, x3 - x2) - atan2(y1 - y2, x1 - x2)) + 360) % 360
            if 30 <= angle <= 150:
                turns += 1
        except KeyError:
            continue

    return i, j, dist, turns, path