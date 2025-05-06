import logging
import math
import os
from typing import List, Tuple

import networkx as nx
import osmnx as ox
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Dodajemy CORS
from pydantic import BaseModel
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()

# Konfiguracja CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Zezwalamy na wszystkie źródła (dla testów, w produkcji można ograniczyć)
    allow_credentials=True,
    allow_methods=["*"],  # Zezwalamy na wszystkie metody (GET, POST, OPTIONS itp.)
    allow_headers=["*"],  # Zezwalamy na wszystkie nagłówki
)


class Point(BaseModel):
    lat: float
    lon: float


class Weights(BaseModel):
    time: float
    distance: float
    left_turns: float


class RouteRequest(BaseModel):
    city: str
    points: List[Point]
    weights: Weights


def get_city_graph(
    city: str, cache_file: str = "city_graph.graphml"
) -> nx.MultiDiGraph:
    if os.path.exists(cache_file):
        logging.info(f"Wczytywanie grafu z pliku: {cache_file}")
        G = ox.load_graphml(cache_file)
    else:
        logging.info(f"Pobieranie danych OSM dla miasta: {city}")
        stages = ["Pobieranie danych", "Przetwarzanie grafu", "Zapis do pliku"]
        for stage in tqdm(stages, desc="Postęp pobierania grafu"):
            if stage == "Pobieranie danych":
                G = ox.graph_from_place(city, network_type="drive")
            elif stage == "Przetwarzanie grafu":
                G = ox.add_edge_speeds(G)
                G = ox.add_edge_travel_times(G)
            elif stage == "Zapis do pliku":
                ox.save_graphml(G, cache_file)
    return G


def calculate_route(
    G: nx.MultiDiGraph, points: List[Tuple[float, float]], weights: dict
) -> List[int]:
    nodes = [ox.distance.nearest_nodes(G, lon, lat) for lat, lon in points]
    cost_matrix = {}
    for i, start_node in enumerate(nodes):
        for j, end_node in enumerate(nodes):
            if i != j:
                try:
                    path = ox.shortest_path(G, start_node, end_node, weight="length")
                    if path:
                        route_gdf = ox.routing.route_to_gdf(G, path)
                        distance = route_gdf["length"].sum()
                        time = route_gdf["travel_time"].sum()
                        left_turns = count_left_turns(G, path)
                        cost_matrix[(i, j)] = (
                            weights["time"] * time
                            + weights["distance"] * distance
                            + weights["left_turns"] * left_turns
                        )
                    else:
                        cost_matrix[(i, j)] = float("inf")
                except nx.NetworkXNoPath:
                    cost_matrix[(i, j)] = float("inf")
    route = nearest_neighbor_tsp(cost_matrix, len(points))
    route = two_opt(route, cost_matrix)
    return route


def count_left_turns(G: nx.MultiDiGraph, path: List[int]) -> int:
    left_turns = 0
    for i in range(len(path) - 2):
        if is_left_turn(G, path[i], path[i + 1], path[i + 2]):
            left_turns += 1
    return left_turns


def is_left_turn(G: nx.MultiDiGraph, node1: int, node2: int, node3: int) -> bool:
    coords1 = (G.nodes[node1]["y"], G.nodes[node1]["x"])
    coords2 = (G.nodes[node2]["y"], G.nodes[node2]["x"])
    coords3 = (G.nodes[node3]["y"], G.nodes[node3]["x"])
    vector1 = (coords2[0] - coords1[0], coords2[1] - coords1[1])
    vector2 = (coords3[0] - coords2[0], coords3[1] - coords2[1])
    cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    mag1 = (vector1[0] ** 2 + vector1[1] ** 2) ** 0.5
    mag2 = (vector2[0] ** 2 + vector2[1] ** 2) ** 0.5
    if mag1 * mag2 == 0:
        return False
    angle = math.degrees(math.acos(min(1.0, max(-1.0, dot_product / (mag1 * mag2)))))
    return cross_product > 0 and angle > 30


def nearest_neighbor_tsp(cost_matrix: dict, n: int) -> List[int]:
    unvisited = set(range(1, n))
    current = 0
    route = [current]
    while unvisited:
        next_node = min(
            unvisited, key=lambda x: cost_matrix.get((current, x), float("inf"))
        )
        route.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    route.append(0)
    return route


def two_opt(route: List[int], cost_matrix: dict) -> List[int]:
    best = route.copy()
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                new_cost = sum(
                    cost_matrix.get((new_route[k], new_route[k + 1]), float("inf"))
                    for k in range(len(new_route) - 1)
                )
                old_cost = sum(
                    cost_matrix.get((best[k], best[k + 1]), float("inf"))
                    for k in range(len(best) - 1)
                )
                if new_cost < old_cost:
                    best = new_route
                    improved = True
    return best


def get_full_route_coords(
    G: nx.MultiDiGraph, points: List[Tuple[float, float]], route: List[int]
) -> List[Tuple[float, float]]:
    nodes = [ox.distance.nearest_nodes(G, lon, lat) for lat, lon in points]
    full_route_coords = []

    for i in range(len(route) - 1):
        start_node = nodes[route[i]]
        end_node = nodes[route[i + 1]]
        try:
            path = ox.shortest_path(G, start_node, end_node, weight="length")
            if path:
                route_gdf = ox.routing.route_to_gdf(G, path)
                for geom in route_gdf.geometry:
                    coords = list(geom.coords)
                    if (
                        i > 0
                        and full_route_coords
                        and coords[0] == full_route_coords[-1]
                    ):
                        coords = coords[1:]
                    full_route_coords.extend(coords)
        except nx.NetworkXNoPath:
            logging.warning(f"Brak ścieżki między {start_node} i {end_node}")

    return [(lat, lon) for lon, lat in full_route_coords]


@app.post("/calculate_route/")
async def calculate_route_endpoint(request: RouteRequest):
    try:
        # Przygotowanie danych
        points = [(point.lat, point.lon) for point in request.points]
        weights = request.weights.dict()

        # Pobierz graf miasta
        cache_file = f"{request.city.lower().replace(' ', '_')}_graph.graphml"
        G = get_city_graph(request.city, cache_file)

        # Oblicz trasę
        route = calculate_route(G, points, weights)
        full_route_coords = get_full_route_coords(G, points, route)

        # Przygotuj odpowiedź
        response = {
            "route": route,
            "coordinates": [{"lat": lat, "lon": lon} for lat, lon in full_route_coords],
            "points": [
                {"lat": point.lat, "lon": point.lon} for point in request.points
            ],
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
