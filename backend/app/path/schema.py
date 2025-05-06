from fastapi_camelcase import CamelModel
from pydantic import ConfigDict


class Point(CamelModel):
    lat: float
    lon: float
    is_start: bool

    model_config = ConfigDict(frozen=True)


class PathRequest(CamelModel):
    landmarks: list[Point]


class Stats(CamelModel):
    distance: float
    time: float
    left_turns: int


class Route(CamelModel):
    path: list[Point]
    stats: Stats


class PathResponse(CamelModel):
    landmarks: list[Point]
    default_route: Route
    optimized_route: Route
