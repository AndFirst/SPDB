from __future__ import annotations

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
    num_yield_directions: int

    def __add__(self, other: Stats) -> Stats:
        return Stats(
            distance=self.distance + other.distance,
            time=self.time + other.time,
            num_yield_directions=self.num_yield_directions + other.num_yield_directions,
        )


class Route(CamelModel):
    path: list[Point]
    stats: Stats


class PathResponse(CamelModel):
    landmarks: list[Point]
    default_route: Route
    optimized_route: Route
