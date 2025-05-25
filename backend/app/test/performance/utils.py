import json

from app.path.schema import PathRequest, Point


def load_points():
    this_path = __file__
    this_dir = this_path.rsplit("/", 1)[0]
    with open(f"{this_dir}/landmarks.json", "r") as f:
        points = json.load(f)
    return points


def get_request_data(
    points: list[dict], district: str, num_points: int = 1
) -> PathRequest:
    filtered_points = [point for point in points if point.get("district") == district][
        :num_points
    ]
    points = [
        Point(
            lon=point["lon"],
            lat=point["lat"],
            is_start=filtered_points.index(point) == 0,
        )
        for point in filtered_points
    ]
    return PathRequest(landmarks=points)


def get_one_point_per_district(points: list[dict], districts: list[str]) -> dict:
    filtered_points = []
    for district in districts:
        district_points = [
            point for point in points if point.get("district") == district
        ]
        if district_points:
            filtered_points.append(district_points[0])
    return PathRequest(
        landmarks=[
            Point(
                lon=point["lon"],
                lat=point["lat"],
                is_start=filtered_points.index(point) == 0,
            )
            for point in filtered_points
        ]
    )


def get_districts(points: list[dict]) -> list[str]:
    return list(set(point["district"] for point in points))
