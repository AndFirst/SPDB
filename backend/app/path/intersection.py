import math
from collections import defaultdict
from typing import Dict, List, Tuple

PRIORITY_ORDER = (
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "living_street",
    "residential",
    "unclassified",
    "road",
)


def angle_diff(in_angle: float, out_angle: float) -> float:
    """Calculate the angle difference between two angles in degrees within [-180, 180].

    A negative value indicates a right turn, a positive value indicates a left turn.

    Args:
        in_angle: Input angle in degrees, must be between -180 and 180.
        out_angle: Output angle in degrees, must be between -180 and 180.

    Returns:
        The angle difference in degrees.

    Raises:
        ValueError: If angles are not within [-180, 180].
    """
    if not -180 <= in_angle <= 180:
        raise ValueError("in_angle must be between -180 and 180 degrees")
    if not -180 <= out_angle <= 180:
        raise ValueError("out_angle must be between -180 and 180 degrees")

    diff = ((out_angle - in_angle + 180) % 360) - 180
    return diff if diff != -180 else 180


def priority_order(highway: str) -> int:
    """Determine the priority of a highway based on PRIORITY_ORDER.

    Lower index indicates higher priority.

    Args:
        highway: The type of highway.

    Returns:
        The priority index, with lower values indicating higher priority.
        Returns len(PRIORITY_ORDER) for unknown highway types.
    """
    return (
        PRIORITY_ORDER.index(highway)
        if highway in PRIORITY_ORDER
        else len(PRIORITY_ORDER)
    )


def get_reversed_angle(angle: float) -> float:
    """Calculate the reversed angle for a given angle.

    Args:
        angle: The input angle in degrees.

    Returns:
        The reversed angle in degrees.
    """
    return -math.copysign(1, angle) * 180 + angle


def count_yield_directions(
    in_edge: Tuple, other_in_edges: List[Tuple], out_edges: List[Tuple]
) -> Dict[int, int]:
    """Count the number of edges to yield to for each outgoing edge.

    Args:
        in_edge: The incoming edge tuple containing angle and highway type.
        other_in_edges: List of other incoming edge tuples.
        out_edges: List of outgoing edge tuples.

    Returns:
        A dictionary mapping outgoing edge OSM IDs to the number of edges to yield to.
    """
    in_angle = in_edge[2]["angle"]
    in_priority = priority_order(in_edge[2]["highway"])

    sorted_in_edges = sorted(
        other_in_edges,
        key=lambda edge: angle_diff(in_angle, edge[2]["angle"]),
    )

    higher_priority_edges = [
        edge
        for edge in sorted_in_edges
        if priority_order(edge[2]["highway"]) < in_priority
    ]

    equal_priority_edges = [
        edge
        for edge in sorted_in_edges
        if priority_order(edge[2]["highway"]) == in_priority
    ]
    equal_priority_edges = [
        (
            edge[0],
            edge[1],
            {**edge[2], "angle": get_reversed_angle(edge[2]["angle"])},
        )
        for edge in equal_priority_edges
    ]
    equal_priority_edges.sort(key=lambda edge: angle_diff(in_angle, edge[2]["angle"]))

    out_edges_sorted = sorted(
        out_edges,
        key=lambda edge: angle_diff(in_angle, edge[2]["angle"]),
    )

    result = defaultdict(int)
    for out_edge in out_edges_sorted:
        osmid = out_edge[2]["osmid"]
        if isinstance(osmid, list):
            osmid = osmid[0]
        result[osmid] = len(higher_priority_edges)

    if out_edges_sorted:
        osmid = out_edges_sorted[0][2]["osmid"]
        if isinstance(osmid, list):
            osmid = osmid[0]
        result[osmid] = max(result[osmid] - 1, 0)

    for out_edge in out_edges_sorted:
        angle_diff_out = angle_diff(in_angle, out_edge[2]["angle"]) - 1
        edges_to_yield = [
            edge
            for edge in equal_priority_edges
            if angle_diff(in_angle, edge[2]["angle"]) < angle_diff_out
        ]
        osmid = out_edge[2]["osmid"]
        if isinstance(osmid, list):
            osmid = osmid[0]
        result[osmid] += len(edges_to_yield)

    return dict(result)
