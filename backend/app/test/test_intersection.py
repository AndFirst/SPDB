import pytest
from app.path.intersection import angle_diff, count_yield_directions


def test_angle_diff():
    assert angle_diff(0, 0) == 0
    assert angle_diff(0, 180) == 180
    assert angle_diff(0, -180) == 180
    assert angle_diff(0, 90) == 90
    assert angle_diff(0, -90) == -90
    assert angle_diff(90, 0) == -90
    assert angle_diff(-90, 0) == 90
    assert angle_diff(45, 135) == 90
    assert angle_diff(-45, -135) == -90
    assert angle_diff(45, -45) == -90
    assert angle_diff(-45, 45) == 90


def test_angle_diff_invalid_inputs():
    with pytest.raises(ValueError):
        angle_diff(-181, 0)
    with pytest.raises(ValueError):
        angle_diff(0, -181)
    with pytest.raises(ValueError):
        angle_diff(181, 0)
    with pytest.raises(ValueError):
        angle_diff(0, 181)


def get_edge(
    start_idx: int, end_idx: int, osmid: int, priority: str, angle: float
) -> tuple:
    return (start_idx, end_idx, {"osmid": osmid, "highway": priority, "angle": angle})


def test_get_num_roads_crossing_main_road_other_worse():
    in_edge = get_edge(1, 0, 1, "motorway", 0)
    other_in_edges = [
        get_edge(2, 0, 2, "trunk", 90),
        get_edge(3, 0, 3, "trunk", 180),
        get_edge(4, 0, 4, "trunk", -90),
    ]

    out_edges = [
        get_edge(0, 1, 11, "motorway", 180),
        get_edge(0, 2, 12, "trunk", -90),
        get_edge(0, 3, 13, "trunk", 0),
        get_edge(0, 4, 14, "trunk", 90),
    ]
    expected_num_roads = {
        11: 0,
        12: 0,
        13: 0,
        14: 0,
    }
    assert expected_num_roads == count_yield_directions(
        in_edge, other_in_edges, out_edges
    )


def test_get_num_roads_crossing_equal_roads():
    in_edge = get_edge(1, 0, 1, "motorway", 0)
    other_in_edges = [
        get_edge(2, 0, 2, "motorway", 90),
        get_edge(3, 0, 3, "motorway", 180),
        get_edge(4, 0, 4, "motorway", -90),
    ]

    out_edges = [
        get_edge(0, 1, 11, "motorway", 180),
        get_edge(0, 2, 12, "motorway", -90),
        get_edge(0, 3, 13, "motorway", 0),
        get_edge(0, 4, 14, "motorway", 90),
    ]

    expected_num_roads = {
        11: 3,
        12: 0,
        13: 1,
        14: 2,
    }

    assert expected_num_roads == count_yield_directions(
        in_edge, other_in_edges, out_edges
    )


def test_get_num_roads_crossing_klobucka():
    in_edge = get_edge(1, 0, 1, "trunk", 0)
    other_in_edges = [
        get_edge(2, 0, 2, "motorway", 90),
        get_edge(3, 0, 3, "trunk", 180),
        get_edge(4, 0, 4, "motorway", -90),
    ]

    out_edges = [
        get_edge(0, 1, 11, "trunk", 180),
        get_edge(0, 2, 12, "motorway", -90),
        get_edge(0, 3, 13, "trunk", 0),
        get_edge(0, 4, 14, "motorway", 90),
    ]

    expected_num_roads = {
        11: 3,
        12: 1,
        13: 2,
        14: 3,
    }

    assert expected_num_roads == count_yield_directions(
        in_edge, other_in_edges, out_edges
    )


def test_get_num_roads_crossing_T_equal():
    in_edge = get_edge(1, 0, 1, "motorway", 0)
    other_in_edges = [
        get_edge(2, 0, 2, "motorway", 180),
        get_edge(3, 0, 3, "motorway", 90),
    ]

    out_edges = [
        get_edge(0, 1, 11, "motorway", 180),
        get_edge(0, 2, 12, "motorway", 0),
        get_edge(0, 3, 13, "motorway", -90),
    ]

    expected_num_roads = {
        11: 2,
        12: 1,
        13: 0,
    }

    assert expected_num_roads == count_yield_directions(
        in_edge, other_in_edges, out_edges
    )


def test_get_num_roads_crossing_T_straight_right_turn():
    in_edge = get_edge(1, 0, 1, "motorway", 0)
    other_in_edges = [
        get_edge(2, 0, 2, "motorway", 180),
        get_edge(3, 0, 3, "trunk", 90),
    ]

    out_edges = [
        get_edge(0, 1, 11, "motorway", 180),
        get_edge(0, 2, 12, "motorway", 0),
        get_edge(0, 3, 13, "trunk", -90),
    ]

    expected_num_roads = {
        11: 1,
        12: 0,
        13: 0,
    }

    assert expected_num_roads == count_yield_directions(
        in_edge, other_in_edges, out_edges
    )


def test_get_num_roads_crossing_T_straight_left_turn():
    in_edge = get_edge(2, 0, 2, "motorway", 180)
    other_in_edges = [
        get_edge(1, 0, 1, "motorway", 0),
        get_edge(3, 0, 3, "trunk", 90),
    ]

    out_edges = [
        get_edge(0, 1, 11, "motorway", 180),
        get_edge(0, 2, 12, "motorway", 0),
        get_edge(0, 3, 13, "trunk", -90),
    ]

    expected_num_roads = {
        11: 0,
        12: 1,
        13: 1,
    }

    assert expected_num_roads == count_yield_directions(
        in_edge, other_in_edges, out_edges
    )


def test_get_num_roads_crossing_T_straight_from_minor():
    in_edge = get_edge(1, 0, 1, "trunk", 0)
    other_in_edges = [
        get_edge(2, 0, 2, "motorway", 180),
        get_edge(3, 0, 3, "motorway", 90),
    ]

    out_edges = [
        get_edge(0, 1, 11, "trunk", 180),
        get_edge(0, 2, 12, "motorway", 0),
        get_edge(0, 3, 13, "motorway", -90),
    ]

    expected_num_roads = {
        11: 2,
        12: 2,
        13: 1,
    }

    assert expected_num_roads == count_yield_directions(
        in_edge, other_in_edges, out_edges
    )


def test_get_num_roads_crossing_T_straight_from_major():
    in_edge = get_edge(2, 0, 2, "motorway", 180)
    other_in_edges = [
        get_edge(1, 0, 1, "trunk", 0),
        get_edge(3, 0, 3, "motorway", 90),
    ]

    out_edges = [
        get_edge(0, 1, 11, "trunk", 180),
        get_edge(0, 2, 12, "motorway", 0),
        get_edge(0, 3, 13, "motorway", -90),
    ]

    expected_num_roads = {
        11: 0,
        12: 1,
        13: 0,
    }

    assert expected_num_roads == count_yield_directions(
        in_edge, other_in_edges, out_edges
    )


def test_get_num_roads_crossing_T_straight_from_major_left_turn():
    in_edge = get_edge(3, 0, 3, "motorway", 90)
    other_in_edges = [
        get_edge(1, 0, 1, "trunk", 0),
        get_edge(2, 0, 2, "motorway", 180),
    ]

    out_edges = [
        get_edge(0, 1, 11, "trunk", 180),
        get_edge(0, 2, 12, "motorway", 0),
        get_edge(0, 3, 13, "motorway", -90),
    ]

    expected_num_roads = {
        11: 1,
        12: 0,
        13: 1,
    }

    assert expected_num_roads == count_yield_directions(
        in_edge, other_in_edges, out_edges
    )
