import pytest
from app.path.schema import Point
from app.path.service import NoStartPointError, TooShortPathError, get_possible_pairs


def get_landmarks_mocks(num: int, start_index: int) -> list[Point]:
    return [Point(lat=i, lon=i + 1, is_start=(i == start_index)) for i in range(num)]


def test_get_possible_path_pairs_one_or_zero_points():
    points = get_landmarks_mocks(1, 0)

    with pytest.raises(TooShortPathError):
        _ = get_possible_pairs(points)

    points = []
    with pytest.raises(TooShortPathError):
        _ = get_possible_pairs(points)


def test_get_possible_path_pairs_no_start_point():
    points = get_landmarks_mocks(2, -1)

    with pytest.raises(NoStartPointError):
        _ = get_possible_pairs(points)


def test_get_possible_path_pairs_two_points():
    points = get_landmarks_mocks(2, 0)

    pairs = get_possible_pairs(points)

    assert sorted(pairs) == sorted([(0, 1)])


def test_get_possible_path_pairs_two_points_second_is_start():
    points = get_landmarks_mocks(2, 1)

    pairs = get_possible_pairs(points)

    assert sorted(pairs) == sorted([(1, 0)])


def test_get_possible_path_pairs_three_points():
    points = get_landmarks_mocks(3, 0)

    pairs = get_possible_pairs(points)

    assert sorted(pairs) == sorted([(0, 1), (0, 2), (1, 2), (2, 1)])


def test_get_possible_path_pairs_three_points_with_start_in_middle():
    points = get_landmarks_mocks(3, 1)

    pairs = get_possible_pairs(points)

    assert sorted(pairs) == sorted([(0, 2), (1, 0), (1, 2), (2, 0)])
